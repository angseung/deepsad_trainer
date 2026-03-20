"""
Deep SAD — 실제 데이터 추론 및 오분류 시각화

데이터 구조:
    TEST_ROOT/
      normal/   → y=+1
      anomaly/  → y=-1

결과:
    result/
      FP/  FN/  TP/         ← 개별 이미지
      FP_grid.png  FN_grid.png  TP_grid.png

사용법:
    CFG의 checkpoint / test_root 경로를 설정한 후 실행:
    python test.py
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import DeepSAD_ResNet50, set_seed, get_real_test_loader
from config import TestConfig, TrainConfig


# ============================================================
# OS 기본 이미지 뷰어로 열기
# ============================================================
def open_image(path: str):
    if sys.platform == "linux":
        for viewer in ["xdg-open", "eog", "feh"]:
            try:
                subprocess.Popen(
                    [viewer, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"  이미지 열기: {viewer} {path}")
                return
            except FileNotFoundError:
                continue
        print(f"  뷰어를 찾을 수 없습니다. 직접 열어주세요: {path}")
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform == "win32":
        os.startfile(path)


# ============================================================
# 이상 점수 계산
# ============================================================
@torch.no_grad()
def compute_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    c: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_scores = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)
        scores = torch.sum((z - c) ** 2, dim=1)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy()


# ============================================================
# 샘플 분류 (FP / FN / TP)
# ============================================================
def find_samples(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    rng: np.random.RandomState,
    num_tp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FP: 정상인데 이상으로 분류 (score > thr, label=+1)
    FN: 이상인데 정상으로 분류 (score < thr, label=-1)
    TP: 이상을 이상으로 올바르게 분류 (score > thr, label=-1) → num_tp 랜덤 샘플링
    """
    pred = np.where(scores > threshold, -1, 1)

    fp_idx = np.where((pred == -1) & (labels == 1))[0]
    fn_idx = np.where((pred == 1) & (labels == -1))[0]
    tp_all = np.where((pred == -1) & (labels == -1))[0]

    n_tp = min(num_tp, len(tp_all))
    tp_idx = rng.choice(tp_all, size=n_tp, replace=False) if n_tp > 0 else tp_all

    print(f"\n[샘플 분석] threshold={threshold:.4f}")
    print(f"  전체 샘플      : {len(labels):>5}개")
    print(f"  정상 샘플      : {(labels ==  1).sum():>5}개")
    print(f"  이상 샘플      : {(labels == -1).sum():>5}개")
    print(f"  FP (정상→이상) : {len(fp_idx):>5}개")
    print(f"  FN (이상→정상) : {len(fn_idx):>5}개")
    print(f"  TP (이상→이상) : {len(tp_all):>5}개 (샘플링: {n_tp}개)")

    return fp_idx, fn_idx, tp_idx


# ============================================================
# 이미지 시각화
# ============================================================
_STYLE = {
    "FP": ((255, 80, 80), (255, 180, 80), "FP: Normal -> Anomaly"),
    "FN": ((80, 80, 255), (80, 180, 255), "FN: Anomaly -> Normal"),
    "TP": ((80, 200, 80), (80, 255, 120), "TP: Anomaly -> Anomaly"),
}


def draw_result_image(
    file_path: str,
    score: float,
    threshold: float,
    sample_type: str,
    canvas_size: int = 300,
) -> np.ndarray:
    border_color, label_color, header_text = _STYLE[sample_type]

    # 원본 이미지 로드 및 리사이즈
    img = cv2.imread(file_path)
    if img is None:
        img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (canvas_size, canvas_size))

    # 테두리
    t = 6
    cv2.rectangle(
        img,
        (t // 2, t // 2),
        (canvas_size - t // 2, canvas_size - t // 2),
        border_color,
        t,
    )

    # 상단 반투명 정보 바
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_size, 62), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    cv2.putText(
        img,
        header_text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        label_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Score: {score:.4f}",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Thr  : {threshold:.4f}",
        (8, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # 파일명 (하단)
    fname = os.path.basename(file_path)
    if len(fname) > 22:
        fname = fname[:19] + "..."
    cv2.putText(
        img,
        fname,
        (8, canvas_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return img


# ============================================================
# 저장 및 그리드 출력
# ============================================================
def save_and_display(
    file_paths: list[str],
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_type: str,
    result_dir: str,
    canvas_size: int = 300,
    grid_cols: int = 5,
    max_display: int = 50,
):
    if len(indices) == 0:
        print(f"  {sample_type}: 해당 샘플 없음")
        return

    display_idx = indices[:max_display]

    img_list = []
    for i, idx in enumerate(display_idx):
        img = draw_result_image(
            file_path=file_paths[idx],
            score=float(scores[idx]),
            threshold=threshold,
            sample_type=sample_type,
            canvas_size=canvas_size,
        )
        fname = os.path.splitext(os.path.basename(file_paths[idx]))[0]
        save_path = os.path.join(
            result_dir,
            sample_type,
            f"{sample_type}_{i:04d}_{fname}_score{scores[idx]:.4f}.png",
        )
        cv2.imwrite(save_path, img)
        img_list.append(img)

    print(
        f"  {sample_type} 이미지 저장: {result_dir}/{sample_type}/ ({len(display_idx)}개)"
    )

    # 그리드 생성
    blank = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    while len(img_list) % grid_cols != 0:
        img_list.append(blank)

    n_rows = len(img_list) // grid_cols
    rows = [
        np.hstack(img_list[r * grid_cols : (r + 1) * grid_cols]) for r in range(n_rows)
    ]
    grid = np.vstack(rows)

    title_bar = np.zeros((52, grid.shape[1], 3), dtype=np.uint8)
    _, label_color, _ = _STYLE[sample_type]
    cv2.putText(
        title_bar,
        f"{sample_type} Samples  ({len(display_idx)} shown, threshold={threshold:.4f})",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        label_color,
        2,
        cv2.LINE_AA,
    )
    grid = np.vstack([title_bar, grid])

    grid_path = os.path.join(result_dir, f"{sample_type}_grid.png")
    cv2.imwrite(grid_path, grid)
    print(f"  {sample_type} 그리드 저장: {grid_path}")
    open_image(grid_path)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Deep SAD 추론 및 시각화")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="추론 설정 YAML 파일 경로 (기본값: test.yaml 이 있으면 로드, 없으면 기본값 사용)",
    )
    args = parser.parse_args()

    _cfg_file = args.config if args.config is not None else Path("test.yaml")
    cfg = TestConfig.from_yaml(_cfg_file) if _cfg_file.exists() else TestConfig()

    for _sub in ("FP", "FN", "TP"):
        os.makedirs(os.path.join(cfg.result_dir, _sub), exist_ok=True)

    set_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # 체크포인트 로드
    print(f"\n[1단계] 체크포인트 로드: {cfg.checkpoint}")
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    saved_cfg = TrainConfig(**ckpt.get("cfg", {}))
    c = ckpt["c"].to(device)
    print(f"  저장 epoch : {ckpt['epoch']}")
    print(f"  Best AUC   : {ckpt['best_auc']:.4f}")

    # 모델 복원
    model = DeepSAD_ResNet50(
        proj_dim=saved_cfg.proj_dim,
        freeze_backbone=saved_cfg.freeze_backbone,
        pretrained=False,  # 가중치를 직접 로드하므로 사전학습 불필요
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  모델 로드 완료")

    # 테스트 데이터 로드
    print("\n[2단계] 테스트 데이터 로드")
    test_loader, file_paths, test_labels = get_real_test_loader(
        test_root=cfg.test_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        img_size=cfg.img_size,
        pretrained=saved_cfg.pretrained,
    )

    # 이상 점수 계산
    print("\n[3단계] 이상 점수 계산")
    scores = compute_scores(model, test_loader, c, device)
    print(
        f"  min={scores.min():.4f}  max={scores.max():.4f}  "
        f"mean={scores.mean():.4f}  std={scores.std():.4f}"
    )

    # 임계값: 정상 샘플 score의 상위 percentile
    threshold = float(np.percentile(scores, cfg.threshold_percentile))
    print(f"  Threshold (p{cfg.threshold_percentile}): {threshold:.4f}")

    # 샘플 분류
    print("\n[4단계] 샘플 탐지")
    fp_idx, fn_idx, tp_idx = find_samples(
        scores, test_labels, threshold, rng, cfg.num_tp_samples
    )

    # FP: score 높은 순 / FN: score 낮은 순 / TP: 랜덤
    fp_idx = fp_idx[np.argsort(scores[fp_idx])[::-1]]
    fn_idx = fn_idx[np.argsort(scores[fn_idx])]

    # 시각화 및 저장
    print("\n[5단계] 시각화 및 저장")
    for idx_arr, stype in [(fp_idx, "FP"), (fn_idx, "FN"), (tp_idx, "TP")]:
        save_and_display(
            file_paths=file_paths,
            indices=idx_arr,
            scores=scores,
            threshold=threshold,
            sample_type=stype,
            result_dir=cfg.result_dir,
        )

    print(f"\n[완료] 결과 저장 경로: {cfg.result_dir}/")
    for stype in ("FP", "FN", "TP"):
        print(f"  {cfg.result_dir}/{stype}/")
        print(f"  {cfg.result_dir}/{stype}_grid.png")


if __name__ == "__main__":
    main()
