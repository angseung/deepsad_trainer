"""
Deep SAD - 오분류 샘플 + TP 샘플 시각화
- deepsad_best.pt 로드
- FP / FN / TP 샘플 탐지 및 score 시각화
- result/ 폴더에 저장
- pip install opencv-python-headless
"""

import os
import sys
import subprocess
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import (
    DeepSAD_MNIST,
    DeepSAD_ResNet50,
    set_seed,
)


# ============================================================
# 설정
# ============================================================
CFG = {
    "seed": 42,
    "normal_class": 1,
    "batch_size": 128,
    "num_workers": 4,
    "checkpoint": "deepsad_best.pt",
    "result_dir": "result",
    "img_size": 224,
    "threshold_percentile": 90,
    "num_tp_samples": 50,  # TP 랜덤 샘플 수
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 결과 저장 폴더
for sub in ("FP", "FN", "TP"):
    os.makedirs(os.path.join(CFG["result_dir"], sub), exist_ok=True)


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
# 1. 테스트 데이터로더
# ============================================================
def get_test_loader(normal_class, batch_size, num_workers, img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_imgs = mnist_test.data.float().unsqueeze(1) / 255.0
    test_labels = mnist_test.targets.numpy()
    raw_imgs = mnist_test.data.numpy()  # [N, 28, 28] uint8, 시각화용

    test_targets_sad = torch.tensor(
        np.where(test_labels == normal_class, 1, -1), dtype=torch.long
    )
    loader = DataLoader(
        DeepSAD_MNIST(test_imgs, test_targets_sad, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, raw_imgs, test_targets_sad.numpy()


# ============================================================
# 2. 이상 점수 계산
# ============================================================
@torch.no_grad()
def compute_scores(model, loader, c, device):
    model.eval()
    all_scores = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)
        scores = torch.sum((z - c) ** 2, dim=1)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy()


# ============================================================
# 3. 샘플 탐지
# ============================================================
def find_samples(scores, labels, threshold, rng, num_tp):
    """
    FP: 정상인데 이상으로 분류 (score > threshold, label=+1)
    FN: 이상인데 정상으로 분류 (score < threshold, label=-1)
    TP: 이상을 이상으로 올바르게 분류 (score > threshold, label=-1)
        → num_tp 개수만큼 랜덤 샘플링
    """
    pred = np.where(scores > threshold, -1, 1)

    fp_idx = np.where((pred == -1) & (labels == 1))[0]
    fn_idx = np.where((pred == 1) & (labels == -1))[0]
    tp_all = np.where((pred == -1) & (labels == -1))[0]

    # TP 랜덤 샘플링
    n_tp = min(num_tp, len(tp_all))
    tp_idx = rng.choice(tp_all, size=n_tp, replace=False)

    print(f"\n[샘플 분석] threshold={threshold:.4f}")
    print(f"  전체 샘플      : {len(labels):>5}개")
    print(f"  정상 샘플      : {(labels ==  1).sum():>5}개")
    print(f"  이상 샘플      : {(labels == -1).sum():>5}개")
    print(f"  FP (정상→이상) : {len(fp_idx):>5}개")
    print(f"  FN (이상→정상) : {len(fn_idx):>5}개")
    print(f"  TP (이상→이상) : {len(tp_all):>5}개 " f"(샘플링: {n_tp}개)")

    return fp_idx, fn_idx, tp_idx


# ============================================================
# 4. 이미지 시각화 생성
# ============================================================
# 샘플 유형별 색상 정의
_STYLE = {
    # (테두리 BGR, 레이블 BGR, 헤더 텍스트)
    "FP": ((255, 80, 80), (255, 180, 80), "FP: Normal -> Anomaly"),
    "FN": ((80, 80, 255), (80, 180, 255), "FN: Anomaly -> Normal"),
    "TP": ((80, 200, 80), (80, 255, 120), "TP: Anomaly -> Anomaly"),
}


def draw_result_image(
    raw_img: np.ndarray,  # [28, 28] uint8
    score: float,
    threshold: float,
    idx: int,
    sample_type: str,  # "FP" | "FN" | "TP"
    canvas_size: int = 300,
) -> np.ndarray:
    border_color, label_color, header_text = _STYLE[sample_type]

    # 28×28 → canvas_size 확대 후 BGR 변환
    img = cv2.resize(
        raw_img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 테두리
    t = 6
    cv2.rectangle(
        img,
        (t // 2, t // 2),
        (canvas_size - t // 2, canvas_size - t // 2),
        border_color,
        t,
    )

    # 상단 반투명 바
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_size, 62), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    # 헤더 텍스트 (유형)
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

    # Score
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

    # Threshold
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

    # 인덱스 (우하단)
    cv2.putText(
        img,
        f"idx={idx}",
        (canvas_size - 75, canvas_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return img


# ============================================================
# 5. 저장 및 그리드 출력 (FP / FN / TP 공용)
# ============================================================
def save_and_display(
    raw_imgs: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    sample_type: str,  # "FP" | "FN" | "TP"
    result_dir: str,
    canvas_size: int = 300,
    grid_cols: int = 5,
    max_display: int = 50,
):
    if len(indices) == 0:
        print(f"  {sample_type}: 해당 샘플 없음")
        return

    display_idx = indices[:max_display]

    # ── 개별 이미지 저장 ──────────────────────────────────────
    img_list = []
    for i, idx in enumerate(display_idx):
        img = draw_result_image(
            raw_img=raw_imgs[idx],
            score=float(scores[idx]),
            threshold=threshold,
            idx=int(idx),
            sample_type=sample_type,
            canvas_size=canvas_size,
        )
        save_path = os.path.join(
            result_dir,
            sample_type,
            f"{sample_type}_{i:04d}_idx{idx}_score{scores[idx]:.4f}.png",
        )
        cv2.imwrite(save_path, img)
        img_list.append(img)

    print(
        f"  {sample_type} 이미지 저장: "
        f"{result_dir}/{sample_type}/ ({len(display_idx)}개)"
    )

    # ── 그리드 생성 ───────────────────────────────────────────
    blank = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    while len(img_list) % grid_cols != 0:
        img_list.append(blank)

    n_rows = len(img_list) // grid_cols
    rows = [
        np.hstack(img_list[r * grid_cols : (r + 1) * grid_cols]) for r in range(n_rows)
    ]
    grid = np.vstack(rows)

    # 타이틀 바
    title_bar = np.zeros((52, grid.shape[1], 3), dtype=np.uint8)
    _, label_color, _ = _STYLE[sample_type]
    cv2.putText(
        title_bar,
        f"{sample_type} Samples  "
        f"({len(display_idx)} shown, threshold={threshold:.4f})",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        label_color,
        2,
        cv2.LINE_AA,
    )
    grid = np.vstack([title_bar, grid])

    # 그리드 저장
    grid_path = os.path.join(result_dir, f"{sample_type}_grid.png")
    cv2.imwrite(grid_path, grid)
    print(f"  {sample_type} 그리드 저장: {grid_path}")

    # OS 기본 뷰어로 열기
    open_image(grid_path)


# ============================================================
# 6. 메인
# ============================================================
def main():
    set_seed(CFG["seed"])
    rng = np.random.RandomState(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    # ── 체크포인트 로드 ───────────────────────────────────────
    print(f"\n[1단계] 체크포인트 로드: {CFG['checkpoint']}")
    ckpt = torch.load(CFG["checkpoint"], map_location=device, weights_only=True)
    saved_cfg = ckpt["cfg"]
    c = ckpt["c"].to(device)
    print(f"  저장 epoch  : {ckpt['epoch']}")
    print(f"  Best AUC    : {ckpt['best_auc']:.4f}")

    # ── 모델 로드 ─────────────────────────────────────────────
    model = DeepSAD_ResNet50(
        proj_dim=saved_cfg["proj_dim"],
        freeze_backbone=saved_cfg["freeze_backbone"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  모델 로드 완료")

    # ── 테스트 데이터 ─────────────────────────────────────────
    print("\n[2단계] 테스트 데이터 로드")
    test_loader, raw_imgs, test_labels = get_test_loader(
        normal_class=CFG["normal_class"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        img_size=CFG["img_size"],
    )

    # ── 이상 점수 계산 ────────────────────────────────────────
    print("\n[3단계] 이상 점수 계산")
    scores = compute_scores(model, test_loader, c, device)
    print(
        f"  min={scores.min():.4f}  max={scores.max():.4f}  "
        f"mean={scores.mean():.4f}  std={scores.std():.4f}"
    )

    # ── 임계값 설정 ───────────────────────────────────────────
    threshold = float(np.percentile(scores, CFG["threshold_percentile"]))
    print(f"  Threshold (p{CFG['threshold_percentile']}): {threshold:.4f}")

    # ── 샘플 탐지 ─────────────────────────────────────────────
    print("\n[4단계] 샘플 탐지")
    fp_idx, fn_idx, tp_idx = find_samples(
        scores, test_labels, threshold, rng, CFG["num_tp_samples"]
    )

    # FP / FN: score 기준 정렬
    fp_idx = fp_idx[np.argsort(scores[fp_idx])[::-1]]  # score 높은 순
    fn_idx = fn_idx[np.argsort(scores[fn_idx])]  # score 낮은 순
    # TP: 이미 랜덤 샘플링됨 (추가 정렬 없음)

    # ── 시각화 & 저장 ─────────────────────────────────────────
    print("\n[5단계] 시각화 및 저장")
    for idx_arr, stype in [(fp_idx, "FP"), (fn_idx, "FN"), (tp_idx, "TP")]:
        save_and_display(
            raw_imgs=raw_imgs,
            indices=idx_arr,
            scores=scores,
            labels=test_labels,
            threshold=threshold,
            sample_type=stype,
            result_dir=CFG["result_dir"],
        )

    print(f"\n[완료] 결과 저장 경로: {CFG['result_dir']}/")
    for stype in ("FP", "FN", "TP"):
        print(f"  {CFG['result_dir']}/{stype}/")
        print(f"  {CFG['result_dir']}/{stype}_grid.png")


if __name__ == "__main__":
    main()
