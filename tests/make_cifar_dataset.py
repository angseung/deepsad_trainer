"""
CIFAR-10 기반 테스트 데이터셋 생성 스크립트

정상 클래스: CIFAR-10 class 0 (airplane)
이상 클래스: class 1-9

생성 구조:
    ./data/test_data/
      unlabeled/     ← train split, class 0의 95%   (y=0)
      normal/        ← train split, class 0의 5%    (y=+1)
      anomaly/       ← train split, class 1-9에서 normal과 동일한 수 랜덤 샘플링 (y=-1)
      test/
        normal/      ← test split, class 0 전체
        anomaly/     ← test split, class 1-9 전체

실행:
    python tests/make_cifar_dataset.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from torchvision import datasets

# ============================================================
# 설정
# ============================================================
NORMAL_CLASS   = 0
OUT_ROOT       = "./data/test_data"
UNLABELED_RATIO = 0.95   # class 0 train 이미지 중 unlabeled 비율
SEED           = 42


# ============================================================
# 유틸리티
# ============================================================
def save_images(imgs: np.ndarray, out_dir: str, prefix: str = "img"):
    """imgs: [N, H, W, C] uint8 numpy 배열 → 개별 PNG 저장"""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(os.path.join(out_dir, f"{prefix}_{i:05d}.png"))


# ============================================================
# 메인
# ============================================================
def main():
    rng = np.random.RandomState(SEED)

    # CIFAR-10 다운로드
    print("[1단계] CIFAR-10 다운로드")
    train_set = datasets.CIFAR10(root="./data", train=True,  download=True)
    test_set  = datasets.CIFAR10(root="./data", train=False, download=True)

    train_imgs   = np.array(train_set.data)    # [50000, 32, 32, 3]
    train_labels = np.array(train_set.targets) # [50000]
    test_imgs    = np.array(test_set.data)     # [10000, 32, 32, 3]
    test_labels  = np.array(test_set.targets)  # [10000]

    # ── train split 분류 ────────────────────────────────────
    normal_idx  = np.where(train_labels == NORMAL_CLASS)[0]
    anomaly_idx = np.where(train_labels != NORMAL_CLASS)[0]

    rng.shuffle(normal_idx)
    n_normal     = len(normal_idx)
    n_labeled    = max(int(n_normal * (1 - UNLABELED_RATIO)), 1)
    n_unlabeled  = n_normal - n_labeled

    unlabeled_idx = normal_idx[:n_unlabeled]
    labeled_normal_idx = normal_idx[n_unlabeled:]          # n_labeled개
    labeled_anomaly_idx = rng.choice(anomaly_idx, size=n_labeled, replace=False)

    # ── test split 분류 ──────────────────────────────────────
    test_normal_idx  = np.where(test_labels == NORMAL_CLASS)[0]
    test_anomaly_idx = np.where(test_labels != NORMAL_CLASS)[0]

    # ── 저장 ────────────────────────────────────────────────
    dirs = {
        "unlabeled": (train_imgs[unlabeled_idx],       "unl"),
        "normal":    (train_imgs[labeled_normal_idx],  "nor"),
        "anomaly":   (train_imgs[labeled_anomaly_idx], "ano"),
        "test/normal":  (test_imgs[test_normal_idx],   "nor"),
        "test/anomaly": (test_imgs[test_anomaly_idx],  "ano"),
    }

    print("\n[2단계] 이미지 저장")
    for subdir, (imgs, prefix) in dirs.items():
        out_dir = os.path.join(OUT_ROOT, subdir)
        save_images(imgs, out_dir, prefix)
        print(f"  {out_dir:<35} {len(imgs):>5}개")

    # ── 요약 ────────────────────────────────────────────────
    print(f"""
[완료] 저장 경로: {OUT_ROOT}

  train/
    unlabeled/ : {len(unlabeled_idx):>5}개  (class 0의 {UNLABELED_RATIO*100:.0f}%)
    normal/    : {len(labeled_normal_idx):>5}개  (class 0의 {(1-UNLABELED_RATIO)*100:.0f}%)
    anomaly/   : {len(labeled_anomaly_idx):>5}개  (class 1-9에서 normal과 동일한 수 샘플링)
  test/
    normal/    : {len(test_normal_idx):>5}개  (test split, class 0 전체)
    anomaly/   : {len(test_anomaly_idx):>5}개  (test split, class 1-9 전체)

  train.py CFG 설정:
    "data_root": "{OUT_ROOT}",
    "test_root": "{os.path.join(OUT_ROOT, "test")}",
""")


if __name__ == "__main__":
    main()
