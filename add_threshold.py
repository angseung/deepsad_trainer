"""
add_threshold.py — 구형 체크포인트에 threshold 추가

구형 체크포인트(threshold 키 없음)와 검증 데이터셋을 입력받아,
학습 시와 동일한 로직(검증 데이터 정상 샘플 score의 percentile)으로
threshold를 계산한 뒤 체크포인트에 추가 저장한다.

사용법:
    python add_threshold.py --checkpoint out/best.pt --test_root data/test/
    python add_threshold.py --checkpoint out/best.pt --test_root data/test/ \\
        --percentile 95 --output out/best_updated.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common import DeepSAD_ResNet50, get_real_test_loader
from config import TrainConfig
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description="구형 체크포인트에 threshold 추가")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="입력 체크포인트 경로 (.pt)")
    parser.add_argument("--test_root", type=Path, required=True,
                        help="검증 데이터 루트 (normal/ anomaly/ 하위 폴더 포함)")
    parser.add_argument("--percentile", type=int, default=95,
                        help="정상 샘플 score 분포의 percentile (기본값: 95)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None,
                        help="연산 장치 (기본값: cuda 있으면 cuda, 없으면 cpu)")
    parser.add_argument("--output", type=Path, default=None,
                        help="저장 경로 (기본값: 입력 파일 덮어쓰기)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_path = args.output if args.output else args.checkpoint

    # Load checkpoint
    print(f"[1단계] 체크포인트 로드: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=True)

    if "threshold" in ckpt and ckpt["threshold"] != 0.0:
        print(f"  ⚠ 이미 threshold가 존재합니다: {ckpt['threshold']:.4f}")
        answer = input("  덮어쓰시겠습니까? [y/N] ").strip().lower()
        if answer != "y":
            print("  취소됨.")
            return

    saved_cfg = TrainConfig(**ckpt.get("cfg", {}))
    c = ckpt["c"].to(device)
    print(f"  저장 epoch : {ckpt.get('epoch', '?')}")
    print(f"  Best AUC   : {ckpt.get('best_auc', 0.0):.4f}")

    # Load model
    print("\n[2단계] 모델 복원")
    model = DeepSAD_ResNet50(
        proj_dim=saved_cfg.proj_dim,
        freeze_backbone=saved_cfg.freeze_backbone_train,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("  모델 로드 완료")

    # Load validation data
    print(f"\n[3단계] 검증 데이터 로드: {args.test_root}")
    test_loader, _, _ = get_real_test_loader(
        test_root=str(args.test_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=saved_cfg.img_size,
        pretrained=saved_cfg.pretrained,
    )

    # Compute threshold
    print("\n[4단계] threshold 계산")
    auc, normal_scores = evaluate(model, test_loader, c, device)
    threshold = float(np.percentile(normal_scores, args.percentile))
    print(f"  AUC               : {auc:.4f}")
    print(f"  정상 샘플 수      : {len(normal_scores)}")
    print(f"  정상 score 범위   : [{normal_scores.min():.4f}, {normal_scores.max():.4f}]")
    print(f"  Threshold (p{args.percentile:02d})  : {threshold:.4f}")

    # Save updated checkpoint
    print(f"\n[5단계] 저장: {output_path}")
    ckpt["threshold"] = threshold
    torch.save(ckpt, str(output_path))
    print("  완료")


if __name__ == "__main__":
    main()
