"""
Deep SAD with ResNet50 — 실제 데이터 학습

데이터 구조:
    DATA_ROOT/
      unlabeled/  → y=0  (레이블 없는 정상 데이터)
      normal/     → y=+1 (레이블 있는 정상 데이터)
      anomaly/    → y=-1 (레이블 있는 이상 데이터)

    TEST_ROOT/
      normal/     → y=+1
      anomaly/    → y=-1

사용법:
    CFG의 data_root / test_root 경로를 설정한 후 실행:
    python train.py
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import (
    set_seed,
    DeepSAD_ResNet50,
    get_real_dataloaders,
    get_real_test_loader,
)
from config import TrainConfig
from tb_logger import TrainingLogger


# ============================================================
# 센터 c 초기화
# ============================================================
@torch.no_grad()
def init_center(
    model: nn.Module,
    normal_loader: DataLoader,
    device: torch.device,
    eps: float = 0.1,
) -> torch.Tensor:
    model.eval()
    c = None
    n_samples = 0

    pbar = tqdm(normal_loader, desc="  c 초기화", leave=False, ncols=80, colour="cyan")
    for imgs, _ in pbar:
        imgs = imgs.to(device)
        z = model(imgs)
        if c is None:
            c = torch.zeros(z.shape[1], device=device)
        c += z.sum(dim=0)
        n_samples += z.shape[0]

    c /= n_samples
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps

    return c


# ============================================================
# Loss 함수
# ============================================================
def deepsad_loss(
    z: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    dist = torch.sum((z - c) ** 2, dim=1)
    loss = torch.where(
        labels == -1,
        1.0 / (dist + 1e-6),
        dist,
    )
    weights = torch.where(
        labels != 0,
        torch.full_like(dist, eta),
        torch.ones_like(dist),
    )
    return (weights * loss).mean()


# ============================================================
# Warmup 학습 루프 (정상 데이터만, 순수 distance 최소화)
# ============================================================
def train_warmup_epoch(
    model: nn.Module,
    normal_loader: DataLoader,
    c: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_warmup: int,
) -> float:
    """pretrained=False 일 때 warmup에서 사용.
    정상 데이터만 사용하여 ||z - c||² 를 최소화한다."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        normal_loader,
        desc=f"  [Warmup ] Epoch {epoch:>3}/{n_warmup}",
        leave=False,
        ncols=100,
        colour="magenta",
    )

    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)
        loss = torch.sum((z - c) ** 2, dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(normal_loader)


# ============================================================
# 학습 루프
# ============================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    c: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    milestone: int,
    eta: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    phase = "Searching  " if epoch <= milestone else "Fine-tuning"

    pbar = tqdm(
        train_loader,
        desc=f"  [{phase}] Epoch {epoch:>3}/{n_epochs}",
        leave=False,
        ncols=100,
        colour="green",
    )

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z = model(imgs)
        loss = deepsad_loss(z, c, labels, eta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


# ============================================================
# 검증
# ============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    c: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    all_scores, all_labels = [], []

    pbar = tqdm(test_loader, desc="  평가 중", leave=False, ncols=80, colour="yellow")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)
        scores = torch.sum((z - c) ** 2, dim=1)
        all_scores.append(scores.cpu())
        all_labels.append(labels)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    binary_labels = (all_labels == -1).astype(int)

    return roc_auc_score(binary_labels, all_scores)


# ============================================================
# 스케줄러
# ============================================================
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    scheduler='multistep' : MultiStepLR — milestone에서 lr × 0.1
    scheduler='onecycle'  : OneCycleLR 개형 — 선형 warmup → 코사인 decay
    scheduler='combined'  : OneCycleLR × MultiStepLR 복합 스케줄

    onecycle / combined 공식 (LambdaLR 기반):
        warmup 구간 [0, pct_start*n): 선형 상승  1/div_factor → 1.0
        decay  구간 [pct_start*n, n): 코사인 하강 1.0 → 1/final_div_factor
        combined만: epoch >= milestone 이후 전체에 × 0.1 추가 적용
    """
    if cfg.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[cfg.milestone], gamma=0.1
        )

    n = cfg.n_epochs
    warmup_end = max(1, int(cfg.onecycle_pct_start * n))
    div = cfg.onecycle_div_factor
    final_div = cfg.onecycle_final_div_factor
    milestone = cfg.milestone

    def _onecycle_factor(epoch: int) -> float:
        if epoch < warmup_end:
            return (1.0 / div) + epoch / warmup_end * (1.0 - 1.0 / div)
        progress = (epoch - warmup_end) / max(n - warmup_end, 1)
        return (1.0 / final_div) + 0.5 * (1.0 - 1.0 / final_div) * (
            1.0 + math.cos(math.pi * progress)
        )

    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _onecycle_factor)

    # combined: onecycle × multistep
    def lr_lambda(epoch: int) -> float:
        ms = 0.1 if epoch >= milestone else 1.0
        return _onecycle_factor(epoch) * ms

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 체크포인트 저장 / 불러오기
# ============================================================
def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    c: torch.Tensor,
    best_auc: float,
    cfg: TrainConfig,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "c": c,
            "best_auc": best_auc,
            "cfg": cfg.model_dump(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, torch.Tensor, float, TrainConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    cfg = TrainConfig(**ckpt.get("cfg", {}))
    print(
        f"  체크포인트 로드: epoch={ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}, "
        f"pretrained={cfg.pretrained}"
    )
    return ckpt["epoch"], ckpt["c"].to(device), ckpt["best_auc"], cfg


# ============================================================
# 전체 학습 파이프라인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Deep SAD 학습")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="학습 설정 YAML 파일 경로 (기본값: train.yaml 이 있으면 로드, 없으면 기본값 사용)",
    )
    args = parser.parse_args()

    _cfg_file = args.config if args.config is not None else Path("train.yaml")
    cfg = TrainConfig.from_yaml(_cfg_file) if _cfg_file.exists() else TrainConfig()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_dir = Path("out") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(out_dir / "best.pt")
    last_path = str(out_dir / "last.pt")

    # 실제 사용된 설정을 출력 디렉터리에 저장
    cfg.to_yaml(out_dir / "train.yaml")

    logger = TrainingLogger(log_dir=str(out_dir))

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(f"출력 경로: {out_dir}")

    # 데이터
    train_loader, normal_loader = get_real_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
    )
    test_loader, _, _ = get_real_test_loader(
        test_root=cfg.test_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
    )

    # freeze_backbone_* 은 pretrained=True 일 때만 유효
    if (cfg.freeze_backbone_warmup or cfg.freeze_backbone_train) and not cfg.pretrained:
        raise ValueError(
            "freeze_backbone_warmup/train=True는 pretrained=True 일 때만 사용할 수 있습니다."
        )

    # 모델 — warmup 동결 여부로 초기화
    model = DeepSAD_ResNet50(
        proj_dim=cfg.proj_dim,
        freeze_backbone=cfg.freeze_backbone_warmup,
        pretrained=cfg.pretrained,
    ).to(device)

    # 본 학습용 Adam 옵티마이저 + 스케줄러
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)
    logger.log_lr_schedule(cfg, str(out_dir / "lr_schedule.png"))

    # 센터 초기화
    print("\n[1단계] 센터 c 초기화")
    c = init_center(model, normal_loader, device)
    print(f"  c shape={c.shape}, norm={c.norm():.4f}")

    # ── Warmup ───────────────────────────────────────────────
    # 공통: warmup 전용 SGD 옵티마이저 사용, 스케줄러 없음
    # backbone 동결 여부: freeze_backbone_warmup (warmup) / freeze_backbone_train (본 학습)
    n_warmup = cfg.warmup_epochs
    if n_warmup > 0:
        warmup_optimizer = torch.optim.SGD(model.parameters(), lr=cfg.warmup_lr)

        freeze_info = ", backbone 동결" if cfg.freeze_backbone_warmup else ""
        print(
            f"\n[1.5단계] Warmup — 정상 데이터로만 {n_warmup} epoch"
            f"{freeze_info}  (SGD lr={cfg.warmup_lr})"
        )

        warmup_pbar = tqdm(
            range(1, n_warmup + 1), desc="Warmup 진행", ncols=100, colour="magenta"
        )
        for w_epoch in warmup_pbar:
            warmup_loss = train_warmup_epoch(
                model, normal_loader, c, warmup_optimizer, device, w_epoch, n_warmup
            )
            warmup_pbar.set_postfix({"loss": f"{warmup_loss:.4f}"})
            logger.log_warmup(warmup_loss, w_epoch)

        # 본 학습용 backbone 동결 상태 적용
        if cfg.pretrained:
            for param in model.backbone.parameters():
                param.requires_grad = not cfg.freeze_backbone_train
            status = "동결 유지" if cfg.freeze_backbone_train else "동결 해제"
            tqdm.write(f"  Warmup 완료 | loss={warmup_loss:.4f} | backbone {status}")
        else:
            tqdm.write(f"  Warmup 완료 | loss={warmup_loss:.4f}")

        # warmup 후 모델이 변경됐으므로 센터 재초기화
        print("  센터 c 재초기화")
        c = init_center(model, normal_loader, device)
        print(f"  c shape={c.shape}, norm={c.norm():.4f}")

    # 학습
    print("\n[2단계] 학습 시작")
    print("=" * 70)

    best_auc = 0.0
    best_epoch = 0

    epoch_pbar = tqdm(
        range(1, cfg.n_epochs + 1),
        desc="전체 진행",
        ncols=100,
        colour="blue",
    )

    for epoch in epoch_pbar:
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_loader,
            c,
            optimizer,
            device,
            epoch,
            cfg.n_epochs,
            cfg.milestone,
            cfg.eta,
        )
        scheduler.step()

        auc = 0.0
        # if epoch % 10 == 0 or epoch == cfg.n_epochs:
        auc = evaluate(model, test_loader, c, device)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            save_checkpoint(
                path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                c=c,
                best_auc=best_auc,
                cfg=cfg,
            )
            tqdm.write(
                f"  ✓ Best 모델 저장 | Epoch {best_epoch:>3} | AUC {best_auc:.4f}"
            )

        # last.pt — 매 에포크 갱신
        save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            c=c,
            best_auc=best_auc,
            cfg=cfg,
        )

        # 주기 저장 (save_interval > 0 일 때만)
        if cfg.save_interval > 0 and epoch % cfg.save_interval == 0:
            interval_path = str(out_dir / f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                path=interval_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                c=c,
                best_auc=best_auc,
                cfg=cfg,
            )
            # tqdm.write(f"  [interval] epoch_{epoch:04d}.pt 저장")

        logger.log_epoch(train_loss, current_lr, auc, best_auc, epoch)

        phase = "Search" if epoch <= cfg.milestone else "FineTune"
        epoch_pbar.set_postfix(
            {
                "phase": phase,
                "loss": f"{train_loss:.4f}",
                "auc": f"{auc:.4f}" if auc > 0 else "-",
                "best": f"{best_auc:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.0e}",
            }
        )

    print("\n" + "=" * 70)
    print(f"학습 완료!")
    print(f"  Best AUC   : {best_auc:.4f}")
    print(f"  Best Epoch : {best_epoch}")
    print(f"  저장 경로  : {save_path}")
    print("=" * 70)

    # Best 모델로 최종 평가
    print("\n[3단계] Best 모델 최종 평가")
    _, c_best, _, _ = load_checkpoint(save_path, model, device)
    final_auc = evaluate(model, test_loader, c_best, device)
    print(f"  최종 테스트 AUC: {final_auc:.4f}")

    logger.log_hparams(cfg, best_auc, final_auc)
    logger.close()


if __name__ == "__main__":
    main()
