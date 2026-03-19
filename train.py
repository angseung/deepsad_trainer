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

import os
from datetime import datetime

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
    cfg: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "c": c,
            "best_auc": best_auc,
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, torch.Tensor, float, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    cfg = ckpt.get("cfg", {})
    print(
        f"  체크포인트 로드: epoch={ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}, "
        f"pretrained={cfg.get('pretrained', 'N/A')}"
    )
    return ckpt["epoch"], ckpt["c"].to(device), ckpt["best_auc"], cfg


# ============================================================
# 전체 학습 파이프라인
# ============================================================
def main():
    CFG = {
        # ── 경로 설정 (필수) ──────────────────────────────────
        "data_root": "./data/test_data",  # unlabeled/ normal/ anomaly/ 가 있는 루트
        "test_root": "./data/test_data/test",  # normal/ anomaly/ 가 있는 테스트 루트
        # ── 학습 설정 ─────────────────────────────────────────
        "seed": 42,
        "batch_size": 64,
        "num_workers": 4,
        "img_size": 224,
        "proj_dim": 128,
        "pretrained": True,  # True: ImageNet 사전학습 / False: scratch 초기화 + warmup
        "freeze_backbone": True,  # pretrained=True 일 때만 유효 (backbone 파라미터 동결)
        "warmup_epochs": 1,  # pretrained=False 일 때만 사용
        "eta": 1.0,
        "lr": 1e-4,
        "n_epochs": 100,
        "milestone": 50,
        "weight_decay": 1e-6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_dir = os.path.join("out", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    CFG["save_path"] = os.path.join(out_dir, "deepsad_best.pt")

    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"Device: {device}")
    print(f"출력 경로: {out_dir}")

    # 데이터
    train_loader, normal_loader = get_real_dataloaders(
        data_root=CFG["data_root"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        seed=CFG["seed"],
        img_size=CFG["img_size"],
        pretrained=CFG["pretrained"],
    )
    test_loader, _, _ = get_real_test_loader(
        test_root=CFG["test_root"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        img_size=CFG["img_size"],
        pretrained=CFG["pretrained"],
    )

    # freeze_backbone은 pretrained=True 일 때만 유효
    if CFG["freeze_backbone"] and not CFG["pretrained"]:
        raise ValueError(
            "freeze_backbone=True는 pretrained=True 일 때만 사용할 수 있습니다."
        )

    # 모델
    # freeze_backbone=True 이면 warmup 동안만 동결 → __init__에서 freeze 적용
    model = DeepSAD_ResNet50(
        proj_dim=CFG["proj_dim"],
        freeze_backbone=CFG["freeze_backbone"],
        pretrained=CFG["pretrained"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[CFG["milestone"]], gamma=0.1
    )

    # 센터 초기화
    print("\n[1단계] 센터 c 초기화")
    c = init_center(model, normal_loader, device)
    print(f"  c shape={c.shape}, norm={c.norm():.4f}")

    # ── Warmup ───────────────────────────────────────────────
    # 케이스 1 (scratch):             정상 데이터만, 전 파라미터 학습
    # 케이스 2 (pretrained+freeze):   정상 데이터만, backbone 동결 → 이후 동결 해제
    # 케이스 3 (pretrained+no freeze): 정상 데이터만, 전 파라미터 학습
    n_warmup = CFG["warmup_epochs"]
    if n_warmup > 0:
        freeze_warmup = CFG["pretrained"] and CFG["freeze_backbone"]

        freeze_info = ", backbone 동결" if freeze_warmup else ""
        print(f"\n[1.5단계] Warmup — 정상 데이터로만 {n_warmup} epoch{freeze_info}")

        warmup_pbar = tqdm(
            range(1, n_warmup + 1), desc="Warmup 진행", ncols=100, colour="magenta"
        )
        for w_epoch in warmup_pbar:
            warmup_loss = train_warmup_epoch(
                model, normal_loader, c, optimizer, device, w_epoch, n_warmup
            )
            warmup_pbar.set_postfix({"loss": f"{warmup_loss:.4f}"})

        if freeze_warmup:
            # backbone 동결 해제 → 이후 전체 파라미터 학습
            for param in model.backbone.parameters():
                param.requires_grad = True
            tqdm.write(f"  Warmup 완료 | loss={warmup_loss:.4f} | backbone 동결 해제")
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
        range(1, CFG["n_epochs"] + 1),
        desc="전체 진행",
        ncols=100,
        colour="blue",
    )

    for epoch in epoch_pbar:
        train_loss = train_one_epoch(
            model,
            train_loader,
            c,
            optimizer,
            device,
            epoch,
            CFG["n_epochs"],
            CFG["milestone"],
            CFG["eta"],
        )
        scheduler.step()

        auc = 0.0
        # if epoch % 10 == 0 or epoch == CFG["n_epochs"]:
        auc = evaluate(model, test_loader, c, device)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            save_checkpoint(
                path=CFG["save_path"],
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                c=c,
                best_auc=best_auc,
                cfg=CFG,
            )
            tqdm.write(
                f"  ✓ Best 모델 저장 | Epoch {best_epoch:>3} | AUC {best_auc:.4f}"
            )

        phase = "Search" if epoch <= CFG["milestone"] else "FineTune"
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
    print(f"  저장 경로  : {CFG['save_path']}")
    print("=" * 70)

    # Best 모델로 최종 평가
    print("\n[3단계] Best 모델 최종 평가")
    _, c_best, _, _ = load_checkpoint(CFG["save_path"], model, device)
    final_auc = evaluate(model, test_loader, c_best, device)
    print(f"  최종 테스트 AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
