"""
Deep SAD with ResNet50 (ImageNet-21k pretrained) on MNIST
- PyTorch 2.7+ 호환
- 재현 가능한 랜덤 시드 고정
- tqdm 진행상황 시각화
- Best 모델 저장
- pip install torch torchvision timm scikit-learn tqdm
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import set_seed, DeepSAD_ResNet50, get_dataloaders


# ============================================================
# 3. 센터 c 초기화
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
# 4. Loss 함수
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
# 5. 학습 루프 (tqdm 배치 진행바 포함)
# ============================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    c: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    eta: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    phase = "Searching  " if epoch <= 50 else "Fine-tuning"

    # 배치 단위 진행바
    pbar = tqdm(
        train_loader,
        desc=f"  [{phase}] Epoch {epoch:>3}/{n_epochs}",
        leave=False,  # 에포크 완료 후 진행바 제거
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

        # 배치 진행바에 현재 loss 표시
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


# ============================================================
# 6. 검증
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
# 7. Best 모델 저장 / 불러오기
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
) -> tuple[int, torch.Tensor, float]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    print(
        f"  체크포인트 로드: epoch={ckpt['epoch']}, " f"best_auc={ckpt['best_auc']:.4f}"
    )
    return ckpt["epoch"], ckpt["c"].to(device), ckpt["best_auc"]


# ============================================================
# 8. 전체 학습 파이프라인
# ============================================================
def main():
    CFG = {
        "seed": 42,
        "normal_class": 1,
        "gamma_l": 0.05,
        "gamma_p": 0.0,
        "batch_size": 128,
        "num_workers": 4,
        "proj_dim": 128,
        "freeze_backbone": False,
        "eta": 1.0,
        "lr": 1e-4,
        "n_epochs": 10,
        "milestone": 50,
        "weight_decay": 1e-6,
        "save_path": "deepsad_best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    # 데이터
    train_loader, normal_loader, test_loader = get_dataloaders(
        normal_class=CFG["normal_class"],
        gamma_l=CFG["gamma_l"],
        gamma_p=CFG["gamma_p"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        seed=CFG["seed"],
    )

    # 모델
    model = DeepSAD_ResNet50(
        proj_dim=CFG["proj_dim"],
        freeze_backbone=CFG["freeze_backbone"],
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

    # 학습
    print("\n[2단계] 학습 시작")
    print("=" * 70)

    best_auc = 0.0
    best_epoch = 0

    # 에포크 단위 진행바
    epoch_pbar = tqdm(
        range(1, CFG["n_epochs"] + 1),
        desc="전체 진행",
        ncols=100,
        colour="blue",
    )

    for epoch in epoch_pbar:
        # 학습
        train_loss = train_one_epoch(
            model,
            train_loader,
            c,
            optimizer,
            device,
            epoch,
            CFG["n_epochs"],
            CFG["eta"],
        )
        scheduler.step()

        # 검증 (10 epoch마다 + 마지막 epoch)
        auc = 0.0
        if epoch % 10 == 0 or epoch == CFG["n_epochs"]:
            auc = evaluate(model, test_loader, c, device)

            # Best 모델 저장
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
                # tqdm과 겹치지 않도록 write 사용
                tqdm.write(
                    f"  ✓ Best 모델 저장 | "
                    f"Epoch {epoch:>3} | "
                    f"AUC {best_auc:.4f} → {auc:.4f}"
                )

        # 에포크 진행바 우측 정보 업데이트
        phase = "Search" if epoch <= CFG["milestone"] else "FineTune"
        epoch_pbar.set_postfix(
            {
                "phase": phase,
                "loss": f"{train_loss:.4f}",
                "auc": f"{auc:.4f}" if auc > 0 else "-",
                "best_auc": f"{best_auc:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.0e}",
            }
        )

    # 최종 결과
    print("\n" + "=" * 70)
    print(f"학습 완료!")
    print(f"  Best AUC   : {best_auc:.4f}")
    print(f"  Best Epoch : {best_epoch}")
    print(f"  저장 경로  : {CFG['save_path']}")
    print("=" * 70)

    # Best 모델로 최종 평가
    print("\n[3단계] Best 모델 최종 평가")
    _, c_best, _ = load_checkpoint(CFG["save_path"], model, device)
    final_auc = evaluate(model, test_loader, c_best, device)
    print(f"  최종 테스트 AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
