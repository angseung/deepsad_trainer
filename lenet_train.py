"""
Deep SAD with LeNet Autoencoder Pretraining on MNIST
- PyTorch 2.7+ 호환
- 오토인코더 사전학습 → 인코더 가중치 이전 → DeepSAD 학습
- pip install torch torchvision scikit-learn tqdm
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import (
    set_seed,
    get_dataloaders_lenet,
    LeNetAutoencoder,
    DeepSAD_LeNet,
)


# ============================================================
# 1. 오토인코더 사전학습
# ============================================================
def pretrain_autoencoder(
    ae: nn.Module,
    ae_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> nn.Module:
    """
    MSE 재구성 손실로 오토인코더 사전학습
    학습 완료 후 ae 반환
    """
    ae.train()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[n_epochs // 2], gamma=0.1
    )
    criterion = nn.MSELoss()

    print("\n[1단계] 오토인코더 사전학습")
    print("=" * 70)

    epoch_pbar = tqdm(
        range(1, n_epochs + 1),
        desc="AE 사전학습",
        ncols=100,
        colour="cyan",
    )

    for epoch in epoch_pbar:
        total_loss = 0.0

        pbar = tqdm(
            ae_loader,
            desc=f"  AE Epoch {epoch:>3}/{n_epochs}",
            leave=False,
            ncols=100,
            colour="green",
        )

        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            x_hat = ae(imgs)  # 재구성 이미지

            # MSE 재구성 손실 = I(X;Z) 최대화
            loss = criterion(x_hat, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        scheduler.step()
        avg_loss = total_loss / len(ae_loader)

        epoch_pbar.set_postfix(
            {
                "loss": f"{avg_loss:.6f}",
                "lr": f"{scheduler.get_last_lr()[0]:.0e}",
            }
        )

    print(f"\n  AE 사전학습 완료 | 최종 Loss: {avg_loss:.6f}")
    return ae


# ============================================================
# 2. 센터 c 초기화
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
# 3. DeepSAD Loss
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
# 4. 학습 루프
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
# 5. 검증
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
# 6. 체크포인트 저장 / 불러오기
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
# 7. 전체 파이프라인
# ============================================================
def main():
    CFG = {
        "seed": 42,
        "normal_class": 0,
        "gamma_l": 0.05,
        "gamma_p": 0.0,
        "batch_size": 128,
        "num_workers": 4,
        "rep_dim": 32,  # 논문 MNIST 설정
        "eta": 1.0,
        # 오토인코더 사전학습
        "ae_epochs": 100,
        "ae_lr": 1e-3,
        "ae_weight_decay": 1e-6,
        "ae_save_path": "ae_pretrained.pt",
        # DeepSAD 학습
        "lr": 1e-4,
        "n_epochs": 150,
        "milestone": 50,
        "weight_decay": 1e-6,
        "save_path": "deepsad_best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    # ----------------------------------------------------------
    # 데이터 로더
    # ----------------------------------------------------------
    train_loader, normal_loader, test_loader, ae_loader = get_dataloaders_lenet(
        normal_class=CFG["normal_class"],
        gamma_l=CFG["gamma_l"],
        gamma_p=CFG["gamma_p"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        seed=CFG["seed"],
        img_size=28,  # LeNet: 28×28
    )

    # ----------------------------------------------------------
    # 1단계: 오토인코더 사전학습
    # ----------------------------------------------------------
    ae = LeNetAutoencoder(rep_dim=CFG["rep_dim"]).to(device)

    ae = pretrain_autoencoder(
        ae=ae,
        ae_loader=ae_loader,
        device=device,
        n_epochs=CFG["ae_epochs"],
        lr=CFG["ae_lr"],
        weight_decay=CFG["ae_weight_decay"],
    )

    # 오토인코더 가중치 저장
    torch.save(ae.state_dict(), CFG["ae_save_path"])
    print(f"  오토인코더 저장: {CFG['ae_save_path']}")

    # ----------------------------------------------------------
    # 2단계: 인코더 가중치 이전 → DeepSAD 모델 초기화
    # ----------------------------------------------------------
    print("\n[2단계] 인코더 가중치 이전")
    model = DeepSAD_LeNet(rep_dim=CFG["rep_dim"]).to(device)
    model.load_encoder_weights(ae)  # 오토인코더 인코더 → DeepSAD

    # ----------------------------------------------------------
    # 3단계: 센터 c 초기화
    # ----------------------------------------------------------
    print("\n[3단계] 센터 c 초기화")
    c = init_center(model, normal_loader, device)
    print(f"  c shape={c.shape}, norm={c.norm():.4f}")

    # ----------------------------------------------------------
    # 4단계: DeepSAD 학습
    # ----------------------------------------------------------
    print("\n[4단계] DeepSAD 학습 시작")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[CFG["milestone"]], gamma=0.1
    )

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
        if epoch % 10 == 0 or epoch == CFG["n_epochs"]:
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
                    f"  ✓ Best 모델 저장 | " f"Epoch {epoch:>3} | AUC: {auc:.4f}"
                )

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

    # ----------------------------------------------------------
    # 최종 결과
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"학습 완료!")
    print(f"  Best AUC   : {best_auc:.4f}")
    print(f"  Best Epoch : {best_epoch}")
    print("=" * 70)

    print("\n[5단계] Best 모델 최종 평가")
    _, c_best, _ = load_checkpoint(CFG["save_path"], model, device)
    final_auc = evaluate(model, test_loader, c_best, device)
    print(f"  최종 테스트 AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
