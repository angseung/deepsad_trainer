"""
TensorBoard logging and LR schedule visualization for Deep SAD training.
"""

import math

import torch
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# LR schedule visualization
# ============================================================
def plot_lr_schedule(cfg, save_path: str):
    """Simulate epoch-wise LR with a dummy optimizer and save the plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dummy_opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=cfg.lr)

    # Rebuild scheduler locally to avoid circular import with train.py
    n = cfg.n_epochs
    if cfg.scheduler == "multistep":
        dummy_sched = torch.optim.lr_scheduler.MultiStepLR(
            dummy_opt, milestones=[cfg.milestone], gamma=0.1
        )
    else:
        warmup_end = max(1, int(cfg.onecycle_pct_start * n))
        div = cfg.onecycle_div_factor
        final_div = cfg.onecycle_final_div_factor

        def _oc(epoch):
            if epoch < warmup_end:
                return (1.0 / div) + epoch / warmup_end * (1.0 - 1.0 / div)
            progress = (epoch - warmup_end) / max(n - warmup_end, 1)
            return (1.0 / final_div) + 0.5 * (1.0 - 1.0 / final_div) * (
                1.0 + math.cos(math.pi * progress)
            )

        if cfg.scheduler == "onecycle":
            dummy_sched = torch.optim.lr_scheduler.LambdaLR(dummy_opt, _oc)
        else:  # combined
            ms = cfg.milestone
            dummy_sched = torch.optim.lr_scheduler.LambdaLR(
                dummy_opt, lambda e: _oc(e) * (0.1 if e >= ms else 1.0)
            )

    lrs = []
    for _ in range(n):
        lrs.append(dummy_opt.param_groups[0]["lr"])
        dummy_sched.step()

    epochs = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color="#4C9BE8", linewidth=2)

    if cfg.scheduler != "onecycle" and cfg.milestone < n:
        ax.axvline(
            cfg.milestone, color="#E84C4C", linestyle="--", linewidth=1.2,
            label=f"milestone={cfg.milestone}",
        )
        ax.legend(fontsize=10)

    _labels = {
        "multistep": f"MultiStepLR  (milestone={cfg.milestone}, γ=0.1)",
        "onecycle":  f"OneCycleLR  (pct_start={cfg.onecycle_pct_start})",
        "combined":  f"OneCycleLR × MultiStepLR  (pct_start={cfg.onecycle_pct_start})",
    }
    ax.set_title(f"LR Schedule — {_labels.get(cfg.scheduler, cfg.scheduler)}", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_xlim(1, n)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  LR schedule 저장: {save_path}")
    return fig


# ============================================================
# TensorBoard logger
# ============================================================
class TrainingLogger:
    """Wraps SummaryWriter with typed logging methods for Deep SAD training."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    # -- LR schedule image -----------------------------------------
    def log_lr_schedule(self, cfg, save_path: str) -> None:
        fig = plot_lr_schedule(cfg, save_path)
        self.writer.add_figure("LR_schedule", fig)
        fig.clf()

    # -- Warmup ----------------------------------------------------
    def log_warmup(self, loss: float, epoch: int) -> None:
        self.writer.add_scalar("Loss/warmup", loss, epoch)

    # -- Per-epoch train metrics -----------------------------------
    def log_epoch(
        self,
        loss: float,
        lr: float,
        auc: float,
        best_auc: float,
        epoch: int,
    ) -> None:
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("LR", lr, epoch)
        self.writer.add_scalar("AUC/val", auc, epoch)
        self.writer.add_scalar("AUC/best", best_auc, epoch)

    # -- Final hparams + metrics -----------------------------------
    def log_hparams(self, cfg, best_auc: float, final_auc: float) -> None:
        self.writer.add_hparams(
            hparam_dict={
                "lr": cfg.lr,
                "n_epochs": cfg.n_epochs,
                "batch_size": cfg.batch_size,
                "eta": cfg.eta,
                "scheduler": cfg.scheduler,
                "milestone": cfg.milestone,
                "warmup_epochs": cfg.warmup_epochs,
                "proj_dim": cfg.proj_dim,
                "pretrained": cfg.pretrained,
            },
            metric_dict={
                "hparam/best_auc": best_auc,
                "hparam/final_auc": final_auc,
            },
        )

    def close(self) -> None:
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
