"""
Deep SAD 설정 데이터클래스.

YAML 로드:
    cfg = TrainConfig.from_yaml("train.yaml")
    cfg = TestConfig.from_yaml("test.yaml")

YAML 저장:
    cfg.to_yaml("train.yaml")
    cfg.to_yaml("test.yaml")
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    # ── paths ────────────────────────────────────────────────
    data_root: str = "./data/test_data"
    test_root: str = "./data/test_data/test"
    # ── model ────────────────────────────────────────────────
    proj_dim: int = 128
    pretrained: bool = True
    freeze_backbone_warmup: bool = True   # warmup 동안 backbone 동결 여부
    freeze_backbone_train: bool = False   # 본 학습 동안 backbone 동결 여부
    # ── training ─────────────────────────────────────────────
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    img_size: int = 224
    warmup_epochs: int = 1
    warmup_lr: float = 1e-3  # warmup 전용 SGD 학습률 (본 학습 lr 과 별도)
    save_interval: int = -1  # -1: best + last 만 저장 / N>0: N 에포크마다 추가 저장
    # ── scheduler ────────────────────────────────────────────
    scheduler: str = "multistep"          # multistep | onecycle | combined
    onecycle_pct_start: float = 0.3      # warmup 비율 (전체 에포크 대비)
    onecycle_div_factor: float = 25.0    # 초기 lr = lr / div_factor
    onecycle_final_div_factor: float = 1.0e4  # 최소 lr = lr / final_div_factor
    eta: float = 1.0
    lr: float = 1e-4
    n_epochs: int = 10
    milestone: int = 50
    weight_decay: float = 1e-6
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )


class TestConfig(BaseModel):
    # ── paths ────────────────────────────────────────────────
    checkpoint: str = "out/deepsad_best.pt"
    test_root: str = "data/test_data/test"
    result_dir: str = "result"
    # ── inference ────────────────────────────────────────────
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    img_size: int = 224
    threshold_percentile: int = 90
    num_tp_samples: int = 50
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
