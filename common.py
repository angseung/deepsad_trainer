import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import timm


# ============================================================
# 0. 랜덤 시드 고정
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 1. 데이터셋
# ============================================================
class DeepSAD_MNIST(Dataset):
    """
    label 규칙:
         0 → unlabeled (정상으로 가정)
        +1 → labeled 정상
        -1 → labeled 이상
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def get_dataloaders(
    normal_class: int = 1,
    gamma_l: float = 0.05,  # labeled 데이터 비율
    gamma_p: float = 0.0,  # unlabeled 오염 비율
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    img_size: int = 224,  # ResNet50 입력 크기
):
    rng = np.random.RandomState(seed)

    # ----------------------------------------------------------
    # ResNet50용 transform
    # MNIST: 흑백(1ch) → 3채널 복제 후 224×224로 리사이즈
    # ----------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch → 3ch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ----------------------------------------------------------
    # MNIST 로드
    # ----------------------------------------------------------
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_imgs = mnist_train.data.float().unsqueeze(1) / 255.0
    train_labels = mnist_train.targets.numpy()
    test_imgs = mnist_test.data.float().unsqueeze(1) / 255.0
    test_labels = mnist_test.targets.numpy()

    normal_idx = np.where(train_labels == normal_class)[0]
    anomaly_idx = np.where(train_labels != normal_class)[0]

    # Unlabeled 데이터
    unlabeled_idx = normal_idx.copy()
    if gamma_p > 0.0:
        n_pollute = int(len(normal_idx) * gamma_p / (1 - gamma_p))
        pollute_idx = rng.choice(anomaly_idx, n_pollute, replace=False)
        unlabeled_idx = np.concatenate([unlabeled_idx, pollute_idx])

    unlabeled_data = train_imgs[unlabeled_idx]
    unlabeled_targets = torch.zeros(len(unlabeled_idx), dtype=torch.long)

    # Labeled 데이터
    n_label = max(int(len(unlabeled_idx) * gamma_l / (1 - gamma_l)), 1)

    lb_normal_idx = rng.choice(normal_idx, n_label, replace=False)
    lb_normal_data = train_imgs[lb_normal_idx]
    lb_normal_targets = torch.ones(len(lb_normal_idx), dtype=torch.long)

    sel_anomaly_cls = [c for c in range(10) if c != normal_class][0]
    sel_anomaly_idx = np.where(train_labels == sel_anomaly_cls)[0]
    lb_anomaly_idx = rng.choice(sel_anomaly_idx, n_label, replace=False)
    lb_anomaly_data = train_imgs[lb_anomaly_idx]
    lb_anomaly_targets = torch.full((n_label,), -1, dtype=torch.long)

    all_train_data = torch.cat([unlabeled_data, lb_normal_data, lb_anomaly_data], dim=0)
    all_train_targets = torch.cat(
        [unlabeled_targets, lb_normal_targets, lb_anomaly_targets], dim=0
    )

    normal_loader = DataLoader(
        DeepSAD_MNIST(unlabeled_data, unlabeled_targets, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        DeepSAD_MNIST(all_train_data, all_train_targets, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    test_targets_sad = torch.tensor(
        np.where(test_labels == normal_class, 1, -1), dtype=torch.long
    )
    test_loader = DataLoader(
        DeepSAD_MNIST(test_imgs, test_targets_sad, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"\n[데이터 구성] normal_class={normal_class}, "
        f"gamma_l={gamma_l}, gamma_p={gamma_p}"
    )
    print(f"  학습 총계  : {len(all_train_data):>6}개")
    print(f"  unlabeled  : {(all_train_targets == 0).sum():>6}개")
    print(f"  labeled +1 : {(all_train_targets == 1).sum():>6}개")
    print(f"  labeled -1 : {(all_train_targets == -1).sum():>6}개")
    print(
        f"  테스트     : {len(test_targets_sad):>6}개 "
        f"(정상 {(test_targets_sad==1).sum()}, "
        f"이상 {(test_targets_sad==-1).sum()})"
    )

    return train_loader, normal_loader, test_loader


# ============================================================
# 2. 모델
# ============================================================
class DeepSAD_ResNet50(nn.Module):
    def __init__(self, proj_dim: int = 128, freeze_backbone: bool = False):
        super().__init__()

        backbone = timm.create_model(
            "resnet50.a1_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone = backbone
        self.backbone_dim = backbone.num_features  # 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.projection(z)
        return z
