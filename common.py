import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
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

    lb_anomaly_idx = rng.choice(anomaly_idx, n_label, replace=False)
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
    def __init__(self, proj_dim: int = 128, freeze_backbone: bool = False, pretrained: bool = True):
        super().__init__()

        backbone = timm.create_model(
            "resnet50.a1_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone = backbone
        self.backbone_dim = backbone.num_features  # 2048

        if pretrained and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif not pretrained and freeze_backbone:
            import warnings
            warnings.warn(
                "freeze_backbone=True는 pretrained=True일 때만 유효합니다. 무시합니다.",
                UserWarning, stacklevel=2,
            )

        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.projection(z)
        return z


def get_dataloaders_lenet(
    normal_class: int = 1,
    gamma_l: float = 0.05,
    gamma_p: float = 0.0,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    img_size: int = 28,  # LeNet: 28×28 그대로 사용
):
    rng = np.random.RandomState(seed)

    # ----------------------------------------------------------
    # LeNet용 transform: MNIST 원본 크기(28×28) 그대로 사용
    # ----------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            # 정규화: MNIST mean/std
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_imgs = mnist_train.data.float().unsqueeze(1) / 255.0  # [N, 1, 28, 28]
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

    lb_anomaly_idx = rng.choice(anomaly_idx, n_label, replace=False)
    lb_anomaly_data = train_imgs[lb_anomaly_idx]
    lb_anomaly_targets = torch.full((n_label,), -1, dtype=torch.long)

    all_train_data = torch.cat([unlabeled_data, lb_normal_data, lb_anomaly_data], dim=0)
    all_train_targets = torch.cat(
        [unlabeled_targets, lb_normal_targets, lb_anomaly_targets], dim=0
    )

    # c 초기화용: 정상 데이터(unlabeled)만
    normal_loader = DataLoader(
        DeepSAD_MNIST(unlabeled_data, unlabeled_targets, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    # 학습용
    train_loader = DataLoader(
        DeepSAD_MNIST(all_train_data, all_train_targets, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    # 테스트용
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

    # 오토인코더 사전학습용: 전체 학습 데이터 (레이블 불필요)
    # unlabeled 데이터만 사용 (정상 데이터 위주)
    ae_loader = DataLoader(
        DeepSAD_MNIST(unlabeled_data, unlabeled_targets, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
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

    return train_loader, normal_loader, test_loader, ae_loader


# ============================================================
# 2. LeNet 인코더 (Deep SAD 네트워크 φ)
# ============================================================
class LeNetEncoder(nn.Module):
    """
    논문 MNIST용 LeNet 인코더
    입력: [B, 1, 28, 28]
    출력: [B, rep_dim]

    - bias=False       : 하이퍼스피어 붕괴 방지
    - LeakyReLU        : unbounded activation
    - MaxPool → 없음   : 디코더 대칭 구성을 위해 stride conv 사용
    """

    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim

        self.encoder = nn.Sequential(
            # 모듈 1: [B, 1, 28, 28] → [B, 8, 12, 12]
            nn.Conv2d(1, 8, kernel_size=5, bias=False, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 → 14
            # 모듈 2: [B, 8, 14, 14] → [B, 4, 5, 5]
            nn.Conv2d(8, 4, kernel_size=5, bias=False, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 → 7
        )
        # FC: 4×7×7 → rep_dim
        self.fc = nn.Linear(4 * 7 * 7, rep_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # [B, 4, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 196]
        x = self.fc(x)  # [B, rep_dim]
        return x


# ============================================================
# 3. LeNet 디코더 (오토인코더 학습용)
# ============================================================
class LeNetDecoder(nn.Module):
    """
    LeNetEncoder의 대칭 디코더
    입력: [B, rep_dim]
    출력: [B, 1, 28, 28]

    인코더 대칭 규칙:
        MaxPool  → Upsample
        Conv     → ConvTranspose2d
        FC(축소) → FC(확장)
    """

    def __init__(self, rep_dim: int = 32):
        super().__init__()

        # FC: rep_dim → 4×7×7
        self.fc = nn.Linear(rep_dim, 4 * 7 * 7, bias=False)

        self.decoder = nn.Sequential(
            # 모듈 2 역순: [B, 4, 7, 7] → [B, 8, 14, 14]
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(4, 8, kernel_size=5, bias=False, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            # 모듈 1 역순: [B, 8, 14, 14] → [B, 1, 28, 28]
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 1, kernel_size=5, bias=False, padding=2),
            nn.Sigmoid(),  # 출력을 [0, 1]로 정규화 (MSE loss용)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)  # [B, 196]
        x = x.view(x.size(0), 4, 7, 7)  # [B, 4, 7, 7]
        x = self.decoder(x)  # [B, 1, 28, 28]
        return x


# ============================================================
# 4. LeNet 오토인코더 (사전학습용)
# ============================================================
class LeNetAutoencoder(nn.Module):
    """
    인코더 + 디코더 통합 모델
    사전학습 완료 후 encoder만 DeepSAD에 사용
    """

    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.encoder = LeNetEncoder(rep_dim)
        self.decoder = LeNetDecoder(rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # 잠재 벡터
        x_hat = self.decoder(z)  # 재구성 이미지
        return x_hat


# ============================================================
# 5. Deep SAD 모델 (인코더만 사용)
# ============================================================
class DeepSAD_LeNet(nn.Module):
    """
    오토인코더 사전학습 후 인코더 가중치를 불러와서 사용
    """

    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.encoder = LeNetEncoder(rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def load_encoder_weights(self, ae: LeNetAutoencoder):
        """오토인코더에서 인코더 가중치 복사"""
        self.encoder.load_state_dict(ae.encoder.state_dict())
        print("  인코더 가중치 로드 완료 (from autoencoder)")


# ============================================================
# 실제 데이터용 유틸리티
# ============================================================
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_image_files(dir_path: str) -> list[str]:
    """디렉터리 내 이미지 파일 경로를 정렬하여 반환. 디렉터리 없으면 빈 리스트."""
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )


class RealImageDataset(Dataset):
    """
    label 규칙:
         0 → unlabeled (정상으로 가정)
        +1 → labeled 정상
        -1 → labeled 이상
    """

    def __init__(self, file_paths: list[str], labels: torch.Tensor, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_real_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    img_size: int = 224,
    pretrained: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    학습용 데이터로더를 반환한다.

    data_root/
      unlabeled/  → y=0  (레이블 없는 정상 데이터)
      normal/     → y=+1 (레이블 있는 정상 데이터)
      anomaly/    → y=-1 (레이블 있는 이상 데이터)

    pretrained=True : ImageNet 통계로 정규화 (mean/std)
    pretrained=False: ToTensor만 적용 ([0, 1] 범위)

    Returns:
        train_loader: 전체 학습 셋 (unlabeled + normal + anomaly)
        normal_loader: 센터 c 초기화용 (unlabeled만)
    """
    normalize = (
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if pretrained else []
    )
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        *normalize,
    ])

    unlabeled_paths = _collect_image_files(os.path.join(data_root, "unlabeled"))
    normal_paths    = _collect_image_files(os.path.join(data_root, "normal"))
    anomaly_paths   = _collect_image_files(os.path.join(data_root, "anomaly"))

    all_paths = unlabeled_paths + normal_paths + anomaly_paths
    all_labels = torch.tensor(
        [0] * len(unlabeled_paths) + [1] * len(normal_paths) + [-1] * len(anomaly_paths),
        dtype=torch.long,
    )

    normal_loader = DataLoader(
        RealImageDataset(
            unlabeled_paths,
            torch.zeros(len(unlabeled_paths), dtype=torch.long),
            transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        RealImageDataset(all_paths, all_labels, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"\n[데이터 구성] {data_root}")
    print(f"  unlabeled  : {len(unlabeled_paths):>6}개")
    print(f"  normal  +1 : {len(normal_paths):>6}개")
    print(f"  anomaly -1 : {len(anomaly_paths):>6}개")
    print(f"  합계       : {len(all_paths):>6}개")

    return train_loader, normal_loader


def get_real_test_loader(
    test_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    pretrained: bool = True,
) -> tuple[DataLoader, list[str], np.ndarray]:
    """
    테스트용 데이터로더를 반환한다.

    test_root/
      normal/   → y=+1
      anomaly/  → y=-1

    pretrained=True : ImageNet 통계로 정규화 (mean/std)
    pretrained=False: ToTensor만 적용 ([0, 1] 범위)

    Returns:
        loader: DataLoader
        file_paths: 원본 파일 경로 리스트 (시각화용)
        labels: numpy int 배열 (+1/-1)
    """
    normalize = (
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if pretrained else []
    )
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        *normalize,
    ])

    normal_paths  = _collect_image_files(os.path.join(test_root, "normal"))
    anomaly_paths = _collect_image_files(os.path.join(test_root, "anomaly"))

    all_paths  = normal_paths + anomaly_paths
    all_labels = torch.tensor(
        [1] * len(normal_paths) + [-1] * len(anomaly_paths),
        dtype=torch.long,
    )

    loader = DataLoader(
        RealImageDataset(all_paths, all_labels, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n[테스트 데이터] {test_root}")
    print(f"  normal  : {len(normal_paths):>6}개")
    print(f"  anomaly : {len(anomaly_paths):>6}개")

    return loader, all_paths, all_labels.numpy()
