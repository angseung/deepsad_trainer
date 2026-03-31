# Deep SAD 학습 가이드

## 데이터 준비

### 디렉터리 구조

학습 데이터와 테스트 데이터를 아래 구조로 준비한다.

```
DATA_ROOT/
  unlabeled/   ← 레이블 없는 정상 이미지  (y = 0)
  normal/      ← 레이블 있는 정상 이미지  (y = +1)
  anomaly/     ← 레이블 있는 이상 이미지  (y = -1)

TEST_ROOT/
  normal/      ← 정상 이미지  (y = +1)
  anomaly/     ← 이상 이미지  (y = -1)
```

지원 이미지 포맷: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

CIFAR-10 기반 테스트 데이터셋은 아래 스크립트로 생성할 수 있다.

```bash
python tests/make_cifar_dataset.py
```

---

## 설정 (TrainConfig / TestConfig)

모든 설정은 `config.py`의 Pydantic 모델(`TrainConfig`, `TestConfig`)로 관리한다.

### YAML 연동

`--config` 인자로 YAML 파일을 명시할 수 있다. 생략하면 현재 디렉터리의 `train.yaml` / `test.yaml`을 자동으로 탐색하고, 없으면 기본값을 사용한다.

```bash
# 기본값으로 실행 (train.yaml 없을 때)
python train.py

# YAML 파일을 명시해 실행
python train.py --config configs/exp1.yaml
python test.py --config configs/test_exp1.yaml

# 첫 실행으로 생성된 설정을 복사해 재사용
cp out/YYYY-MM-DD-HH:MM:SS/train.yaml train.yaml
python train.py          # train.yaml 자동 로드
```

코드에서 직접 다루는 경우:

```python
from config import TrainConfig, TestConfig

cfg = TrainConfig.from_yaml("train.yaml")   # YAML → 인스턴스
cfg.to_yaml("train.yaml")                   # 인스턴스 → YAML
```

### 경로

| 필드 | 설명 |
|----|------|
| `data_root` | 학습 데이터 루트 (`unlabeled/` `normal/` `anomaly/` 포함) |
| `test_root` | 테스트 데이터 루트 (`normal/` `anomaly/` 포함) |

체크포인트는 실행 시각 기준 타임스탬프 디렉터리에 자동 저장된다.

```
out/YYYY-MM-DD-HH:MM:SS/
  train.yaml          ← 학습 시작 시 사용된 설정 자동 저장
  lr_schedule.png     ← 학습 시작 시 에포크별 lr 그래프 자동 저장
  best.pt             ← Best AUC 갱신 시
  last.pt             ← 매 에포크 갱신
  epoch_XXXX.pt       ← save_interval 설정 시 주기 저장
```

### 학습 하이퍼파라미터

| 필드 | 기본값 | 설명 |
|----|--------|------|
| `seed` | `42` | 재현성을 위한 랜덤 시드 |
| `batch_size` | `64` | 미니배치 크기 |
| `num_workers` | `4` | DataLoader 워커 수 |
| `img_size` | `224` | 입력 이미지 크기 (ResNet50 기본값) |
| `proj_dim` | `128` | projection head 출력 차원 |
| `eta` | `1.0` | 레이블 샘플의 loss 가중치 (η) |
| `lr` | `1e-4` | Adam 초기 학습률 |
| `weight_decay` | `1e-6` | Adam weight decay |
| `n_epochs` | `10` | 본 학습 총 에포크 수 |
| `milestone` | `50` | lr 스텝다운 시점 (×0.1) — 모든 스케줄러 공통 |
| `warmup_epochs` | `1` | Warmup 에포크 수 (0이면 warmup 생략) |
| `warmup_lr` | `1e-3` | Warmup 전용 SGD 학습률 (본 학습 `lr`과 별도) |
| `save_interval` | `-1` | 주기 체크포인트 저장 간격 (-1: best + last 만 저장) |
| `freeze_backbone_warmup` | `true` | warmup 동안 backbone 동결 여부 (`pretrained=true` 시만 유효) |
| `freeze_backbone_train` | `false` | 본 학습 동안 backbone 동결 여부 (`pretrained=true` 시만 유효) |
| `scheduler` | `"multistep"` | 스케줄링 방식: `multistep` \| `onecycle` \| `combined` |
| `onecycle_pct_start` | `0.3` | OneCycleLR warmup 비율 (전체 에포크 대비) — `onecycle` / `combined` 시 사용 |
| `onecycle_div_factor` | `25.0` | 초기 lr = `lr / div_factor` — `onecycle` / `combined` 시 사용 |
| `onecycle_final_div_factor` | `1e4` | 최소 lr = `lr / final_div_factor` — `onecycle` / `combined` 시 사용 |

---

## 학습 모드

`pretrained` / `freeze_backbone_warmup` / `freeze_backbone_train` 조합에 따라 동작한다.

### 케이스 1 — Scratch

```yaml
pretrained: false
freeze_backbone_warmup: false   # pretrained=false 시 무시됨
freeze_backbone_train: false
warmup_epochs: 10               # 권장: 충분한 warmup 필요
```

랜덤 초기화된 ResNet50에서 출발한다.

| 단계 | 옵티마이저 | 데이터 | 학습 대상 | Loss |
|------|-----------|--------|-----------|------|
| Warmup | SGD (`warmup_lr`) | `unlabeled` (정상) | backbone + projection | `‖z − c‖²` |
| 본 학습 | Adam (`lr`) | 전체 | backbone + projection | Deep SAD |

> feature space가 무의미한 상태에서 시작하므로 warmup 에포크를 충분히 설정한다.
> 도메인이 ImageNet과 매우 다를 때 (의료, 산업 특수 이미지 등) 유리하다.

---

### 케이스 2 — Pretrained + Warmup 동결 + 본 학습 해제

```yaml
pretrained: true
freeze_backbone_warmup: true
freeze_backbone_train: false
warmup_epochs: 5
```

warmup 동안 backbone을 동결해 projection을 안정적으로 초기화하고, 이후 backbone도 함께 fine-tune한다.

| 단계 | 옵티마이저 | 데이터 | 학습 대상 | Loss |
|------|-----------|--------|-----------|------|
| Warmup | SGD (`warmup_lr`) | `unlabeled` (정상) | **projection만** (backbone 동결) | `‖z − c‖²` |
| 본 학습 | Adam (`lr`) | 전체 | backbone + projection (동결 해제) | Deep SAD |

> backbone fine-tune으로 도메인 특화가 가능하지만, lr이 높으면 feature drift로 AUC가 하락할 수 있다.
> `lr`을 낮게 설정하거나 `onecycle` 스케줄러와 함께 사용하는 것을 권장한다.

---

### 케이스 3 — Pretrained + 전구간 동결 (projection only)

```yaml
pretrained: true
freeze_backbone_warmup: true
freeze_backbone_train: true
warmup_epochs: 5
```

warmup과 본 학습 모두 backbone을 동결하고 projection head만 학습한다.

| 단계 | 옵티마이저 | 데이터 | 학습 대상 | Loss |
|------|-----------|--------|-----------|------|
| Warmup | SGD (`warmup_lr`) | `unlabeled` (정상) | **projection만** (backbone 동결) | `‖z − c‖²` |
| 본 학습 | Adam (`lr`) | 전체 | **projection만** (backbone 동결 유지) | Deep SAD |

> backbone feature가 고정되므로 feature drift가 없고 center `c`와의 정렬이 유지된다.
> pretrained feature 품질이 충분한 경우 (예: CIFAR-10, ImageNet 유사 도메인) 가장 안정적이다.

---

### 케이스 4 — Pretrained + 동결 없음

```yaml
pretrained: true
freeze_backbone_warmup: false
freeze_backbone_train: false
warmup_epochs: 3
```

warmup부터 전체 파라미터를 함께 학습한다.

| 단계 | 옵티마이저 | 데이터 | 학습 대상 | Loss |
|------|-----------|--------|-----------|------|
| Warmup | SGD (`warmup_lr`) | `unlabeled` (정상) | backbone + projection | `‖z − c‖²` |
| 본 학습 | Adam (`lr`) | 전체 | backbone + projection | Deep SAD |

> backbone이 warmup 초반부터 drift될 수 있다.
> 데이터가 충분하고 도메인이 ImageNet과 유사할 때 적합하다.

---

## 전처리

`pretrained` 값에 따라 입력 정규화 방식이 달라진다.

| `pretrained` | 정규화 |
|---|---|
| `True` | ImageNet 통계 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| `False` | `ToTensor`만 적용 (픽셀값 `[0, 1]`) |

학습과 테스트에 동일한 전처리가 자동으로 적용된다.
체크포인트에 `pretrained` 값이 저장되므로 `test.py`에서 별도 설정 없이 일치가 보장된다.

---

## Loss 함수

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \eta_i \cdot \ell(\mathbf{z}_i, \mathbf{c})
$$

$$
\ell(\mathbf{z}, \mathbf{c}) = \begin{cases} \|\mathbf{z} - \mathbf{c}\|^2 & y_i \in \{0, +1\} \\ \left(\|\mathbf{z} - \mathbf{c}\|^2 + \varepsilon\right)^{-1} & y_i = -1 \end{cases}
\qquad
\eta_i = \begin{cases} \eta & y_i \neq 0 \\ 1 & y_i = 0 \end{cases}
$$

- 정상 샘플은 센터 **c** 쪽으로 당긴다.
- 이상 샘플은 센터에서 밀어낸다.
- 레이블 있는 샘플(`y ≠ 0`)에 η 배 가중치를 부여한다.

---

## 스케줄러

`scheduler` 파라미터로 세 가지 방식 중 하나를 선택한다.

### `multistep` (기본)

`milestone` 에포크에서 lr을 ×0.1 스텝다운한다.

```
lr ──────────────────┐ milestone
                      └── lr × 0.1 ──────────
```

```yaml
scheduler: multistep
milestone: 50
```

### `onecycle`

선형 warmup → 코사인 decay로 구성된 OneCycleLR 개형 스케줄이다.

```
lr      ╱‾‾‾╲
       ╱      ╲________________________
lr/div_factor                   lr/final_div_factor
```

- **warmup 구간** `[0, pct_start × n_epochs)` : `lr/div_factor` → `lr` 선형 상승
- **decay 구간** `[pct_start × n_epochs, n_epochs)` : `lr` → `lr/final_div_factor` 코사인 하강

```yaml
scheduler: onecycle
onecycle_pct_start: 0.3
onecycle_div_factor: 25.0
onecycle_final_div_factor: 1.0e4
```

### `combined`

OneCycleLR 개형 위에 `milestone`에서 ×0.1 스텝다운을 곱한 복합 스케줄이다.

```
lr      ╱‾‾‾╲
       ╱      ╲____________┐ milestone
      ╱                     └── × 0.1 (코사인 계속)
lr/div_factor           lr/final_div_factor
```

- **warmup 구간** `[0, pct_start × n_epochs)` : `lr/div_factor` → `lr` 선형 상승
- **decay 구간** `[pct_start × n_epochs, n_epochs)` : `lr` → `lr/final_div_factor` 코사인 하강
- `milestone` 이후 전체 lr에 ×0.1 추가 적용

```yaml
scheduler: combined
milestone: 50
onecycle_pct_start: 0.3
onecycle_div_factor: 25.0
onecycle_final_div_factor: 1.0e4
```

학습 시작 시 `lr_schedule.png`가 출력 디렉터리에 자동 저장된다.

---

## 학습 흐름

```
[1단계]   센터 c 초기화   ← unlabeled 데이터의 네트워크 출력 평균
[1.5단계] Warmup          ← 정상 데이터만, ‖z − c‖² 최소화
          센터 c 재초기화 ← warmup 후 변경된 모델 기준으로 재계산
[2단계]   본 학습         ← 전체 데이터, Deep SAD loss, n_epochs epoch
[3단계]   최종 평가       ← Best 체크포인트 로드 후 테스트셋 AUC
```

검증(AUC 계산)은 매 에포크 수행되며, Best AUC 갱신 시 `best.pt`가 저장된다.
`last.pt`는 매 에포크마다 덮어쓴다. `save_interval > 0`이면 해당 주기마다 `epoch_XXXX.pt`를 추가로 저장한다.

---

## 실행

```bash
# 데이터 준비 (CIFAR-10 기반 테스트 데이터셋)
python tests/make_cifar_dataset.py

# 학습 — 기본값 사용 (train.yaml 없을 때)
python train.py

# 학습 — YAML 파일 명시
python train.py --config configs/exp1.yaml

# 추론 및 시각화 — 기본값 사용 (test.yaml 없을 때)
python test.py

# 추론 및 시각화 — YAML 파일 명시
python test.py --config configs/test_exp1.yaml

# 도움말
python train.py --help
python test.py --help
```
