# test.py — Deep SAD 추론 및 시각화

## 개요

학습된 Deep SAD 모델로 실제 테스트 데이터를 추론하고, 오분류 샘플(FP/FN)과 정탐 이상 샘플(TP)을 주석이 달린 이미지 그리드로 시각화합니다.

## 데이터 구조

```
TEST_ROOT/
  normal/    # label = +1 (정상)
  anomaly/   # label = -1 (이상)
```

## 출력 구조

```
result/
  FP/                  # FP 개별 이미지
  FN/                  # FN 개별 이미지
  TP/                  # TP 개별 이미지
  FP_grid.png          # FP 샘플 그리드
  FN_grid.png          # FN 샘플 그리드
  TP_grid.png          # TP 샘플 그리드
```

| 레이블 | 정의 |
|--------|------|
| **FP** | 정상인데 이상으로 분류 (score > threshold, label = +1) |
| **FN** | 이상인데 정상으로 분류 (score < threshold, label = -1) |
| **TP** | 이상을 이상으로 올바르게 분류 (score > threshold, label = -1) |

## 사용법

```bash
# 기본 실행: test.yaml이 있으면 로드, 없으면 기본값 사용
python test.py

# 설정 파일 직접 지정
python test.py --config configs/test.yaml
```

## 설정 파라미터 (`configs/test.yaml`)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `checkpoint` | `out/deepsad_best.pt` | 로드할 체크포인트 경로 |
| `test_root` | `data/test_data/test` | `normal/`, `anomaly/` 하위 폴더가 있는 테스트 루트 경로 |
| `result_dir` | `result` | 결과 이미지 저장 루트 경로 |
| `seed` | `42` | 랜덤 시드 |
| `batch_size` | `64` | 추론 배치 크기 |
| `num_workers` | `4` | DataLoader 워커 수 |
| `img_size` | `224` | 입력 이미지 크기 (픽셀) |
| `threshold_percentile` | `90` | 이상 임계값 = 전체 점수의 N번째 백분위수 |
| `num_tp_samples` | `50` | 시각화할 TP 샘플 최대 수 (랜덤 샘플링) |
| `device` | `cuda` | `cuda` 또는 `cpu` |

## 실행 단계

1. **체크포인트 로드** — `.pt` 파일에서 모델 가중치와 초구 중심 `c` 복원
2. **데이터 로드** — `test_root/normal/`, `test_root/anomaly/`로부터 테스트 DataLoader 구성
3. **이상 점수 계산** — 각 샘플의 이상 점수 `||z - c||²` 계산
4. **임계값 설정** — 전체 점수의 `threshold_percentile`번째 백분위수를 임계값으로 사용
5. **샘플 분류** — FP / FN / TP 인덱스 식별; FP는 점수 내림차순, FN은 오름차순, TP는 랜덤 샘플링
6. **시각화 및 저장** — 주석이 달린 개별 이미지와 그리드 PNG 저장 후 OS 기본 이미지 뷰어로 자동 열기

## 체크포인트 형식

체크포인트(`.pt`) 파일에 다음 키가 포함되어야 합니다.

| 키 | 내용 |
|----|------|
| `model_state` | `DeepSAD_ResNet50` state dict |
| `c` | 초구 중심 텐서 |
| `cfg` | `TrainConfig` 필드 (dict) |
| `epoch` | 마지막 저장 에포크 |
| `best_auc` | 최고 검증 AUC |
