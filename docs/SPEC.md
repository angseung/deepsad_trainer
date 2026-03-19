# 구현에 사용하는 패키지

- Ultralytics Yolo : 모델 학습, 추론
- Gradio : web ui 빌드
- PyQT6 : web ui에서 사용하는 파일 브라우저 호출

---

# 기능 동작 설계

## UI

- 아래 5개의 탭으로 구성됨
    - 데이터 재구성
    - 학습 설정
    - 학습 터미널
    - 정량적 성능 검증
    - 테스트

## 데이터 재구성
- 데이터셋의 클래스 별 샘플 수의 불균형을 해소한 데이터셋으로 재구성하는 기능
- 동작 방식
    - 원본 데이터셋에서 복원 추출하고, 표본의 클래스 별 샘플 수를 역수로 하여 가중치 벡터를 생성
    - 해당 가중치만큼의 비율로 데이터셋의 클래스 별 샘플 수를 재구성
        - 샘플 수가 많은 클래스 -> 일부 제거
        - 샘플 수가 적은 클래스 -> 이미지 복사됨


## 학습 설정

1. UI에서 학습 관련 파라미터, 하이퍼 파라미터, Augmentation 설정 등을 사용자가 입력하도록 함
    1. Resume 관련 기능
        1. Resume 기능을 사용하지 않을 때에는 유저가 모든 설정 값을 직접 설정할 수 있음
        2. Resume 기능을 사용할 때에는 중단된 pt 파일만 입력하며, 다른 설정 값은 변경할 수 없음
    2. 사전 학습 모델 관련 기능
        1. 사전 학습 모델을 사용할 경우에는 해당 모델의 pt파일을 입력
        2. 사전 학습 모델을 사용하지 않는 경우에는 모델 설정  YAML 파일 경로를 입력
2. 설정  값을 YAML 파일로 구성
    1. 구성된 YAML 파일 예시
    
    ```yaml
    model: yolov8n.pt
    data: coco128.yaml
    epochs: 100
    batch: 16
    imgsz: 640
    lr0: 0.01
    lrf: 0.01
    momentum: 0.937
    weight_decay: 0.0005
    project: runs/detect
    name: coco_exp
    save_period: -1
    device: '0'
    workers: 8
    patience: 50
    val: true
    resume: false
    amp: true
    rect: false
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 0.0
    translate: 0.1
    scale: 0.5
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.0
    copy_paste: 0.0
    erasing: 0.4
    crop_fraction: 1.0
    auto_augment: randaugment
    ```
    
3. 구성한 YAML 파일을 입력으로 하는 Ultralytics YOLO 학습 Python 코드 생성
4. 생성한 Python 코드를 실행

## 학습 터미널

1. 학습 코드의 터미널 출력을 Web UI의 학습 로그 탭에 출력

## 정량적 성능 검증

1. 검증할 모델의 pt 파일 경로, 데이터셋 YAML 파일 경로, Confidence/IoU 임계 값을 유저가 설정
2. 유저 설정을 바탕으로 Ultalytics YOLO 추론 코드 실행
3. 전체 클래스의 Metric 및 클래스 별 Meric을 Web UI의 검증 결과 탭에 출력

## 테스트

1. 검증할 모델의 pt 파일 경로, 원본 테스트 이미지 경로, 테스트 결과 이미지 저장 경로, Confidence/IoU 임계 값을 유저가 설정
    1. 테스트 이미지는 파일, 또는 이미지로 구성된 폴더로 입력
2. 유저 설정을 바탕으로 Ultalytics YOLO 추론 코드 실행 → 원본 이미지 위에 검출 결과 (Bbox, Class, Score)가 그려진 이미지가 저장됨
3. 검출 결과가 그려진 이미지 경로에서 이미지를 불러와 Web UI에 출력
    1. 이미지 폴더로 입력한 경우에는 좌우 이동 버튼으로 이전/다음 이미지를 출력할 수 있도록 함