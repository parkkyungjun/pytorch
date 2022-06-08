# 2022 인공지능 온라인 경진대회
## [이미지] 준지도학습 기반의 항만 구조물 객체 분할 문제

해당 코드는 https://github.com/lorenmt/reco 을 참조하여 경진대회 문제에 맞게 변형한 코드임을 밝힙니다.
### 코드 구조

```
${PROJECT}
├── config/
│   ├── train_config.yml
│   └── predict_config.yml
├── models/
│   ├── deeplabv3/
│   ├── deeplabv2.py
│   └── utils.py
├── modules/
│   ├── datasets.py
│   ├── earlystopper.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── recorders.py
│   ├── schedulers.py
│   ├── trainer.py
│   └── utils.py
├── README.md
├── train.py
└── predict.py
```

- config : 학습/추론에 필요한 파라미터 등을 기록하는 yml 파일
- models  
    - deeplabv3/ : deeplab v3에 대한 source code
    - deeplabv2.py : deeplab v2에 대한 source code
    - utils.py : config에서 지정한 모델을 리턴
- modules
    - datasets.py : dataset 클래스
    - earlystopper.py : loss가 특정 에폭 이상 개선되지 않을 경우 멈춤
    - losses.py : config에서 지정한 loss function을 리턴
    - metrics.py : config에서 지정한 metric을 리턴
    - optimizers.py : config에서 지정한 optimizer를 리턴
    - recorders.py : log, learning curve 등을 기록
    - schedulers.py
    - trainer.py : 에폭 별로 수행할 학습 과정
    - utils.py : 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 등의 함수
- train.py : 학습 시 실행하는 코드
- predict.py : 추론 시 실행하는 코드

---
### 데이터 경로
```
${DATADIR}
├── train/
│   ├── labeled_images/ : fully label이 존재하는 image 폴더
│   ├── labels/ : labeled_images에 대응하는 라벨
│   └── unlabeled_images/
└── test/
    └── images/
```
1. train
   1. labeled_images : fully label이 존재하는 image 폴더
   2. labels : labeled_images에 대응하는 라벨
   3. unlabeled_images : unlabeled image가 존재하는 폴더
2. test
   1. images : 학습한 모델로 추론해야 하는 image가 존재하는 폴더


### 학습

1. `config/train_config.yaml` 수정
    1. DIRECTORY/dataset : 데이터 경로 지정 (DATADIR에 해당하는 경로)
    2. 이외 파라미터 조정
2. `python train.py`
3. `results/train/` 내에 결과(weight, log, plot 등)가 저장됨


### 추론

1. `config/predict_config.yaml` 수정
    1. DIRECTORY/dataset : 데이터 경로 지정 (sample_submission.csv이 위치한 디렉토리)
    2. TRAIN/train_serial : weight를 불러올 train serial number (result/train 내 폴더명) 지정
2. `python predict.py`
3. `results/predict/` 내에 결과 파일(submission.csv)이 저장됨

