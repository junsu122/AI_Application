# 🚀 Advanced Computer Vision & Generative AI Library

> **[NOTICE]** 본 저장소는 모든 파일이 코랩을 이용한 코드로 이루어져 있습니다. 원활한 실행을 위해서는 코랩 사용과 GPU(T4이상)사용을 추천합니다. (최근 업데이트: 2024-05-22)

---

## 📌 Project Overview
본 내용들은 두산 ROKEY 7기 내용이며, AI를 이용한 여러 기술들을 보여줍니다. 

---

## 🛠 Core Technologies

### 1. GAN (Generative Adversarial Networks)
데이터 부족 문제를 해결하기 위한 이미지 증강 및 생성 모델입니다.
* **Models:** DCGAN, CycleGAN, Pix2Pix
* **Target:** MNIST 숫자 생성 및 도심 환경/기상 변화 시뮬레이션 데이터 구축
* **Key Concept:** Generator와 Discriminator의 경쟁적 학습을 통한 고해상도 이미지 합성



### 2. XAI (Explainable AI)
AI 모델의 의사결정 과정을 시각화하여 엔지니어링 결과의 신뢰성을 확보합니다.
* **Techniques:** Grad-CAM, LIME, SHAP
* **Target:** 자율주행 판단 근거 시각화 및 산업 시설물 결함 탐지 부위 특정
* **Key Concept:** 신경망의 마지막 컨볼루션 레이어 활성화를 역추적하여 중요도 Heatmap 생성



### 3. Neural Style Transfer (신경망 스타일 전이)
콘텐츠의 구조는 유지하면서 스타일만 변경하는 기법입니다.
* **Models:** VGG-19 based NST
* **Target:** 주간 도로 영상을 야간/우천 주행 영상으로 변환하여 자율주행 모델 학습용 가상 데이터 생성
* **Key Concept:** Content Loss와 Style Loss(Gram Matrix)의 최적화



### 4. Image Captioning
이미지의 상황을 텍스트로 설명하는 멀티모달(Multimodal) 기술입니다.
* **Architecture:** CNN Encoder (Feature Extraction) + RNN/Transformer Decoder
* **NLP Processing:** KoNLPy(Mecab)를 활용한 한국어 토큰화 및 임베딩 최적화
* **Target:** 산업 현장 CCTV 영상의 자동 상황 보고서 생성



### 5. Object Tracking (추적 알고리즘)
연속적인 프레임에서 객체의 ID를 유지하며 위치를 추적합니다.
* **Algorithms:** DeepSORT, ByteTrack, Kalman Filter
* **Target:** 스마트 도시 내 차량/보행자 흐름 분석 및 작업자 안전 거리 모니터링
* **Key Concept:** 객체 탐지(Detection) 결과에 헝가리안 알고리즘(Hungarian Algorithm)을 적용한 데이터 연관(Data Association)



---

## 📊 Performance & Verification
각 모델은 코랩(Google Colab) T4 GPU 환경에서 학습되었으며, 구체적인 Loss Curve 및 정확도 지표는 각 폴더의 `.ipynb` 파일 내에서 확인할 수 있습니다.

* **Dataset:** MNIST, COCO, AI Hub Industry Dataset
* **Optimization:** Adam Optimizer, Learning Rate Scheduling

---

## 📂 Directory Structure
```text
.
├── 📂 GAN/                # DCGAN, CycleGAN 구현
├── 📂 XAI/                # Grad-CAM 시각화 코드
├── 📂 Style-Transfer/     # VGG기반 스타일 전이
├── 📂 Captioning/         # Encoder-Decoder 캡셔닝 모델
├── 📂 Tracking/           # 객체 추적 알고리즘
└── README.md              # Project Description
