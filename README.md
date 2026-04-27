# PPE Detection on Jetson Orin Nano

실시간 웹캠 영상에서 안전 장비(헬멧, 마스크, 장갑)를 감지하는 PPE(Personal Protective Equipment) 탐지 시스템입니다.  
YOLO-World와 YOLO Pose 모델을 활용하며, NVIDIA Jetson Orin Nano Super에서 동작합니다.

---

## 실험 환경 (Hardware Spec)

| 항목 | 사양 |
|------|------|
| 보드 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| CPU | Cortex-A78AE (6-core) |
| RAM | 8GB (CPU/GPU 공유) |
| 저장공간 | 128GB microSD |
| OS | Ubuntu 22.04 (JetPack R36.4.7) |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| 카메라 | ArduCAM IMX477 (CSI) |

---

## 파일 구조

```
capstone_2601/
├── models/                          # 모델 가중치 파일 (별도 준비 필요)
│   ├── yolov8s-worldv2.pt           # YOLO-World pt 모델
│   ├── yolov8s-worldv2.engine       # TensorRT 변환 모델
│   ├── yolo11n-pose.pt              # YOLO Pose pt 모델
│   └── yolo11n-pose.engine          # TensorRT 변환 모델
├── ppe_detection.py                 # PPE 탐지 (.pt 버전)
├── ppe_detection_with_trt.py        # PPE 탐지 (TensorRT .engine 버전, 권장)
├── export_trt.py                    # .pt → .engine 변환 스크립트
└── requirements.txt                 # 패키지 목록
```

---

## 설치 과정

### 1. JetPack 버전 확인

```bash
sudo apt show nvidia-jetpack -a
cat /etc/nv_tegra_release
```

### 2. 시스템 패키지 설치

```bash
sudo apt-get -y update
sudo apt-get install -y python3-pip libopenblas-dev nano
```

### 3. cusparseLt 설치 (CUDA 12.6용)

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.2.3-archive.tar.xz
tar xf libcusparse_lt-linux-aarch64-0.6.2.3-archive.tar.xz
sudo cp libcusparse_lt-linux-aarch64-0.6.2.3-archive/include/* /usr/local/cuda/include/
sudo cp libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig
```

### 4. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 5. 가상환경 생성

```bash
cd ~/Documents/capstone_2601
uv venv PPEDetector --python 3.10
source PPEDetector/bin/activate
```

### 6. Jetson용 PyTorch 설치

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
mv torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl torch-2.5.0a0-cp310-cp310-linux_aarch64.whl
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install torch-2.5.0a0-cp310-cp310-linux_aarch64.whl
```

CUDA 연동 확인:
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 출력: 2.5.0a0+872d972e41.nv24.08 / True
```

### 7. torchvision 소스 빌드

```bash
# 빌드 의존성 설치
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

# 소스 다운로드
git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
cd torchvision

# 빌드 (약 30~60분 소요)
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
export BUILD_VERSION=0.20.0
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"
python3 setup.py bdist_wheel

# 설치
uv pip install dist/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl

cd ..
```

torchvision `_meta_registrations.py` 패치 (nms 오류 방지):
```bash
python3 -c "
import re
path = 'PPEDetector/lib/python3.10/site-packages/torchvision/_meta_registrations.py'
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'@register_meta.*?def meta_nms.*?return torch\.empty.*?\n', '', content, flags=re.DOTALL)
with open(path, 'w') as f:
    f.write(content)
print('Done')
"
```

### 8. numpy 버전 고정

```bash
uv pip install "numpy<2"
```

### 9. TensorRT 파이썬 바인딩 연결

```bash
ln -s /usr/lib/python3.10/dist-packages/tensorrt \
  ~/Documents/capstone_2601/PPEDetector/lib/python3.10/site-packages/tensorrt

python3 -c "import tensorrt as trt; print(trt.__version__)"
# 출력: 10.3.0
```

### 10. GStreamer 지원 OpenCV 연결

```bash
# 기존 OpenCV 제거
uv pip uninstall opencv-python opencv-contrib-python

# 시스템 OpenCV(GStreamer 포함) 설치
sudo apt install -y python3-opencv

# 심볼릭 링크 연결
rm ~/Documents/capstone_2601/PPEDetector/lib/python3.10/site-packages/cv2
ln -s /usr/lib/python3.10/dist-packages/cv2 \
  ~/Documents/capstone_2601/PPEDetector/lib/python3.10/site-packages/cv2

# GStreamer 지원 확인
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
# 출력: GStreamer: YES
```

### 11. 나머지 패키지 설치

```bash
# requirements.txt에서 충돌 패키지 제거 후 설치
sed 's/==.*//' requirements.txt | sed 's/@ .*//' > requirements_nover.txt
uv pip install -r requirements_nover.txt
```

### 12. 최대 성능 모드 설정

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 모델 파일 준비

### pt 파일 다운로드 (Windows/Linux PC에서)

```python
from ultralytics import YOLO
YOLO('yolov8s-worldv2.pt')   # 자동 다운로드
YOLO('yolo11n-pose.pt')       # 자동 다운로드
```

자동 다운로드 실패 시 아래 링크에서 수동 다운로드:
- [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt)
- [yolo11n-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt)

다운로드된 파일을 `models/` 폴더에 복사.

### TensorRT 엔진 변환 (Jetson에서 직접 실행)

```bash
source PPEDetector/bin/activate
python3 export_trt.py
# 약 10~20분 소요
# models/yolov8s-worldv2.engine, models/yolo11n-pose.engine 생성
```

---

## 실행

```bash
cd ~/Documents/capstone_2601
source PPEDetector/bin/activate

# TensorRT 엔진 버전 (권장, 더 빠름)
python3 ppe_detection_with_trt.py

# PyTorch pt 버전
python3 ppe_detection.py
```

종료: `q` 키

---

## 탐지 항목

| 항목 | 색상 | 설명 |
|------|------|------|
| Person | 파란색 | 사람 바운딩 박스 |
| Helmet | 노란색 | 헬멧 감지 |
| Mask | 보라색 | 마스크 감지 |
| Glove | 초록색 | 장갑 바운딩 박스 |
| Wrist ROI (좌) | 하늘색 | 왼손 손목 ROI |
| Wrist ROI (우) | 핑크색 | 오른손 손목 ROI |

PPE 상태는 화면 상단에 `P1 0.93 | O | O | O | O` 형식으로 표시됩니다.  
(순서: 헬멧 | 마스크 | 왼손 장갑 | 오른손 장갑, O=착용, X=미착용)

---

## 성능

| 모드 | FPS (탐지 없을 때) | FPS (탐지 시) |
|------|-------------------|--------------|
| .pt 버전 | ~14 | ~11 |
| .engine 버전 | ~20 | ~13 |

---

## 탐지 결과

![PPE Detection Result](assets/result_image.png)

> `.engine` 버전으로 탐지한 결과 이미지.  
> 헬멧(노란색), 마스크(보라색), 장갑(초록색) 탐지 및 손목 ROI(하늘/핑크색) 표시.
