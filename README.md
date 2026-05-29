# PPE Detection on Jetson Orin Nano

실시간 카메라 영상에서 작업자의 PPE(안전모, 마스크, 왼손 장갑, 오른손 장갑) 착용 상태를 판단하고, 구역별 단속 기준에 따라 미착용 위반 사건을 MariaDB에 자동 적재하는 Jetson 기반 온디바이스 PPE 감지 시스템입니다.

YOLO-World 기반 PPE 객체 탐지, YOLO Pose 기반 관절 추정, ByteTrack 기반 사람 ID 추적, MJPEG 스트리밍, 구역별 PPE 단속 기준 반영, 위반 이미지 BLOB 저장을 하나의 파이프라인으로 구성합니다.

---

## 주요 기능

- **실시간 PPE 탐지**: YOLO-World TensorRT 엔진으로 `person`, `helmet`, `mask`, `glove`, `vest` 객체 탐지
- **Pose 기반 손 영역 판단**: YOLO Pose의 손목/팔꿈치 keypoint를 활용해 왼손·오른손 장갑 착용 여부 판별
- **person 단위 추적**: `supervision.ByteTrack`으로 카메라별 작업자 ID 부여
- **구역별 PPE 단속 기준 반영**: `areas.enforce_*` 컬럼에 따라 안전모, 마스크, 왼손 장갑, 오른손 장갑 단속 여부를 구역별로 제어
- **단속 제외 장비 표시/판정 제외**: 단속하지 않는 장비는 ROI와 탐지 박스를 표시하지 않고, 위반 판정에서도 착용 상태로 보정
- **동적 설정 재로드**: DB의 구역-카메라 매핑 및 PPE 단속 기준을 주기적으로 다시 로드
- **위반 자동 적재**: 미착용 상태가 일정 시간 이상 지속되면 person bbox 크롭 이미지를 MariaDB `violations` 테이블에 BLOB으로 저장
- **MJPEG 스트리밍**: Jetson 보드에서 5001 포트로 카메라별 MJPEG 스트림 제공
- **다중 카메라 지원**: CSI / USB / RealSense 계열 V4L2 컬러 노드 자동 탐지 및 순차 추론

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
| 카메라 | ArduCAM IMX477 (CSI) + USB 웹캠 + Intel RealSense 계열 USB 컬러 노드 |
| DB | AWS Lightsail MariaDB 10.11 |

---

## 파일 구조

```text
ppe_detector_capstone_2601/
├── models/                          # 모델 가중치 및 TensorRT 엔진
│   ├── yolov8s-worldv2.pt / .engine # YOLO-World (person/glove/helmet/mask/vest)
│   └── yolo11n-pose.pt / .engine    # YOLO Pose
├── ppe_detection.py                 # PPE 탐지 (.pt 버전)
├── ppe_detection_with_trt.py        # PPE 탐지 + 스트리밍 + 위반 적재 (TensorRT, 권장)
├── export_trt.py                    # .pt → .engine 변환
├── db_config.py                     # DB 접속 설정 (.env 로드)
├── area_loader.py                   # areas 테이블 → camera_key 매핑 및 PPE 단속 기준 로드
├── violation_tracker.py             # 위반 상태 추적 (지속 시간 + cooldown)
├── image_utils.py                   # person bbox 크롭 + JPEG 인코딩
├── violation_logger.py              # 비동기 BLOB INSERT 워커
├── extract_image.py                 # DB BLOB → 파일 추출 (검증용)
├── .env                             # DB 접속 정보 (gitignore)
└── requirements.txt
```

---

## 전체 처리 흐름

```text
카메라 프레임 수집
  ↓
YOLO-World 객체 탐지
  - person / helmet / mask / glove / vest
  ↓
YOLO Pose 관절 추정
  - nose / shoulder / wrist / elbow 등
  ↓
ByteTrack으로 person_id 부여
  ↓
camera_key → area_id 매핑
  ↓
area_id → 구역별 PPE 단속 기준 로드
  ↓
단속 대상 장비만 ROI/박스/상태 표시
  ↓
단속 제외 장비는 착용(True)으로 보정
  ↓
ViolationStateTracker가 미착용 지속 시간 판단
  ↓
위반 발생 시 person crop JPEG를 MariaDB violations 테이블에 저장
  ↓
MJPEG 스트림으로 관제 서버/WinForms 클라이언트에 송출
```

---

## 구역별 PPE 단속 기준

### 개념

각 구역은 `areas` 테이블의 `enforce_*` 컬럼을 통해 단속할 장비를 선택합니다.

| 컬럼 | 의미 |
|------|------|
| `enforce_helmet` | 1이면 안전모 단속 |
| `enforce_mask` | 1이면 마스크 단속 |
| `enforce_glove_left` | 1이면 왼손 장갑 단속 |
| `enforce_glove_right` | 1이면 오른손 장갑 단속 |

예를 들어 B구역에서 마스크 단속을 끄고 나머지 장비만 단속하려면 다음과 같이 저장됩니다.

```text
area_id=20, area_name=B구역
enforce_helmet      = 1
enforce_mask        = 0
enforce_glove_left  = 1
enforce_glove_right = 1
```

이 경우 Jetson 화면에서는 마스크 ROI/마스크 박스를 표시하지 않고, `no_mask` 위반도 저장하지 않습니다. 상태바 역시 단속 대상 장비만 표시하므로 `안전모 | 왼손 장갑 | 오른손 장갑` 3개 항목만 O/X로 표시됩니다.

### areas 테이블 마이그레이션

기존 DB에 `enforce_*` 컬럼이 없다면 아래 SQL을 실행합니다.

```sql
ALTER TABLE areas
ADD COLUMN enforce_helmet TINYINT(1) NOT NULL DEFAULT 1,
ADD COLUMN enforce_mask TINYINT(1) NOT NULL DEFAULT 1,
ADD COLUMN enforce_glove_left TINYINT(1) NOT NULL DEFAULT 1,
ADD COLUMN enforce_glove_right TINYINT(1) NOT NULL DEFAULT 1;
```

컬럼이 이미 일부 존재한다면 없는 컬럼만 개별 추가해야 합니다.

### area_loader.py 동작

`area_loader.py`는 두 가지 정보를 DB에서 로드합니다.

| 함수 | 반환 |
|------|------|
| `load_camera_area_map()` | `camera_key → area_id` 매핑 |
| `load_area_ppe_rules()` | `area_id → {helmet, mask, left_glove, right_glove}` 단속 기준 |

`enforce_*` 컬럼이 없는 구버전 DB에서는 예외를 처리하고 모든 장비를 단속하는 기본값으로 복구합니다.

---

## 위반 적재 시스템

### 알고리즘

`(cam_key, person_id, violation_type)` 단위로 미착용 상태를 추적합니다.

| 설정 | 현재 값 | 의미 |
|------|--------|------|
| `SUSTAIN_SECONDS` | 60초 | 미착용이 이 시간 이상 지속되면 위반 발생 |
| `COOLDOWN_SECONDS` | 300초 | 같은 `(카메라, person_id, type)`은 이 시간 동안 재기록 안 함 |

PPE 위반 type은 4종입니다.

```text
no_helmet
no_mask
no_glove_left
no_glove_right
```

단속 제외 장비는 `ppe_detection_with_trt.py`에서 착용 상태(`True`)로 보정한 뒤 `ViolationStateTracker.update()`에 전달되므로, 해당 장비의 위반 이벤트는 발생하지 않습니다.

### 증거 이미지 처리

- 원본 frame에서 person bbox 크롭
- crop 및 JPEG 인코딩은 `image_utils.crop_and_encode()`에서 수행
- MariaDB `violations.image_data` 컬럼에 MEDIUMBLOB으로 비동기 INSERT
- 위반 저장은 `ViolationLogger` 워커가 담당

### DB 스키마 (`capstone_db.violations`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | INT AUTO_INCREMENT | PK |
| `violation_type` | VARCHAR(50) | `no_helmet` / `no_mask` / `no_glove_left` / `no_glove_right` |
| `detected_at` | DATETIME | 위반 발생 시각 |
| `area_id` | INT | `areas.area_id` |
| `person_id` | INT | ByteTrack 부여 ID |
| `image_data` | MEDIUMBLOB | 크롭된 person JPEG |
| `image_mime` | VARCHAR(20) | 보통 `image/jpeg` |
| `is_checked` | TINYINT(1) | 처리 여부 (0=미처리, 1=처리됨) |

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
sudo apt-get install -y python3-pip libopenblas-dev nano autossh putty-tools v4l-utils
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
cd ~/Documents/ppe_detector_capstone_2601
uv venv PPEDetector --python 3.10
source PPEDetector/bin/activate
```

### 6. Jetson용 PyTorch 설치

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
mv torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl torch-2.5.0a0-cp310-cp310-linux_aarch64.whl
```

> `$(which python3)`는 시스템 Python(`/usr/bin/python3`)을 가리킬 수 있으므로, 가상환경 Python 경로를 직접 지정합니다.

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  --python ~/Documents/ppe_detector_capstone_2601/PPEDetector/bin/python3 \
  torch-2.5.0a0-cp310-cp310-linux_aarch64.whl
```

#### python3 alias 등록

```bash
echo 'alias python3="~/Documents/ppe_detector_capstone_2601/PPEDetector/bin/python3"' >> ~/.bashrc
source ~/.bashrc

type python3
```

CUDA 연동 확인:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 7. torchvision 소스 빌드

```bash
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
cd torchvision

export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
export BUILD_VERSION=0.20.0
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"
python3 setup.py bdist_wheel

uv pip install dist/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl

cd ..
```

torchvision `_meta_registrations.py` 패치:

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
  ~/Documents/ppe_detector_capstone_2601/PPEDetector/lib/python3.10/site-packages/tensorrt

python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### 10. GStreamer 지원 OpenCV 연결

```bash
uv pip uninstall opencv-python opencv-contrib-python

sudo apt install -y python3-opencv

rm ~/Documents/ppe_detector_capstone_2601/PPEDetector/lib/python3.10/site-packages/cv2
ln -s /usr/lib/python3.10/dist-packages/cv2 \
  ~/Documents/ppe_detector_capstone_2601/PPEDetector/lib/python3.10/site-packages/cv2

python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
```

### 11. 일반 패키지 설치

```bash
sed 's/==.*//' requirements.txt | sed 's/@ .*//' > requirements_nover.txt
uv pip install -r requirements_nover.txt
```

### 12. DB / 트래커 관련 추가 패키지

```bash
uv pip install pymysql python-dotenv
uv pip install --no-deps supervision
uv pip install --no-deps defusedxml pydeprecate
```

`supervision`은 `--no-deps`로 설치하여 numpy 2.x / opencv-python 충돌을 피합니다.

### 13. `.env` 파일 작성

```bash
cat > .env << 'EOF'
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=<MariaDB 비밀번호>
DB_NAME=capstone_db
EOF
chmod 600 .env
```

`.env`는 반드시 `.gitignore`에 포함합니다.

### 14. SSH 터널 (DB 접속용) systemd 서비스 등록

AWS MariaDB는 외부로 직접 노출하지 않고 SSH 터널을 통해 접속합니다.

```text
Jetson:3306 → AWS:3308 → capstone_db
```

```bash
sudo tee /etc/systemd/system/ppe-db-tunnel.service > /dev/null << 'EOF'
[Unit]
Description=PPE DB SSH Forward Tunnel to AWS MariaDB
After=network-online.target
Wants=network-online.target
Before=ppe-stream.service

[Service]
Type=simple
User=nana1124
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/bin/autossh -M 0 -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -i /home/nana1124/Documents/ppe_detector_capstone_2601/.ssh/my_key.pem \
    -L 3306:localhost:3308 \
    ubuntu@<AWS_IP>
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ppe-db-tunnel
sudo systemctl start ppe-db-tunnel
sudo systemctl status ppe-db-tunnel
```

### 15. 최대 성능 모드 설정

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 모델 파일 준비

### pt 파일 다운로드

```python
from ultralytics import YOLO
YOLO('yolov8s-worldv2.pt')
YOLO('yolo11n-pose.pt')
```

다운로드된 파일을 `models/` 폴더에 복사합니다.

### TensorRT 엔진 변환

```bash
source PPEDetector/bin/activate
python3 export_trt.py
```

생성 파일:

```text
models/yolov8s-worldv2.engine
models/yolo11n-pose.engine
```

---

## 실행

### 개발 모드

```bash
cd ~/Documents/ppe_detector_capstone_2601
source PPEDetector/bin/activate

ss -tnlp | grep 3306

python3 ppe_detection_with_trt.py
```

종료는 `Ctrl+C`입니다.

### 운영 모드 (systemd)

```bash
sudo systemctl start ppe-stream
sudo systemctl stop ppe-stream
sudo systemctl status ppe-stream
sudo journalctl -u ppe-stream -f -o cat
```

### 중복 실행 방지

`ppe_detection_with_trt.py`는 `/tmp/ppe_detection.lock` 파일 잠금을 사용합니다. 이미 다른 인스턴스가 실행 중이면 즉시 종료하며, 필요 시 아래 명령으로 5001 포트 프로세스를 정리합니다.

```bash
sudo fuser -k 5001/tcp
```

---

## 카메라 자동 탐지 및 스트리밍

`ppe_detection_with_trt.py`는 `v4l2-ctl --list-devices` 출력과 `/dev/videoN --list-formats` 결과를 기반으로 카메라를 자동 탐지합니다.

| 대상 | 처리 방식 |
|------|-----------|
| CSI | `nvarguscamerasrc sensor-id=N` GStreamer 파이프라인 사용 |
| USB | MJPG GStreamer → raw GStreamer → V4L2 직접 접근 순으로 fallback |
| RealSense 계열 | USB 장치 중 컬러 포맷 노드만 통과, depth/IR/metadata 노드 제외 |

카메라별 key는 다음 우선순위로 생성됩니다.

```text
USB_{vid}_{pid}_{serial}
USB_{vid}_{pid}_PORT_{port_path}
USB_UNKNOWN_video{N}
CSI_{sensor_id}
```

Jetson은 `/cameras` API와 `/stream/<cam_name>` API를 제공합니다.

```text
GET http://<JETSON_IP>:5001/cameras
GET http://<JETSON_IP>:5001/stream/CAM0(USB0)
```

### 실행 로그 예시

```text
▶ areas 매핑 로드 중...
✓ 2개 카메라 매핑 로드:
  USB_2304_4922_PORT_1-2.3 → area_id=21
  USB_2304_4922_PORT_1-2.4 → area_id=20
✓ 2개 구역 PPE 단속 기준 로드:
  area_id=20 → {'helmet': True, 'mask': False, 'left_glove': True, 'right_glove': True}
  area_id=21 → {'helmet': True, 'mask': True, 'left_glove': True, 'right_glove': True}
▶ 카메라 탐지 중...
  → USB /dev/video0 시도 중... ✓ CAM0(USB0) 추가 [key=USB_...]
✓ MJPEG 스트리밍 서버 시작 (포트 5001)
✓ 추론 스레드 시작 (카메라 순차 처리)
```

---

## 설정값

| 설정값 | 위치 | 설명 |
|--------|------|------|
| `CSI_W`, `CSI_H`, `CSI_FPS` | `ppe_detection_with_trt.py` | CSI 카메라 해상도/프레임레이트 |
| `WB_MODE` | `ppe_detection_with_trt.py` | CSI 화이트밸런스 모드 |
| `CSI_B_GAIN`, `CSI_G_GAIN`, `CSI_R_GAIN` | `ppe_detection_with_trt.py` | CSI B/G/R 보정 게인 |
| `USB_W`, `USB_H`, `USB_FPS` | `ppe_detection_with_trt.py` | USB 웹캠 해상도/프레임레이트 |
| `OVERLAP_THRESHOLD` | `ppe_detection_with_trt.py` | 손목 ROI와 glove bbox 겹침 판정 기준 |
| `WRIST_ROI_SIZE` | `ppe_detection_with_trt.py` | 기본 손목 ROI 크기 |
| `WRIST_ROI_MIN` | `ppe_detection_with_trt.py` | 동적 손목 ROI 최소 크기 |
| `MASK_ROI_HEIGHT` | `ppe_detection_with_trt.py` | 마스크 ROI fallback 높이 |
| `AREA_CONFIG_REFRESH_INTERVAL` | `ppe_detection_with_trt.py` | 구역/단속 기준 재로드 주기 |
| `SUSTAIN_SECONDS` | `violation_tracker.py` | 위반 판정 지속 시간 |
| `COOLDOWN_SECONDS` | `violation_tracker.py` | 동일 위반 재기록 방지 시간 |
| `PADDING_RATIO` | `image_utils.py` | crop 여백 |
| `MAX_LONG_SIDE` | `image_utils.py` | crop 이미지 긴 변 제한 |
| `JPEG_QUALITY` | `image_utils.py` | JPEG 압축 품질 |

---

## 화면 표시

### PPE 상태바

person 박스 상단에 다음 형식으로 표시됩니다.

```text
ID7 0.93 | O | X | O | O
```

| 항목 | 의미 |
|------|------|
| `ID7` | ByteTrack이 부여한 person ID |
| `ID?` | 아직 tracker ID가 없는 상태 |
| `0.93` | person confidence |
| `O` | 착용 또는 정상 |
| `X` | 미착용 또는 미탐지 |

표시 순서는 항상 다음과 같습니다.

```text
안전모 → 마스크 → 왼손 장갑 → 오른손 장갑
```

다만 구역별 단속 기준에서 제외된 장비는 상태바에 표시하지 않습니다.

| 단속 기준 | 표시 예 |
|-----------|---------|
| 4개 모두 단속 | `ID7 0.93 | O | X | O | O` |
| 마스크 단속 제외 | `ID7 0.93 | O | O | O` |
| 안전모, 오른손 장갑만 단속 | `ID7 0.93 | O | X` |
| 4개 모두 단속 제외 | `ID7 0.93` |

### ROI / bbox 표시 제어

단속 제외 장비는 ROI와 탐지 박스를 그리지 않습니다.

| 설정 | 화면 표시 | 위반 저장 |
|------|-----------|-----------|
| `enforce_mask = 1` | mask ROI, mask box 표시 | `no_mask` 저장 가능 |
| `enforce_mask = 0` | mask ROI, mask box 미표시 | `no_mask` 저장 안 함 |
| `enforce_glove_left = 0` | 왼손 wrist ROI 미표시 | `no_glove_left` 저장 안 함 |
| `enforce_glove_right = 0` | 오른손 wrist ROI 미표시 | `no_glove_right` 저장 안 함 |

### 탐지 항목 색상

| 항목 | 색상 | 설명 |
|------|------|------|
| Person | 파란색 | 사람 바운딩 박스 |
| Helmet | 노란색 | 안전모 감지 |
| Helmet ROI | 연노란색 | 안전모 탐지 영역 |
| Mask | 보라색 | 마스크 감지 |
| Mask ROI | 진한 보라색 | 마스크 탐지 영역 |
| Glove | 초록색 | 장갑 바운딩 박스 |
| Wrist ROI (좌) | 하늘색 | 왼손 손목 ROI |
| Wrist ROI (우) | 보라색 | 오른손 손목 ROI |
| Pose | 노란색 선 / 빨간색 점 | 관절 연결선 및 keypoint |

---

## 위반 데이터 확인

### SQL 조회

최근 위반 조회:

```sql
SELECT id, violation_type, detected_at, area_id, person_id,
       LENGTH(image_data) AS bytes, is_checked
FROM violations
ORDER BY id DESC
LIMIT 20;
```

구역별 특정 위반 확인:

```sql
SELECT id, violation_type, area_id, person_id, detected_at
FROM violations
WHERE area_id = 20
ORDER BY detected_at DESC
LIMIT 20;
```

예를 들어 `area_id=20`에서 `enforce_mask=0`이면 신규 `no_mask`가 저장되지 않아야 합니다.

```sql
SELECT id, violation_type, area_id, person_id, detected_at
FROM violations
WHERE area_id = 20
  AND violation_type = 'no_mask'
ORDER BY detected_at DESC
LIMIT 20;
```

### Python 유틸로 이미지 추출

```bash
python3 extract_image.py
python3 extract_image.py 42
```

HeidiSQL은 BLOB을 부분 로드할 수 있으므로, 증거 이미지 검증은 `extract_image.py` 사용을 권장합니다.

---

## 디버깅 로그

`ppe_detection_with_trt.py`는 DB 연동 및 위반 상태를 주기적으로 로그에 출력합니다.

| 로그 | 의미 |
|------|------|
| `[DB-SKIP] area_id 없음` | `camera_key`가 `areas.camera_key`에 매핑되지 않음 |
| `[DB-SKIP] person_id 없음` | ByteTrack ID가 아직 부여되지 않음 |
| `[DB-WAIT] 이벤트 없음` | 위반 지속 시간이 아직 임계값 미만 |
| `[DB-TRY]` | 위반 저장 시도 |
| `[DB-OK]` | 위반 저장 성공 |
| `[DB-FAIL]` | 위반 저장 실패 |
| `[RULE-LOAD-FAIL]` | 구역/PPE 단속 기준 재로드 실패 |
| `[VT-START]` | 특정 위반 type 미착용 시작 |
| `[VT-HOLD]` | 미착용 지속 중 |
| `[VT-EVENT]` | 지속 시간 충족으로 위반 이벤트 발생 |
| `[VT-COOLDOWN]` | 동일 위반 cooldown 중 |
| `[VT-RESET]` | 착용 감지로 미착용 상태 리셋 |

운영 중 로그 확인:

```bash
sudo journalctl -u ppe-stream -f -o cat
```

---

## 성능

| 모드 | FPS (탐지 없을 때) | FPS (탐지 시) |
|------|-------------------|--------------|
| `.pt` 버전 | ~14 | ~11 |
| `.engine` 버전 | ~20 | ~13 |
| `.engine + ByteTrack + 구역별 PPE 기준 + 위반 적재` | ~16 | ~13 |

프레임 추론은 카메라별 순차 처리 구조이며, MJPEG 인코딩은 해당 카메라 스트림 접속자가 있을 때만 수행합니다.

---

## 탐지 결과

![PPE Detection Result](assets/result_image.png)

> TensorRT `.engine` 버전으로 탐지한 결과 예시입니다. 구역별 PPE 단속 기준에 따라 단속 제외 장비는 ROI/box/상태바 표시와 위반 저장 대상에서 제외됩니다.
