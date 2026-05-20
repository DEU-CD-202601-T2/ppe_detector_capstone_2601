import supervision as sv
import os
import cv2
import numpy as np
import time
import re
import subprocess
import threading
import atexit
import signal
import sys
import torch
from ultralytics import YOLO
from flask import Flask, Response
from area_loader        import load_camera_area_map
from violation_tracker  import ViolationStateTracker
from violation_logger   import ViolationLogger
from image_utils        import crop_and_encode

# RealSense 지원 (선택적 — 없으면 graceful degradation)
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠ pyrealsense2 import 실패 — RealSense 카메라 미지원 모드로 동작")

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

# ── CSI 카메라 공통 설정 ─────────────────────────────────────
CSI_W, CSI_H, CSI_FPS = 1280, 720, 30
WB_MODE               = 1      # 0=Auto 1=Incandescent 3=Daylight 4=Fluorescent
CSI_B_GAIN, CSI_G_GAIN, CSI_R_GAIN = 0.78, 0.88, 1.10

# ── USB 웹캠 공통 설정 ───────────────────────────────────────
USB_W, USB_H, USB_FPS = 1280, 720, 30
USB_B_GAIN, USB_G_GAIN, USB_R_GAIN = 1.0, 1.0, 1.0

# ── 추론 설정 ────────────────────────────────────────────────
OVERLAP_THRESHOLD = 0.3
WRIST_ROI_SIZE    = 120
WRIST_ROI_MIN     = 20
MASK_ROI_HEIGHT   = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORLD_CLASSES = ["person", "glove", "helmet", "mask", "vest"]
CLASS_CONF    = {"person": 0.50, "helmet": 0.10, "mask": 0.10, "glove": 0.10, "vest": 0.10}

COLOR_PERSON     = (255,   0,   0)
COLOR_POSE_LINE  = (  0, 255, 255)
COLOR_POSE_DOT   = (  0,   0, 255)
COLOR_GLOVE      = (  0, 255,   0)
COLOR_LEFT_ROI   = (255, 128,   0)
COLOR_RIGHT_ROI  = (128,   0, 255)
COLOR_HELMET     = (  0, 215, 255)
COLOR_HELMET_ROI = ( 80, 215, 255)
COLOR_MASK       = (255, 105, 180)
COLOR_MASK_ROI   = (200,  80, 160)
COLOR_OK         = (  0, 220,   0)
COLOR_NG         = (  0,   0, 220)

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ══════════════════════════════════════════════════════════════════════════════
# ★ 카메라 자동 탐지
#
# v4l2-ctl --list-devices 출력 예시:
#
#   NVIDIA Tegra Video Input Device (platform:tegra-camrtc-ca):
#       /dev/media0
#   vi-output, imx477 9-001a (platform:tegra-capture-vi:2):   ← CSI
#       /dev/video0
#   vi-output, imx477 10-001a (platform:tegra-capture-vi:1):  ← CSI
#       /dev/video1
#   KS2A418-2.0 (usb-3610000.usb-2.3):                        ← USB
#       /dev/video2
#       /dev/video3
#
# 파싱 결과:
#   csi_count = 2  → sensor-id 0, 1 순서로 시도
#   usb_devices = [2, 3] → /dev/video2, /dev/video3 순서로 시도
# ══════════════════════════════════════════════════════════════════════════════

def parse_v4l2_devices() -> tuple[int, list[int]]:
    """
    v4l2-ctl --list-devices 를 파싱해
    (csi_count, usb_video_indices) 를 반환.

    csi_count  : 감지된 CSI(vi-output) 카메라 수
    usb_video_indices : USB 그룹 아래의 /dev/videoN 번호 리스트
                        (RealSense/Depth 카메라는 제외)
    """
    try:
        raw = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode(errors="replace")
    except Exception:
        print("  ⚠ v4l2-ctl 실행 실패")
        return 0, []

    csi_count = 0
    usb_indices: list[int] = []
    current_type = None   # "csi" | "usb" | None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            current_type = None
            continue

        # 장치 그룹 헤더 판별
        if stripped.endswith(":"):
            low = stripped.lower()
            # RealSense / Depth 카메라는 일반 V4L2로 못 다루므로 제외
            if "realsense" in low or "depth" in low:
                current_type = None
                continue
            if "vi-output" in low:
                current_type = "csi"
                csi_count += 1
            elif "usb-" in low:
                current_type = "usb"
            else:
                current_type = None
            continue

        # /dev/videoN 수집 (USB 그룹만)
        if current_type == "usb":
            m = re.search(r"/dev/video(\d+)", stripped)
            if m:
                usb_indices.append(int(m.group(1)))

    return csi_count, usb_indices


def get_usb_camera_key(dev_idx: int) -> str:
    """
    /dev/videoN 에 대한 변하지 않는 식별자.
    우선순위: vid:pid:serial > vid:pid:port_path > unknown fallback
    """
    sys_iface = f"/sys/class/video4linux/video{dev_idx}/device"
    try:
        # /sys/.../device 는 USB 인터페이스를 가리킴
        # → 한 단계 위가 실제 USB device (idVendor/idProduct/serial 보유)
        iface_real = os.path.realpath(sys_iface)
        usb_dev    = os.path.dirname(iface_real)

        def _read(name):
            try:
                with open(os.path.join(usb_dev, name)) as f:
                    return f.read().strip()
            except OSError:
                return None

        vid    = _read('idVendor')
        pid    = _read('idProduct')
        serial = _read('serial')

        if vid and pid:
            if serial:
                return f"USB_{vid}_{pid}_{serial}"
            port = os.path.basename(usb_dev)   # 예: '1-2.1'
            return f"USB_{vid}_{pid}_PORT_{port}"
    except Exception as e:
        print(f"  ⚠ video{dev_idx} key 추출 실패: {e}")

    return f"USB_UNKNOWN_video{dev_idx}"


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 열기
# ══════════════════════════════════════════════════════════════════════════════

def open_csi_camera(sensor_id: int) -> cv2.VideoCapture | None:
    """CSI 카메라 열기 (nvarguscamerasrc sensor-id=N)."""
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} wbmode={WB_MODE} ! "
        f"video/x-raw(memory:NVMM),width={CSI_W},height={CSI_H},"
        f"framerate={CSI_FPS}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink max-buffers=1 drop=true"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def open_usb_camera(device_idx: int) -> cv2.VideoCapture | None:
    """
    USB 웹캠 열기. MJPG 우선 → raw → V4L2 fallback 순으로 시도.
    실제 프레임이 나오는지까지 검증해 메타 노드를 걸러냄.
    """
    # ① GStreamer + MJPG (USB 캠 표준 경로)
    pipeline_mjpg = (
        f"v4l2src device=/dev/video{device_idx} ! "
        f"image/jpeg,width={USB_W},height={USB_H},framerate={USB_FPS}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink max-buffers=1 drop=true"
    )
    cap = cv2.VideoCapture(pipeline_mjpg, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"  [video{device_idx}] GStreamer MJPG 경로 사용", flush=True)

    # ② GStreamer + raw (MJPG 미지원 캠 fallback)
    if not cap.isOpened():
        cap.release()
        pipeline_raw = (
            f"v4l2src device=/dev/video{device_idx} ! "
            f"video/x-raw,width={USB_W},height={USB_H},framerate={USB_FPS}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )
        cap = cv2.VideoCapture(pipeline_raw, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"  [video{device_idx}] GStreamer raw 경로 사용", flush=True)

    # ③ V4L2 직접 접근 fallback (MJPG fourcc 명시)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  USB_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_H)
            cap.set(cv2.CAP_PROP_FPS,          USB_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            print(f"  [video{device_idx}] V4L2 직접 접근 경로 사용", flush=True)
        else:
            cap.release()
            return None

    # ④ 실제 프레임 수신 확인
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.05)

    cap.release()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RealSense 카메라 (librealsense2 → cv2.VideoCapture 호환 어댑터)
#
# RealSense 는 컬러/깊이/IR/IMU 스트림이 /dev/video0~5 에 흩어져 있어
# 일반 V4L2 로 다루기 어려움. librealsense2 의 pipeline API 를 사용해
# 컬러 스트림만 BGR8 로 받아서 cv2.VideoCapture 인터페이스로 노출한다.
# 이렇게 하면 기존 CameraThread 가 USB 캠과 동일하게 처리할 수 있다.
# ══════════════════════════════════════════════════════════════════════════════

class RealSenseColorSource:
    """librealsense2 컬러 스트림을 cv2.VideoCapture 호환 인터페이스로 감싸는 어댑터.

    CameraThread 가 호출하는 메서드만 구현:
      - grab()     : 다음 프레임을 내부 버퍼로 가져옴 (bool 반환)
      - retrieve() : 가장 최근 프레임을 반환 ((ret, frame))
      - isOpened() : 파이프라인 활성 여부
      - release()  : 파이프라인 정지
    """

    def __init__(self, serial: str = "",
                 width: int = 1280, height: int = 720, fps: int = 30):
        self.serial = serial
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipe.start(cfg)
        self._opened = True
        self._latest_frame = None

        # 자동 노출/화이트밸런스 안정화 워밍업
        for _ in range(5):
            try:
                self.pipe.wait_for_frames(timeout_ms=2000)
            except Exception:
                break

    def grab(self) -> bool:
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=500)
            color = frames.get_color_frame()
            if color:
                self._latest_frame = np.asanyarray(color.get_data())
                return True
        except Exception:
            pass
        return False

    def retrieve(self):
        if self._latest_frame is not None:
            return True, self._latest_frame.copy()
        return False, None

    def isOpened(self) -> bool:
        return self._opened

    def release(self):
        if self._opened:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self._opened = False


def detect_realsense_devices() -> list[str]:
    """연결된 RealSense 장치의 serial number 목록 반환."""
    if not REALSENSE_AVAILABLE:
        return []
    try:
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number)
                for d in ctx.query_devices()]
    except Exception as e:
        print(f"  ⚠ RealSense 탐지 실패: {e}")
        return []


def open_realsense_camera(serial: str):
    """RealSense 컬러 스트림 열기. 실패 시 None 반환."""
    if not REALSENSE_AVAILABLE:
        return None
    try:
        return RealSenseColorSource(
            serial=serial,
            width=USB_W, height=USB_H, fps=USB_FPS,
        )
    except Exception as e:
        print(f"  RealSense({serial[-6:]}) 열기 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 스레드 (Non-blocking 캡처 + USB/RealSense 자동 재연결)
# ══════════════════════════════════════════════════════════════════════════════

class CameraThread:
    """
    별도 스레드에서 프레임을 지속적으로 읽어 최신 프레임을 유지.

    reopen_fn : USB 카메라에만 전달. 장치가 끊기면 호출해 재연결 시도.
                CSI 는 None → 재연결 로직 건너뜀.
    FAIL_LIMIT: 연속 실패가 이 횟수를 넘으면 stale 프레임 폐기 후 재연결.
    """

    FAIL_LIMIT = 30

    def __init__(self, cap, name, key,
                 reopen_fn=None,
                 b_gain: float = 1.0,
                 g_gain: float = 1.0,
                 r_gain: float = 1.0):
        self.cap       = cap
        self.name      = name
        self.key       = key
        self.reopen_fn = reopen_fn
        self.b_gain    = b_gain
        self.g_gain    = g_gain
        self.r_gain    = r_gain

        self._frame      = None
        self._lock       = threading.Lock()
        self._running    = True
        self._fail_count = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            # _running 체크를 빠르게 하기 위해 grab을 한 번씩만
            if not self._running:
                break
            self.cap.grab()
            if not self._running:
                break
            ret, frame = self.cap.retrieve()

            if not ret:
                self._fail_count += 1
                if self._fail_count >= self.FAIL_LIMIT:
                    with self._lock:
                        self._frame = None
                    if self.reopen_fn is not None:
                        print(f"  [{self.name}] 장치 끊김 → 재연결 대기 중...")
                        self.cap.release()
                        new_cap = None
                        while self._running and new_cap is None:
                            time.sleep(2.0)
                            new_cap = self.reopen_fn()
                        if new_cap is not None:
                            self.cap = new_cap
                            self._fail_count = 0
                            print(f"  [{self.name}] 재연결 성공")
                time.sleep(0.005)
                continue

            self._fail_count = 0

            if not (self.b_gain == 1.0 and self.g_gain == 1.0 and self.r_gain == 1.0):
                b, g, r = cv2.split(frame)
                b = np.clip(b.astype(np.float32) * self.b_gain, 0, 255).astype(np.uint8)
                g = np.clip(g.astype(np.float32) * self.g_gain, 0, 255).astype(np.uint8)
                r = np.clip(r.astype(np.float32) * self.r_gain, 0, 255).astype(np.uint8)
                frame = cv2.merge([b, g, r])

            with self._lock:
                self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """
        스레드의 자연 종료를 기다린 후 release.
        daemon 스레드가 cap.grab() 안에 블로킹된 상태로 release 되면
        OS 레벨에서 V4L2 핸들이 깔끔히 닫히지 않는 race condition 방지.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 추론 유틸
# ══════════════════════════════════════════════════════════════════════════════

def filter_boxes_in_region(results, class_name,
                           rx1=0, ry1=0, rx2=99999, ry2=99999):
    boxes = []
    r     = results[0]
    names = r.names if r.names else {i: c for i, c in enumerate(WORLD_CLASSES)}
    thr   = CLASS_CONF.get(class_name, 0.1)
    for b in r.boxes:
        if names[int(b.cls)] == class_name and float(b.conf) >= thr:
            x1, y1, x2, y2 = [round(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                boxes.append((x1, y1, x2, y2, float(b.conf)))
    return boxes


def filter_person_boxes_with_id(results, rx1=0, ry1=0, rx2=99999, ry2=99999):
    """person 박스 + 트래커 ID 함께 추출.

    반환: [(x1, y1, x2, y2, conf, person_id), ...]
    ID가 없는 경우(트래커가 아직 부여 안 함) person_id = -1
    """
    boxes = []
    r     = results[0]
    names = r.names if r.names else {i: c for i, c in enumerate(WORLD_CLASSES)}
    thr   = CLASS_CONF.get("person", 0.1)
    ids   = r.boxes.id  # tensor or None

    for i, b in enumerate(r.boxes):
        if names[int(b.cls)] != "person" or float(b.conf) < thr:
            continue
        x1, y1, x2, y2 = [round(v) for v in b.xyxy[0].tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue
        pid = int(ids[i].item()) if ids is not None else -1
        boxes.append((x1, y1, x2, y2, float(b.conf), pid))
    return boxes


def get_keypoints_for_person(pose_results, px1, py1, px2, py2):
    if not pose_results or pose_results[0].keypoints is None:
        return None
    kps_all = pose_results[0].keypoints
    if kps_all.xy is None or len(kps_all.xy) == 0:
        return None

    best_kp, best_score = None, -1
    for i in range(len(kps_all.xy)):
        xy   = kps_all.xy[i].cpu().numpy()
        conf = kps_all.conf[i].cpu().numpy() if kps_all.conf is not None \
               else np.ones(17, dtype=np.float32)
        nx, ny = xy[KP_NOSE]
        if px1 <= nx <= px2 and py1 <= ny <= py2:
            score = conf[KP_NOSE]
            if score > best_score:
                best_score = score
                best_kp = np.concatenate([xy, conf[:, np.newaxis]], axis=1)

    if best_kp is None:
        for i in range(len(kps_all.xy)):
            xy   = kps_all.xy[i].cpu().numpy()
            conf = kps_all.conf[i].cpu().numpy() if kps_all.conf is not None \
                   else np.ones(17, dtype=np.float32)
            valid = xy[conf > 0.3]
            if len(valid) == 0:
                continue
            cx, cy = valid[:, 0].mean(), valid[:, 1].mean()
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                best_kp = np.concatenate([xy, conf[:, np.newaxis]], axis=1)
                break
    return best_kp


def box_overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    return inter / max((ax2-ax1)*(ay2-ay1), 1)


def wrist_roi_box(kp_xy, img_w, img_h, elbow_xy=None, size=None):
    s    = size if size is not None else WRIST_ROI_SIZE
    half = s // 2
    wx, wy = int(kp_xy[0]), int(kp_xy[1])
    if wx == 0 and wy == 0:
        return None
    x1 = max(0,     wx - half)
    x2 = min(img_w, wx + half)
    if elbow_xy is not None and int(elbow_xy[1]) > wy:
        y2 = wy; y1 = max(0, wy - s)
    else:
        y1 = wy; y2 = min(img_h, wy + s)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def dynamic_wrist_size(keypoints, wrist_idx, elbow_idx):
    if keypoints is None or keypoints[elbow_idx, 2] < 0.3:
        return WRIST_ROI_SIZE
    wx, wy = keypoints[wrist_idx, :2]
    ex, ey = keypoints[elbow_idx, :2]
    dist   = float(np.hypot(wx - ex, wy - ey))
    return int(np.clip(dist, WRIST_ROI_MIN, WRIST_ROI_SIZE))


def judge(best_overlap):
    return ("GLOVE", (0, 200, 0)) if best_overlap >= OVERLAP_THRESHOLD \
           else ("HAND", (0, 0, 255))


def draw_helmet(frame, all_results, px1, py1, px2, py2, keypoints):
    crop_h = py2 - py1
    if keypoints is not None and keypoints[KP_NOSE, 2] > 0.3:
        roi_y2 = int(keypoints[KP_NOSE, 1])
    else:
        roi_y2 = py1 + int(crop_h * 0.30)
    roi_y1, roi_x1, roi_x2 = py1, px1, px2
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), COLOR_HELMET_ROI, 1, cv2.LINE_AA)
    cv2.putText(frame, "helmet ROI", (roi_x1+4, roi_y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HELMET_ROI, 1)
    detected = False
    for x1, y1, x2, y2, conf in filter_boxes_in_region(
            all_results, "helmet", rx1=px1, ry1=py1, rx2=px2, ry2=roi_y2):
        detected = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HELMET, 2)
        label = f"helmet {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HELMET, 2)
    return detected


def draw_mask(frame, all_results, px1, py1, px2, py2, keypoints):
    crop_h = py2 - py1
    if keypoints is not None and keypoints[KP_NOSE, 2] > 0.3:
        roi_y1 = int(keypoints[KP_NOSE, 1])
        sh_ys  = [keypoints[sh, 1] for sh in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
                  if keypoints[sh, 2] > 0.3]
        roi_y2 = min(py2, int(min(sh_ys))) if sh_ys \
                 else min(py2, roi_y1 + MASK_ROI_HEIGHT)
    else:
        roi_y1 = py1 + int(crop_h * 0.40)
        roi_y2 = py1 + int(crop_h * 0.70)

    def best_x(priority_indices, kps):
        if kps is None:
            return None
        for idx in priority_indices:
            if kps[idx, 2] > 0.3:
                return int(kps[idx, 0])
        return None

    left_x  = best_x([KP_LEFT_EAR,  KP_LEFT_EYE,  KP_NOSE], keypoints)
    right_x = best_x([KP_RIGHT_EAR, KP_RIGHT_EYE, KP_NOSE], keypoints)

    if left_x is not None and right_x is not None:
        roi_x1 = max(px1, min(left_x, right_x))
        roi_x2 = min(px2, max(left_x, right_x))
    else:
        roi_x1, roi_x2 = px1, px2

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), COLOR_MASK_ROI, 1, cv2.LINE_AA)
    cv2.putText(frame, "mask ROI", (roi_x1+4, roi_y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MASK_ROI, 1)
    detected = False
    for x1, y1, x2, y2, conf in filter_boxes_in_region(
            all_results, "mask",
            rx1=roi_x1, ry1=roi_y1, rx2=roi_x2, ry2=roi_y2):
        detected = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MASK, 2)
        label = f"mask {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MASK, 2)
    return detected


def draw_pose(frame, keypoints, img_w, img_h):
    if keypoints is None:
        return
    lm_r  = max(2, int(min(img_w, img_h) * 0.005))
    ln_th = max(1, int(min(img_w, img_h) * 0.002))
    def pt(idx):
        return (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
    for s, e in COCO_SKELETON:
        if keypoints[s, 2] < 0.3 or keypoints[e, 2] < 0.3:
            continue
        cv2.line(frame, pt(s), pt(e), COLOR_POSE_LINE, ln_th)
    for i in range(len(keypoints)):
        if keypoints[i, 2] < 0.3:
            continue
        p = pt(i)
        cv2.circle(frame, p, lm_r, COLOR_POSE_DOT, -1)
        cv2.circle(frame, p, lm_r, (255, 255, 255), max(1, lm_r//3))


def draw_ppe_status(frame, bx1, by1, person_id, person_conf, status):
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_base = 0.6
    font_scale_ox   = 0.65
    thickness_base  = 2
    thickness_ox    = 2
    base_text = f"ID{person_id} {person_conf:.2f} " if person_id >= 0 else f"ID? {person_conf:.2f} "
    (bw, bh), _ = cv2.getTextSize(base_text, font, font_scale_base, thickness_base)
    sep_text = "| "
    (sw, _), _ = cv2.getTextSize(sep_text, font, font_scale_ox, thickness_ox)
    (ow, oh), _ = cv2.getTextSize("O", font, font_scale_ox, thickness_ox)
    total_w = bw + (sw + ow + 4) * 4 + 8
    tx, ty  = bx1, by1 - 8
    cv2.rectangle(frame, (tx-2, ty-bh-4), (tx+total_w+2, ty+4), (0, 0, 0), -1)
    cv2.putText(frame, base_text, (tx, ty), font, font_scale_base, COLOR_PERSON, thickness_base)
    items = [status["helmet"], status["mask"], status["left_glove"], status["right_glove"]]
    cx = tx + bw
    for ok in items:
        sym   = "O" if ok else "X"
        color = COLOR_OK if ok else COLOR_NG
        cv2.putText(frame, "| ", (cx, ty), font, font_scale_ox, (180, 180, 180), 1)
        cx += sw
        cv2.putText(frame, sym, (cx, ty), font, font_scale_ox, color, thickness_ox)
        cx += ow + 4


def process_person(frame, px1, py1, px2, py2, img_w, img_h,
                   all_results, pose_results):
    keypoints = get_keypoints_for_person(pose_results, px1, py1, px2, py2)
    helmet_ok = draw_helmet(frame, all_results, px1, py1, px2, py2, keypoints)
    mask_ok   = draw_mask(frame, all_results, px1, py1, px2, py2, keypoints)

    glove_boxes = [(x1, y1, x2, y2)
                   for x1, y1, x2, y2, _ in filter_boxes_in_region(
                       all_results, "glove", rx1=px1, ry1=py1, rx2=px2, ry2=py2)]
    for gx1, gy1, gx2, gy2 in glove_boxes:
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), COLOR_GLOVE, 2)

    glove_status = {"left": False, "right": False}
    wrist_cfg = [
        (KP_LEFT_WRIST,  KP_LEFT_ELBOW,  "left",  "L", COLOR_LEFT_ROI),
        (KP_RIGHT_WRIST, KP_RIGHT_ELBOW, "right", "R", COLOR_RIGHT_ROI),
    ]
    for kp_idx, elbow_idx, side_key, side_label, roi_color in wrist_cfg:
        if keypoints is None or keypoints[kp_idx, 2] < 0.3:
            continue
        roi_size = dynamic_wrist_size(keypoints, kp_idx, elbow_idx)
        elbow_xy = keypoints[elbow_idx, :2] if keypoints[elbow_idx, 2] > 0.3 else None
        roi = wrist_roi_box(keypoints[kp_idx, :2], img_w, img_h,
                            elbow_xy=elbow_xy, size=roi_size)
        if roi is None:
            continue
        rx1, ry1, rx2, ry2 = roi
        best_overlap = max((box_overlap_ratio(roi, gb) for gb in glove_boxes), default=0.0)
        verdict, txt_color = judge(best_overlap)
        glove_status[side_key] = (verdict == "GLOVE")
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cx, cy = (rx1+rx2)//2, ry1-8
        label  = f"{side_label}:{verdict}"
        detail = f"ovlp:{best_overlap:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
        cv2.rectangle(frame, (cx-4, cy-th-4), (cx+tw+4, cy+4), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx, cy), font, 0.65, txt_color, 2)
        (dw, dh), _ = cv2.getTextSize(detail, font, 0.45, 1)
        cv2.rectangle(frame, (cx-4, cy+4), (cx+dw+4, cy+dh+8), (0, 0, 0), -1)
        cv2.putText(frame, detail, (cx, cy+dh+6), font, 0.45, (200, 200, 200), 1)

    draw_pose(frame, keypoints, img_w, img_h)
    return {
        "helmet":      helmet_ok,
        "mask":        mask_ok,
        "left_glove":  glove_status["left"],
        "right_glove": glove_status["right"],
    }


def run_inference_on_frame(frame, cam_key: str):
    img_h, img_w = frame.shape[:2]
    original_frame = frame.copy()

    # 일반 detect() — PPE 인식 정확도 유지
    all_results = yolo_model(frame, imgsz=640,
                             conf=min(CLASS_CONF.values()),
                             iou=0.45, verbose=False, device=0)
    pose_results = pose_model(frame, imgsz=640, verbose=False, device=0)

    # person 박스만 추출해서 별도 트래커에 적용
    dets = sv.Detections.from_ultralytics(all_results[0])
    person_idx = WORLD_CLASSES.index("person")
    person_dets = dets[(dets.class_id == person_idx) &
                       (dets.confidence >= CLASS_CONF["person"])]
    person_dets = person_tracker.update_with_detections(person_dets)

    area_id = CAMERA_AREA_MAP.get(cam_key)

    for i in range(len(person_dets)):
        bx1, by1, bx2, by2 = person_dets.xyxy[i].astype(int).tolist()
        conf = float(person_dets.confidence[i])
        tid = person_dets.tracker_id
        person_id = int(tid[i]) if tid is not None and tid[i] is not None else -1

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_PERSON, 2)
        status = process_person(frame, bx1, by1, bx2, by2, img_w, img_h,
                                all_results, pose_results)
        draw_ppe_status(frame, bx1, by1, person_id, conf, status)

        if area_id is None or person_id < 0:
            continue
        events = violation_tracker.update(cam_key, person_id, status)
        for ev in events:
            try:
                jpeg = crop_and_encode(original_frame, bx1, by1, bx2, by2)
                violation_logger.log(
                    violation_type=ev.violation_type,
                    area_id=area_id,
                    person_id=ev.person_id,
                    image_jpeg=jpeg,
                )
            except Exception as e:
                print(f"  ⚠ 위반 처리 실패: {type(e).__name__}: {e}")

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════

print("▶ 모델 로드 중...")
yolo_model = YOLO("models/yolov8s-worldv2.engine", task="detect")
pose_model = YOLO("models/yolo11n-pose.engine",    task="pose")
person_tracker = sv.ByteTrack()
print("✓ 모델 로드 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 위반 추적/저장 초기화
# ══════════════════════════════════════════════════════════════════════════════

print("▶ areas 매핑 로드 중...")
CAMERA_AREA_MAP = load_camera_area_map()
print(f"✓ {len(CAMERA_AREA_MAP)}개 카메라 매핑 로드:")
for ck, aid in CAMERA_AREA_MAP.items():
    print(f"  {ck} → area_id={aid}")

violation_tracker = ViolationStateTracker()  # 기본값: 3초/30초
violation_logger  = ViolationLogger()
print("✓ 위반 추적기/로거 준비 완료")


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG 스트리밍 서버 (Flask)
# ══════════════════════════════════════════════════════════════════════════════

app              = Flask(__name__)
latest_frames:   dict[str, bytes] = {}
frame_counters:  dict[str, int]   = {}
active_streams:  dict[str, int]   = {}   # 카메라별 현재 접속자 수
stream_lock      = threading.Lock()
frame_lock       = threading.Lock()


def update_annotated_frame(cam_name: str, frame):
    """접속자가 있을 때만 JPEG 인코딩 후 스트리밍 버퍼 업데이트."""
    with stream_lock:
        has_viewer = active_streams.get(cam_name, 0) > 0
    if not has_viewer:
        return   # 아무도 안 보면 인코딩 생략
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ret:
        with frame_lock:
            latest_frames[cam_name]  = buf.tobytes()
            frame_counters[cam_name] = frame_counters.get(cam_name, 0) + 1


def inference_loop(cam_list):
    """단일 스레드에서 모든 카메라를 순차 추론 — 모델 1개 공유."""
    while True:
        for cam in cam_list:
            frame = cam.get_frame()
            if frame is None:
                blank = NO_SIGNAL_FRAME.copy()
                cv2.putText(blank, f"[{cam.name}]",
                            (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
                cv2.putText(blank, "NO SIGNAL",
                            (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 3)
                cv2.putText(blank, "reconnecting...",
                            (60, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                update_annotated_frame(cam.name, blank)
                continue

            # ── 추론 (항상 실행, 모델 공유) ──────────────────
            t0  = time.time()
            out = run_inference_on_frame(frame, cam.key)
            fps_map[cam.name] = 1.0 / max(time.time() - t0, 1e-6)

            cv2.putText(out, f"[{cam.name}] FPS: {fps_map[cam.name]:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(out,
                        f"OVERLAP_THR={OVERLAP_THRESHOLD}  WRIST_ROI={WRIST_ROI_SIZE}px",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ── 스트리밍 (접속자 있을 때만 인코딩) ──────────
            update_annotated_frame(cam.name, out)


def _generate(cam_name: str):
    # 접속자 수 증가
    with stream_lock:
        active_streams[cam_name] = active_streams.get(cam_name, 0) + 1
    print(f"  [{cam_name}] 스트림 접속 (접속자: {active_streams[cam_name]})")

    last_counter = -1
    try:
        while True:
            with frame_lock:
                counter = frame_counters.get(cam_name, 0)
                data    = latest_frames.get(cam_name)
            if data and counter != last_counter:
                last_counter = counter
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            else:
                time.sleep(0.01)
    finally:
        # 접속 종료 시 접속자 수 감소
        with stream_lock:
            active_streams[cam_name] = max(0, active_streams.get(cam_name, 1) - 1)
        print(f"  [{cam_name}] 스트림 종료 (접속자: {active_streams[cam_name]})")

@app.route('/stream/<path:cam_name>')
def stream(cam_name):
    return Response(
        _generate(cam_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/cameras')
def cameras_list():
    return {
        'cameras': [
            {'name': cam.name, 'key': cam.key}
            for cam in cameras
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# ★ 카메라 자동 탐지 및 초기화
#
# 1. v4l2-ctl 파싱 → CSI 개수 / USB 장치 번호 추출 (RealSense 노드는 제외)
# 2. CSI: sensor-id 0 부터 순서대로 시도 → CAM0, CAM1, ...
# 3. USB: /dev/videoN 순서대로 시도 → 프레임 확인 후 추가
# 4. RealSense: pyrealsense2 로 컬러 스트림 열기 (가능한 경우)
# 5. 발견 순서대로 전역 카메라 ID(cam_id) 부여
# ══════════════════════════════════════════════════════════════════════════════

cameras: list[CameraThread] = []

print("▶ 카메라 탐지 중...")
csi_count, usb_candidates = parse_v4l2_devices()
print(f"  감지된 CSI 카메라: {csi_count}개 | USB 후보 노드: {usb_candidates}")

cam_id = 0   # 전역 순번 (발견될 때마다 증가)

# ── CSI 카메라 ───────────────────────────────────────────────
for sensor_id in range(csi_count):
    print(f"  → CSI sensor-id={sensor_id} 시도 중...", end=" ", flush=True)
    cap = open_csi_camera(sensor_id)
    if cap is not None:
        name = f"CAM{cam_id}(CSI{sensor_id})"
        key  = f"CSI_{sensor_id}"
        cameras.append(CameraThread(
            cap, name, key,
            reopen_fn=None,          # CSI는 재연결 미지원
            b_gain=CSI_B_GAIN,
            g_gain=CSI_G_GAIN,
            r_gain=CSI_R_GAIN,
        ))
        print(f"✓ {name} 추가  [key={key}]")
        cam_id += 1
    else:
        print("✗ 열기 실패")

# ── USB 카메라 ───────────────────────────────────────────────
for dev_idx in usb_candidates:
    print(f"  → USB /dev/video{dev_idx} 시도 중...", end=" ", flush=True)
    cap = open_usb_camera(dev_idx)
    if cap is not None:
        name = f"CAM{cam_id}(USB{dev_idx})"
        key  = get_usb_camera_key(dev_idx)
        cameras.append(CameraThread(
            cap, name, key,
            reopen_fn=lambda i=dev_idx: open_usb_camera(i),  # 클로저로 idx 고정
            b_gain=USB_B_GAIN,
            g_gain=USB_G_GAIN,
            r_gain=USB_R_GAIN,
        ))
        print(f"✓ {name} 추가  [key={key}]")
        cam_id += 1
    else:
        print("✗ (메타 노드이거나 프레임 없음)")

# ── RealSense 카메라 ─────────────────────────────────────────
rs_serials = detect_realsense_devices()
if rs_serials:
    print(f"  감지된 RealSense: {len(rs_serials)}대 (S/N: {[s[-6:] for s in rs_serials]})")
    for serial in rs_serials:
        print(f"  → RealSense {serial[-6:]} 시도 중...", end=" ", flush=True)
        rs_source = open_realsense_camera(serial)
        if rs_source is not None:
            name = f"CAM{cam_id}(RS_{serial[-6:]})"
            key  = f"RS_{serial}"
            cameras.append(CameraThread(
                rs_source, name, key,
                reopen_fn=lambda s=serial: open_realsense_camera(s),
                b_gain=USB_B_GAIN,
                g_gain=USB_G_GAIN,
                r_gain=USB_R_GAIN,
            ))
            print(f"✓ {name} 추가  [key={key}]")
            cam_id += 1
        else:
            print("✗ 열기 실패")
elif REALSENSE_AVAILABLE:
    print("  감지된 RealSense: 0대")

# ── 결과 요약 ─────────────────────────────────────────────────
if not cameras:
    print("\n❌ 사용 가능한 카메라가 없습니다. 연결 상태를 확인하세요.")
    exit(1)

print(f"\n■ 활성 카메라 {len(cameras)}개: {[c.name for c in cameras]}")
print("  스트림 요청 시 해당 카메라만 추론 시작\n")

# ── MJPEG 서버 기동 ───────────────────────────────────────────
threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True),
    daemon=True
).start()
print("✓ MJPEG 스트리밍 서버 시작 (포트 5001)\n")


# ══════════════════════════════════════════════════════════════════════════════
# 카메라별 온디맨드 추론 스레드 기동
# ══════════════════════════════════════════════════════════════════════════════

NO_SIGNAL_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
fps_map         = {cam.name: 0.0 for cam in cameras}

# ── 추론 스레드 1개만 기동 (모델 공유) ───────────────────────
threading.Thread(target=inference_loop, args=(cameras,), daemon=True).start()
print("✓ 추론 스레드 시작 (카메라 순차 처리)\n")

print("\n종료하려면 Ctrl+C\n")

# ── 모든 종료 경로에서 카메라 해제 보장 ──────────────────
_cleanup_done = False

def _cleanup_cameras():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    print("\n[cleanup] 카메라 해제 중...")
    for cam in cameras:
        try:
            cam.stop()
        except Exception as e:
            print(f"  [{cam.name}] stop 실패: {e}")
    print("[cleanup] 완료")

atexit.register(_cleanup_cameras)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGHUP,  lambda s, f: sys.exit(0))

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    _cleanup_cameras()

print("종료")
