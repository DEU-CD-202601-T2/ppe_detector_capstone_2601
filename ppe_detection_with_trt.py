import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════

CAM_INDEX    = 0
CAM_W, CAM_H = 1280, 720

OVERLAP_THRESHOLD = 0.3
WRIST_ROI_SIZE    = 120
WRIST_ROI_MIN     = 20
MASK_ROI_HEIGHT   = 80
PERSON_PAD        = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORLD_CLASSES = ["person", "glove", "helmet", "mask"]

CLASS_CONF = {
    "person": 0.50,
    "helmet": 0.10,
    "mask":   0.10,
    "glove":  0.10,
}

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

KP_NOSE            = 0
KP_LEFT_EYE        = 1
KP_RIGHT_EYE       = 2
KP_LEFT_EAR        = 3
KP_RIGHT_EAR       = 4
KP_LEFT_SHOULDER   = 5
KP_RIGHT_SHOULDER  = 6
KP_LEFT_ELBOW      = 7
KP_RIGHT_ELBOW     = 8
KP_LEFT_WRIST      = 9
KP_RIGHT_WRIST     = 10

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ══════════════════════════════════════════════════════════
# 전체 프레임 결과에서 클래스별 박스 추출 (좌표 필터링 포함)
# ══════════════════════════════════════════════════════════

def filter_boxes_in_region(results, class_name, rx1=0, ry1=0, rx2=99999, ry2=99999):
    """
    전체 프레임 추론 결과에서 class_name 박스만 추출.
    rx1~ry2: 관심 영역 (person bbox). 박스 중심이 영역 안에 있는 것만 반환.
    """
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


# ══════════════════════════════════════════════════════════
# 전체 프레임 pose 결과에서 person bbox에 해당하는 keypoints 추출
# ══════════════════════════════════════════════════════════

def get_keypoints_for_person(pose_results, px1, py1, px2, py2):
    """
    전체 프레임 pose 추론 결과에서 person bbox 안에 있는 keypoints 반환.
    nose 또는 중심점이 person bbox 안에 있는 것 선택.
    """
    if not pose_results or pose_results[0].keypoints is None:
        return None

    kps_all  = pose_results[0].keypoints
    if kps_all.xy is None or len(kps_all.xy) == 0:
        return None

    best_kp   = None
    best_score = -1

    for i in range(len(kps_all.xy)):
        xy   = kps_all.xy[i].cpu().numpy()
        conf = kps_all.conf[i].cpu().numpy() if kps_all.conf is not None \
               else np.ones(17, dtype=np.float32)

        # nose 좌표가 person bbox 안에 있는지 확인
        nx, ny = xy[KP_NOSE]
        if px1 <= nx <= px2 and py1 <= ny <= py2:
            score = conf[KP_NOSE]
            if score > best_score:
                best_score = score
                best_kp = np.concatenate([xy, conf[:, np.newaxis]], axis=1)

    # nose 못 찾으면 키포인트 중심이 bbox 안에 있는 것 선택
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


# ══════════════════════════════════════════════════════════
# 박스 overlap ratio  (box_a 면적 기준)
# ══════════════════════════════════════════════════════════

def box_overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    return inter / max((ax2-ax1)*(ay2-ay1), 1)


# ══════════════════════════════════════════════════════════
# wrist ROI 박스
# ══════════════════════════════════════════════════════════

def wrist_roi_box(kp_xy, img_w, img_h, elbow_xy=None, size=None):
    s    = size if size is not None else WRIST_ROI_SIZE
    half = s // 2
    wx, wy = int(kp_xy[0]), int(kp_xy[1])
    if wx == 0 and wy == 0:
        return None
    x1 = max(0,     wx - half)
    x2 = min(img_w, wx + half)

    # 팔꿈치 위치로 방향 결정
    if elbow_xy is not None and int(elbow_xy[1]) > wy:
        # 팔꿈치가 손목보다 아래 = 팔 내림 → 박스 위쪽
        y2 = wy
        y1 = max(0, wy - s)
    else:
        # 팔꿈치가 손목보다 위 = 팔 올림 → 박스 아래쪽
        y1 = wy
        y2 = min(img_h, wy + s)

    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def dynamic_wrist_size(keypoints, wrist_idx, elbow_idx):
    if keypoints is None or keypoints[elbow_idx, 2] < 0.3:
        return WRIST_ROI_SIZE
    wx, wy = keypoints[wrist_idx, :2]
    ex, ey = keypoints[elbow_idx, :2]
    dist   = float(np.hypot(wx - ex, wy - ey))
    return int(np.clip(dist, WRIST_ROI_MIN, WRIST_ROI_SIZE))


def judge(best_overlap):
    return ("GLOVE", (0,200,0)) if best_overlap >= OVERLAP_THRESHOLD \
           else ("HAND",  (0,0,255))


# ══════════════════════════════════════════════════════════
# 헬멧 ROI 시각화 → 감지 여부(bool) 반환
# ══════════════════════════════════════════════════════════

def draw_helmet(frame, all_results, px1, py1, px2, py2, keypoints):
    crop_h = py2 - py1
    crop_w = px2 - px1

    if keypoints is not None and keypoints[KP_NOSE, 2] > 0.3:
        roi_y2 = int(keypoints[KP_NOSE, 1])
    else:
        roi_y2 = py1 + int(crop_h * 0.30)

    roi_y1 = py1
    roi_x1, roi_x2 = px1, px2

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), COLOR_HELMET_ROI, 1, cv2.LINE_AA)
    cv2.putText(frame, "helmet ROI", (roi_x1+4, roi_y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HELMET_ROI, 1)

    detected = False
    for x1, y1, x2, y2, conf in filter_boxes_in_region(all_results, "helmet",
                                                         rx1=px1, ry1=py1, rx2=px2, ry2=roi_y2):
        detected = True
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_HELMET, 2)
        label = f"helmet {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1,y1-th-6),(x1+tw+4,y1),(0,0,0),-1)
        cv2.putText(frame, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HELMET, 2)
    return detected


# ══════════════════════════════════════════════════════════
# 마스크 ROI 시각화 → 감지 여부(bool) 반환
# ══════════════════════════════════════════════════════════

def draw_mask(frame, all_results, px1, py1, px2, py2, keypoints):
    crop_h = py2 - py1
    crop_w = px2 - px1

    if keypoints is not None and keypoints[KP_NOSE, 2] > 0.3:
        roi_y1 = int(keypoints[KP_NOSE, 1])
        sh_ys = [keypoints[sh, 1] for sh in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
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
    for x1, y1, x2, y2, conf in filter_boxes_in_region(all_results, "mask",
                                                         rx1=roi_x1, ry1=roi_y1,
                                                         rx2=roi_x2, ry2=roi_y2):
        detected = True
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_MASK, 2)
        label = f"mask {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1,y1-th-6),(x1+tw+4,y1),(0,0,0),-1)
        cv2.putText(frame, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MASK, 2)
    return detected


# ══════════════════════════════════════════════════════════
# Pose 시각화
# ══════════════════════════════════════════════════════════

def draw_pose(frame, keypoints, img_w, img_h):
    if keypoints is None:
        return
    lm_r  = max(2, int(min(img_w, img_h) * 0.005))
    ln_th = max(1, int(min(img_w, img_h) * 0.002))

    def pt(idx):
        return (int(keypoints[idx,0]), int(keypoints[idx,1]))

    for s, e in COCO_SKELETON:
        if keypoints[s,2] < 0.3 or keypoints[e,2] < 0.3:
            continue
        cv2.line(frame, pt(s), pt(e), COLOR_POSE_LINE, ln_th)
    for i in range(len(keypoints)):
        if keypoints[i,2] < 0.3:
            continue
        p = pt(i)
        cv2.circle(frame, p, lm_r, COLOR_POSE_DOT, -1)
        cv2.circle(frame, p, lm_r, (255,255,255), max(1,lm_r//3))


# ══════════════════════════════════════════════════════════
# PPE 상태 라벨 렌더링
# ══════════════════════════════════════════════════════════

def draw_ppe_status(frame, bx1, by1, person_idx, person_conf, status):
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_base = 0.6
    font_scale_ox   = 0.65
    thickness_base  = 2
    thickness_ox    = 2

    base_text = f"P{person_idx} {person_conf:.2f} "
    (bw, bh), _ = cv2.getTextSize(base_text, font, font_scale_base, thickness_base)
    sep_text = "| "
    (sw, _), _ = cv2.getTextSize(sep_text, font, font_scale_ox, thickness_ox)
    (ow, oh), _ = cv2.getTextSize("O", font, font_scale_ox, thickness_ox)

    cell_w  = sw + ow + 4
    total_w = bw + cell_w * 4 + 8
    tx, ty  = bx1, by1 - 8

    cv2.rectangle(frame, (tx-2, ty-bh-4), (tx+total_w+2, ty+4), (0,0,0), -1)
    cv2.putText(frame, base_text, (tx, ty), font, font_scale_base, COLOR_PERSON, thickness_base)

    items = [status["helmet"], status["mask"], status["left_glove"], status["right_glove"]]
    cx = tx + bw
    for ok in items:
        sym   = "O" if ok else "X"
        color = COLOR_OK if ok else COLOR_NG
        cv2.putText(frame, "| ", (cx, ty), font, font_scale_ox, (180,180,180), 1)
        cx += sw
        cv2.putText(frame, sym, (cx, ty), font, font_scale_ox, color, thickness_ox)
        cx += ow + 4


# ══════════════════════════════════════════════════════════
# 한 사람 처리 (전체 프레임 추론 결과 재사용)
# ══════════════════════════════════════════════════════════

def process_person(frame, px1, py1, px2, py2, img_w, img_h,
                   all_results, pose_results):

    # keypoints는 전체 프레임 pose 결과에서 이 사람 것만 추출
    keypoints = get_keypoints_for_person(pose_results, px1, py1, px2, py2)

    # ① 헬멧
    helmet_ok = draw_helmet(frame, all_results, px1, py1, px2, py2, keypoints)

    # ② 마스크
    mask_ok   = draw_mask(frame, all_results, px1, py1, px2, py2, keypoints)

    # ③ Glove bbox (person bbox 안에 있는 것만)
    glove_boxes = [(x1,y1,x2,y2)
                   for x1,y1,x2,y2,_ in filter_boxes_in_region(
                       all_results, "glove", rx1=px1, ry1=py1, rx2=px2, ry2=py2)]
    for gx1,gy1,gx2,gy2 in glove_boxes:
        cv2.rectangle(frame, (gx1,gy1), (gx2,gy2), COLOR_GLOVE, 2)

    # ④ wrist ROI ↔ Glove overlap
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
        roi = wrist_roi_box(keypoints[kp_idx, :2], img_w, img_h, elbow_xy=elbow_xy, size=roi_size)
        if roi is None:
            continue

        rx1,ry1,rx2,ry2 = roi
        best_overlap = max((box_overlap_ratio(roi, gb) for gb in glove_boxes), default=0.0)
        verdict, txt_color = judge(best_overlap)
        glove_status[side_key] = (verdict == "GLOVE")

        cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), roi_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cx, cy = (rx1+rx2)//2, ry1-8
        label  = f"{side_label}:{verdict}"
        detail = f"ovlp:{best_overlap:.2f}"
        (tw,th),_ = cv2.getTextSize(label, font, 0.65, 2)
        cv2.rectangle(frame, (cx-4,cy-th-4),(cx+tw+4,cy+4),(0,0,0),-1)
        cv2.putText(frame, label, (cx,cy), font, 0.65, txt_color, 2)
        (dw,dh),_ = cv2.getTextSize(detail, font, 0.45, 1)
        cv2.rectangle(frame, (cx-4,cy+4),(cx+dw+4,cy+dh+8),(0,0,0),-1)
        cv2.putText(frame, detail, (cx,cy+dh+6), font, 0.45, (200,200,200), 1)

    # ⑤ Pose 시각화
    draw_pose(frame, keypoints, img_w, img_h)

    return {
        "helmet":      helmet_ok,
        "mask":        mask_ok,
        "left_glove":  glove_status["left"],
        "right_glove": glove_status["right"],
    }


# ══════════════════════════════════════════════════════════
# 모델 초기화
# ══════════════════════════════════════════════════════════

yolo_model = YOLO("models/yolov8s-worldv2.engine", task="detect")
pose_model = YOLO("models/yolo11n-pose.engine",    task="pose")


# ══════════════════════════════════════════════════════════
# 웹캠 처리
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# wbmode 선택 가이드:
#   0 = Auto
#   1 = Incandescent  (백열등)
#   3 = Daylight      (야외 낮)
#   4 = Fluorescent   (형광등/LED 실내) ← 일반 실내 추천
#   5 = Cloudy
#   8 = Manual
# ──────────────────────────────────────────────
WB_MODE = 1  # 조명 환경에 맞게 변경

cap = cv2.VideoCapture(
    f"nvarguscamerasrc sensor-id=0 wbmode={WB_MODE} ! "
    f"video/x-raw(memory:NVMM),width={CAM_W},height={CAM_H},framerate=30/1,format=NV12 ! "
    f"nvvidconv ! "
    f"video/x-raw,format=BGRx ! "
    f"videoconvert ! "
    f"video/x-raw,format=BGR ! "
    f"appsink max-buffers=1 drop=true",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    print(f"❌ 카메라를 열 수 없습니다. (CAM_INDEX={CAM_INDEX})")
    exit()

print(f"✓ 웹캠 시작 (CAM_INDEX={CAM_INDEX}) | q 키로 종료")

t_prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # ── 푸른빛 보정 ──────────────────────────────────
    # wbmode=1 기준 미세 보정
    # 아직 푸르면: B_GAIN 낮추기(0.75까지), R_GAIN 올리기(1.15까지)
    B_GAIN = 0.78
    G_GAIN = 0.88
    R_GAIN = 1.10
    b, g, r = cv2.split(frame)
    b = np.clip(b.astype(np.float32) * B_GAIN, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * G_GAIN, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.float32) * R_GAIN, 0, 255).astype(np.uint8)
    frame = cv2.merge([b, g, r])
    # ─────────────────────────────────────────────────
    
    img_h, img_w = frame.shape[:2]

    # ★ 핵심 최적화: 전체 프레임에 대해 추론 2회만 실행 ★
    # 1회: YOLO-World → person/helmet/mask/glove 모두 한번에
    all_results  = yolo_model(frame, imgsz=640,
                              conf=min(CLASS_CONF.values()),
                              iou=0.45, verbose=False, device=0)
    # 2회: Pose → 전체 프레임 모든 사람 keypoints
    pose_results = pose_model(frame, imgsz=640,
                              verbose=False, device=0)

    # person 박스만 필터링
    person_boxes = filter_boxes_in_region(all_results, "person")

    for i, (bx1, by1, bx2, by2, conf) in enumerate(person_boxes):
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), COLOR_PERSON, 2)

        # 추론 결과 재사용 (추가 추론 없음!)
        status = process_person(frame, bx1, by1, bx2, by2, img_w, img_h,
                                all_results, pose_results)

        draw_ppe_status(frame, bx1, by1, i+1, conf, status)

    t_now  = time.time()
    fps    = 1.0 / max(t_now - t_prev, 1e-6)
    t_prev = t_now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame,
                f"OVERLAP_THR={OVERLAP_THRESHOLD}  WRIST_ROI={WRIST_ROI_SIZE}px",
                (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("PPE Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("종료")
