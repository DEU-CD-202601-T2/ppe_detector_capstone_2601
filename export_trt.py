"""
TensorRT 엔진 변환 스크립트
실행: python export_trt.py

생성 파일:
  models/yolov8m-worldv2.engine   ← YOLO-World (person/glove/helmet/mask 고정)
  models/yolo11n-pose.engine      ← YOLO Pose
"""

from ultralytics import YOLO

# ══════════════════════════════════════════════════════════
# 공통 export 옵션
# ══════════════════════════════════════════════════════════
EXPORT_OPTS = dict(
    format  = "engine",
    device  = 0,        # GPU 0
    half    = True,     # FP16 (속도↑, 정밀도 소폭↓)
    imgsz   = 640,
    workspace = 2,      # TRT 빌드 메모리 (GB), VRAM 부족 시 줄이세요
)

# ══════════════════════════════════════════════════════════
# 1) YOLO-World  →  클래스 4개 고정 후 export
# ══════════════════════════════════════════════════════════
# TRT 엔진은 컴파일 시점에 클래스가 고정됩니다.
# 추론 코드에서는 set_classes() 없이 cls 인덱스로 필터링합니다.
WORLD_CLASSES = ["person", "glove", "helmet", "mask"]

print("▶ YOLO-World 변환 중...")
world_model = YOLO("models/yolov8s-worldv2.pt")
world_model.set_classes(WORLD_CLASSES)
world_model.export(**EXPORT_OPTS)
print("✓ models/yolov8s-worldv2.engine 생성 완료\n")

# ══════════════════════════════════════════════════════════
# 2) YOLO-Pose
# ══════════════════════════════════════════════════════════
print("▶ YOLO-Pose 변환 중...")
pose_model = YOLO("models/yolo11n-pose.pt")
pose_model.export(**EXPORT_OPTS)
print("✓ models/yolo11n-pose.engine 생성 완료\n")

print("모든 변환 완료.")
print("엔진 파일은 동일한 GPU/드라이버 환경에서만 사용 가능합니다.")
