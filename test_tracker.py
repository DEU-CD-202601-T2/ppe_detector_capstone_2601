"""YOLO-World engine이 track() 지원하는지 검증."""
from ultralytics import YOLO
import numpy as np
import traceback

model = YOLO("models/yolov8s-worldv2.engine", task="detect")

# 더미 프레임 (검은 이미지)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

try:
    results = model.track(
        dummy,
        imgsz=640,
        conf=0.5,
        iou=0.45,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )
    print("✓ track() 호출 성공")
    print(f"  결과 타입: {type(results)}")
    boxes = results[0].boxes
    print(f"  박스에 id 속성: {hasattr(boxes, 'id')}")
    if hasattr(boxes, 'id'):
        print(f"  boxes.id = {boxes.id}")
except Exception as e:
    print(f"✗ track() 실패: {type(e).__name__}: {e}")
    traceback.print_exc()
