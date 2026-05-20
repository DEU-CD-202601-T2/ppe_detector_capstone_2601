"""DB에서 위반 이미지 1개 꺼내서 파일로 저장 + 검증."""
import sys
import pymysql
import cv2
import numpy as np
from db_config import DB_CONFIG

row_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

conn = pymysql.connect(**DB_CONFIG)
with conn.cursor() as cur:
    if row_id:
        cur.execute("SELECT id, violation_type, image_data "
                    "FROM violations WHERE id = %s", (row_id,))
    else:
        cur.execute("SELECT id, violation_type, image_data "
                    "FROM violations ORDER BY id DESC LIMIT 1")
    result = cur.fetchone()
conn.close()

if result is None:
    print("row 없음")
    sys.exit()

rid, vtype, blob = result
print(f"id={rid}, type={vtype}, blob size={len(blob)} bytes")
print(f"첫 4바이트(hex): {blob[:4].hex()}")  # JPEG는 ff d8 ff e0 으로 시작

# 파일로 저장
fname = f"violation_{rid}_{vtype}.jpg"
with open(fname, "wb") as f:
    f.write(blob)
print(f"파일 저장: {fname}")

# OpenCV로 디코딩 검증
img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
if img is None:
    print("⚠ JPEG 디코딩 실패 — 데이터 손상")
else:
    print(f"✓ 정상 이미지: {img.shape[1]}x{img.shape[0]} (W x H)")
