"""ViolationLogger 동작 검증 (실제 DB INSERT 1건)."""
import time
import numpy as np

from image_utils import crop_and_encode
from violation_logger import ViolationLogger


def main():
    # 가짜 프레임 (640x480, 회색)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    # 가짜 person 박스
    bx1, by1, bx2, by2 = 200, 100, 440, 400

    jpeg_bytes = crop_and_encode(frame, bx1, by1, bx2, by2)
    print(f"인코딩된 JPEG 크기: {len(jpeg_bytes)/1024:.1f}KB")

    logger = ViolationLogger()

    # 테스트 위반 1건 큐에 적재
    logger.log(
        violation_type="no_helmet",
        area_id=5,            # B구역 (이미 있는 area_id)
        person_id=999,        # 테스트용
        image_jpeg=jpeg_bytes,
    )

    # 워커가 처리할 시간 좀 줌
    print("DB INSERT 대기 중...")
    time.sleep(3.0)
    logger.stop()


if __name__ == "__main__":
    main()
