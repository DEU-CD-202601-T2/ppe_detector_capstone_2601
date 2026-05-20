"""위반 증거 이미지 크롭 및 JPEG 인코딩 유틸."""
import cv2
import numpy as np


# ── 설정값 ────────────────────────────────────────────────
PADDING_RATIO  = 0.10   # 박스 주변 10% 여유
JPEG_QUALITY   = 85     # 1-100, 높을수록 화질↑ 용량↑
MAX_LONG_SIDE  = 480    # 긴 변 최대 픽셀 (None이면 리사이즈 안 함)


def crop_person(frame: np.ndarray,
                bx1: int, by1: int, bx2: int, by2: int) -> np.ndarray:
    """원본 프레임에서 person 박스 + padding 영역 크롭.

    프레임 경계 밖으로 나가지 않게 클램프.
    """
    h, w = frame.shape[:2]
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)

    cx1 = max(0, bx1 - pad_x)
    cy1 = max(0, by1 - pad_y)
    cx2 = min(w, bx2 + pad_x)
    cy2 = min(h, by2 + pad_y)
    return frame[cy1:cy2, cx1:cx2].copy()


def resize_keep_aspect(img: np.ndarray, max_long_side: int) -> np.ndarray:
    """긴 변 기준 리사이즈 (비율 유지)."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """OpenCV 이미지 → JPEG 바이너리."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buf = cv2.imencode(".jpg", img, params)
    if not success:
        raise RuntimeError("JPEG 인코딩 실패")
    return buf.tobytes()


def crop_and_encode(frame: np.ndarray,
                    bx1: int, by1: int, bx2: int, by2: int) -> bytes:
    """크롭 + 리사이즈 + JPEG 압축을 한 번에."""
    crop = crop_person(frame, bx1, by1, bx2, by2)
    if MAX_LONG_SIDE:
        crop = resize_keep_aspect(crop, MAX_LONG_SIDE)
    return encode_jpeg(crop)
