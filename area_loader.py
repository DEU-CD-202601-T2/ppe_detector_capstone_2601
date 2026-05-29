"""areas 테이블에서 camera_key → area_id 및 구역별 PPE 단속 기준을 로드."""
import pymysql
from db_config import DB_CONFIG


DEFAULT_PPE_RULE = {
    "helmet": True,
    "mask": True,
    "left_glove": True,
    "right_glove": True,
}


def load_camera_area_map() -> dict[str, int]:
    """활성 카메라(is_active=1)의 camera_key → area_id 매핑 반환."""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT camera_key, area_id
                FROM areas
                WHERE is_active = 1 AND camera_key IS NOT NULL
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {camera_key: area_id for camera_key, area_id in rows}


def load_area_ppe_rules() -> dict[int, dict[str, bool]]:
    """활성 구역의 area_id → PPE 단속 기준 반환.

    enforce_* 컬럼이 없는 구버전 DB에서는 전체 단속(True) 기본값으로 복구한다.
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT
                        area_id,
                        enforce_helmet,
                        enforce_mask,
                        enforce_glove_left,
                        enforce_glove_right
                    FROM areas
                    WHERE is_active = 1
                """)
                rows = cur.fetchall()
            except Exception as e:
                print(f"  ⚠ PPE 단속 기준 로드 실패 → 기본값 전체 단속 사용: {type(e).__name__}: {e}", flush=True)
                cur.execute("""
                    SELECT area_id
                    FROM areas
                    WHERE is_active = 1
                """)
                return {
                    int(row[0]): DEFAULT_PPE_RULE.copy()
                    for row in cur.fetchall()
                }
    finally:
        conn.close()

    return {
        int(area_id): {
            "helmet": bool(enforce_helmet),
            "mask": bool(enforce_mask),
            "left_glove": bool(enforce_glove_left),
            "right_glove": bool(enforce_glove_right),
        }
        for area_id, enforce_helmet, enforce_mask, enforce_glove_left, enforce_glove_right in rows
    }
