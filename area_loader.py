"""areas 테이블에서 camera_key → area_id 매핑 로드."""
import pymysql
from db_config import DB_CONFIG


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
