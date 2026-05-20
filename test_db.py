"""DB 접속 및 areas 매핑 로드 동작 확인."""
from area_loader import load_camera_area_map


if __name__ == "__main__":
    try:
        mapping = load_camera_area_map()
        print(f"✓ 접속 성공 — 총 {len(mapping)}개 카메라 매핑 로드됨:\n")
        for cam_key, area_id in mapping.items():
            print(f"  {cam_key:50s} → area_id={area_id}")
    except Exception as e:
        print(f"✗ 오류: {type(e).__name__}: {e}")
