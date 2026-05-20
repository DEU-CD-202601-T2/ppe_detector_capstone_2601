"""위반 사건을 비동기로 DB에 적재.

추론 스레드에서 log()를 호출하면 큐에 넣고 즉시 리턴.
별도 워커 스레드가 큐를 폴링하며 DB INSERT.
"""
import queue
import threading
import traceback
from datetime import datetime

import pymysql

from db_config import DB_CONFIG


# ── 설정값 ────────────────────────────────────────────────
QUEUE_MAX_SIZE = 500   # 가득 차면 신규 위반 drop (추론 정지 방지)


class ViolationLogger:
    """위반 사건을 받아 큐에 쌓고 워커 스레드가 DB INSERT."""

    def __init__(self, queue_max: int = QUEUE_MAX_SIZE):
        self.queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop,
                                        daemon=True, name="ViolationLogger")
        self._worker.start()
        print("✓ ViolationLogger 워커 스레드 시작")

    def log(self,
            violation_type: str,
            area_id: int,
            person_id: int,
            image_jpeg: bytes,
            detected_at: datetime | None = None):
        """추론 스레드에서 호출. 즉시 리턴 (큐 적재만)."""
        item = {
            "violation_type": violation_type,
            "area_id":        area_id,
            "person_id":      person_id,
            "image_data":     image_jpeg,
            "detected_at":    detected_at or datetime.now(),
        }
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            print(f"  ⚠ 위반 큐 가득 참 → drop ({violation_type}, "
                  f"area={area_id}, pid={person_id})")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._insert(item)
            except Exception:
                print("  ⚠ 위반 저장 실패:")
                traceback.print_exc()

    def _insert(self, item: dict):
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO violations "
                    "(violation_type, detected_at, area_id, person_id, "
                    " image_data, image_mime, is_checked) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 0)",
                    (item["violation_type"],
                     item["detected_at"],
                     item["area_id"],
                     item["person_id"],
                     item["image_data"],
                     "image/jpeg"),
                )
            conn.commit()
            size_kb = len(item["image_data"]) / 1024
            print(f"  ✓ 위반 저장: {item['violation_type']} | "
                  f"area={item['area_id']} | pid={item['person_id']} | "
                  f"img={size_kb:.1f}KB")
        finally:
            conn.close()

    def stop(self):
        self._stop_event.set()
        self._worker.join(timeout=3.0)
