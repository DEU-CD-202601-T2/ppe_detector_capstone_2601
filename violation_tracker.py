"""(카메라, person_id, type) 단위 위반 상태 추적.

규칙:
  - 미착용이 10초 연속 지속되면 → 위반 발생 1건
  - 같은 (cam, pid, type)는 300초간 재기록 안 함 (cooldown)
  - 미착용 끊기면(착용 감지) 시작 시각 리셋
"""
import time
from typing import NamedTuple


# ── 설정값 ────────────────────────────────────────────────
SUSTAIN_SECONDS = 10.0     # 미착용 지속 시간 임계값
COOLDOWN_SECONDS = 300.0   # 같은 (cam, pid, type) 재기록 방지 시간

# PPE type 정의
PPE_TYPES = ("no_helmet", "no_mask", "no_glove_left", "no_glove_right")


class ViolationEvent(NamedTuple):
    """발생한 위반 사건."""
    cam_key: str
    person_id: int
    violation_type: str
    timestamp: float


class ViolationStateTracker:
    """(cam_key, person_id, type)별 시작 시각과 마지막 기록 시각을 추적."""

    def __init__(
        self,
        sustain_sec: float = SUSTAIN_SECONDS,
        cooldown_sec: float = COOLDOWN_SECONDS,
    ):
        self.sustain_sec = sustain_sec
        self.cooldown_sec = cooldown_sec

        # key: (cam_key, person_id, type)
        # value: {"started_at": float | None, "last_logged": float | None}
        self._state: dict[tuple, dict] = {}

    def update(self, cam_key: str, person_id: int, status: dict) -> list[ViolationEvent]:
        """프레임마다 호출.

        status:
            {
                "helmet": bool,
                "mask": bool,
                "left_glove": bool,
                "right_glove": bool,
            }

        True = 착용, False = 미착용

        반환:
            이번 프레임에 새로 발생한 위반 사건 리스트
        """
        if person_id < 0:
            return []

        now = time.time()
        events: list[ViolationEvent] = []

        # PPE type별 미착용 여부
        violations_now = {
            "no_helmet": not status["helmet"],
            "no_mask": not status["mask"],
            "no_glove_left": not status["left_glove"],
            "no_glove_right": not status["right_glove"],
        }

        for vt, is_missing in violations_now.items():
            key = (cam_key, person_id, vt)
            st = self._state.setdefault(
                key,
                {
                    "started_at": None,
                    "last_logged": None,
                },
            )

            if not is_missing:
                # 착용 감지 → 시작 시각 리셋
                if st["started_at"] is not None:
                    print(
                        f"[VT-RESET] cam={cam_key} pid={person_id} type={vt} "
                        f"elapsed={now - st['started_at']:.1f}s",
                        flush=True,
                    )
                st["started_at"] = None
                continue

            # cooldown 중이면 재기록하지 않음
            if (
                st["last_logged"] is not None
                and now - st["last_logged"] < self.cooldown_sec
            ):
                remain = self.cooldown_sec - (now - st["last_logged"])
                print(
                    f"[VT-COOLDOWN] cam={cam_key} pid={person_id} type={vt} "
                    f"remain={remain:.1f}s",
                    flush=True,
                )
                continue

            # 미착용 시작 시각 기록
            if st["started_at"] is None:
                st["started_at"] = now
                print(
                    f"[VT-START] cam={cam_key} pid={person_id} type={vt}",
                    flush=True,
                )
                continue

            elapsed = now - st["started_at"]
            print(
                f"[VT-HOLD] cam={cam_key} pid={person_id} type={vt} "
                f"elapsed={elapsed:.1f}s/{self.sustain_sec:.1f}s",
                flush=True,
            )

            # 미착용 지속 시간이 임계값 이상이면 이벤트 발생
            if elapsed >= self.sustain_sec:
                print(
                    f"[VT-EVENT] cam={cam_key} pid={person_id} type={vt}",
                    flush=True,
                )
                events.append(
                    ViolationEvent(
                        cam_key=cam_key,
                        person_id=person_id,
                        violation_type=vt,
                        timestamp=now,
                    )
                )
                st["last_logged"] = now
                st["started_at"] = None

        return events

    def cleanup_stale(
        self,
        active_ids: set[tuple[str, int]],
        timeout_sec: float = 60.0,
    ):
        """오랫동안 안 보이는 (cam_key, person_id) 상태 정리.

        active_ids:
            현재 프레임에 등장한 (cam_key, person_id) 집합
        """
        now = time.time()
        to_delete = []

        for key, st in self._state.items():
            cam_key, pid, _ = key
            if (cam_key, pid) in active_ids:
                continue

            last_activity = st["last_logged"] or st["started_at"] or 0
            if now - last_activity > timeout_sec:
                to_delete.append(key)

        for key in to_delete:
            del self._state[key]
