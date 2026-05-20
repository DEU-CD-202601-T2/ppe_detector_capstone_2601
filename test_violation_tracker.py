"""ViolationStateTracker 동작 검증."""
import time
from violation_tracker import ViolationStateTracker


def main():
    # 빠른 테스트를 위해 임계값을 줄임
    tracker = ViolationStateTracker(sustain_sec=1.0, cooldown_sec=3.0)
    cam = "CAM0"
    pid = 7

    # 시나리오: helmet 없이 들어와서 2초 머무름
    print("=== 시나리오 1: helmet 미착용 2초 ===")
    status = {"helmet": False, "mask": True,
              "left_glove": True, "right_glove": True}

    for i in range(20):    # 0.1초 간격으로 20회 → 2초
        events = tracker.update(cam, pid, status)
        if events:
            for e in events:
                elapsed = time.time() - (e.timestamp - 0)
                print(f"  ⚠ {e.violation_type} 발생 (t={i*0.1:.1f}s)")
        time.sleep(0.1)

    # 시나리오: cooldown 동안은 기록 안 됨
    print("\n=== 시나리오 2: cooldown 중 (1초 추가 미착용) ===")
    for i in range(10):
        events = tracker.update(cam, pid, status)
        if events:
            print(f"  ⚠ 또 발생?! (t={i*0.1:.1f}s)")
        time.sleep(0.1)
    print("  (위반 발생 없음이 정상)")

    # 시나리오: 3초 더 기다림 → cooldown 풀림, 또 발생
    print("\n=== 시나리오 3: cooldown 후 다시 미착용 ===")
    time.sleep(3.0)
    for i in range(20):
        events = tracker.update(cam, pid, status)
        if events:
            for e in events:
                print(f"  ⚠ {e.violation_type} 또 발생 (t={i*0.1:.1f}s)")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
