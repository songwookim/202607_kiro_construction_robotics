"""시험만: 프로세스 풀 자식 안에서만 다르게 도는 함수 (spawn 자식이 이름으로 import 한다 — 시험 파일보다 가볍게)."""
import os
import signal


def die_in_child(owner_pid: int, x: int) -> int:
    """풀의 자식에서는 스스로 죽는다 (SIGKILL), 주인 (owner_pid) 에서는 2·x."""
    if os.getpid() != owner_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    return 2 * x


def sleep_or_pow(*a):
    """인자 하나면 그만큼 잔다 (살았지만 답이 없는 자식), 둘이면 pow."""
    import time
    return time.sleep(a[0]) if len(a) == 1 else pow(*a)
