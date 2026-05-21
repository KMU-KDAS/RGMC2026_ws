import time # 시간 측정을 위한 파이썬 내장 라이브러리
from collections import defaultdict # 딕셔너리(사전)에 없는 키를 찾을 때 에러를 내는 대신 기본값을 만들어주는 자료구조
from contextlib import contextmanager # 파이썬의 'with' 구문을 쉽게 만들 수 있게 해주는 데코레이터

import config # 환경설정 파일 임포트


# 전역(Global) 변수: 각 함수별로 실행 횟수(count)와 누적 실행 시간(total_ms)을 저장하는 딕셔너리입니다.
# lambda: {"count": 0, "total_ms": 0.0} 를 통해, 새로운 이름의 타이머가 불리면 자동으로 0으로 초기화된 값을 세팅합니다.
_PERF_ACCUM = defaultdict(lambda: {"count": 0, "total_ms": 0.0})


def perf_enabled():
    # config.py 파일에 "PERF_LOG" 설정이 켜져 있는지(True) 꺼져 있는지(False) 확인합니다.
    # getattr를 써서 만약 PERF_LOG 변수 자체가 config에 없더라도 에러 없이 False를 반환하도록 안전하게 짰습니다.
    return bool(getattr(config, "PERF_LOG", False))


def log_perf(name, elapsed_ms, every=1):
    # 측정된 시간(elapsed_ms)을 받아서 화면에 프린트할지 말지 결정하고 누적하는 함수입니다.
    
    # 1. 성능 측정이 꺼져있으면 아무것도 안 하고 바로 종료합니다.
    if not perf_enabled():
        return

    # 2. every 값이 1 이하이면, 함수가 불릴 때마다 매번 화면에 즉시 로그를 찍습니다.
    if every <= 1:
        print(f"[PERF] {name}: {elapsed_ms:.1f} ms")
        return

    # 3. every 값이 1보다 크면 (예: every=20), 매번 찍지 않고 통계를 누적합니다.
    bucket = _PERF_ACCUM[name] # 해당 타이머 이름(name)의 통계 바구니를 가져옵니다.
    bucket["count"] += 1 # 실행 횟수 1 증가
    bucket["total_ms"] += float(elapsed_ms) # 걸린 시간을 누적 시간에 더합니다.
    
    # 누적된 실행 횟수가 설정된 주기(every)의 배수가 될 때만 화면에 평균값을 출력합니다. (터미널 도배 방지)
    if bucket["count"] % every == 0:
        avg_ms = bucket["total_ms"] / bucket["count"] # 누적 시간 / 실행 횟수 = 평균 시간
        print(f"[PERF] {name}: avg {avg_ms:.1f} ms over {bucket['count']} calls")


@contextmanager
def perf_timer(name, every=1):
    # 이 파일의 핵심! 'with perf_timer("이름"):' 형태로 감싸진 코드 블록의 실행 시간을 측정하는 마법의 함수입니다.
    
    # 1. 블록 실행 직전: 현재 시간을 극도로 정밀하게(마이크로초 단위까지) 기록합니다.
    start = time.perf_counter() 
    try:
        # yield는 'with' 블록 안에 있는 실제 메인 코드들이 실행되도록 제어권을 넘겨주는 역할을 합니다.
        yield 
    finally:
        # 2. 블록 실행 직후: 메인 코드 실행이 끝나면(또는 에러가 나더라도 finally 덕분에) 시간을 다시 잽니다.
        # (끝난 시간 - 시작 시간) * 1000을 해서 초(s) 단위를 밀리초(ms) 단위로 변환합니다.
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        # 측정된 시간을 위에서 만든 log_perf 함수로 보내 기록/출력합니다.
        log_perf(name, elapsed_ms, every=every)