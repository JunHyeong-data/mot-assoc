# -*- coding: utf-8 -*-
"""
실측: 프레임당 M(트랙), N(검출) 이 어느 체제인가. 임계값에 얼마나 걸리는가.

theory/ 의 두 단서 조항이 실제로 얼마나 자주 발동하는지를 센다.
  (1) 모양 조건  : M > N 이면 행상수 불변성이 깨진다
  (2) 임계값 조건: cost <= thresh 로 자르면 M,N 과 무관하게 깨진다

방법: ultralytics 의 matching.linear_assignment 를 가로채서 호출마다
      비용행렬 모양·임계값·최적쌍·절단된 쌍을 기록한다. 트래커 로직 자체는
      건드리지 않는다.

주의 두 가지
  - `lap` 이 없으면 ultralytics 가 런타임에 pip install 을 시도한다.
    여기서는 stub 을 끼워 그걸 막고, 문서화된 scipy 경로(use_lap=False)로 고정한다.
    lap 경로와 결과가 같은지는 아직 미확인이다 (notes/questions.md 열린 질문).
  - 행상수 민감도는 '비용에 행상수를 더하면 최종 매칭이 바뀌는가' 를 잰다.
    크기 a 를 여러 개 놓고 재야 정직하다. 하나만 고르면 원하는 답이 나온다.

사용법:
    python experiments/exp00_mn_census/census.py <영상경로> [모델경로]
"""
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ---- lap 자동설치 차단. 반드시 ultralytics.trackers 임포트 전에. ----------
if "lap" not in sys.modules:
    _stub = types.ModuleType("lap")
    _stub.__version__ = "0.0.0-stub-blocked"

    def _no(*a, **k):
        raise RuntimeError("census.py 는 scipy 경로만 쓴다")

    _stub.lapjv = _no
    sys.modules["lap"] = _stub

from scipy.optimize import linear_sum_assignment            # noqa: E402
from ultralytics import YOLO                                 # noqa: E402
from ultralytics.trackers.utils import matching              # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "../VLMDrivingAssis/assets/sample_autobahn.mp4"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "../BSDsystem/yolov8m.pt"
# 3번째 인자로 클래스 제한. MOT17 은 보행자 전용이므로 보행자 영상엔 "0" 을 준다.
CLASSES = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
PERTURB = (0.01, 0.05, 0.15)   # 행상수 크기 a_i ~ U(0, a)
DRAWS = 5                      # 크기마다 난수 시행 수

rng = np.random.default_rng(0)
log = []
_orig = matching.linear_assignment


def matched_set(cost, thresh):
    """ultralytics matching.linear_assignment 의 scipy 경로와 같은 판정."""
    x, y = linear_sum_assignment(cost)
    return frozenset((int(x[i]), int(y[i])) for i in range(len(x))
                     if cost[x[i], y[i]] <= thresh)


def patched(cost_matrix, thresh, use_lap=True):
    C = np.asarray(cost_matrix, dtype=float)
    if C.size:
        M, N = C.shape
        x, y = linear_sum_assignment(C)
        opt = len(x)
        kept = int(sum(C[x[i], y[i]] <= thresh for i in range(opt)))
        base = matched_set(C, thresh)

        # 두 지표를 같이 낸다.
        #   sens  = 호출 단위. '하나라도 바뀌었나'. 행렬이 크면 자동으로 커진다
        #   plost = 쌍 단위. 바뀐 쌍 수 / 원래 쌍 수. 크기에 robust 하다
        sens, lost, tot = {}, {}, {}
        for a in PERTURB:
            diff = nl = nt = 0
            for _ in range(DRAWS):
                Ca = C + rng.uniform(0, a, M)[:, None]
                pert = matched_set(Ca, thresh)
                diff += pert != base
                nl += len(base - pert)
                nt += len(base)
            sens[a], lost[a], tot[a] = diff / DRAWS, nl, nt

        log.append(dict(M=M, N=N, thresh=float(thresh), opt=opt, kept=kept,
                        gate=float((C <= thresh).mean()),
                        sens=sens, lost=lost, tot=tot))
    # 트래커에는 원래 함수의 결과를 그대로 돌려준다 (scipy 경로 고정)
    return _orig(cost_matrix, thresh, use_lap=False)


matching.linear_assignment = patched

print("=" * 74)
print("실측: M, N 체제와 임계값 절단률")
print("=" * 74)
print(f"  영상  : {VIDEO}")
print(f"  모델  : {MODEL}")
print(f"  트래커: bytetrack.yaml (match_thresh 0.8, fuse_score True)")
print(f"  할당  : scipy 경로 고정 (lap 미설치)")
print(f"  클래스: {'전체' if CLASSES is None else CLASSES}")
print()

model = YOLO(MODEL)
frames = 0
for _ in model.track(source=VIDEO, tracker="bytetrack.yaml", stream=True,
                     persist=True, verbose=False, classes=CLASSES):
    frames += 1

print(f"  처리 프레임 {frames}, linear_assignment 호출 {len(log)}회")
print()

# ---- 단계별 집계. bytetrack 은 프레임당 최대 3회 부른다 -------------------
STAGE = {0.8: "1단계 (고신뢰 검출)", 0.5: "2단계 (저신뢰 검출)", 0.7: "3단계 (미확정 트랙)"}
by_stage = defaultdict(list)
for r in log:
    by_stage[r["thresh"]].append(r)

print("=" * 74)
print("[1] M, N 체제")
print("=" * 74)
print(f"  {'단계':>22}{'호출':>7}{'M 중앙':>8}{'N 중앙':>8}{'M>N':>9}{'M=N':>9}{'M<N':>9}")
print("  " + "-" * 72)
for th in sorted(by_stage, reverse=True):
    rs = by_stage[th]
    M = np.array([r["M"] for r in rs]); N = np.array([r["N"] for r in rs])
    n = len(rs)
    print(f"  {STAGE.get(th, f'thresh={th}'):>22}{n:>7}"
          f"{np.median(M):>8.0f}{np.median(N):>8.0f}"
          f"{f'{(M > N).mean():.1%}':>9}{f'{(M == N).mean():.1%}':>9}"
          f"{f'{(M < N).mean():.1%}':>9}")

allM = np.array([r["M"] for r in log]); allN = np.array([r["N"] for r in log])
print("  " + "-" * 72)
print(f"  {'전체':>22}{len(log):>7}{np.median(allM):>8.0f}{np.median(allN):>8.0f}"
      f"{f'{(allM > allN).mean():.1%}':>9}{f'{(allM == allN).mean():.1%}':>9}"
      f"{f'{(allM < allN).mean():.1%}':>9}")

print()
print("=" * 74)
print("[2] 임계값 절단률")
print("=" * 74)
print("  헝가리안이 고른 최적쌍 중 cost > thresh 라서 버려지는 비율.")
print()
print(f"  {'단계':>22}{'최적쌍':>9}{'채택':>8}{'절단':>8}{'절단률':>9}{'게이트밀도':>12}")
print("  " + "-" * 72)
for th in sorted(by_stage, reverse=True):
    rs = by_stage[th]
    opt = sum(r["opt"] for r in rs); kept = sum(r["kept"] for r in rs)
    gate = np.mean([r["gate"] for r in rs])
    cut = opt - kept
    print(f"  {STAGE.get(th, f'thresh={th}'):>22}{opt:>9}{kept:>8}{cut:>8}"
          f"{f'{cut / max(opt, 1):.1%}':>9}{f'{gate:.1%}':>12}")
opt = sum(r["opt"] for r in log); kept = sum(r["kept"] for r in log)
print("  " + "-" * 72)
print(f"  {'전체':>22}{opt:>9}{kept:>8}{opt - kept:>8}"
      f"{f'{(opt - kept) / max(opt, 1):.1%}':>9}"
      f"{f'{np.mean([r[chr(103)+chr(97)+chr(116)+chr(101)] for r in log]):.1%}':>12}")

print()
print("=" * 74)
print("[3] 행상수 민감도 -- 단서 조항이 실제로 발동하는 빈도")
print("=" * 74)
print("  비용에 행상수 a_i ~ U(0, a) 를 더했을 때 최종 매칭이 바뀐 정도.")
print("  이론상 임계값이 없고 M<=N 이면 0% 여야 한다.")
print()
print("  (가) 호출 단위 -- '하나라도 바뀐' 호출의 비율. 행렬이 크면 자동으로 커진다.")
print(f"  {'단계':>22}" + "".join(f"{f'a={a}':>12}" for a in PERTURB))
print("  " + "-" * 60)
for th in sorted(by_stage, reverse=True):
    rs = by_stage[th]
    print(f"  {STAGE.get(th, f'thresh={th}'):>22}"
          + "".join(f"{np.mean([r['sens'][a] for r in rs]):>12.1%}" for a in PERTURB))
print("  " + "-" * 60)
print(f"  {'전체':>22}" + "".join(
    f"{np.mean([r['sens'][a] for r in log]):>12.1%}" for a in PERTURB))

print()
print("  (나) 쌍 단위 -- 바뀐 쌍 / 원래 쌍. 행렬 크기에 robust 하다. 이쪽이 정직하다.")
print(f"  {'단계':>22}" + "".join(f"{f'a={a}':>12}" for a in PERTURB))
print("  " + "-" * 60)
for th in sorted(by_stage, reverse=True):
    rs = by_stage[th]
    print(f"  {STAGE.get(th, f'thresh={th}'):>22}" + "".join(
        f"{sum(r['lost'][a] for r in rs) / max(sum(r['tot'][a] for r in rs), 1):>12.1%}"
        for a in PERTURB))
print("  " + "-" * 60)
print(f"  {'전체':>22}" + "".join(
    f"{sum(r['lost'][a] for r in log) / max(sum(r['tot'][a] for r in log), 1):>12.1%}"
    for a in PERTURB))

print()
print("=" * 74)
print("해석")
print("=" * 74)
mn = (allM > allN).mean()
print(f"  M > N 인 호출이 {mn:.1%}. 이 구간에서는 행상수 불변성이 원래 깨진다.")
print(f"  최적쌍의 {(opt - kept) / max(opt, 1):.1%} 가 임계값에서 잘린다. 절단이 일어나는 한")
print(f"  분리되는 항도 '채택/기각' 을 통해 결과에 도달한다.")
print(f"  행상수 민감도가 0 이 아니면, 그 자체가 '분리되면 기여 0' 이 실제")
print(f"  파이프라인에서는 성립하지 않는다는 직접 증거다.")
