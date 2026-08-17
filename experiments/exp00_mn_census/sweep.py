# -*- coding: utf-8 -*-
"""
census.py 의 결과가 설정 부작용인지 검증한다.

의심: "M>N 이 77%" 는 발견이 아니라 track_buffer=30 때문일 수 있다.
      버퍼를 줄이면 잃은 트랙이 빨리 사라져 M 이 작아지고, M>N 비율도 떨어질 것이다.
      만약 그렇다면 '가림 중 M>N' 이라는 서술은 설정 의존적이라고 못박아야 한다.

의심 2: 2단계가 죽은 게 fuse_score 때문이라는 주장. fuse_score=False 로 두면
        정말 살아나는지 직접 확인한다.

방법: 검출기를 한 번만 돌려 캐시하고, 설정만 바꿔 BYTETracker 에 재생한다.
      검출이 동일하므로 차이는 순수하게 트래커 설정에서 온다.

사용법:
    python experiments/exp00_mn_census/sweep.py <영상경로> [모델경로]
"""
import sys
import types
from types import SimpleNamespace

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

if "lap" not in sys.modules:                      # census.py 와 같은 이유
    _stub = types.ModuleType("lap")
    _stub.__version__ = "0.0.0-stub-blocked"
    _stub.lapjv = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scipy only"))
    sys.modules["lap"] = _stub

from scipy.optimize import linear_sum_assignment              # noqa: E402
from ultralytics import YOLO                                   # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker      # noqa: E402
from ultralytics.trackers.utils import matching                # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "../VLMDrivingAssis/assets/sample_autobahn.mp4"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "../BSDsystem/yolov8m.pt"
PERTURB = 0.15
DRAWS = 5

BASE = dict(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
            new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)

rng = np.random.default_rng(0)
LOG = []
_orig = matching.linear_assignment


class Det:
    """BYTETracker 가 요구하는 최소 인터페이스 (.xywh/.conf/.cls, 불리언 인덱싱)."""

    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __getitem__(self, m):
        return Det(self.xywh[m], self.conf[m], self.cls[m])

    def __len__(self):
        return len(self.conf)


def matched_set(cost, thresh):
    x, y = linear_sum_assignment(cost)
    return frozenset((int(x[i]), int(y[i])) for i in range(len(x))
                     if cost[x[i], y[i]] <= thresh)


def patched(cost_matrix, thresh, use_lap=True):
    C = np.asarray(cost_matrix, dtype=float)
    if C.size:
        M, N = C.shape
        x, y = linear_sum_assignment(C)
        kept = int(sum(C[x[i], y[i]] <= thresh for i in range(len(x))))
        base = matched_set(C, thresh)
        sens = sum(matched_set(C + rng.uniform(0, PERTURB, M)[:, None], thresh) != base
                   for _ in range(DRAWS)) / DRAWS
        LOG.append(dict(M=M, N=N, thresh=float(thresh), opt=len(x), kept=kept,
                        gate=float((C <= thresh).mean()), sens=sens))
    return _orig(cost_matrix, thresh, use_lap=False)


matching.linear_assignment = patched

# ---- 검출을 한 번만 돌려 캐시 --------------------------------------------
print("=" * 74)
print("설정 스윕 -- census.py 결과가 설정 부작용인지 검증")
print("=" * 74)
print(f"  영상: {VIDEO}")
print(f"  모델: {MODEL}")
print()
print("  검출기 1회 실행해 캐시하는 중...")

model = YOLO(MODEL)
cache = []
# conf=0.1 은 track() 이 쓰는 값이다 (engine/model.py:577). predict 기본값 0.25 로
# 두면 2단계용 저신뢰 검출이 미리 잘려나가 2단계가 빈 행렬만 받는다.
for r in model.predict(source=VIDEO, stream=True, verbose=False, conf=0.1):
    b = r.boxes
    cache.append(Det(b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()))
print(f"  프레임 {len(cache)}, 검출 총 {sum(len(d) for d in cache)}개. 이후는 재생만 한다.")
print()


def run(**over):
    """캐시된 검출을 주어진 설정의 트래커에 재생하고 집계를 낸다.

    난수를 설정마다 되감는다. 안 되감으면 track_buffer 갈래들이 서로 다른
    교란 표본을 쓰게 되어, 민감도 열의 차이에 난수 잡음이 섞인다. 검출은
    캐시라 같은데 교란만 다른 것은 비교를 흐린다. (2026-08-17)
    """
    global rng
    rng = np.random.default_rng(0)
    LOG.clear()
    args = SimpleNamespace(**{**BASE, **over})
    tr = BYTETracker(args, frame_rate=30)
    for d in cache:
        tr.update(d)
    st1 = [r for r in LOG if r["thresh"] == args.match_thresh]
    st2 = [r for r in LOG if r["thresh"] == 0.5]
    M = np.array([r["M"] for r in st1]); N = np.array([r["N"] for r in st1])
    opt = sum(r["opt"] for r in st1); kept = sum(r["kept"] for r in st1)
    return dict(
        mgt=(M > N).mean() if len(M) else float("nan"),
        medM=np.median(M) if len(M) else 0, medN=np.median(N) if len(N) else 0,
        cut=(opt - kept) / max(opt, 1),
        sens=np.mean([r["sens"] for r in st1]) if st1 else float("nan"),
        s2opt=sum(r["opt"] for r in st2), s2kept=sum(r["kept"] for r in st2),
        s2gate=np.mean([r["gate"] for r in st2]) if st2 else float("nan"),
    )


print("=" * 74)
print("[1] track_buffer 를 바꾸면 M>N 이 무너지는가")
print("=" * 74)
print("  나머지 설정은 기본값 고정. 1단계 주 연관 기준.")
print()
print(f"  {'track_buffer':>13}{'M 중앙':>8}{'N 중앙':>8}{'M>N':>9}{'절단률':>9}{'행상수민감도':>14}")
print("  " + "-" * 62)
for tb in (1, 5, 15, 30, 60, 120):
    r = run(track_buffer=tb)
    tag = "  <- 기본값" if tb == 30 else ""
    mgt_s = f"{r['mgt']:.1%}"
    cut_s = f"{r['cut']:.1%}"
    sens_s = f"{r['sens']:.1%}"
    print(f"  {tb:>13}{r['medM']:>8.0f}{r['medN']:>8.0f}"
          f"{mgt_s:>9}{cut_s:>9}{sens_s:>14}{tag}")

print()
print("  버퍼를 1 로 줄여도 M>N 이 크게 남으면, M>N 은 버퍼 부작용이 아니라")
print("  '검출이 트랙보다 자주 사라진다' 는 실제 성질이다.")

print()
print("=" * 74)
print("[2] fuse_score 를 끄면 2단계가 살아나는가")
print("=" * 74)
print()
print(f"  {'fuse_score':>12}{'2단계 최적쌍':>14}{'채택':>8}{'게이트밀도':>12}"
      f"{'1단계 M>N':>12}{'1단계 절단률':>14}")
print("  " + "-" * 74)
for fs in (True, False):
    r = run(fuse_score=fs)
    g_s = f"{r['s2gate']:.1%}"
    mgt_s = f"{r['mgt']:.1%}"
    cut_s = f"{r['cut']:.1%}"
    print(f"  {str(fs):>12}{r['s2opt']:>14}{r['s2kept']:>8}"
          f"{g_s:>12}{mgt_s:>12}{cut_s:>14}")

print()
print("  fuse_score=True 에서 채택이 0 이고 False 에서 0 이 아니면,")
print("  2단계 사망의 원인이 fuse_score 라는 것이 실증된다.")
