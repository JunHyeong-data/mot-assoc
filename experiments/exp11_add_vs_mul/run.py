# -*- coding: utf-8 -*-
"""실험 11 (E1) -- **가산 대 곱, 실데이터에서.**

사전 등록은 `PREREG.md` (커밋 `2ac0db4`, 자료보다 먼저).

원고 2절의 설계 제약을 합성에서 실측으로 옮긴다:

    검출별 스칼라를 비용에 **더하면** 순수 열상수라 할당 기여가 정확히 0.
    **단 N <= M (모든 열이 배정될 때) 에 한한다.**

**한 번의 재생 안에서** 매 연관 호출마다 base/add/mul 세 비용을 만들고 셋 다
푼다. 트래커는 base 로 진행하므로 관측이 개입을 안 바꾼다.

사용법:
    python experiments/exp11_add_vs_mul/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))                       # experiments/
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                # noqa: E402
from stage_util import which_stage, stage_thresh                # noqa: E402
from replay import WTracker, Det, load, SEQS, BASE             # noqa: E402
import evaluate as EV                                          # noqa: E402
from tracker.eval.collections.hota import HOTA                 # noqa: E402

TARGET_DELTA = 0.20        # add 의 목표 평균 |Δcost| (사전 등록)
MIN_CALLS = 1000
OUT = Path("data/exp11/tracks")

LOG = []
LOG_NOTE = None   # (LOG 는 연관 호출마다 한 줄)
CALLS = {}        # 단계별 호출 수 (감사 정정)


def unit(dets):
    """검출별 스칼라. 프레임 안에서 [0,1] 로 표준화한 sigma."""
    v = np.array([float(d.det_var[0]) + float(d.det_var[1]) for d in dets])
    u = np.sqrt(np.maximum(v, 0.0))
    lo, hi = float(u.min()), float(u.max())
    return np.zeros_like(u) if hi - lo < 1e-12 else (u - lo) / (hi - lo)


def perturb(base, u):
    """add 와 mul 을 만든다. 섭동 크기를 서로 맞춘다 (사전 등록 [0b])."""
    a = TARGET_DELTA / max(float(np.mean(u)), 1e-9)
    add = base + a * u[None, :]
    d_add = float(np.mean(np.abs(add - base)))
    denom = max(float(np.mean(np.abs(base * u[None, :]))), 1e-9)
    mul = base * (1.0 + (d_add / denom) * u[None, :])
    return add, mul


def solve(cost, thresh):
    """(임계값 전 할당, 임계값 후 채택쌍)."""
    m_all = np.asarray(matching.linear_assignment(cost, 1e9)[0]).reshape(-1, 2)
    m_thr = np.asarray(matching.linear_assignment(cost, thresh)[0]).reshape(-1, 2)
    return frozenset(map(tuple, m_all)), frozenset(map(tuple, m_thr))


class E1Tracker(WTracker):
    """1단계 비용에서 세 조건을 나란히 풀어 기록한다. 진행은 base 로."""

    def get_dists(self, tracks, detections):
        base = super().get_dists(tracks, detections)
        if base.ndim != 2 or 0 in base.shape:
            return base
        M, N = base.shape
        # 감사 정정 (2026-08-18): get_dists 는 1단계와 3단계에서 둘 다 불린다.
        # 예전에는 3단계 호출에도 match_thresh(0.8)를 썼는데 실제는 0.7 이다.
        stage = which_stage(tracks)
        CALLS[stage] = CALLS.get(stage, 0) + 1
        u = unit(detections)
        add, mul = perturb(base, u)
        th = stage_thresh(self.args, stage)
        p0, t0 = solve(base, th)
        p1, t1 = solve(add, th)
        p2, t2 = solve(mul, th)
        # **동점 검사.** 쌍 집합이 달라도 base 비용 합이 같으면 최적성은 보존된다.
        # 헝가리안은 동점에서 어느 최적해를 돌려줄지 정하지 않는다.
        opt_ok = True
        if p1 != p0:
            c0 = sum(base[i, j] for i, j in p0)
            c1 = sum(base[i, j] for i, j in p1)
            opt_ok = abs(c0 - c1) < 1e-6
        LOG.append(dict(
            stage=stage, M=M, N=N, le=(N <= M), opt_ok=opt_ok,
            d_add=float(np.mean(np.abs(add - base))),
            d_mul=float(np.mean(np.abs(mul - base))),
            same_add=(p1 == p0), same_mul=(p2 == p0),
            thr_add=(t1 == t0), thr_mul=(t2 == t0)))
        return base


class AddTracker(WTracker):
    def get_dists(self, tracks, detections):
        c = super().get_dists(tracks, detections)
        if c.ndim != 2 or 0 in c.shape:
            return c
        return perturb(c, unit(detections))[0]


class MulTracker(WTracker):
    def get_dists(self, tracks, detections):
        c = super().get_dists(tracks, detections)
        if c.ndim != 2 or 0 in c.shape:
            return c
        return perturb(c, unit(detections))[1]


def replay(cls, tag, arm="iou"):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        c = load(seq, arm)
        tr = cls(SimpleNamespace(**BASE), arm, 1.0, frame_rate=30)
        lines = []
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            det = Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                      c["sxx"][m], c["syy"][m])
            for row in tr.update(det):
                x1, y1, x2, y2 = row[:4]
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                             % (f, int(row[4]), x1, y1, x2 - x1, y2 - y1, float(row[5])))
        (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")


def hota_of(tag):
    """evaluate.TRACKS 를 잠시 이 실험 폴더로 돌린다."""
    keep = EV.TRACKS
    EV.TRACKS = OUT
    try:
        metric, per = HOTA(), {}
        for seq in SEQS:
            d = EV.build_data(seq, tag)
            if d is None:
                return float("nan")
            per[seq] = metric.eval_sequence(d)
        return 100 * float(np.mean(metric.combine_sequences(per)["HOTA"]))
    finally:
        EV.TRACKS = keep


def pct(xs):
    return 100.0 * float(np.mean(xs)) if len(xs) else float("nan")


def main():
    print("=" * 92)
    print("실험 11 (E1) -- 가산 대 곱, 실데이터에서")
    print("=" * 92)
    print("사전 등록 PREREG.md (커밋 2ac0db4, 자료보다 먼저)")
    print("한 번의 재생 안에서 매 연관 호출마다 base/add/mul 셋을 나란히 푼다.")
    print()

    replay(E1Tracker, "base")
    # 감사 정정: 사전 등록 범위는 **1단계**다. 3단계 호출은 빼고 판정한다.
    tot = sum(CALLS.values())
    print("  [진단] get_dists 호출: " + ",  ".join(
        "%s단계 %d (%.1f%%)" % (k, v, 100.0 * v / tot) for k, v in sorted(
            CALLS.items(), key=lambda z: (z[0] is None, z[0]))))
    print("         **예전 판은 둘을 섞어 세었고 3단계에도 0.8 을 썼다 (실제 0.7)**")
    print()
    L = [r for r in LOG if r["stage"] == 1]
    if not L:
        print("연관 호출이 없다")
        return 1

    print("=" * 92)
    print("사전 등록한 사전 점검")
    print("=" * 92)
    da = float(np.mean([r["d_add"] for r in L]))
    dm = float(np.mean([r["d_mul"] for r in L]))
    ratio = dm / max(da, 1e-12)
    ok = True
    g0a = da >= 0.1
    ok &= g0a
    print("  [0a] 섭동이 크다 -- add 평균 |Δcost| = %.4f  %s"
          % (da, "OK (>=0.1)" if g0a else "** 실패: 작은 섭동으로는 무의미 **"))
    g0b = 0.5 <= ratio <= 2.0
    ok &= g0b
    print("  [0b] 두 섭동 크기가 비슷하다 -- mul 평균 %.4f, 비 %.2f  %s"
          % (dm, ratio, "OK" if g0b else "** 범위 밖 **"))
    g0c = len(L) >= MIN_CALLS
    ok &= g0c
    print("  [0c] 연관 호출 %d 건  %s" % (len(L), "OK" if g0c else "** 1000 미만 **"))
    if not ok:
        print()
        print("  ** 사전 점검 실패. 판정하지 않는다 **")
        return 1

    le = [r for r in L if r["le"]]
    gt = [r for r in L if not r["le"]]
    print()
    print("  호출 구성: N<=M %d건 (%.1f%%),  N>M %d건 (%.1f%%)"
          % (len(le), 100.0 * len(le) / len(L), len(gt), 100.0 * len(gt) / len(L)))

    print()
    print("=" * 92)
    print("사전 등록한 평가지표")
    print("=" * 92)
    e1 = pct([r["same_add"] for r in le])
    e2 = pct([r["same_add"] for r in gt])
    e3 = pct([r["same_mul"] for r in L])
    e3le = pct([r["same_mul"] for r in le])
    e4 = 100.0 - pct([r["thr_add"] for r in L])
    e4m = 100.0 - pct([r["thr_mul"] for r in L])
    print("  [1] N<=M 에서 add 의 임계값 전 할당 일치율 = %.4f %%   (예측 100)" % e1)
    print("  [2] N>M  에서 add 의 임계값 전 할당 일치율 = %.4f %%   (예측 <100)" % e2)
    print("  [3] mul 의 임계값 전 할당 일치율          = %.4f %%   (N<=M 만 %.4f %%)"
          % (e3, e3le))
    print("  [4] add 의 임계값 후 채택쌍이 **다른** 호출 = %.4f %%" % e4)
    print("      (참고) mul 의 임계값 후가 다른 호출     = %.4f %%" % e4m)

    print()
    print("=" * 92)
    print("참고 -- 끝까지 돌리면 (성능 주장 아님)")
    print("=" * 92)
    replay(AddTracker, "add")
    replay(MulTracker, "mul")
    hb, ha, hm = hota_of("base"), hota_of("add"), hota_of("mul")
    print("  HOTA  base %.3f   add %+.3f   mul %+.3f" % (hb, ha - hb, hm - hb))

    print()
    print("=" * 92)
    print("판정 -- 사전 등록한 읽는 법")
    print("=" * 92)
    if abs(e1 - 100.0) < 1e-9:
        print("  [1] **정확히 100%%.** 가산은 N<=M 에서 할당 기여가 0 이다.")
        print("      평균 |Δcost| %.3f 의 **큰** 섭동을 주고도 한 칸도 안 움직인다." % da)
        if e3 < 99.0:
            print("  [3] 곱은 %.2f%% 만 같다 -> **곱은 도달한다.**" % e3)
            print("      => **설계 제약이 실데이터에서 확인된다.**")
        else:
            print("  [3] 곱도 거의 안 바꾼다 (%.2f%%). 섭동이 약했을 수 있다" % e3)
    else:
        # **여기서 끝내면 안 된다.** 헝가리안은 동점에서 어느 최적해를 줄지
        # 정하지 않는다. 쌍 집합이 달라도 base 비용 합이 같으면 명제(기여 0)는
        # 살아 있다. 그걸 검사한 뒤에 판정한다 (감사 정정 2026-08-18).
        bad = [r for r in le if not r["opt_ok"]]
        n_diff = int(round(len(le) * (100.0 - e1) / 100.0))
        print("  [1] 쌍 집합이 다른 호출 %d 건 (%.4f%% 일치)" % (n_diff, e1))
        print("      그중 **최적성이 깨진 것 = %d 건**" % len(bad))
        if not bad:
            print("      => **전부 동점이다. 명제는 살아 있다** --")
            print("         가산은 최적 할당의 *비용* 을 안 바꾼다. 솔버가 동점에서")
            print("         다른 최적해를 골랐을 뿐이다")
            if e3 < 99.0:
                print("  [3] 곱은 %.2f%% 만 같다 -> **곱은 도달한다.**" % e3)
                print("      => **설계 제약이 실데이터에서 확인된다.**")
        else:
            print("      => **최적성이 실제로 깨졌다. 명제나 구현이 틀렸다.**")
            print("         원고 2절을 고쳐야 한다")
    if e4 > 0:
        print("  [4] 임계값 경로가 살아 있다 -- 할당은 같은데 채택이 %.2f%% 에서 다르다" % e4)
    else:
        print("  [4] 임계값 예외가 이 자료에서는 발현되지 않았다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
