# -*- coding: utf-8 -*-
"""실험 20 — **게이팅 대조군은 무엇을 맞춰야 하는가.**

사전 등록은 `PREREG.md` (보정 3 까지, 전부 자료 전에 커밋).

원고 5.3 의 88% 주장이 어느 기준에 걸려 있는지 잰다.

    A  평균 선형 확장량 (px)   exp03 이 맞춘 것
    B  총 확장 면적            exp19 가 맞춘 것
    C  1단계 채택률            거리 함수 경로가 쓰는 것 (**주 기준**)

## 왜 다시 썼나 (보정 3)

앞판은 `solve()` 가 **양 끝점 둘로 방향을 정하고 이분**했다. 그런데 exp03
방향 측정이 **채택률은 α 에 단조가 아님**을 확정했다 (뒤집힌 U 자).
끝점만 보면 곡선의 꼭대기를 못 본다.

그래서 **소스마다 α 격자를 한 번 훑고** 그 곡선 위에서 판정한다.
`measure()` 가 세 양을 한 번에 주므로 **훑기 한 번이 세 기준을 감당한다.**

**목표가 곡선 밖이면 이분하지 않고 「불능」으로 적는다.** 앞판이
`ALPHA 2000 -> 목표의 1.2%` 로 포화한 것은 탐색 실패가 아니라 **해의 부재**였다.

사용법:
    python experiments/exp20_gatecriterion/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp19_grid"))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import importlib.util as _iu                                    # noqa: E402
_spec = _iu.spec_from_file_location("g19", HERE.parents[0] / "exp19_grid" / "run.py")
G = _iu.module_from_spec(_spec)
_spec.loader.exec_module(G)                                     # noqa: E402

from ultralytics.trackers.utils import matching                 # noqa: E402
from replay import Det, SEQS, BASE                              # noqa: E402

G.OUT = Path("data/exp20/tracks")

# **자료를 보기 전에 고정한 격자** (PREREG 보정 3)
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
NAMES = ["A 평균 선형 확장량", "B 총 확장 면적", "C 1단계 채택률"]
TOL = 0.05                       # 목표 대비 5% 밖이면 그 칸은 쓰지 않는다
REF_ALPHA = 0.5                  # 박스 크기 기준 조건 (앵커). 보정 2 (5) 참고
ITERS = 10                       # 구간이 2^10 배 줄면 5% 문턱에 충분하다

_CACHE = {}


def measure(source, alpha):
    """한 조건에서 세 양을 한 번에 잰다: (평균 선형 확장량, 총 면적, 채택률).

    **훑기와 이분이 같은 점을 여러 번 묻는다.** 재생이 비싸므로 기억해 둔다.
    """
    key = (source, round(float(alpha), 9))
    if key in _CACHE:
        return _CACHE[key]
    pad_sum = pad_n = 0.0
    area = 0.0
    num = den = 0
    for seq in SEQS:
        c = G.load(seq, source)
        tr = G.GridTracker(SimpleNamespace(**BASE), "gate", source, alpha)
        orig = tr.get_dists

        def counted(tracks, dets, _o=orig):
            d = _o(tracks, dets)
            nonlocal num, den
            if d.ndim == 2 and 0 not in d.shape and G._stage1(tracks):
                m, _, _ = matching.linear_assignment(d, float(BASE["match_thresh"]))
                num += len(np.asarray(m).reshape(-1, 2))
                den += min(len(tracks), len(dets))
            return d

        tr.get_dists = counted
        for f in range(1, c["n_frames"] + 1):
            msk = c["frame"] == f
            det = Det(c["xyxy"][msk], c["conf"][msk], np.zeros(int(msk.sum())),
                      c["sxx"][msk], c["syy"][msk])
            tr.update(det)
            # 선형 확장량은 검출별 sigma 로 직접 잰다 (트래커와 무관)
            dv = G.size_var(np.asarray(c["xyxy"][msk], float)) if source == "size" \
                else np.stack([c["sxx"][msk], c["syy"][msk]], 1)
            if len(dv):
                p = alpha * np.sqrt(np.maximum(dv, 0.0))
                pad_sum += float(p.sum()); pad_n += p.size
        area += float(np.sum(tr.pad_area))
    v = (pad_sum / max(pad_n, 1), area, num / max(den, 1))
    _CACHE[key] = v
    return v


def scan(source):
    """α 격자를 한 번 훑어 세 곡선을 동시에 얻는다."""
    out = []
    for a in ALPHAS:
        v = measure(source, a)
        out.append((a, v))
        print("    α %8.4g  ->  선형 %10.4g  면적 %10.4g  채택률 %.6f"
              % (a, v[0], v[1], v[2]), flush=True)
    return out


def shape_of(curve, which):
    """곡선이 단조인가 -- 부수 종말점 [3].

    exp03 에서 채택률이 **뒤집힌 U 자**였다. 이 파이프라인에서도 그런지는
    안 쟀으므로 훑기가 주는 것을 그대로 적는다.
    """
    ys = [v[which] for _, v in curve]
    d = np.diff(ys)
    if np.all(d >= -1e-12):
        return "단조 증가"
    if np.all(d <= 1e-12):
        return "단조 감소"
    top = int(np.argmax(ys))
    return "**비단조** (꼭대기 α=%g, 최대 %.6f)" % (curve[top][0], ys[top])


def solve_on_curve(source, curve, which, target):
    """도달 가능성을 먼저 보고, 가능하면 **단조 구간에서만** 이분한다.

    반환 `(alpha, value, note)`. `alpha` 가 None 이면 **불능**이다 --
    측정 실패가 아니라 **그 대조가 존재하지 않는다**는 뜻이다.
    """
    xs = [a for a, _ in curve]
    ys = [v[which] for _, v in curve]
    lo_y, hi_y = min(ys), max(ys)
    if target > hi_y + 1e-12:
        return None, hi_y, ("**불능** -- 목표가 곡선 위다 "
                            "(도달 최대 %.6g, 간격 %+.6g)" % (hi_y, target - hi_y))
    if target < lo_y - 1e-12:
        return None, lo_y, ("**불능** -- 목표가 곡선 아래다 "
                            "(도달 최소 %.6g, 간격 %+.6g)" % (lo_y, target - lo_y))

    # 목표를 가로지르는 구간을 전부 찾는다
    br = [i for i in range(len(ys) - 1)
          if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]]
    if not br:
        return None, target, "**불능** -- 격자가 목표를 가로지르지 않는다"
    note = " (해가 %d 개 -- **작은 α 를 쓴다**)" % len(br) if len(br) > 1 else ""

    i = br[0]                                  # 개입이 작은 쪽
    lo, hi = xs[i], xs[i + 1]
    inc = ys[i + 1] > ys[i]                    # **이 구간 안에서의** 방향
    v = ys[i + 1]
    for _ in range(ITERS):
        mid = 0.5 * (lo + hi)
        v = measure(source, mid)[which]
        if (v > target) == inc:
            hi = mid
        else:
            lo = mid
    a = 0.5 * (lo + hi)
    return a, measure(source, a)[which], note


def hota(source, alpha, tag):
    G.run("gate", source, alpha, tag)
    return G.evaluate([tag])[tag][0]


def main():
    print("=" * 92)
    print("실험 20 -- 게이팅 대조군은 무엇을 맞춰야 하는가 (보정 3 재설계)")
    print("=" * 92)
    print("주 기준은 C(채택률). **다만 그것이 정의되는지부터 본다.**")
    G.OUT.mkdir(parents=True, exist_ok=True)

    base_h = hota("dfl", 0.0, "base")
    print()
    print("[사전 점검] 기준선 HOTA %.3f (기록 61.002)" % base_h, flush=True)
    if abs(base_h - 61.002) > 0.01:
        print("  **멈춘다.** (CLAUDE.md 규칙 2)")
        return 1

    print()
    print("[훑기] 박스 크기 (대조군이자 목표의 출처)", flush=True)
    ref_curve = scan("size")
    ref = measure("size", REF_ALPHA)
    h_size = hota("size", REF_ALPHA, "size")
    print("  기준 조건(박스 크기, α=%g): 선형 %.4g  면적 %.4g  채택률 %.6f"
          % (REF_ALPHA, ref[0], ref[1], ref[2]))
    print("  기준 조건 HOTA %.3f  (dHOTA %+.3f)" % (h_size, h_size - base_h))

    curves = {}
    for src in ("nms", "dfl"):
        print()
        print("[훑기] %s" % src, flush=True)
        curves[src] = scan(src)

    # ---- 부수 종말점 [3] ----
    print()
    print("=" * 92)
    print("[3] 채택률 곡선의 모양 -- exp03 에서 뒤집힌 U 자였다. 여기서는?")
    print("=" * 92)
    for src, cv in [("size", ref_curve), ("nms", curves["nms"]),
                    ("dfl", curves["dfl"])]:
        print("  %-6s %s" % (src, shape_of(cv, 2)))

    # ---- 세 기준 ----
    out, notes = {}, {}
    for which in (0, 1, 2):
        print()
        print("-" * 92)
        print("기준 %s   목표 %.6g" % (NAMES[which], ref[which]), flush=True)
        for src in ("nms", "dfl"):
            a, v, note = solve_on_curve(src, curves[src], which, ref[which])
            if a is None:
                print("  %-5s %s" % (src, note))
                notes[(which, src)] = note
                continue
            off = abs(v - ref[which]) / max(abs(ref[which]), 1e-12)
            if off > TOL:
                print("  %-5s α %10.4g -> %.6g (목표의 %.1f%%)  !! 5%% 밖 -- 쓰지 않는다"
                      % (src, a, v, 100 * v / ref[which]))
                notes[(which, src)] = "**불능** -- 맞추기가 5% 밖"
                continue
            print("  %-5s α %10.4g -> %.6g  (목표의 %.1f%%)  OK%s"
                  % (src, a, v, 100 * v / ref[which], note), flush=True)
            h = hota(src, a, "%s_%d" % (src, which))
            out[(which, src)] = h - h_size
            print("     HOTA %.3f  =>  **σ - 크기 = %+.3f**" % (h, h - h_size))

    # ---- 표 ----
    print()
    print("=" * 92)
    print("[1][2] 기준별 `검출기 σ - 박스 크기`")
    print("=" * 92)
    print("  %-22s %14s %14s" % ("기준", "NMS", "DFL"))
    print("  " + "-" * 52)
    for which in (0, 1, 2):
        cells = ["%+14.3f" % out[(which, s)] if (which, s) in out
                 else "%14s" % "불능" for s in ("nms", "dfl")]
        print("  %-22s %s %s" % (NAMES[which], cells[0], cells[1]))

    # ---- 판정 (자료 전에 정한 읽는 법) ----
    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md 보정 3)")
    print("=" * 92)
    if (2, "nms") not in out:
        print("  **[1] 주 종말점을 얻지 못했다** -- 기준 C 가 NMS 에 정의되지 않는다.")
        print("     %s" % notes.get((2, "nms"), ""))
        print()
        print("     **이것이 결과다**: 채택률을 맞추는 대조는 NMS 게이팅에")
        print("     원리적으로 구성할 수 없다. 격자의 「개입 불성립」 판정을")
        print("     **다른 잣대로 확인**해 준다 (규칙 3).")
        print("     88% 감사는 A·B 로 내려가고, **주 기준을 적용하지 못했다는")
        print("     것을 원고에 함께 적는다.**")
    else:
        print("  주 판정(기준 C): NMS %+.3f,  DFL %+.3f"
              % (out[(2, "nms")], out[(2, "dfl")]))

    vals = list(out.values())
    print()
    if len(vals) < 2:
        print("  얻어진 칸이 %d 개뿐이라 [2] 를 판정하지 않는다." % len(vals))
        return 0
    rng = max(vals) - min(vals)
    same = all(v < 0 for v in vals) or all(v > 0 for v in vals)
    print("  얻어진 칸 %d 개.  값 범위 %.3f,  부호 %s"
          % (len(vals), rng, "같음" if same else "**갈림**"))
    if not same:
        print("  => **88% 를 철회한다.** 귀속 자체가 기준의 산물이다.")
    elif rng >= 1.5:
        print("  => 방향은 견고하나 크기가 기준 의존이다.")
        print("     **88% 라는 수치를 철회**하고 방향만 적는다.")
    else:
        print("  => 기준에 안 민감하다.")
        print("     **88% 를 유지하되 어느 기준의 값인지 밝힌다.**")
    print()
    print("  **빈 칸을 채운 것처럼 쓰지 않는다.** 위 표의 「불능」 은 측정 실패가")
    print("  아니라 **그 대조가 존재하지 않는다**는 뜻이다.")
    print("  **A 를 exp03 의 -3.81 과 비교하지 않는다** (보정 2 (5) -- 앵커가 반대다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
