# -*- coding: utf-8 -*-
"""실험 20 — **게이팅 대조군은 무엇을 맞춰야 하는가.**

사전 등록은 `PREREG.md` (자료보다 먼저 커밋, 읽는 법 포함).

원고 5.3 의 88% 주장이 어느 기준에 걸려 있는지 잰다. 세 기준 각각으로
`검출기 sigma - 박스 크기` 를 내고, **주 판정은 채택률 기준(C)** 으로 한다 --
거리 함수 경로가 이미 쓰는 기준이므로 게이팅만 다르게 할 근거가 없다.

    A  평균 선형 확장량 (px)   exp03 이 맞춘 것
    B  총 확장 면적            exp19 가 맞춘 것
    C  1단계 채택률            4.2 절이 거리 함수에 쓰는 것

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

import importlib.util as _iu                                    # noqa: E402
_spec = _iu.spec_from_file_location("g19", HERE.parents[0] / "exp19_grid" / "run.py")
G = _iu.module_from_spec(_spec)
_spec.loader.exec_module(G)                                     # noqa: E402

from ultralytics.trackers.utils import matching                 # noqa: E402
from replay import Det, SEQS, BASE                              # noqa: E402

G.OUT = Path("data/exp20/tracks")
SOURCES = ["nms", "dfl", "size"]
TOL = 0.05                       # 사전 등록: 목표 대비 5% 벗어나면 멈춘다


def measure(source, alpha):
    """한 조건에서 세 양을 한 번에 잰다: (평균 선형 확장량, 총 면적, 채택률)."""
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
    return (pad_sum / max(pad_n, 1), area, num / max(den, 1))


def solve(source, which, target, hi=4000.0, iters=18):
    """이분 탐색. which 는 0=선형, 1=면적, 2=채택률.

    **방향을 자동으로 잡는다.** 예전 판은 "v 가 alpha 에 증가한다" 를 가정했는데
    **채택률은 감소한다** (상자를 키우면 트랙 예측 상자와 크기가 안 맞아 IoU 가
    나빠진다). 그래서 첫 판이 상한으로 달아나 포화했다 -- 사전 등록 가드가 잡았다.
    """
    lo = hi * 1e-5
    inc = measure(source, hi)[which] > measure(source, lo)[which]
    best = (None, None)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = measure(source, mid)[which]
        if best[0] is None or abs(v - target) < abs(best[1] - target):
            best = (mid, v)
        if (v > target) == inc:
            hi = mid
        else:
            lo = mid
    return best


def hota(source, alpha, tag):
    G.run("gate", source, alpha, tag)
    return G.evaluate([tag])[tag][0]


def main():
    print("=" * 92)
    print("실험 20 -- 게이팅 대조군은 무엇을 맞춰야 하는가")
    print("=" * 92)
    print("주 판정은 **채택률 기준(C)** -- 거리 함수 경로가 이미 쓰는 기준이다.")
    print()

    G.OUT.mkdir(parents=True, exist_ok=True)
    base_h = hota("dfl", 0.0, "base")
    base = measure("dfl", 0.0)
    print("[사전 점검] 기준선 HOTA %.3f (기록 61.002), 채택률 %.4f"
          % (base_h, base[2]))
    if abs(base_h - 61.002) > 0.01:
        print("  **멈춘다.**")
        return 1

    # 박스 크기 ALPHA=0.5 를 기준 조건으로 삼고 그 세 양을 목표로 쓴다
    ref = measure("size", 0.5)
    print("  기준 조건(박스 크기, ALPHA=0.5): 선형 %.4g,  면적 %.4g,  채택률 %.4f"
          % ref)
    h_size = hota("size", 0.5, "size")
    print("  기준 조건 HOTA %.3f  (dHOTA %+.3f)" % (h_size, h_size - base_h))
    print()

    names = ["A 평균 선형 확장량", "B 총 확장 면적", "C 1단계 채택률"]
    out = {}
    for which in (0, 1, 2):
        print("-" * 92)
        print("기준 %s   목표 %.6g" % (names[which], ref[which]))
        for src in ("nms", "dfl"):
            a, v = solve(src, which, ref[which])
            off = abs(v - ref[which]) / max(abs(ref[which]), 1e-12)
            ok = off <= TOL
            print("  %-5s ALPHA %10.4g -> %.6g  (목표의 %.1f%%)  %s"
                  % (src, a, v, 100 * v / ref[which], "OK" if ok else "!! 포화"))
            if not ok:
                print("     **사전 등록대로 멈춘다.** 통제가 안 걸린 값으로 판정 안 한다.")
                return 1
            h = hota(src, a, "%s_%d" % (src, which))
            out[(which, src)] = h - h_size
            print("     HOTA %.3f  =>  **sigma - 크기 = %+.3f**" % (h, h - h_size))

    print()
    print("=" * 92)
    print("[2] 기준별 `검출기 sigma - 박스 크기`")
    print("=" * 92)
    print("  %-22s %12s %12s" % ("기준", "NMS", "DFL"))
    print("  " + "-" * 48)
    for which in (0, 1, 2):
        print("  %-22s %+12.3f %+12.3f"
              % (names[which], out[(which, "nms")], out[(which, "dfl")]))

    vals = list(out.values())
    rng = max(vals) - min(vals)
    same = all(v < 0 for v in vals) or all(v > 0 for v in vals)
    print()
    print("  값 범위 %.3f,  부호 %s" % (rng, "같음" if same else "**갈림**"))

    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md)")
    print("=" * 92)
    if not same:
        print("  **부호가 갈린다 => 88%% 를 철회한다.**")
        print("  '확장량을 무엇으로 정했는가' 라는 귀속 자체가 기준의 산물이다.")
    elif rng >= 1.5:
        print("  부호는 같은데 범위 %.3f >= 1.5 => **88%% 라는 수치를 철회**하고" % rng)
        print("  방향만 적는다. 크기는 기준 의존이다.")
    else:
        print("  부호 같고 범위 %.3f < 1.5 => 기준에 안 민감하다." % rng)
        print("  **88%% 를 유지하되 기준 C 값으로 다시 적는다.**")
    print()
    print("  주 판정(기준 C, 채택률): NMS %+.3f,  DFL %+.3f"
          % (out[(2, "nms")], out[(2, "dfl")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
