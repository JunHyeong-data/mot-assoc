# -*- coding: utf-8 -*-
"""실험 11 [진단] -- **[1] 이 왜 100% 가 아닌가.**

`run.py` 가 낸 [1] = 99.61% 는 사전 선언한 읽는 법으로는 "명제나 구현이 틀렸다"
이다. **판정하기 전에 원인을 가른다** (CLAUDE.md 규칙 2).

## 가릴 것

할당이 달라졌다고 해서 정리가 깨진 것은 아니다. **최적해가 여럿이면**
솔버가 섭동 뒤 다른 최적해를 고를 수 있고, 그때도 **원래 할당은 여전히 최적**이다.

    P0 = base 의 최적 할당,  P1 = add 의 최적 할당

  - `cost_add(P0) == cost_add(P1)`  ->  **동점.** P0 도 add 에서 최적이다.
    정리는 성립한다. 솔버가 동점 중 다른 것을 골랐을 뿐
  - `cost_add(P1) <  cost_add(P0)`  ->  **진짜 위반.** 명제가 틀렸다

같은 검사를 base 방향으로도 한다 (`cost_base(P0)` 대 `cost_base(P1)`).

**이 검사는 사전 선언한 종말점을 바꾸지 않는다.** [1] 은 이미 99.61% 로
기록됐고 그대로 둔다. 여기서 답하는 것은 **그 4/1000 이 무엇인가**이다.

사용법:
    python experiments/exp11_add_vs_mul/diagnose.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                # noqa: E402
from replay import WTracker, Det, load, SEQS, BASE             # noqa: E402
from run import unit, perturb                                  # noqa: E402

TOL = 1e-9
REC = []


def total(cost, pairs):
    return float(sum(cost[i, j] for i, j in pairs))


class DiagTracker(WTracker):
    def get_dists(self, tracks, detections):
        base = super().get_dists(tracks, detections)
        if base.ndim != 2 or 0 in base.shape:
            return base
        M, N = base.shape
        if N > M:
            return base                       # 조건부 명제의 범위 밖
        u = unit(detections)
        add, _ = perturb(base, u)
        P0 = np.asarray(matching.linear_assignment(base, 1e9)[0]).reshape(-1, 2)
        P1 = np.asarray(matching.linear_assignment(add, 1e9)[0]).reshape(-1, 2)
        s0, s1 = frozenset(map(tuple, P0)), frozenset(map(tuple, P1))
        if s0 == s1:
            return base
        REC.append(dict(
            M=M, N=N,
            add_P0=total(add, s0), add_P1=total(add, s1),
            base_P0=total(base, s0), base_P1=total(base, s1),
            n0=len(s0), n1=len(s1)))
        return base


def main():
    print("=" * 92)
    print("실험 11 [진단] -- [1] 의 불일치 4/1000 은 동점인가 위반인가")
    print("=" * 92)
    print("**사전 선언한 종말점을 바꾸지 않는다.** [1] = 99.61%% 는 그대로 둔다.")
    print()

    for seq in SEQS:
        c = load(seq, "iou")
        tr = DiagTracker(SimpleNamespace(**BASE), "iou", 1.0, frame_rate=30)
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            tr.update(Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                          c["sxx"][m], c["syy"][m]))

    if not REC:
        print("  N<=M 에서 불일치가 하나도 없다")
        return 0

    print("  N<=M 인데 할당이 달라진 호출 = %d 건" % len(REC))
    print()
    print("  %-6s %-6s %14s %14s %12s" % ("M", "N", "cost_add(P0)", "cost_add(P1)", "차이"))
    print("  " + "-" * 60)
    tie = strict = 0
    for r in REC[:20]:
        d = r["add_P0"] - r["add_P1"]
        print("  %-6d %-6d %14.9f %14.9f %12.2e" % (r["M"], r["N"], r["add_P0"], r["add_P1"], d))
    for r in REC:
        d = r["add_P0"] - r["add_P1"]
        if d <= TOL:
            tie += 1                      # P0 도 최적 (동점 또는 수치오차 안)
        else:
            strict += 1                   # P1 이 진짜로 더 싸다 -> 위반
    if len(REC) > 20:
        print("  ... (%d 건 더)" % (len(REC) - 20))

    print()
    print("=" * 92)
    print("진단")
    print("=" * 92)
    print("  동점/수치오차 안 (P0 도 add 에서 최적) : %d / %d" % (tie, len(REC)))
    print("  **진짜 위반** (P1 이 엄밀히 더 싸다)   : %d / %d" % (strict, len(REC)))
    n0n1 = sum(1 for r in REC if r["n0"] != r["n1"])
    print("  배정된 쌍 수가 다른 호출               : %d / %d" % (n0n1, len(REC)))
    print()
    if strict == 0:
        print("  => **정리는 성립한다.** 불일치는 전부 동점이고, 솔버가 섭동 뒤")
        print("     동점 중 다른 최적해를 골랐을 뿐이다. 원래 할당도 여전히 최적이다.")
        print("     **원고 2절을 고칠 필요가 없다.** 다만 '할당이 비트 단위로 같다'")
        print("     가 아니라 **'최적성이 보존된다'** 로 정확히 써야 한다.")
    else:
        print("  => **진짜 위반이 %d 건 있다. 명제나 구현이 틀렸다.**" % strict)
        print("     원고 2절을 고쳐야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
