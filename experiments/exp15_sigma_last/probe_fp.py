# -*- coding: utf-8 -*-
"""**심사 지적 #7 — 오탐 쌍을 빼고 잰 것이 구멍이다.**

## 무엇이 지적됐나

원고 5.6 의 AUC 는 채택 쌍 47,194건 중 **29,571건(62.7%)** 으로만 계산했다.
`미상(오탐)` 15,671건(33.2%)을 *"판별 불가"* 로 뺐기 때문이다.

그런데 5.5 는 여지 +3.122 중 **+0.770(24.7%)이 오탐 행이 빠져서 오른 몫**이라고
적었다. 즉:

> **여지의 4분의 1이 오탐에서 오는데, σ 를 오탐에 대해서는 시험하지 않았다.**

그리고 오탐 표시야말로 검출 불확실성의 **가장 전형적인 용도**다.
"여지가 있다(5.5) -> σ 로는 못 간다(5.6)" 의 연결에 구멍이 있다.

## 무엇을 재는가

`data/exp15/pairs-{nms,dfl}.npz` 에 라벨이 이미 있다. **재생이 필요 없다.**

    AUC(σ -> 미상 대 옳음)      σ 가 오탐 매칭을 짚는가
    AUC(σ_C -> 미상 대 옳음)    상자 크기로도 되는가 (기준 비교)
    높이 통제 편상관             크기 효과를 뺀 뒤에도 남는가

## 읽는 법 — 자료를 보기 전에 정한다

    |AUC-0.5| >= 0.10   σ 가 오탐을 짚는다. **5.6 의 결론에 단서를 달아야 한다**
    0.05 ~ 0.10         약하게 짚는다. 원고에 적되 결론은 유지
    < 0.05              못 짚는다. 5.6 의 결론이 오탐 population 으로 확장된다

**방향이 아니라 크기로 읽는다** — 심사 지적 #1 이 옳다. AUC 0.46 과 0.54 는
같은 양의 정보이고 부호는 규칙의 방향만 정한다. 5.6 의 사전 등록이 방향성
기준(<0.55)이었던 것과 별개로, **여기서는 |AUC-0.5| 로 정한다.**

사용법:
    python experiments/exp15_sigma_last/probe_fp.py
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("data/exp15")


def auc(pos, neg):
    """Mann-Whitney AUC. run.py 와 같은 정의 (동점은 절반)."""
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.empty(len(a), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def pcorr(x, y, z):
    """z 를 통제한 x,y 의 스피어만 편상관. exp01 aggregate.py 와 같은 정의."""
    def rk(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r
    rx, ry, rz = rk(x), rk(y), rk(z)
    for a in (rx, ry, rz):
        a -= a.mean()
    bx = (rx @ rz) / (rz @ rz)
    by = (ry @ rz) / (rz @ rz)
    ex, ey = rx - bx * rz, ry - by * rz
    return float((ex @ ey) / np.sqrt((ex @ ex) * (ey @ ey)))


def one(tag):
    f = SRC / ("pairs-%s.npz" % tag)
    if not f.exists():
        print("없다: %s -- run.py 를 먼저 돌려라" % f)
        return None
    d = np.load(f, allow_pickle=True)
    lab, sig, hgt = d["lab"], d["sig"].astype(float), d["hgt"].astype(float)
    ok = lab == "옳음"
    fp = lab == "미상"
    fix = lab == "틀림_고칠수있음"
    sig_c = hgt / 2.0                       # 5.6 절과 같은 상자 크기 모형

    print("=" * 92)
    print("[%s]  옳음 %d / 미상(오탐) %d / 틀림(고칠수있음) %d"
          % (tag.upper(), ok.sum(), fp.sum(), fix.sum()))
    print("=" * 92)

    a_fp = auc(sig[fp], sig[ok])
    a_fp_c = auc(sig_c[fp], sig_c[ok])
    a_fix = auc(sig[fix], sig[ok])

    m = ok | fp
    p_fp = pcorr(sig[m], fp[m].astype(float), hgt[m])

    print("  %-38s %8s %10s" % ("무엇", "AUC", "|AUC-0.5|"))
    print("  " + "-" * 58)
    print("  %-38s %8.4f %10.4f" % ("sigma -> 오탐 매칭", a_fp, abs(a_fp - 0.5)))
    print("  %-38s %8.4f %10.4f" % ("상자 크기 sigma_C -> 오탐 매칭",
                                    a_fp_c, abs(a_fp_c - 0.5)))
    print("  %-38s %8.4f %10.4f" % ("(참고) sigma -> 수정 가능한 오류",
                                    a_fix, abs(a_fix - 0.5)))
    print()
    print("  높이 통제 편상관(sigma, 오탐) = %+.4f" % p_fp)
    print()

    band = ("짚는다" if abs(a_fp - 0.5) >= 0.10 else
            "약하게 짚는다" if abs(a_fp - 0.5) >= 0.05 else "못 짚는다")
    print("  => |AUC-0.5| = %.4f  ->  **%s**" % (abs(a_fp - 0.5), band))
    return a_fp, a_fp_c, p_fp, band


def main():
    print("심사 지적 #7 -- 오탐 쌍(33.2%)을 빼고 잰 것이 구멍인가")
    print("여지의 24.7% 가 오탐에서 오는데 sigma 를 거기서 시험하지 않았다.")
    print()
    res = {}
    for tag in ("dfl", "nms"):
        r = one(tag)
        if r:
            res[tag] = r
        print()

    if len(res) == 2:
        print("=" * 92)
        print("두 소스 나란히")
        print("=" * 92)
        print("  %-6s %10s %12s %12s %s" % ("소스", "AUC(오탐)", "|AUC-0.5|",
                                            "편상관", "판정"))
        for tag in ("nms", "dfl"):
            a, ac, p, band = res[tag]
            print("  %-6s %10.4f %12.4f %+12.4f %s" % (tag.upper(), a,
                                                       abs(a - 0.5), p, band))
    return 0


if __name__ == "__main__":
    sys.exit(main())
