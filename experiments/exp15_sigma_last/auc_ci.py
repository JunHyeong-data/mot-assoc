# -*- coding: utf-8 -*-
"""**주 평가지표 AUC 의 신뢰구간.** 그림·표 검토가 지적한 것이다.

## 왜

원고의 판정이 AUC 하나에 걸려 있다 -- 사전 등록 기준은 `< 0.55` 이고 NMS 가
`0.5355` 다. **간발의 차이인데 원고에도 그림에도 구간이 없다.**

이 저장소는 추적 수준에 대해서는 해상도 규율을 엄격히 적용하면서
(6.5 절: "여섯 개 판정이 모두 자기 해상도를 넘지 못한다") **검출 수준의 핵심
수치에는 같은 잣대를 안 댔다.** 그건 일관성이 없다.

## 어떻게

DeLong 의 구조적 성분(placement value)으로 AUC 의 분산을 해석적으로 낸다.
부트스트랩보다 빠르고 이 표본 크기에서는 사실상 같다.

    V10[i] = P(X_neg < x_pos[i])   (동점 0.5)
    V01[j] = P(x_neg[j] < X_pos)
    Var = Var(V10)/n_pos + Var(V01)/n_neg

**규칙 3**: 씨앗 고정 부트스트랩 400회로 교차 확인한다. 둘이 어긋나면 멈춘다.

사용법:
    python experiments/exp15_sigma_last/auc_ci.py
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("data/exp15")
THRESH = 0.55                    # 사전 등록한 판정선


def auc_delong(pos, neg):
    """AUC 와 DeLong 표준오차. 동점은 0.5 로 센다 (run.py 와 같은 규약)."""
    pos = np.sort(np.asarray(pos, float))
    neg = np.sort(np.asarray(neg, float))
    n1, n0 = len(pos), len(neg)
    # V10[i] = (neg 중 pos[i] 보다 작은 수 + 동점/2) / n0
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    v10 = (lo + (hi - lo) / 2.0) / n0
    lo2 = np.searchsorted(pos, neg, side="left")
    hi2 = np.searchsorted(pos, neg, side="right")
    v01 = ((n1 - hi2) + (hi2 - lo2) / 2.0) / n1
    a = float(v10.mean())
    var = v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0
    return a, float(np.sqrt(var))


def auc_boot(pos, neg, B=400, seed=20260819):
    """규칙 3 -- 같은 양을 다른 경로로. 씨앗 고정."""
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    out = np.empty(B)
    for b in range(B):
        out[b] = auc_delong(rng.choice(p, len(p), True),
                            rng.choice(n, len(n), True))[0]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    print("=" * 92)
    print("주 평가지표 AUC 의 신뢰구간 -- 판정이 값 하나에 걸려 있다")
    print("=" * 92)
    print("사전 등록 판정선 %.2f. DeLong 해석적 SE 와 부트스트랩(400, 씨앗 고정)을"
          % THRESH)
    print("**둘 다** 내서 어긋나면 멈춘다 (규칙 3).")
    print()
    print("  %-16s %8s %8s %20s %s" % ("신호", "AUC", "SE", "95% CI", "판정선"))
    print("  " + "-" * 74)

    rows = []
    for tag, name in (("nms", "NMS 후보 분산"), ("dfl", "DFL 분포 분산")):
        f = SRC / ("pairs-%s.npz" % tag)
        if not f.exists():
            print("없다: %s" % f)
            return 1
        d = np.load(f, allow_pickle=True)
        lab, sig, hgt = d["lab"], d["sig"].astype(float), d["hgt"].astype(float)
        ok, bad = lab == "옳음", lab == "틀림_고칠수있음"
        rows.append((name, sig[bad], sig[ok]))
        if tag == "dfl":
            rows.append(("박스 크기 sigma_C", (hgt / 2.0)[bad], (hgt / 2.0)[ok]))

    bad_flag = False
    for name, p, n in rows:
        a, se = auc_delong(p, n)
        lo, hi = a - 1.96 * se, a + 1.96 * se
        blo, bhi = auc_boot(p, n)
        # 규칙 3: 두 경로가 어긋나면 멈춘다
        if abs(blo - lo) > 0.01 or abs(bhi - hi) > 0.01:
            print("  %-16s !! DeLong [%.4f,%.4f] 대 부트스트랩 [%.4f,%.4f] 어긋남"
                  % (name, lo, hi, blo, bhi))
            bad_flag = True
            continue
        verdict = ("포함" if lo <= THRESH <= hi else
                   "아래" if hi < THRESH else "위")
        print("  %-16s %8.4f %8.4f   [%.4f, %.4f]   %s"
              % (name, a, se, lo, hi, verdict))

    if bad_flag:
        print()
        print("  **두 경로가 어긋났다. 판정하지 말 것.**")
        return 1

    print()
    print("=" * 92)
    print("읽는 법")
    print("=" * 92)
    print("  구간이 판정선을 **포함**하면 그 판정은 자료가 지지하는 것보다 세다.")
    print("  구간이 판정선 **아래**에 온전히 들어가면 판정이 선다.")
    print("  어느 쪽이든 **원고에 구간을 함께 적어야 한다** -- 지금은 점추정만 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
