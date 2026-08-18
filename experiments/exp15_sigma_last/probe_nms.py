# -*- coding: utf-8 -*-
"""**탐색 — 판정이 아니다.** NMS 소스에서 합친 값과 시퀀스별이 반대로 나왔다.

## 무엇이 생겼나

`run.py -nms`:

    합친 AUC          0.5355   (사전 선언 기준 <0.55 -> "정보 없음")
    시퀀스별 중앙값     0.6275   (사전 선언 기준 >=0.60 -> "짚는다")
    시퀀스별 0.5 초과   **4/7**  (MOT17-04 는 **0.897**)

**두 읽기가 반대다.** DFL 은 이런 일이 없었다 (합친 0.457, 7/7 이 0.5 미만).

## 왜 갈릴 수 있는가 — 이미 우리가 잰 것이 있다

`exp01` 이 **σ 눈금이 시퀀스마다 62~1840배 갈린다**고 했다. 시퀀스를 합쳐
하나의 순위로 세우면 **눈금 차이가 순위를 지배**해서 시퀀스 안의 판별력이
씻긴다. 표준적인 층화 문제다.

## 보는 것 — **전부 탐색. 사전 선언 밖이다**

    [a] 시퀀스 안에서 순위정규화한 뒤 합쳐 AUC   (눈금 차이 제거)
    [b] 시퀀스별 AUC 의 오류 수 가중 평균         (층화 추정)
    [c] MOT17-04 가 왜 0.897 인가                (오류가 141건뿐이다)
    [d] 같은 것을 DFL 로도                        (DFL 은 안 갈리는가)

> **사전 선언한 주 종말점은 [1] 합친 AUC 하나이고 그 값은 0.5355 다.**
> 아래 값으로 **판정을 바꾸지 않는다.** 원고에는 **갈렸다는 사실**을 적는다.

사용법:
    python experiments/exp15_sigma_last/probe_nms.py
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


def rankit(x):
    """0~1 로 순위정규화. 동점은 평균순위."""
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    r = np.empty(len(x), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r / max(len(x) - 1, 1)


def one(tag):
    f = SRC / ("pairs-%s.npz" % tag)
    if not f.exists():
        print("없다: %s  -- run.py 를 먼저 돌려라" % f)
        return None
    d = np.load(f, allow_pickle=True)
    lab, sig, seq = d["lab"], d["sig"].astype(float), d["seq"]
    ok, bad = lab == "옳음", lab == "틀림_고칠수있음"

    print("=" * 92)
    print("[%s] 쌍 %d건  (옳음 %d / 틀림_고칠수있음 %d)"
          % (tag.upper(), len(lab), ok.sum(), bad.sum()))
    print("=" * 92)

    pooled = auc(sig[bad], sig[ok])
    print("  사전 선언 [1] 합친 AUC              %.4f" % pooled)

    # [a] 시퀀스 안 순위정규화
    z = np.empty(len(sig), float)
    for s in np.unique(seq):
        m = seq == s
        z[m] = rankit(sig[m])
    a_z = auc(z[bad], z[ok])
    print("  [a] 시퀀스 안 순위정규화 후 합침      %.4f   (탐색)" % a_z)

    # [b] 시퀀스별 + 층화
    rows = []
    for s in sorted(np.unique(seq)):
        m = seq == s
        o, b = ok & m, bad & m
        if o.sum() and b.sum():
            rows.append((str(s).replace("-FRCNN", ""), auc(sig[b], sig[o]),
                         int(o.sum()), int(b.sum())))
    av = np.array([r[1] for r in rows])
    nb = np.array([r[3] for r in rows], float)
    print("  [b] 시퀀스별 중앙 %.4f,  오류수 가중 %.4f,  0.5 초과 %d/%d"
          % (np.median(av), float((av * nb / nb.sum()).sum()),
             int((av > 0.5).sum()), len(av)))
    print()
    print("      %-10s %8s %8s %8s   %s" % ("시퀀스", "AUC", "옳음", "틀림", "sigma 중앙"))
    for nm, v, no, nbb in rows:
        m = seq == (nm + "-FRCNN")
        print("      %-10s %8.4f %8d %8d   %10.4g" % (nm, v, no, nbb,
                                                      np.median(sig[m])))

    print()
    print("      sigma 중앙값이 시퀀스마다 %.0f배 갈린다 -- 합치면 이게 순위를 지배한다"
          % (max(np.median(sig[seq == (r[0] + "-FRCNN")]) for r in rows)
             / max(min(np.median(sig[seq == (r[0] + "-FRCNN")]) for r in rows), 1e-12)))
    return pooled, a_z, np.median(av), int((av > 0.5).sum())


def main():
    print("탐색 -- 사전 선언 밖이다. **판정을 바꾸지 않는다**")
    print()
    res = {}
    for tag in ("nms", "dfl"):
        r = one(tag)
        if r:
            res[tag] = r
        print()

    if len(res) == 2:
        print("=" * 92)
        print("두 소스 나란히")
        print("=" * 92)
        print("  %-6s %10s %12s %12s %10s" % ("소스", "합침", "순위정규화", "시퀀스중앙", ">0.5"))
        for tag in ("nms", "dfl"):
            p, z, m, n = res[tag]
            print("  %-6s %10.4f %12.4f %12.4f %8d/7" % (tag.upper(), p, z, m, n))
        print()
        print("  **DFL 은 어느 자로 봐도 0.5 아래다. NMS 는 자에 따라 갈린다.**")
        print("  원고의 '연관 오류에 정보가 없다' 는 **DFL 에 대해서만** 깨끗하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
