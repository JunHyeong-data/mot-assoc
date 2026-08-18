# -*- coding: utf-8 -*-
"""**논문의 그림.** 신호는 있는데 연관으로 안 옮겨간다.

## 무엇을 보이는가

원고의 한 문장을 한 장에 넣는다:

> **σ 의 순위 정보는 검출에 대한 것이지 연관에 대한 것이 아니다.**

  왼쪽  **위치 오차**   -- 시퀀스별 편상관. 0 위에 있다 (신호가 있다)
  오른쪽 **연관 오류**   -- 시퀀스별 AUC.   0.5 아래에 있다 (정보가 없다)

**같은 σ, 같은 시퀀스, 다른 종말점.** 두 판이 나란히 있어야 뜻이 산다.

## 자료 출처 (전부 재현 가능)

  왼쪽  `data/exp15/loc_pcorr.json`  <- exp01 정의(`aggregate.py:26`)로 재계산.
        중앙값이 기록(DFL +0.151, NMS +0.322)과 정확히 일치하는 것을 확인했다
  오른쪽 `data/exp15/perseq-*.json`  <- `exp15_sigma_last/run.py` 가 저장

라벨은 영어로 쓴다 -- 논문 그림이고, Windows 에 한글 글꼴이 없을 수 있다.

사용법:
    python experiments/figures/fig_signal_transfer.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data/exp15")
OUT = Path("figures")
SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]
SHORT = [s.replace("MOT17-", "") for s in SEQS]
CLR = {"nms": "#1f4e79", "dfl": "#c0504d"}
NAME = {"nms": "NMS candidate spread", "dfl": "DFL distribution variance"}


def main():
    loc_f = DATA / "loc_pcorr.json"
    if not loc_f.exists():
        print("먼저 편상관을 계산해라 (fig 주석 참고). 없음: %s" % loc_f)
        return 1
    loc = json.loads(loc_f.read_text())

    assoc = {}
    for src in ("nms", "dfl"):
        f = DATA / ("perseq-%s.json" % src)
        if f.exists():
            assoc[src] = json.loads(f.read_text())
    if not assoc:
        print("먼저 실험 15 를 돌려라: python experiments/exp15_sigma_last/run.py")
        return 1

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(SEQS))

    # ---- 왼쪽: 위치 오차 (신호가 있다) ----
    a = ax[0]
    a.axhline(0.0, color="0.35", lw=1.2, ls="--", zorder=1)
    for src in ("nms", "dfl"):
        if src not in loc:
            continue
        y = [loc[src][s] for s in SEQS]
        a.plot(x, y, "o-", color=CLR[src], ms=6, lw=1.6,
               label="%s  (median %+.3f)" % (NAME[src], np.median(y)), zorder=3)
    a.set_title("(a) Localization error\n$\\sigma$ ranks which box is wrong", fontsize=11)
    a.set_ylabel("partial Spearman $\\rho$  (height controlled)")
    a.set_ylim(-0.13, 0.50)
    a.text(0.99, 0.13, "no signal", transform=a.transAxes, fontsize=8,
           color="0.35", ha="right")

    # ---- 오른쪽: 연관 오류 (정보가 없다) ----
    b = ax[1]
    b.axhline(0.5, color="0.35", lw=1.2, ls="--", zorder=1)
    for src in ("nms", "dfl"):
        if src not in assoc:
            continue
        y = [assoc[src]["per"][s + "-FRCNN"]["auc"] for s in SEQS]
        b.plot(x, y, "o-", color=CLR[src], ms=6, lw=1.6,
               label="%s  (overall %.3f)" % (NAME[src], assoc[src]["overall"]), zorder=3)
    b.set_title("(b) Association error\n$\\sigma$ does not rank which match is wrong",
                fontsize=11)
    b.set_ylabel("AUC ( $\\sigma$ $\\rightarrow$ fixable association error )")
    b.set_ylim(0.25, 0.72)
    b.text(0.99, 0.56, "no information", transform=b.transAxes, fontsize=8,
           color="0.35", ha="right")

    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(SHORT)
        a_.set_xlabel("MOT17 sequence (val_half)")
        a_.grid(axis="y", color="0.9", zorder=0)
        a_.legend(fontsize=8, loc="lower left", framealpha=0.95,
                  handlelength=1.6, borderpad=0.4)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    fig.suptitle("Detection uncertainty carries a localization signal that does not "
                 "reach association", fontsize=12, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / ("fig_signal_transfer.%s" % ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("저장: %s" % p)

    print()
    print("왼쪽 (위치 오차, 0 위에 있어야 신호):")
    for src in loc:
        y = [loc[src][s] for s in SEQS]
        print("  %-28s 중앙 %+.3f   0 초과 %d/7" % (NAME[src], np.median(y), sum(v > 0 for v in y)))
    print("오른쪽 (연관 오류, 0.5 미만이면 정보 없음):")
    for src in assoc:
        y = [assoc[src]["per"][s + "-FRCNN"]["auc"] for s in SEQS]
        print("  %-28s 전체 %.3f   0.5 미만 %d/7" % (NAME[src], assoc[src]["overall"],
                                                   sum(v < 0.5 for v in y)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
