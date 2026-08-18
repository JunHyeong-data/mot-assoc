# -*- coding: utf-8 -*-
"""**논문의 그림.** 위치 신호가 연관으로는 거의 안 옮겨간다.

## 무엇을 보이는가 -- **NMS 를 재고 나서 바뀌었다 (2026-08-18)**

예전 판은 DFL 만 있었고 오른쪽이 전부 0.5 아래여서 이렇게 적었다:

> ~~*"σ 의 순위 정보는 검출에 대한 것이지 연관에 대한 것이 아니다."*~~

**NMS 소스를 재니 4/7 이 0.5 위다** (MOT17-04 는 0.897). 그림이 말하는
것이 소스에 따라 갈린다:

  왼쪽  **위치 오차**  두 소스 다 0 위. 신호가 있다 (NMS +0.322 > DFL +0.151)
  오른쪽 **연관 오류**  **DFL 은 7/7 이 0.5 아래, NMS 는 3/7 뿐이다**

**같은 σ, 같은 시퀀스, 다른 평가 대상.** 두 판이 나란히 있어야 뜻이 산다.
그리고 **오른쪽 판이 소스에 따라 갈린다는 것 자체가 결과다.**

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
    b.set_title("(b) Association error\nDFL: 0/7 above chance    NMS: 4/7",
                fontsize=11)
    b.set_ylabel("AUC ( $\\sigma$ $\\rightarrow$ fixable association error )")
    b.set_ylim(0.25, 0.95)
    # 0.5 파선 바로 위 **왼쪽**. 오른쪽은 시퀀스 13 의 두 점과 겹쳤다
    b.text(0.015, 0.375, "chance", transform=b.transAxes, fontsize=8,
           color="0.35", ha="left")

    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(SHORT)
        a_.set_xlabel("MOT17 sequence (val_half)")
        a_.grid(axis="y", color="0.9", zorder=0)
        a_.legend(fontsize=8, loc="lower left", framealpha=0.95,
                  handlelength=1.6, borderpad=0.4)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    fig.suptitle("The localization signal is clear in both sources; the association "
                 "signal is absent (DFL) or weak and uneven (NMS)",
                 fontsize=12, y=1.02)
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
