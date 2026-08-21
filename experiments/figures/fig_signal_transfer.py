# -*- coding: utf-8 -*-
"""**그림 `fig:transfer`** -- 위치 신호가 연관으로는 거의 안 옮겨간다.

  (a) **위치 오차**  두 소스 다 0 위. 신호가 있다 (NMS +0.322 > DFL +0.151)
  (b) **연관 오류**  DFL 은 7/7 이 0.5 아래, **NMS 는 3/7 뿐이다**

**같은 σ, 같은 시퀀스, 다른 평가 대상.** 두 판이 나란히 있어야 뜻이 산다.
그리고 **오른쪽 판이 소스에 따라 갈린다는 것 자체가 결과다.**

예전 판은 DFL 만 있어서 오른쪽이 전부 0.5 아래였고 제목에
~~"σ does not rank which match is wrong"~~ 이라고 박아 놨었다. NMS 를 재니
**거짓이 됐다.** 그래서 **제목을 아예 뺐다** -- 할 말은 캡션이 한다.

## 자료 출처 (전부 재현 가능)

  (a) `data/exp15/loc_pcorr.json`  <- exp01 정의(`aggregate.py:26`)로 재계산.
      중앙값이 기록(DFL +0.151, NMS +0.322)과 정확히 일치하는 것을 확인했다
  (b) `data/exp15/perseq-*.json`   <- `exp15_sigma_last/run.py` 가 저장

사용법:
    python experiments/figures/fig_signal_transfer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup, save, panel, bare, C, WIDTH        # noqa: E402

import matplotlib.pyplot as plt                             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data/exp15")
SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]
SHORT = [s.replace("MOT17-", "") for s in SEQS]
SRC = [("nms", "NMS candidate spread", C["blue"], "o", "-"),
       ("dfl", "DFL distribution variance", C["red"], "s", "--")]


def main():
    loc_f = DATA / "loc_pcorr.json"
    if not loc_f.exists():
        print("먼저 편상관을 계산해라. 없음: %s" % loc_f)
        return 1
    loc = json.loads(loc_f.read_text())

    assoc = {}
    for tag, _, _, _, _ in SRC:
        f = DATA / ("perseq-%s.json" % tag)
        if f.exists():
            assoc[tag] = json.loads(f.read_text())
    if not assoc:
        print("먼저 실험 15 를 돌려라: python experiments/exp15_sigma_last/run.py")
        return 1

    setup()
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 1.19, 2.35 * 1.19))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(SEQS))

    # ---- (a) 위치 오차 ----
    a = ax[0]
    a.axhline(0.0, color=C["rule"], lw=0.6, ls=(0, (4, 3)), zorder=1)
    for tag, name, col, mk, ls in SRC:
        if tag not in loc:
            continue
        y = [loc[tag][s] for s in SEQS]
        a.plot(x, y, marker=mk, ls=ls, color=col, zorder=3,
               markerfacecolor="white", markeredgewidth=0.9,
               label="%s (%+.3f)" % (name, np.median(y)))
    a.set_ylabel("partial Spearman $\\rho$ | height")
    a.set_ylim(-0.10, 0.50)
    a.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    a.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.03))

    # ---- (b) 연관 오류 ----
    b = ax[1]
    b.axhline(0.5, color=C["rule"], lw=0.6, ls=(0, (4, 3)), zorder=1)
    # 0.5 파선 바로 위 **왼쪽**. 오른쪽 끝은 시퀀스 13 의 두 점과 겹친다
    b.text(0.012, 0.335, "chance", transform=b.transAxes, ha="left", va="bottom",
           fontsize=6.5, color=C["gray"])
    for tag, name, col, mk, ls in SRC:
        if tag not in assoc:
            continue
        y = [assoc[tag]["per"][s + "-FRCNN"]["auc"] for s in SEQS]
        b.plot(x, y, marker=mk, ls=ls, color=col, zorder=3,
               markerfacecolor="white", markeredgewidth=0.9,
               label="%s (%.3f)" % (name, assoc[tag]["overall"]))
    b.set_ylabel("AUC ( $\\sigma$ $\\rightarrow$ fixable error )")
    b.set_ylim(0.28, 0.95)
    b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    b.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03))

    for tag, a_ in (("(a) localization error", a), ("(b) association error", b)):
        a_.set_xticks(x)
        a_.set_xticklabels(SHORT)
        a_.set_xlabel("MOT17 sequence (val\\_half)".replace("\\_", "_"))
        a_.set_xlim(-0.4, 6.4)
        a_.grid(axis="y", zorder=0)
        a_.set_axisbelow(True)
        bare(a_)
        panel(a_, tag)

    save(fig, "fig_signal_transfer")

    print()
    print("(a) 위치 오차 -- 0 위에 있어야 신호")
    for tag, name, _, _, _ in SRC:
        if tag in loc:
            y = [loc[tag][s] for s in SEQS]
            print("    %-26s 중앙 %+.3f   0 초과 %d/7"
                  % (name, np.median(y), sum(v > 0 for v in y)))
    print("(b) 연관 오류 -- 0.5 미만이면 정보 없음")
    for tag, name, _, _, _ in SRC:
        if tag in assoc:
            y = [assoc[tag]["per"][s + "-FRCNN"]["auc"] for s in SEQS]
            print("    %-26s 전체 %.3f   0.5 미만 %d/7"
                  % (name, assoc[tag]["overall"], sum(v < 0.5 for v in y)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
