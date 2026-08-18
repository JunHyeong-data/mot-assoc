# -*- coding: utf-8 -*-
"""**그림 1 — 소스 × 통로 지도.** 문헌이 무엇을 어디에 넣었고 무엇이 이겼나.

## 왜 이 그림인가

이 연구의 논지가 **소스 × 통로**다. 두 축을 한 장에 놓으면 문헌의 성패가
**소스를 따라 갈린다**는 것이 보인다 -- 같은 통로(칼만 R)에서도
**상자 크기를 쓴 쪽만 이긴다.**

## 자료 -- 전부 **원문을 직접 읽고** 채웠다 (`notes/primary_reading.md`)

  ByteTrack 논문   점수는 **문지기만**. `cost = 1 − IoU·s` 는 논문에 **없다**
  ByteTrack 구현   `fuse_score` 곱. **근거 없음**
  DeepSORT         검출별 스칼라 **없음**. 캐스케이드는 **트랙 나이**
  Bae (CVPR'14)    tracklet confidence = **트랙 이력** 양. **라우팅**
  LG-Track         `C_iou·d_l` (식 1). **4단계 캐스케이드**. 근거 없음
  UCMCTrack        `R = diag((σ_m·w)², (σ_m·h)²)` (식 2) + `ln|S|` (식 8)
  NSA Kalman       `R̃ = (1−c)·R` (StrongSORT 식 9)
  NWD              `Σ = diag(w²/4, h²/4)` (식 4)
  UncertaintyTrack 칼만 R **−0.1** / 상자확장 **+2.2** / 엔트로피 순서 **+0.2**

**우리 것은 회색으로, 문헌은 색으로 칠한다.** 우리가 채운 칸이 어디인지 보여야 한다.

사용법:
    python experiments/figures/fig_source_channel.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Rectangle                       # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("figures")

CHANNELS = ["Cost: additive", "Cost: multiplicative", "Kalman R",
            "Gating / box expansion", "Association ordering", "Matching threshold"]
SOURCES = ["Box size\n(geometry)", "Detection\nconfidence",
           "Localization\nquality", "Detector $\\sigma$\n(NMS / DFL)", "Track-side\nhistory"]

# (통로, 소스) -> (라벨, 결과, 우리것인가)
#   결과: "+" 이득 보고,  "-" 손해,  "0" 중립/기준선,  "?" 결과 없음
CELLS = {
    (0, 3): ("ours\n$-3.75$", "-", True),
    (1, 1): ("ByteTrack impl\n(no rationale)", "0", False),
    (1, 2): ("LG-Track\neq.1, 3", "+", False),
    (1, 3): ("ours\n$-1.01$", "-", True),
    (2, 0): ("UCMCTrack\n$R=(\\sigma_m w,\\sigma_m h)$", "+", False),
    (2, 1): ("NSA Kalman\n$\\tilde R=(1-c)R$", "+", False),
    (2, 3): ("UTrack $-0.62$\nUncTrack $-0.1$", "-", False),
    (3, 0): ("ours (control)\nbeats $\\sigma$ by 3.81", "0", True),
    (3, 3): ("UncTrack $+2.2$\nours $-4.33$", "+", False),
    (4, 2): ("LG-Track\n4-level cascade", "+", False),
    (4, 3): ("UncTrack\nentropy $+0.2$", "+", False),
    (4, 4): ("DeepSORT (age)\nBae (routing)", "+", False),
    (5, 3): ("ours\n$-0.21$ (oracle $+0.89$)", "-", True),
}
FILL = {"+": "#d6ead6", "-": "#f6d9d6", "0": "#e8e8e8", "m": "#fdf0d5", "?": "#ffffff"}
EDGE = {"+": "#4a7a4a", "-": "#a04a44", "0": "#888888", "m": "#b8860b", "?": "#cccccc"}


def main():
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    nr, nc = len(CHANNELS), len(SOURCES)

    for i in range(nr):
        for j in range(nc):
            e = CELLS.get((i, j))
            face = FILL[e[1]] if e else "#fbfbfb"
            edge = EDGE[e[1]] if e else "#e4e4e4"
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, facecolor=face,
                                   edgecolor=edge, lw=1.4 if e else 0.8, zorder=2))
            if e:
                ax.text(j + 0.5, nr - 1 - i + 0.5, e[0], ha="center", va="center",
                        fontsize=7.6, zorder=3,
                        fontstyle="italic" if e[2] else "normal",
                        color="#333333" if e[2] else "#111111")

    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels(SOURCES, fontsize=9)
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)])
    ax.set_yticklabels(CHANNELS, fontsize=9)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_xlabel("source of the signal", fontsize=10, labelpad=10)
    ax.set_ylabel("channel into the association", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    hs = [Rectangle((0, 0), 1, 1, facecolor=FILL[k], edgecolor=EDGE[k])
          for k in ("+", "-", "0", "m")]
    ax.legend(hs, ["gain reported", "loss measured", "neutral / baseline",
                   "mixed (different quantities)"],
              loc="lower center", bbox_to_anchor=(0.5, -0.21), ncol=4, fontsize=8.5,
              frameon=False)
    ax.text(0.0, -0.135, "italic = this work", transform=ax.transAxes,
            fontsize=8, color="#555555")

    fig.suptitle("Where each method injects the signal — and what it bought",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / ("fig_source_channel.%s" % ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("저장: %s" % p)

    print()
    print("칸이 %d개 찼다 (%d x %d 중). **빈 칸이 정보다** --" % (len(CELLS), nr, nc))
    print("  '상자 크기 × 순서', '검출 신뢰도 × 게이팅' 같은 조합은 아무도 안 했다")
    print("  그리고 **가산 열은 우리 것 하나뿐이다** -- 아무도 덧셈으로 안 넣는다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
