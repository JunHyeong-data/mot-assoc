# -*- coding: utf-8 -*-
"""**그림 1 — 산출 방식 × 주입 경로 지도.** 문헌이 무엇을 어디에 넣고 무엇을 얻었나.

## 왜 이 그림인가

이 연구의 논지가 **산출 방식 × 주입 경로**다. 두 축을 한 장에 놓으면 문헌의
성패가 **산출 방식을 따라 갈린다**는 것이 보인다 -- 같은 경로(칼만 R)에서도
**박스 크기를 쓴 쪽만 이긴다.** 열을 따라 읽으면 검출기 sigma 열이 거의 전부
손실이다.

## 자료 -- 전부 **원문을 직접 읽고** 채웠다 (`notes/primary_reading.md`)

  ByteTrack 논문   점수는 **문지기만**. `cost = 1 - IoU*s` 는 논문에 **없다**
  ByteTrack 구현   `fuse_score` 곱. **근거 없음**
  DeepSORT         검출별 스칼라 **없음**. 캐스케이드는 **트랙 나이**
  Bae (CVPR'14)    tracklet confidence = **트랙 이력** 양. **라우팅**
  LG-Track         `C_iou*d_l` (식 1). **4단계 캐스케이드**. 근거 없음
  UCMCTrack        `R = diag((s_m*w)^2, (s_m*h)^2)` (식 2) + `ln|S|` (식 8)
  NSA Kalman       `R~ = (1-c)*R` (StrongSORT 식 9)
  NWD              `Sigma = diag(w^2/4, h^2/4)` (식 4). **검출용이다**
  UncertaintyTrack 칼만 R **-0.1** / 상자확장 **+2.2** / 엔트로피 순서 **+0.2**

**우리 것은 기울임체다.** 어느 칸을 우리가 채웠는지 보여야 한다.

**거리 함수 행은 2026-08-18 에 넣었다** -- 원고 2.3 이 주입 경로 넷 중 하나로
세는데 지도에 그 행이 없었다.

사용법:
    python experiments/figures/fig_source_channel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup, save, C, WIDTH                     # noqa: E402

import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.patches import Rectangle                    # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHANNELS = ["cost: additive", "cost: multiplicative", "distance function",
            "Kalman $R$", "gating / box expansion", "association ordering",
            "matching threshold"]
SOURCES = ["box size\n(geometry)", "detection\nconfidence", "localization\nquality",
           "detector $\\sigma$\n(NMS / DFL)", "track-side\nhistory"]

# (경로, 산출방식) -> (라벨, 결과, 우리것인가)
#   결과: "+" 이득 보고,  "-" 손해 측정,  "0" 중립/기준선,  "m" 섞임
CELLS = {
    (0, 3): ("ours $-3.75$", "-", True),
    (1, 1): ("ByteTrack impl\n(no rationale)", "0", False),
    (1, 2): ("LG-Track eq. 1, 3", "+", False),
    (1, 3): ("ours $-1.01$", "-", True),
    # NWD 는 **검출용**이라 MOT 연관 결과가 아니다 -- 중립으로 칠한다.
    (2, 0): ("NWD (detection)\nours $-4.98$", "0", False),
    (2, 3): ("ours $-8.90$\nloses to size by 3.92", "-", True),
    (3, 0): ("UCMCTrack\n$R=(\\sigma_m w,\\sigma_m h)$", "+", False),
    (3, 1): ("NSA Kalman\n$\\tilde R=(1-c)R$", "+", False),
    (3, 3): ("UTrack $-0.62$\nUncTrack $-0.1$", "-", False),
    (4, 0): ("ours (control)\nbeats $\\sigma$ by 3.81", "0", True),
    # **한 칸에 두 이야기가 있다.** UncTrack 의 +2.2 는 상자확장 전체의 이득이고,
    # 우리 -4.33 은 확장량을 맞췄을 때 sigma 로 정한 쪽이 크기로 정한 쪽보다
    # 얼마나 나쁜가다. **다른 양이므로 같은 색으로 칠하면 오해를 부른다.**
    (4, 3): ("UncTrack $+2.2$\nours $-4.33$", "m", False),
    (5, 2): ("LG-Track\n4-level cascade", "+", False),
    (5, 3): ("UncTrack\nentropy $+0.2$", "+", False),
    (5, 4): ("DeepSORT (age)\nBae (routing)", "+", False),
    (6, 3): ("ours $-0.21$\n(oracle $+0.89$)", "-", True),
}
FILL = {"+": "#dfe8dc", "-": "#eedcd9", "0": "#e9e9e9", "m": "#f2e8d4"}
EDGE = {"+": C["green"], "-": C["red"], "0": C["gray"], "m": C["amber"]}
NAME = {"+": "gain reported", "-": "loss measured", "0": "neutral / baseline",
        "m": "mixed (different quantities)"}


def main():
    setup()
    nr, nc = len(CHANNELS), len(SOURCES)
    fig, ax = plt.subplots(figsize=(WIDTH * 1.05, 3.9 * 1.05))

    for i in range(nr):
        for j in range(nc):
            e = CELLS.get((i, j))
            ax.add_patch(Rectangle(
                (j, nr - 1 - i), 1, 1,
                facecolor=FILL[e[1]] if e else "#fcfcfc",
                edgecolor=EDGE[e[1]] if e else "#ececec",
                lw=0.7 if e else 0.5, zorder=2))
            if e:
                ax.text(j + 0.5, nr - 1 - i + 0.5, e[0], ha="center", va="center",
                        fontsize=5.9, zorder=3, linespacing=1.35,
                        fontstyle="italic" if e[2] else "normal",
                        color=C["ink"])

    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels(SOURCES, fontsize=7.0, linespacing=1.3)
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)])
    ax.set_yticklabels(CHANNELS, fontsize=7.0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("source of the signal", fontsize=7.8, labelpad=8)
    ax.set_ylabel("channel into the association", fontsize=7.8, labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    hs = [Rectangle((0, 0), 1, 1, facecolor=FILL[k], edgecolor=EDGE[k], lw=0.7)
          for k in ("+", "-", "0", "m")]
    ax.legend(hs, [NAME[k] for k in ("+", "-", "0", "m")],
              loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=4,
              fontsize=6.6, handlelength=1.2, columnspacing=1.3,
              handletextpad=0.5)
    ax.text(0.0, -0.028, "italic = this work", transform=ax.transAxes,
            fontsize=6.3, color=C["gray"], va="top")

    save(fig, "fig_source_channel")

    print()
    print("칸이 %d개 찼다 (%d x %d 중). **빈 칸이 정보다** --" % (len(CELLS), nr, nc))
    print("  '박스 크기 x 연관 순서', '검출 신뢰도 x 게이팅' 은 아무도 안 했다")
    print("  **가산 행은 우리 것 하나뿐이다** -- 아무도 덧셈으로 안 넣는다")
    ours = sum(1 for v in CELLS.values() if v[2])
    loss = sum(1 for (i, j), v in CELLS.items() if j == 3 and v[1] == "-")
    print("  우리가 채운 칸 %d개.  검출기 sigma 열에서 손실로 측정된 칸 %d개" % (ours, loss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
