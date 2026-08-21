# -*- coding: utf-8 -*-
"""**그림 `fig:grid`** -- 추정 방식 x 적용 경로 격자 (원고 표 `tab:grid`).

표로는 여섯 칸을 눈으로 빼야 "무엇이 무엇을 이겼나" 가 보인다. 그림으로 놓으면
세 가지가 한 번에 읽힌다:

  * **해상도 띠**(+-3.31, 최소 검출 가능 효과)를 깔면 **어느 칸이 0 과
    구별되는지**가 즉시 보인다 -- 박스 크기 x 게이팅 `+0.074` 는 띠 안이다
  * **개입 불성립 칸**(NMS x 게이팅)은 막대를 비워 그린다. 값은 적되
    "넣어 봤더니 졌다" 와 **다른 종류의 사실**임을 형태로 구분한다
  * 비가중 값을 같은 줄에 눈금으로 얹는다 (규칙 5). **여섯 칸 전부 부호가 같다**

## 자료 출처

exp19 `run.py` 가 낸 track 출력을 평가한 값이다. **요약 JSON 이 없으므로**
값을 여기 적고, 아래 `verify()` 가 **원고에서 같은 숫자를 찾지 못하면 그리지
않는다.** 그림과 표가 갈리는 것을 막는 장치다 (규칙 3).

**철회판 값이 아니다.** 3 판은 확장을 검출과 트랙 양쪽에 주며 게이팅 열이
통째로 바뀌었다 (6.777/9.622/6.898 -> 8.470/1.960/+0.074).

사용법:
    python experiments/figures/fig_grid.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup, save, bare, C, WIDTH                  # noqa: E402

import matplotlib.pyplot as plt                                # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEX = Path("paper/report.tex")
MDE = 3.31          # exp07. 최소 검출 가능 효과
BASE = 61.002       # 기준선 HOTA

# (경로, 추정 방식, 가중 dHOTA, 비가중 dHOTA, 나빠진 시퀀스 수, 개입 성립?)
CELLS = [
    ("distance function", "NMS candidate var.",  -10.568, -13.949, 6, True),
    ("distance function", "DFL distribution var.", -8.573, -11.029, 6, True),
    ("distance function", "box size $\\sigma_C$",  -4.922,  -5.974, 6, True),
    ("gating",            "NMS candidate var.",   -8.470,  -7.658, 7, False),
    ("gating",            "DFL distribution var.", -1.960,  -0.853, 6, True),
    ("gating",            "box size $\\sigma_C$",  +0.074,  +0.636, 2, True),
]
FILL = {"NMS candidate var.": C["red"],
        "DFL distribution var.": C["red_pale"],
        "box size $\\sigma_C$": C["blue_pale"]}


def verify():
    """**그림에 찍는 값이 원고에 없으면 그리지 않는다.**

    exp19 는 요약 파일을 안 남긴다. 손으로 적은 값이 표와 갈리는 것이 이
    저장소의 반복된 실패이므로(letterbox, NMS in-place, 격자 확장 방향),
    원고를 정본으로 삼아 대조한다.
    """
    if not TEX.exists():
        print("원고를 찾을 수 없다: %s (저장소 뿌리에서 돌려라)" % TEX)
        return False
    t = TEX.read_text(encoding="utf-8")
    miss = []
    for _, _, w, u, _, _ in CELLS:
        for v in (w, u):
            if ("%.3f" % abs(v)) not in t:
                miss.append("%.3f" % abs(v))
    for v in ("%.2f" % MDE, "%.3f" % BASE):
        if v not in t:
            miss.append(v)
    if miss:
        print("!! 원고에 없는 값: %s" % ", ".join(miss))
        print("   tab:grid 를 고쳤다면 이 스크립트도 같이 고쳐라.")
        return False
    print("[검산] 격자 12값 + MDE + 기준선이 원고와 일치한다.")
    return True


def main():
    if not verify():
        return 1

    setup()
    # 값·시퀀스 수 열이 축 밖에 있어 tight bbox 가 지면 폭을 넘었다. 그만큼 줄인다
    fig, ax = plt.subplots(figsize=(WIDTH - 0.50, 2.30))

    # 경로마다 3줄. 위에서부터 거리 함수 -> 게이팅
    y, lab, seen = [], [], []
    pos = 0.0
    for ch, src, w, u, bad, ok in CELLS:
        if ch not in seen:
            if seen:
                pos -= 0.55          # 경로 사이를 벌린다
            seen.append(ch)
        y.append(pos)
        lab.append(src)
        pos -= 1.0
    y = np.array(y)

    # 값과 시퀀스 수는 **고정 열**에 적는다. 막대 끝에 붙이면 비가중 눈금과 겹친다
    COL_V, COL_N = 5.0, 6.7

    # 해상도 띠 -- 이 안은 0 과 구별되지 않는다. **양쪽 끝이 다 보여야** 띠로 읽힌다.
    # axvspan 은 y 를 축 좌표로 잡아 값 열까지 덮으므로 fill_betweenx 로 명시한다
    ylo, yhi = y[-1] - 0.55, y[0] + 0.55
    ax.fill_betweenx([ylo, yhi], -MDE, MDE, color=C["grid"], lw=0, zorder=0)
    ax.axvline(0.0, color=C["rule"], lw=0.6, zorder=2,
               ymin=0.03, ymax=0.90)

    for i, (ch, src, w, u, bad, ok) in enumerate(CELLS):
        if ok:
            ax.barh(y[i], w, height=0.62, color=FILL[src], edgecolor="none",
                    zorder=3)
        else:
            # **개입 불성립.** 속을 비우고 파선으로 -- 다른 종류의 사실이다
            ax.barh(y[i], w, height=0.62, facecolor="none",
                    edgecolor=FILL[src], lw=0.9, ls=(0, (2.2, 1.4)), zorder=3)
        ax.plot([u], [y[i]], marker="|", ms=6.5, mew=0.9, color=C["ink"],
                zorder=5)
        ax.text(COL_V, y[i], "%+.3f" % w, va="center", ha="right",
                fontsize=6.8, color=C["ink"], zorder=5)
        ax.text(COL_N, y[i], "%d/7" % bad, va="center", ha="center",
                fontsize=6.6, color=C["gray"], zorder=5)

    # 열 머리
    ax.text(COL_V, y[0] + 0.80, "$\\Delta$HOTA", va="center", ha="right",
            fontsize=6.6, color=C["gray"])
    ax.text(COL_N, y[0] + 0.80, "worse", va="center", ha="center",
            fontsize=6.6, color=C["gray"])
    ax.text(-7.6, y[0] + 0.80,
            "grey band: $\\pm3.31$, the minimum detectable effect",
            va="center", ha="center", fontsize=6.6, color=C["gray"])

    # **개입 불성립 칸은 범례가 아니라 그 자리에서 설명한다**
    ax.annotate("expansion does not open the gate:\nnot a measurement",
                xy=(-8.470, y[3] - 0.30), xytext=(-14.8, y[3] - 1.05),
                fontsize=6.4, color=C["red"], va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["red"],
                                shrinkA=0, shrinkB=2))

    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-15.5, 7.4)
    ax.set_xticks([-14, -12, -10, -8, -6, -4, -2, 0, 2])
    ax.set_xlabel("$\\Delta$HOTA vs baseline "
                  "(bar: detection-weighted, tick: unweighted)")
    ax.set_ylim(y[-1] - 0.75, y[0] + 1.15)

    # 경로 이름을 왼쪽 바깥에
    for ch, rows in (("distance\nfunction", y[:3]), ("gating", y[3:])):
        ax.text(-0.34, float(rows.mean()), ch,
                transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=7.6, color=C["ink"],
                linespacing=1.15)

    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)
    bare(ax, keep=("bottom",))
    # 눈금은 자료 구간에만. 값 열까지 축선이 뻗으면 표처럼 안 읽힌다
    ax.spines["bottom"].set_bounds(-15.0, 2.4)
    save(fig, "fig_grid")

    print()
    print("여섯 칸 중 다섯 유효, 그중 넷 음수. 양수 한 칸(+0.074)은 대조군이고")
    print("해상도 띠(+-3.31) 안이므로 이득이 아니라 '사실상 공짜' 다.")
    print("비가중 눈금은 여섯 칸 전부 가중과 부호가 같다 (규칙 5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
