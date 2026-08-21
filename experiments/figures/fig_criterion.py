# -*- coding: utf-8 -*-
"""**그림 `fig:criterion`** -- 게이팅의 우열은 무엇을 맞추었는가에 달려 있다
(원고 표 `tab:criterion`).

이 그림이 있어야 하는 이유는 하나다. **표로 읽으면 숫자 다섯 개이고, 그림으로
보면 부호가 갈리는 것이 한눈에 보인다.** 원고는 이 자리에서 ``손실의 88%%가
확장량을 무엇으로 정했는가에서 온다'' 는 서술을 **철회했다.**

  * DFL 선이 **0 을 가로지른다** -- 같은 대조, 같은 자료, 맞춤 기준만 바꿨다
  * NMS 는 주 기준에서 **점이 아예 없다.** 측정 실패가 아니라 **그 대조군이
    존재하지 않는다** -- 채택률이 alpha=2 에서 0.9639 로 정점이라 목표 0.9692
    에 닿지 못한다
  * 값의 범위 8.735 는 해상도(3.31)의 **두 배가 넘는다**

## 자료 출처

exp20 `run.py`. **요약 JSON 이 없으므로** 값을 여기 적고 `verify()` 가 원고에서
같은 숫자를 찾지 못하면 그리지 않는다 (규칙 3).

사용법:
    python experiments/figures/fig_criterion.py
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
MDE = 3.31
PEAK, TARGET = 0.9639, 0.9692     # NMS 채택률 정점 / 박스 크기 조건의 채택률

# 맞춘 양 -> (NMS, DFL). None 은 **그 대조군이 존재하지 않는다**
ROWS = [
    ("mean linear expansion (px)", -7.630, -1.947),
    ("total expanded area",        -8.543, -2.034),
    ("stage-1 adoption rate\n(preregistered primary)", None, +0.191),
]
SPAN = 8.735                      # +0.191 - (-8.543)


def verify():
    if not TEX.exists():
        print("원고를 찾을 수 없다: %s (저장소 뿌리에서 돌려라)" % TEX)
        return False
    t = TEX.read_text(encoding="utf-8")
    want = ["%.3f" % abs(v) for _, a, b in ROWS for v in (a, b) if v is not None]
    want += ["%.3f" % SPAN, "%.2f" % MDE, "%.4f" % PEAK, "%.4f" % TARGET]
    miss = [v for v in want if v not in t]
    if miss:
        print("!! 원고에 없는 값: %s" % ", ".join(miss))
        print("   tab:criterion 을 고쳤다면 이 스크립트도 같이 고쳐라.")
        return False
    print("[검산] 기준 5값 + 범위 + MDE + 채택률 둘이 원고와 일치한다.")
    return True


def main():
    if not verify():
        return 1

    setup()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))

    y = np.arange(len(ROWS))[::-1].astype(float)   # 위에서부터 ROWS 순서
    nms = [r[1] for r in ROWS]
    dfl = [r[2] for r in ROWS]

    ax.fill_betweenx([y[-1] - 0.62, y[0] + 0.62], -MDE, MDE,
                     color=C["grid"], lw=0, zorder=0)
    ax.axvline(0.0, color=C["rule"], lw=0.8, zorder=2)

    # 같은 추정 방식을 잇는다 -- **기준을 바꾸면 어디로 움직이는가**가 논지다
    ok = [(v, yy) for v, yy in zip(nms, y) if v is not None]
    ax.plot([v for v, _ in ok], [yy for _, yy in ok], "-", lw=0.9,
            color=C["red"], zorder=3)
    ax.plot(dfl, y, "-", lw=0.9, color=C["blue"], zorder=3)
    ax.plot([v for v, _ in ok], [yy for _, yy in ok], "o", ms=4.6,
            color=C["red"], zorder=4)
    ax.plot(dfl, y, "s", ms=4.2, color=C["blue"], zorder=4)

    # **범례 대신 직접 라벨한다.** 계열이 둘뿐이고 범례는 주석과 겹친다
    ax.text(-7.30, y[0] - 0.34, "NMS candidate var.\n$-$ box size",
            fontsize=6.6, color=C["red"], ha="left", va="center",
            linespacing=1.2, zorder=5)
    ax.text(-1.70, y[0] - 0.34, "DFL distribution var.\n$-$ box size",
            fontsize=6.6, color=C["blue"], ha="left", va="center",
            linespacing=1.2, zorder=5)

    # 라벨을 전부 점 위에 두면 **잇는 선 위에 앉는다.** 선 반대쪽으로 민다
    def mark(vals, col, off):
        for (v, yy), (dx, dy, ha) in zip(
                [(v, yy) for v, yy in zip(vals, y) if v is not None], off):
            ax.text(v + dx, yy + dy, "%+.3f" % v, ha=ha, va="center",
                    fontsize=6.6, color=col, zorder=5)

    mark(nms, C["red"], [(0.0, 0.30, "center"), (-0.22, 0.0, "right")])
    mark(dfl, C["blue"], [(0.0, 0.30, "center"), (-0.22, 0.0, "right"),
                          (0.0, 0.30, "center")])

    # **없는 점.** 빈 자리에 왜 없는지 적는다 -- 측정 실패와 다르다
    ax.plot([-8.1], [y[2]], marker="x", ms=5.2, mew=1.1, color=C["red"],
            alpha=0.55, zorder=4)
    ax.text(-7.55, y[2],
            "no such control exists: NMS adoption rate\n"
            "peaks at %.4f, short of the %.4f target" % (PEAK, TARGET),
            fontsize=6.4, color=C["gray"], ha="left", va="center",
            linespacing=1.25, zorder=5)

    # 범위 표시 -- 해상도의 두 배가 넘는다
    ax.annotate("", xy=(-8.543, y[0] + 0.86), xytext=(+0.191, y[0] + 0.86),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=C["ink"],
                                shrinkA=0, shrinkB=0))
    ax.text(-4.18, y[0] + 0.99,
            "range %.3f  ($2.6\\times$ the minimum detectable effect)" % SPAN,
            ha="center", va="bottom", fontsize=6.6, color=C["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], linespacing=1.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-10.3, 4.6)
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2])
    ax.set_xlabel("$\\Delta$HOTA, detector $\\sigma$ $-$ box size $\\sigma_C$ "
                  "(gating channel)")
    ax.set_ylim(y[-1] - 0.95, y[0] + 1.45)
    ax.text(-10.1, y[-1] - 0.62,
            "grey band: within $\\pm3.31$, indistinguishable from zero",
            fontsize=6.4, color=C["gray"], ha="left", va="center")

    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)
    bare(ax, keep=("bottom",))
    save(fig, "fig_criterion")

    print()
    print("DFL 은 주 기준에서 부호가 뒤집힌다 (-2.034 -> +0.191).")
    print("NMS 는 주 기준에 대조군 자체가 없다. 범위 %.3f 은 해상도의 2.6배." % SPAN)
    print("**하나의 비율로 귀속하는 서술(88%)은 이 그림 위에서 지탱되지 않는다.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
