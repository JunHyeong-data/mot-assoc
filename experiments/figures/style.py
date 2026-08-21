# -*- coding: utf-8 -*-
"""논문 그림 공통 서식.

## 왜 이 파일이 있는가

그림 셋을 각자 만들었더니 **발표 슬라이드처럼** 됐다 -- 결론을 제목에 박고,
굵은 글씨에 색이 세고, 판마다 글꼴 크기가 달랐다. 학술지 그림은 그렇게 안 한다:

  * **그림에 제목을 달지 않는다.** 할 말은 캡션이 한다
  * 판 이름은 `(a)` `(b)` 정도로 짧게, 왼쪽 위에
  * 본문과 같은 **세리프** 글꼴, 8pt 안팎
  * 색은 묽게, 선은 얇게, 굵은 글씨는 안 쓴다
  * 격자는 아주 옅게 또는 없이

## 크기를 실제 인쇄 크기로 잡는다

`\\includegraphics[width=\\textwidth]` 로 넣으므로 **그림을 6.5인치로 만들면
축소가 없다** -- 즉 여기 8pt 로 찍은 글자가 지면에서도 8pt 다. 예전에는
11인치로 만들어 0.6배로 줄이는 바람에 글자가 5pt 로 앉았다.

사용법:
    from style import setup, C
    setup()
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

# 지면 폭. geometry margin=1in, a4 -> 6.27in. 안전하게 6.2 로 잡는다
WIDTH = 6.2

# 묽은 색. 인쇄와 흑백 복사에서 명도로도 갈린다
C = {
    "ink": "#1a1a1a",          # 본문 검정
    "rule": "#9a9a9a",         # 축·기준선
    "grid": "#e6e6e6",
    "blue": "#31556e",         # 진한 청회색 -- 주 계열
    "blue_pale": "#a8bcc9",
    "red": "#9c4a42",          # 벽돌 -- 대비 계열
    "red_pale": "#d6aaa5",
    "gray": "#8c8c8c",
    "gray_pale": "#d8d8d8",
    "green": "#4f6b4a",
    "amber": "#9a7b3f",
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Cambria", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C["rule"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # 글꼴 내장. 학술지 요구사항이다
        "ps.fonttype": 42,
    })


def panel(ax, tag, dy=1.02):
    """판 이름을 왼쪽 위 **축 바깥**에 단다. 제목이 아니라 표지다."""
    ax.text(0.0, dy, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=C["ink"])


def bare(ax, keep=("left", "bottom")):
    """쓸데없는 테두리를 없앤다."""
    for k, sp in ax.spines.items():
        sp.set_visible(k in keep)
    ax.tick_params(length=2.5)


def save(fig, name, out="figures", also="paper/figures"):
    """PDF 는 두 곳에 쓰고 PNG 는 보기용으로 `out` 에만 쓴다.

    **PDF 에서 `CreationDate` 를 뺀다.** 안 빼면 같은 그림을 다시 그려도
    바이트가 달라져서 (가) git 이 매번 새 blob 을 쌓고 (나) *"이 그림 진짜
    바뀌었나"* 를 해시로 못 묻는다. 실제로 두 번 그려 대조했다 --- 빼기 전은
    해시가 다르고 빼면 같다. **PNG 는 원래 타임스탬프를 안 넣어 손댈 것이 없다.**

    `paper/figures/` 는 예전에 README 가 **사람에게 복사를 시켰고** 여섯 중
    둘이 갈렸다 (풀어 보니 내용은 같고 메타데이터만 달랐다). 손을 없앤다.

    **PNG 는 git 에서 제외한다** --- 원고는 PDF 만 쓰고 어디서도 참조하지
    않는데 추적 바이트의 38% 였다. 눈으로 볼 용도로 만들기만 한다.
    """
    from pathlib import Path
    dirs = [Path(out)] + ([Path(also)] if also else [])
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        f = d / ("%s.pdf" % name)
        fig.savefig(f, metadata={"CreationDate": None})
        print("저장: %s" % f)
    f = Path(out) / ("%s.png" % name)
    fig.savefig(f)
    print("저장: %s (보기용, git 제외)" % f)
