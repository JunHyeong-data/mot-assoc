# -*- coding: utf-8 -*-
"""**그림 `fig:reversal`** -- 잣대를 바꾸면 박스 크기만 우연 수준을 넘는다.

원고의 두 표(`tab:auc` 주변부 AUC, `tab:withinrow` 행 안 정답률)를 한 장에 겹친다.
표로는 두 쪽에 떨어져 있어 **둘이 우연 수준의 반대쪽이라는 게 안 보인다.**

  (a) 같은 형태의 확률(오답 쪽 검출의 신호가 더 클 확률)을 두 비교 대상으로
      잰다. **박스 크기만 0.5 를 가로지른다** (주변부 0.4661, 행 안 0.6451).
      **주변부 값을 방향으로 읽으면 안 된다** -- 두 분포가 교차하므로 이동이
      아니라 교차의 요약이다 (철회 18). `crossing()` 이 그 근거를 다시 잰다
  (b) 그런데 시퀀스별로 보면 박스 크기는 **4/7 만** 0.5 위다. 군집 구간이
      0.5 를 포함하므로 **탐색적 관측이다** (규칙 6)

## 자료 출처 -- (a)(b) 의 점은 전부 실측이다

  * 행 안: `data/exp21/withinrow-{nms,dfl}.npz` (exp18 `paired.py` 산출).
    동점은 제외한다 -- exp21 과 같은 규약
  * 주변부: `data/exp15/perseq-{nms,dfl}.json` (exp15 `run.py`)
  * **군집 구간만 원고에서 옮겨 적는다.** exp21 은 rng 를 여섯 번 이어 쓰므로
    호출 순서까지 맞춰야 같은 값이 나온다. 사전 등록이 정한 **주 판독은
    시퀀스별 부호 개수**이고 그건 여기서 실측한다

`verify()` 가 실측값과 원고를 대조해 어긋나면 **그리지 않는다** (규칙 3).

사용법:
    python experiments/figures/fig_reversal.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup, save, panel, bare, C, WIDTH            # noqa: E402

import matplotlib.pyplot as plt                                 # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEX = Path("paper/report.tex")
P15, P21 = Path("data/exp15"), Path("data/exp21")

# (표시 이름, 색, 원고의 군집 95% 구간)
SIG = [
    ("NMS candidate var.",   "nms",  C["red"],       (0.427, 0.510)),
    ("DFL distribution var.", "dfl", C["red_pale"],  (0.366, 0.504)),
    ("box size $\\sigma_C$",  "size", C["blue"],     (0.482, 0.813)),
]


def within_row(tag):
    """행 안 정답률. 동점 제외 -- exp21 `cluster_ci.py` 와 같은 규약."""
    f = P21 / ("withinrow-%s.npz" % ("dfl" if tag == "size" else tag))
    w = np.load(f, allow_pickle=True)
    col = "size" if tag == "size" else "sig"
    a, b = w[col + "_wrong"].astype(float), w[col + "_right"].astype(float)
    keep = a != b
    seq = w["seq"][keep]
    a, b = a[keep], b[keep]
    seqs = sorted(set(seq.tolist()))
    per = [(s, float((b[seq == s] < a[seq == s]).mean())) for s in seqs]
    return float((b < a).mean()), per, int(keep.size - keep.sum())


def marginal(tag):
    """채택 쌍 전체의 주변부 AUC. exp15 가 이미 낸 값을 읽는다."""
    j = json.loads((P15 / ("perseq-%s.json"
                           % ("dfl" if tag == "size" else tag))).read_text(
        encoding="utf-8"))
    return j["overall_c"] if tag == "size" else j["overall"]


def crossing():
    """**주변부 AUC 를 방향으로 읽으면 틀린다.** 그 근거를 여기서 다시 잰다.

    원고는 한때 `0.4661` 을 "박스가 작을수록 오류" 로 옮겨 적었다. 그런데
    두 분포가 **교차**한다 -- 오류 쪽이 아래에 더 몰리지만 중앙값은 오히려
    높고 위쪽 꼬리는 얇다. 이동이 아니라 교차이므로 방향이 정의되지 않는다.

    5.8 절이 인용하는 여섯 값을 돌려준다 (25/50/90 분위, 오류/정답).
    """
    d = np.load(P15 / "pairs-dfl.npz", allow_pickle=True)
    lab, sc = d["lab"], d["hgt"].astype(float) / 2.0
    err, ok = sc[lab == "틀림_고칠수있음"], sc[lab == "옳음"]
    q = [25, 50, 90]
    return np.percentile(err, q), np.percentile(ok, q)


def main():
    for p in (P15, P21, TEX):
        if not p.exists():
            print("없다: %s  (저장소 뿌리에서 돌려라)" % p)
            return 1
    t = TEX.read_text(encoding="utf-8")

    data, miss, ties = {}, [], 0
    for name, tag, col, ci in SIG:
        w, per, nt = within_row(tag)
        m = marginal(tag)
        ties += nt
        data[tag] = dict(name=name, color=col, ci=ci, within=w, per=per,
                         marg=m)
        # **실측이 원고와 맞는가.** 어긋나면 그림이 아니라 원고를 의심해야 한다
        for v in ("%.4f" % w, "%.4f" % m):
            if v not in t:
                miss.append("%s %s" % (tag, v))
        for v in ci:
            if ("%.3f" % v) not in t:
                miss.append("%s CI %.3f" % (tag, v))
    if miss:
        print("!! 실측/구간이 원고에 없다: %s" % ", ".join(miss))
        return 1
    print("[검산] 행 안 3값 + 주변부 3값 + 군집 구간 6값이 원고와 일치한다.")
    print("[검산] 동점 %d 건 (원고: 동률 0건)." % ties)
    if ties:
        print("  !! 원고는 동률 0건이라 적었다.")
        return 1

    # **교차 근거.** 5.8 절이 인용하는 분위수 여섯을 다시 재어 대조한다
    qe, qk = crossing()
    bad = [("%.1f" % v) for v in list(qe) + list(qk) if ("%.1f" % v) not in t]
    print("[검산] 분위수 25/50/90  오류 %s  정답 %s"
          % ("/".join("%.1f" % v for v in qe), "/".join("%.1f" % v for v in qk)))
    if bad:
        print("  !! 원고에 없는 분위수: %s" % ", ".join(bad))
        return 1
    if not (qe[0] < qk[0] and qe[1] > qk[1] and qe[2] < qk[2]):
        print("  !! 교차 양상이 깨졌다. 5.8 절의 ``교차'' 서술을 다시 봐야 한다.")
        return 1
    print("  교차 확인 -- 하위는 오류가 작고, 중앙은 크고, 상위는 다시 작다.")
    print("  **그러므로 주변부 0.4661 을 하나의 방향으로 옮겨 적을 수 없다.**")

    setup()
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 1.14, 2.6),
                           gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(wspace=0.34)

    # ---------- (a) 잣대를 바꾸면 ----------
    a = ax[0]
    a.axhline(0.5, color=C["rule"], lw=0.7, ls=(0, (3, 2)), zorder=2)
    a.text(0.005, 0.5, "0.5", fontsize=6.6, color=C["gray"], va="bottom",
           ha="left")
    # 점을 **안쪽으로 들인다** -- 0/1 에 두면 값 라벨이 판 밖으로 잘린다
    XL, XR = 0.13, 0.87
    for tag in ("nms", "dfl", "size"):
        d = data[tag]
        lw = 1.5 if tag == "size" else 0.9
        a.plot([XL, XR], [d["marg"], d["within"]], "-o", lw=lw, ms=4.0,
               color=d["color"], zorder=4)
        a.text(XL - 0.035, d["marg"], "%.4f" % d["marg"], ha="right",
               va="center", fontsize=6.6, color=d["color"])
        a.text(XR + 0.035, d["within"], "%.4f" % d["within"], ha="left",
               va="center", fontsize=6.6, color=d["color"])
    a.set_xlim(0, 1)
    a.set_xticks([XL, XR])
    a.set_xticklabels(["marginal:\nagainst correct matches\nanywhere in the data",
                       "within row:\nagainst the correct\ndetection in the same call"],
                      fontsize=6.8, linespacing=1.25)
    a.set_ylim(0.40, 0.70)
    a.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    a.set_ylabel("P(signal is larger on the erroneous match)")
    # **부호가 뒤집히는 자리를 그 자리에서 말한다**
    a.annotate("only box size crosses 0.5,\nand it crosses upward",
               xy=(0.47, 0.573), xytext=(0.20, 0.672), fontsize=6.4,
               color=C["blue"], ha="left", va="center", linespacing=1.25,
               arrowprops=dict(arrowstyle="-", lw=0.6, color=C["blue"],
                               shrinkA=2, shrinkB=2))
    a.tick_params(axis="x", length=0)

    # ---------- (b) 시퀀스별 ----------
    b = ax[1]
    b.axhline(0.5, color=C["rule"], lw=0.7, ls=(0, (3, 2)), zorder=2)
    for i, tag in enumerate(("nms", "dfl", "size")):
        d = data[tag]
        lo, hi = d["ci"]
        b.plot([i, i], [lo, hi], "-", lw=5.0, color=d["color"], alpha=0.20,
               solid_capstyle="butt", zorder=2)
        vals = np.array([v for _, v in d["per"]])
        jit = np.linspace(-0.16, 0.16, len(vals))
        b.plot(i + jit, vals, "o", ms=3.0, mfc="none", mew=0.8,
               color=d["color"], zorder=4)
        b.plot([i - 0.28, i + 0.28], [d["within"]] * 2, "-", lw=1.4,
               color=d["color"], zorder=5)
        n_up = int((vals > 0.5).sum())
        b.text(i, 0.965, "%d/7" % n_up, ha="center", va="center",
               fontsize=7.2, color=d["color"])
    b.set_xlim(-0.55, 2.45)
    b.set_xticks([0, 1, 2])
    b.set_xticklabels(["NMS", "DFL", "box size"], fontsize=7.0)
    b.set_ylim(0.28, 1.0)
    b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    b.tick_params(axis="x", length=0)

    for tag, ax_ in (("(a) the same signal under two rulers", a),
                     ("(b) within-row rate by sequence", b)):
        ax_.grid(axis="y", zorder=0)
        ax_.set_axisbelow(True)
        bare(ax_)
        panel(ax_, tag)
    a.spines["bottom"].set_visible(False)
    b.spines["bottom"].set_visible(False)

    save(fig, "fig_reversal")

    print()
    for tag in ("nms", "dfl", "size"):
        d = data[tag]
        print("  %-22s 주변부 %.4f -> 행 안 %.4f   시퀀스별 0.5 위 %d/7"
              % (d["name"].replace("$\\sigma_C$", "sigma_C"), d["marg"],
                 d["within"], sum(1 for _, v in d["per"] if v > 0.5)))
    print()
    print("**박스 크기만 0.5 를 가로지르고, 두 잣대의 방향이 반대다.**")
    print("그러나 시퀀스별로는 4/7 이고 군집 구간이 0.5 를 포함한다 -- 탐색적이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
