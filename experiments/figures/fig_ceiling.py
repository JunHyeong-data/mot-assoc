# -*- coding: utf-8 -*-
"""**그림 3 -- 여지는 있는데 우리는 못 갔다.**

## 왜 이 그림인가

원고의 전환점이 5.5~5.6 절이다. "네 경로가 다 음성" 은 "**딸 게 없었다**" 로도
읽힌다. 그걸 **오라클 상한**으로 닫았다 -- 여지는 +3.122 HOTA 로 **있다.**

  왼쪽  시퀀스별 기준선과 천장. **여지가 어디 몰려 있는가** (0.91 ~ 12.17)
  오른쪽 결합 기준 dHOTA. **천장 하나만 0 위에 있고 우리 넷은 전부 아래다**

한 장에 나란히 놓아야 "못 간 것" 과 "없던 것" 이 갈린다.

## 자료 출처 -- 전부 재현 가능

  왼쪽   `data/exp14/recovery.json` 의 `per[].base` 와 `room`
         (exp12 가 낸 값을 exp14 가 받아 적은 것. 감사 정정 후 값)
  오른쪽 exp12 `run.py` 의 [4] 표와 같다. 기록은 `notes/progress.md`

라벨은 영어로 쓴다 -- 논문 그림이고 Windows 에 한글 글꼴이 없을 수 있다.

사용법:
    python experiments/figures/fig_ceiling.py
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

SRC = Path("data/exp14/recovery.json")
OUT = Path("figures")

# 오른쪽 판. exp12 run.py 의 [4] 와 **같은 값이어야 한다** -- 아래에서 검산한다.
TRIED = [
    ("Association ceiling\n(oracle, exp12)", +3.122, "ceiling"),
    ("Threshold oracle\n(exp06 upper bound)", +0.892, "oracle"),
    ("Camera motion comp.\n(exp14, not $\\sigma$)", +0.430, "other"),
    ("Matching threshold\n(exp06 LOSO)", -0.207, "ours"),
    ("Kalman $R$\n(exp02, NMS $\\sigma$)", -0.620, "ours"),
    ("Gating\n(exp03, NMS $\\sigma$)", -4.330, "ours"),
    # **정정 (심사 2차)**: 예전에 거리 함수를 -4.98 로 그렸는데 그건 `wn_size`,
    # 즉 **박스 크기 소스**다. 검출기 sigma 갈래는 `wn_dfl` 이고
    # 52.10 - 61.00 = **-8.90** 이다. 둘을 갈라 그린다.
    ("Distance function\n(exp05, box-size $\\sigma_C$)", -4.980, "size"),
    ("Distance function\n(exp05, DFL $\\sigma$)", -8.900, "ours"),
]
CLR = {"ceiling": "#1f4e79", "oracle": "#7f9fbf", "other": "#9a9a9a",
       "ours": "#c0504d", "size": "#e8b4b1"}   # size = 크기 소스. 우리 sigma 와 구별


def main():
    if not SRC.exists():
        print("먼저 실험 14 를 돌려라: python experiments/exp14_cmc/run.py")
        return 1
    d = json.loads(SRC.read_text())
    seqs = sorted(d["room"], key=lambda s: d["room"][s])
    short = [s.replace("MOT17-", "").replace("-FRCNN", "") for s in seqs]
    base = np.array([d["per"][s]["base"] for s in seqs])
    room = np.array([d["room"][s] for s in seqs])

    # **검산 (규칙 3).** 결합 여지는 시퀀스 여지의 가중 평균이지 단순 평균이 아니다.
    # 단순 평균이 exp12 가 찍은 "가중 없는 시퀀스 평균 +5.372" 와 맞는지만 본다.
    unw = float(room.mean())
    print("[검산] 시퀀스 여지의 단순 평균 = %.3f   (exp12 기록 5.372)" % unw)
    if abs(unw - 5.372) > 0.01:
        print("  !! exp12 기록과 어긋난다. 그림을 믿지 말 것.")
        return 1
    print("  일치. 왼쪽 판의 자료가 exp12 것과 같다.")

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.8),
                           gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- 왼쪽: 시퀀스별 기준선 + 여지 ----
    a = ax[0]
    y = np.arange(len(seqs))
    a.barh(y, base, color="#d9d9d9", edgecolor="#bbbbbb", height=0.62, zorder=2,
           label="baseline (IoU)")
    a.barh(y, room, left=base, color="#1f4e79", edgecolor="#16395a", height=0.62,
           zorder=3, label="room to ceiling (oracle)")
    for i, (b, r) in enumerate(zip(base, room)):
        a.text(b + r + 0.6, i, "+%.2f" % r, va="center", fontsize=8.4,
               color="#1f4e79", fontweight="bold", zorder=4)
    a.set_yticks(y)
    a.set_yticklabels(short)
    a.set_xlim(0, 96)
    a.set_xlabel("HOTA")
    a.set_ylabel("MOT17 sequence (val_half)")
    a.set_title("(a) Where the room is\n13x spread across scenes", fontsize=11)
    # **판 안에 빈 구석이 없다** -- lower right 는 04 의 값을, upper right 는
    # 13 의 막대를 가렸다. 축 **바깥 아래**로 뺀다
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17),
             ncol=2, frameon=False, handlelength=1.5, columnspacing=1.4)

    # ---- 오른쪽: 결합 dHOTA ----
    b = ax[1]
    lab = [x[0] for x in TRIED][::-1]
    val = [x[1] for x in TRIED][::-1]
    kind = [x[2] for x in TRIED][::-1]
    yy = np.arange(len(lab))
    b.barh(yy, val, color=[CLR[k] for k in kind], height=0.62, zorder=3,
           edgecolor="#00000022")
    b.axvline(0.0, color="0.3", lw=1.2, zorder=4)
    for i, v in enumerate(val):
        b.text(v + (0.16 if v >= 0 else -0.16), i, "%+.2f" % v, va="center",
               ha="left" if v >= 0 else "right", fontsize=8.4, zorder=5,
               fontweight="bold" if kind[i] == "ceiling" else "normal")
    b.set_yticks(yy)
    b.set_yticklabels(lab, fontsize=8.2)
    b.set_xlim(-10.8, 4.6)
    b.set_xlabel("$\\Delta$HOTA vs baseline (combined)")
    b.set_title("(b) What we reached\nthe ceiling is positive; every detector-$\\sigma$ channel is not",
                fontsize=11)

    for a_ in ax:
        a_.grid(axis="x", color="0.92", zorder=0)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(length=0)

    fig.suptitle("The association ceiling is +3.12 HOTA -- the negative results are "
                 "\"did not reach\", not \"nothing to reach\"", fontsize=11.5, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / ("fig_ceiling.%s" % ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("저장: %s" % p)

    print()
    print("여지 범위 %.2f (%s) ~ %.2f (%s)"
          % (room[0], short[0], room[-1], short[-1]))
    print("0 위에 있는 것은 오라클 둘과 카메라 보상뿐이다. **검출기 sigma 경로 넷은 전부 아래.**")
    print("거리 함수는 소스를 갈라 둘로 그렸다 -- 크기 -4.98 대 DFL -8.90.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
