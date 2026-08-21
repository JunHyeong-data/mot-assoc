# -*- coding: utf-8 -*-
"""**그림 3.** 여지는 있는데 우리는 못 갔다.

  (a) 시퀀스별 기준선과 오라클 상한. **여지가 어디 몰려 있는가** (0.91 ~ 12.17)
  (b) 결합 기준 dHOTA. **오라클 둘과 카메라 보상만 0 위, 검출기 sigma 는 전부 아래**

한 장에 나란히 놓아야 "못 간 것" 과 "없던 것" 이 갈린다.

## 자료 출처

  (a) `data/exp14/recovery.json` 의 `per[].base` 와 `room`
      (exp12 가 낸 값을 exp14 가 받아 적은 것. 감사 정정 후 값)
  (b) exp12 `run.py` 의 [4] 표. 기록은 `notes/progress.md`

**거리 함수는 소스를 갈라 둘로 그린다** -- 예전에 -4.98 하나만 그렸는데
그건 `wn_size`, 즉 **박스 크기 소스**다. 검출기 sigma 조건은 `wn_dfl` 이고
52.10 - 61.00 = **-8.90** 이다 (심사 2차에서 잡음).

사용법:
    python experiments/figures/fig_ceiling.py
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

SRC = Path("data/exp14/recovery.json")

# (라벨, 값, 조건). exp12 run.py 의 [4] 와 같아야 한다 -- 아래에서 검산한다.
TRIED = [
    ("association ceiling (oracle)", +3.122, "ceiling"),
    ("threshold oracle (upper bound)", +0.892, "oracle"),
    ("camera motion comp. (not $\\sigma$)", +0.430, "other"),
    ("matching threshold", -0.207, "ours"),
    ("Kalman $R$", -0.620, "ours"),
    ("gating / box expansion", -4.330, "ours"),
    ("distance function, box-size $\\sigma_C$", -4.980, "size"),
    ("distance function, detector $\\sigma$", -8.900, "ours"),
]
FILL = {"ceiling": C["blue"], "oracle": C["blue_pale"], "other": C["gray_pale"],
        "ours": C["red"], "size": C["red_pale"]}


def main():
    if not SRC.exists():
        print("먼저 실험 14 를 돌려라: python experiments/exp14_cmc/run.py")
        return 1
    d = json.loads(SRC.read_text())
    seqs = sorted(d["room"], key=lambda s: d["room"][s])
    short = [s.replace("MOT17-", "").replace("-FRCNN", "") for s in seqs]
    base = np.array([d["per"][s]["base"] for s in seqs])
    room = np.array([d["room"][s] for s in seqs])

    # **검산 (규칙 3).** 시퀀스 여지의 단순 평균이 exp12 기록과 맞는가
    unw = float(room.mean())
    print("[검산] 시퀀스 여지의 단순 평균 = %.3f   (exp12 기록 5.372)" % unw)
    if abs(unw - 5.372) > 0.01:
        print("  !! exp12 기록과 어긋난다. 그림을 믿지 말 것.")
        return 1
    print("  일치.")

    setup()
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 1.20, 2.5 * 1.20),
                           gridspec_kw={"width_ratios": [1.0, 1.15]})
    # (b) 의 눈금 라벨이 길어서 **왼쪽 판을 침범한다.** 간격을 넉넉히 준다
    fig.subplots_adjust(wspace=0.72)

    # ---- (a) 시퀀스별 기준선 + 여지 ----
    a = ax[0]
    y = np.arange(len(seqs))
    a.barh(y, base, color=C["gray_pale"], edgecolor="none", height=0.68, zorder=2)
    a.barh(y, room, left=base, color=C["blue"], edgecolor="none", height=0.68,
           zorder=3)
    for i, (bs, r) in enumerate(zip(base, room)):
        a.text(bs + r + 1.4, i, "%.2f" % r, va="center", fontsize=6.8,
               color=C["blue"], zorder=4)
    a.set_yticks(y)
    a.set_yticklabels(short)
    a.set_xlim(0, 92)
    a.set_xticks([0, 20, 40, 60, 80])
    a.set_xlabel("HOTA")
    a.set_ylabel("MOT17 sequence")

    # ---- (b) 결합 dHOTA ----
    b = ax[1]
    lab = [t[0] for t in TRIED][::-1]
    val = [t[1] for t in TRIED][::-1]
    kind = [t[2] for t in TRIED][::-1]
    yy = np.arange(len(lab))
    b.barh(yy, val, color=[FILL[k] for k in kind], height=0.68, zorder=3,
           edgecolor="none")
    b.axvline(0.0, color=C["rule"], lw=0.6, zorder=4)
    for i, v in enumerate(val):
        b.text(v + (0.3 if v >= 0 else -0.3), i, "%+.2f" % v, va="center",
               ha="left" if v >= 0 else "right", fontsize=6.8, zorder=5,
               color=C["ink"])
    b.set_yticks(yy)
    b.set_yticklabels(lab)
    b.set_xlim(-11.4, 5.6)
    b.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4])
    b.set_xlabel("$\\Delta$HOTA vs baseline (combined)")

    for tag, a_ in (("(a) where the room is", a),
                    ("(b) what each channel reached", b)):
        a_.grid(axis="x", zorder=0)
        a_.set_axisbelow(True)
        bare(a_)
        panel(a_, tag)
    b.tick_params(axis="y", length=0)

    save(fig, "fig_ceiling")

    print()
    print("여지 범위 %.2f (%s) ~ %.2f (%s)" % (room[0], short[0], room[-1], short[-1]))
    print("0 위: 오라클 둘 + 카메라 보상. **검출기 sigma 경로는 전부 아래.**")
    print("거리 함수는 소스를 갈라 둘 -- 크기 -4.98 대 DFL -8.90.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
