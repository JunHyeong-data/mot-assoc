# -*- coding: utf-8 -*-
"""실험 16 -- **원고 6.2 절의 미측정 주장을 검사한다.**

## 왜

원고 6.2 절에 이렇게 적었다:

> *"NWD 는 diag(w^2/4, h^2/4), UCMCTrack 은 (sigma_m w, sigma_m h) 의 **비등방**
> 형태를 쓰는 반면 우리 C 는 diag(h^2, h^2) 의 **등방** 형태다.
> **비등방을 쓰면 C 가 더 강해질 것이므로 우리 결론은 보수적이다.**"*

**뒷 문장은 재지 않고 적은 것이다.** 이 저장소 규범(CLAUDE.md)에 어긋난다.

## 무엇을 재면 갈리는가

C 가 옳은 모양이려면 **실제 중심 오차의 이방성**과 맞아야 한다.

    등방  diag(h^2, h^2)      -> Var(e_x)/Var(e_y) = 1 을 가정
    비등방 diag(w^2, h^2)     -> Var(e_x)/Var(e_y) = (w/h)^2 를 가정

보행자는 w/h 가 대략 0.4 이므로 비등방은 **약 0.16** 을 가정하는 셈이다.
실제 비가 1 에 가까우면 **등방이 오히려 맞는 모양**이고, 그러면 원고의
"보수적" 은 **틀린 말**이 된다.

`ex`, `ey` (= dcx, dcy) 가 이미 `data/exp01/*.npz` 에 있으므로 **검출기를 다시
돌릴 필요가 없다.** 박스 폭 w 는 npz 에 없어서 GT 에서 가져온다.

## 읽는 법 -- 자료를 보기 전에 정한다

    비 >= 0.7      등방이 맞다.       원고의 "보수적" 을 **철회**하고 뒤집어 적는다
    0.25 ~ 0.7     중간. 어느 쪽도 못 박는다. 문장을 **사실 서술로만** 남긴다
    비 <= 0.25     비등방이 맞다.     원고의 "보수적" 이 **선다**

**분석 단위는 검출**이고 n 이 수만이라 검정력 문제가 없다 (규칙 6).
시퀀스별로도 내서 7/7 이 같은 구간에 드는지 본다.

사용법:
    python experiments/exp16_anisotropy/measure.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp01_nms_variance"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import student_t as ST                                          # noqa: E402

GT_ROOT = Path("data/MOT17_A/ablation")
PEDESTRIAN = 1


def gt_aspect(seq):
    """GT 보행자 박스의 w/h 중앙값. 판정과 무관한 참고량이지만 스케일이 필요하다."""
    wh = []
    p = GT_ROOT / ("%s-FRCNN" % seq) / "gt" / "gt.txt"
    if not p.exists():
        return float("nan")
    for line in open(p):
        f = line.strip().split(",")
        if len(f) < 8 or int(f[6]) == 0 or int(f[7]) != PEDESTRIAN:
            continue
        w, h = float(f[4]), float(f[5])
        if h > 0:
            wh.append(w / h)
    return float(np.median(wh)) if wh else float("nan")


def robust_var(x):
    """MAD 기반 분산. 오차 꼬리가 두꺼워서(꼬리비 12~213) 표본분산은 못 믿는다."""
    m = np.median(x)
    return (1.4826 * np.median(np.abs(x - m))) ** 2


def main():
    print("=" * 92)
    print("실험 16 -- 중심 오차의 이방성. **원고 6.2 의 미측정 주장을 검사한다**")
    print("=" * 92)
    print("등방 C   = diag(h^2, h^2)  -> Var(e_x)/Var(e_y) = 1 을 가정")
    print("비등방 C = diag(w^2, h^2)  -> Var(e_x)/Var(e_y) = (w/h)^2 를 가정")
    print()

    print("%-12s %8s %10s %10s %10s %10s"
          % ("시퀀스", "n", "sd(e_x)", "sd(e_y)", "**비**", "(w/h)^2"))
    print("-" * 92)

    ratios, targets, alle = [], [], []
    for s in ST.SEQS:
        d = ST.load(s)
        ex, ey = d["ex"], d["ey"]
        vx, vy = robust_var(ex), robust_var(ey)
        r = vx / vy
        a = gt_aspect(s)
        ratios.append(r)
        targets.append(a * a)
        alle.append((ex, ey))
        print("%-12s %8d %10.3f %10.3f %10.3f %10.3f"
              % (s, len(ex), np.sqrt(vx), np.sqrt(vy), r, a * a))

    print("-" * 92)
    med = float(np.median(ratios))
    medt = float(np.median(targets))
    ex = np.concatenate([a for a, _ in alle])
    ey = np.concatenate([b for _, b in alle])
    pooled = robust_var(ex) / robust_var(ey)
    print("%-12s %8d %10.3f %10.3f %10.3f %10.3f"
          % ("합침", len(ex), np.sqrt(robust_var(ex)), np.sqrt(robust_var(ey)),
             pooled, medt))
    print()
    print("  시퀀스별 비 중앙값 = %.3f   (범위 %.3f ~ %.3f)"
          % (med, min(ratios), max(ratios)))
    print("  비등방이 가정하는 값 (w/h)^2 중앙 = %.3f" % medt)

    # 표본분산으로도 낸다 -- 같은 양을 다른 경로로 (규칙 3)
    print()
    print("  [대조] 표본분산으로 낸 합침 비 = %.3f  (MAD 기준 %.3f)"
          % (np.var(ex) / np.var(ey), pooled))
    print("       두 값이 크게 다르면 꼬리가 지배하는 것이다. MAD 쪽을 판정에 쓴다.")

    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법")
    print("=" * 92)
    n_iso = sum(r >= 0.7 for r in ratios)
    n_ani = sum(r <= 0.25 for r in ratios)
    print("  시퀀스별: 등방 구간(>=0.7) %d/7,  중간 %d/7,  비등방 구간(<=0.25) %d/7"
          % (n_iso, 7 - n_iso - n_ani, n_ani))
    print()
    if med >= 0.7:
        print("  비 %.3f >= 0.7 => **등방이 맞다.**" % med)
        print("     원고 6.2 의 '비등방을 쓰면 C 가 더 강해진다' 는 **철회한다.**")
        print("     실제 오차가 거의 등방이므로 (w/h)^2=%.2f 를 가정하는 비등방 형태는" % medt)
        print("     x 축 분산을 %.0f배 **과소** 잡는다. 우리 C 가 오히려 유리한 모양이고,"
              % (1.0 / medt))
        print("     그렇다면 **우리 결론은 보수적이 아니라 관대하다.**")
    elif med <= 0.25:
        print("  비 %.3f <= 0.25 => **비등방이 맞다.**" % med)
        print("     원고 6.2 의 '보수적이다' 가 **선다.** 그대로 둔다.")
    else:
        print("  비 %.3f 는 0.25~0.7 사이 => **어느 쪽도 못 박는다.**" % med)
        print("     원고 6.2 를 **사실 서술로만** 남기고 방향 예측을 뺀다.")
    print()
    print("  한계: 이건 *모양*에 대한 측정이지 NLL 비교가 아니다. 실제 비등방 C 를")
    print("  적합해 보려면 박스 폭이 npz 에 있어야 하는데 없다. **후속 과제로 남긴다.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
