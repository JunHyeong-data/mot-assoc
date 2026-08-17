# -*- coding: utf-8 -*-
"""실험 6 LOSO [2단계] -- **판정**. 사전 선언은 `PREREG-loso.md`.

물음: **장면을 보고(GT 없이) 매칭 임계값을 고르면, 본 적 없는 장면에서
기본값 0.80 을 이기는가.**

## 갈래 넷 -- R1 이 진짜 비교 대상이다

  R0  0.80 고정 (ByteTrack 기본값)          장면 정보 없음
  R1  훈련 6개 최적값의 중앙값               장면 정보 **없음**
  R2  훈련 6개로 적합한 예측 규칙            장면 정보 있음
  R3  held-out 자신의 argmax (신탁)          정답을 봄

실험 1e 에서 검출별 정보가 **0** 인 상수 Sigma 가 NLL 최고를 찍은 적이 있다.
같은 함정이 여기 있다 -- 이득이 "장면을 봐서" 가 아니라 "0.80 이라는 숫자가
나빠서" 일 수 있다. **R1 을 빼면 이 실험은 무효다.**

## 예측변수를 사람이 고르지 않는다

`predictors.py` 의 상관표를 이미 봤으므로 거기서 최고를 골라 쓰면 자료 재사용이다.
각 fold 에서 **훈련 6개만으로** 고른다. 선택 편향이 fold 안에 갇힌다.

사용법:
    python experiments/exp06_levers/grid.py      # 먼저. 표를 만든다
    python experiments/exp06_levers/loso.py
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from replay import SEQS                                       # noqa: E402
from evaluate import build_data                               # noqa: E402
from predictors import scene_stats                            # noqa: E402
from tracker.eval.collections.hota import HOTA                # noqa: E402

GRIDJSON = Path("data/exp06/grid.json")
DEFAULT = 0.80
BAND = 0.3                      # 판정폭. exp03/exp05/exp06[A] 와 같은 자
# 사전 선언 목록 순서 그대로. 6~8 (0-based 5~7) 이 sigma 계열
SIGMA_KEYS = ("sigma 중앙값 (px)", "**sigma/sqrt(area) 중앙값**", "sigma 산포 CV")


def tag(th):
    return "th%03d" % int(round(th * 100))


def snap(v, grid):
    """그리드로 스냅. 동률이면 작은 쪽 (사전 선언)."""
    return sorted((abs(g - v), g) for g in grid)[0][1]


def fit_rule(x_tr, y_tr, grid):
    """중앙값 이분. n=6 이므로 이보다 복잡하면 안 된다."""
    m = float(np.median(x_tr))
    lo = [y for x, y in zip(x_tr, y_tr) if x <= m]
    hi = [y for x, y in zip(x_tr, y_tr) if x > m]
    allmed = float(np.median(y_tr))
    lo_v = snap(float(np.median(lo)) if lo else allmed, grid)
    hi_v = snap(float(np.median(hi)) if hi else allmed, grid)
    return m, lo_v, hi_v


def combined_hota(assign, metric):
    """시퀀스마다 다른 임계값의 트랙 파일을 모아 결합 HOTA (검출 수 가중)."""
    per = {}
    for seq, th in assign.items():
        d = build_data(seq, "_headroom/" + tag(th))
        if d is not None:
            per[seq] = metric.eval_sequence(d)
    if not per:
        return float("nan")
    return 100 * float(np.mean(metric.combine_sequences(per)["HOTA"]))


def sign_test(d):
    n = len(d)
    win = int((np.asarray(d) > 0).sum())
    k = max(win, n - win)
    p = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return win, n, min(p, 1.0)


def main():
    if not GRIDJSON.exists():
        print("먼저 grid.py 를 돌려라: python experiments/exp06_levers/grid.py")
        return 1
    G = json.loads(GRIDJSON.read_text())
    grid = G["grid"]
    # per_seq[seq][th] -> HOTA
    H = {s: {float(t): v for t, v in G["per_seq"][s].items()} for s in G["seqs"]}
    seqs = [s for s in SEQS if s in H]

    best = {s: max(H[s], key=H[s].get) for s in seqs}

    print("=" * 96)
    print("실험 6 LOSO [2단계] 판정 -- 사전 선언 PREREG-loso.md")
    print("=" * 96)
    print("장면을 보고 임계값을 고르면 본 적 없는 장면에서 0.80 을 이기는가.")
    print()

    # 장면 관측량 (GT 를 안 쓴다)
    print("장면 관측량을 잰다...")
    stats = {}
    for s in seqs:
        st = scene_stats(s)
        if st is not None:
            stats[s] = st
    keys = list(next(iter(stats.values())))
    print("  후보 %d개, 시퀀스 %d개" % (len(keys), len(stats)))
    print()

    # ---------------- fold 별 ----------------
    print("=" * 96)
    print("fold 별 -- 훈련 6개로 정하고 본 적 없는 1개에서 잰다")
    print("=" * 96)
    print("%-16s %-26s %6s %6s %6s %6s   %7s %7s %7s %7s"
          % ("held-out", "고른 예측변수", "R0", "R1", "R2", "R3",
             "R0", "R1", "R2", "R3"))
    print("%-16s %-26s %6s %6s %6s %6s   %7s %7s %7s %7s"
          % ("", "(훈련 6개로 고름)", "임계", "임계", "임계", "임계",
             "HOTA", "HOTA", "HOTA", "HOTA"))
    print("-" * 96)

    picks, rows = [], {}
    assign = {"R0": {}, "R1": {}, "R2": {}, "R3": {}}
    for s in seqs:
        tr = [t for t in seqs if t != s]
        y_tr = [best[t] for t in tr]

        # 예측변수 선택: 훈련 6개만으로. 동률이면 목록 앞선 것
        scoreboard = []
        for k in keys:
            x = np.array([stats[t][k] for t in tr], float)
            if np.all(np.isnan(x)) or np.nanstd(x) == 0:
                scoreboard.append(0.0)
                continue
            r, _ = spearmanr(x, y_tr)
            scoreboard.append(0.0 if np.isnan(r) else abs(float(r)))
        kbest = keys[int(np.argmax(scoreboard))]
        picks.append(kbest)

        x_tr = [stats[t][kbest] for t in tr]
        m, lo_v, hi_v = fit_rule(x_tr, y_tr, grid)
        th2 = hi_v if stats[s][kbest] > m else lo_v

        th = {"R0": DEFAULT,
              "R1": snap(float(np.median(y_tr)), grid),
              "R2": th2,
              "R3": best[s]}
        for a in assign:
            assign[a][s] = th[a]
        h = {a: H[s][th[a]] for a in th}
        rows[s] = (th, h)
        print("%-16s %-26s %6.2f %6.2f %6.2f %6.2f   %7.2f %7.2f %7.2f %7.2f"
              % (s.replace("-FRCNN", ""), kbest[:26],
                 th["R0"], th["R1"], th["R2"], th["R3"],
                 h["R0"], h["R1"], h["R2"], h["R3"]))

    # ---------------- 집계 ----------------
    metric = HOTA()
    comb_h = {a: combined_hota(assign[a], metric) for a in assign}
    unw = {a: float(np.mean([rows[s][1][a] for s in seqs])) for a in assign}

    print()
    print("=" * 96)
    print("집계")
    print("=" * 96)
    print("%-34s %10s %10s" % ("", "결합 HOTA", "시퀀스 평균"))
    print("%-34s %10s %10s" % ("", "(검출 수 가중)", "(가중 없음)"))
    print("-" * 96)
    label = {"R0": "R0  기본값 0.80",
             "R1": "R1  훈련 최적값의 중앙값 (상수)",
             "R2": "R2  예측 규칙",
             "R3": "R3  신탁 (도달 불가 상한)"}
    for a in ("R0", "R1", "R2", "R3"):
        print("%-34s %10.3f %10.3f" % (label[a], comb_h[a], unw[a]))

    # 검산: R0 는 grid.json 의 0.80 결합값과 같아야 한다
    ref = float(G["combined"]["0.80"])
    print()
    print("  검산: R0 결합 %.3f vs grid.json 의 0.80 %.3f  %s"
          % (comb_h["R0"], ref, "OK" if abs(comb_h["R0"] - ref) < 1e-6 else "** 불일치 **"))

    # ---------------- 사전 선언한 판정 ----------------
    print()
    print("=" * 96)
    print("사전 선언한 판정  (판정폭 %.1f HOTA)" % BAND)
    print("=" * 96)
    e1 = comb_h["R2"] - comb_h["R0"]
    e2 = comb_h["R2"] - comb_h["R1"]
    e3 = comb_h["R3"] - comb_h["R0"]
    e1u = unw["R2"] - unw["R0"]
    e2u = unw["R2"] - unw["R1"]

    d1 = [rows[s][1]["R2"] - rows[s][1]["R0"] for s in seqs]
    d2 = [rows[s][1]["R2"] - rows[s][1]["R1"] for s in seqs]
    w1, n1, p1 = sign_test(d1)
    w2, n2, p2 = sign_test(d2)

    print("  [1] 주 종말점   R2 - R0 = %+.3f  (가중 없음 %+.3f)   부호 %d/%d, p=%.3f"
          % (e1, e1u, w1, n1, p1))
    print("  [2] 기여 귀속   R2 - R1 = %+.3f  (가중 없음 %+.3f)   부호 %d/%d, p=%.3f"
          % (e2, e2u, w2, n2, p2))
    print("  [3] 신탁 상한   R3 - R0 = %+.3f  (참고. 주장 아님)" % e3)

    # 아래 두 줄은 **결과를 보고 추가했다.** 판정 로직은 안 건드렸다 --
    # evaluate.py 가 실험 5 에서 같은 상황에 찍던 경고와 회수율을 옮긴 것뿐이다.
    if e1 * e1u < 0:
        print("      ** 가중 여부에 따라 [1] 의 부호가 뒤집힌다. 큰 시퀀스가 지배한다는 뜻이다. **")
        print("         사전 선언한 주 판정은 **가중 결합**이다. 둘 다 판정폭 안이므로 결론은 같다")
    if e3 > 0:
        print("      신탁 회수율: 가중 %.0f%%, 가중 없음 %.0f%%   (R2-R0 을 R3-R0 으로 나눈 것)"
              % (100 * e1 / e3, 100 * e1u / max(unw["R3"] - unw["R0"], 1e-9)))

    nsig = sum(1 for k in picks if k in SIGMA_KEYS)
    print("  [4] sigma 계열이 뽑힌 fold = %d/%d" % (nsig, len(picks)))
    uniq = {}
    for k in picks:
        uniq[k] = uniq.get(k, 0) + 1
    print("      선택 분포: " + ", ".join("%s x%d" % (k[:24], v)
                                          for k, v in sorted(uniq.items(), key=lambda z: -z[1])))

    print()
    print("=" * 96)
    print("결론 -- 사전 선언한 표를 그대로 적용한다")
    print("=" * 96)
    if e1 > BAND and e2 > BAND:
        print("  [1] 통과, [2] 통과 => **적응 규칙이 실재한다.**")
        print("      다음은 외부 데이터셋 검증 (MOT20 / DanceTrack / KITTI)")
    elif e1 > BAND:
        print("  [1] 통과, [2] 실패 => **상수만 바꾸면 된다.** 장면 관측량의 기여는 0.")
        print("      남는 건 '0.80 이 최적이 아니다' 뿐이고 그건 튜닝이지 연구가 아니다")
    elif e1 < -BAND:
        print("  [1] 음의 방향 => **규칙이 해롭다.** 레버 2 가 죽는다")
    else:
        print("  [1] 판정폭 안 => **레버 2 가 죽는다.**")
        print("      신탁 %+.2f 는 도달 불가였다. 임계값 통로도 닫힌다" % e3)

    if nsig == 0:
        print()
        print("  [4] sigma 계열이 한 번도 안 뽑혔다 => 눈금이 필요 없는 자리에서도")
        print("      sigma 는 밀도/겹침 같은 단순 관측량에 진다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
