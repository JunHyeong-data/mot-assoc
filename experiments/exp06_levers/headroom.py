# -*- coding: utf-8 -*-
"""실험 6 [관문] -- 매칭 임계값에 **여유가 남아 있는가**.

## 이건 제안이 아니라 여유 측정이다

실험 5b 에서 채택률 보정만으로 HOTA 가 **+7.14** 움직였다. exp00 도
"민감도의 원인은 임계값" 이라고 했다. 그런데 그건 전부 **와서스타인 갈래** 얘기다.
**기준선(맨 IoU, match_thresh=0.8)에도 여유가 있는가**는 아직 모른다.

여유가 없으면(HOTA 가 0.8 근처에서 평평하면) 레버 2 는 죽은 것이다. **끝낸다.**
여유가 있으면 다음 질문이 열린다: **최적 임계값이 장면마다 다른가.**

## 미리 못박는다 -- 최적값을 골라 "이겼다" 고 하지 않는다

이 스윕은 **검정 집합 위에서 도는 것**이다. 여기서 가장 좋은 값을 골라 보고하면
그건 그냥 검정 집합 튜닝이다. **이 스크립트가 답하는 것은 딱 둘이다:**

  [A] 기준선 0.8 이 이미 꼭대기인가 (여유의 유무)
  [B] 꼭대기가 시퀀스마다 다른가 (적응 규칙이 존재할 수 있는가)

[B] 가 '다르다' 로 나와도 그것 자체는 성능 주장이 아니다. 규칙은 **장면 고유의
관측량**(밀도, 카메라 운동 등)에서 나와야 하고 **leave-one-sequence-out** 으로
검증해야 한다. 그게 실험 1e 에서 쓴 절차다.

사용법:
    python experiments/exp06_levers/headroom.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from replay import WTracker, Det, load, SEQS, BASE          # noqa: E402
from evaluate import build_data, TRACKS                     # noqa: E402
from tracker.eval.collections.hota import HOTA              # noqa: E402

# 0.8 을 가운데 두고 양쪽으로. 촘촘하게 볼 필요는 없다 -- 여유의 유무만 본다.
GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ARM = "iou"
OUTROOT = TRACKS / "_headroom"


def run(thresh):
    """기준선을 주어진 match_thresh 로 재생하고 HOTA 를 낸다."""
    cfg = dict(BASE)
    cfg["match_thresh"] = thresh
    tag = "th%03d" % int(round(thresh * 100))
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)

    for seq in SEQS:
        c = load(seq, ARM)
        if c is None:
            continue
        tr = WTracker(SimpleNamespace(**cfg), ARM, 1.0, frame_rate=30)
        lines = []
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            det = Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                      c["sxx"][m], c["syy"][m])
            for row in tr.update(det):
                x1, y1, x2, y2 = row[:4]
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                             % (f, int(row[4]), x1, y1, x2 - x1, y2 - y1, float(row[5])))
        (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")

    metric = HOTA()
    per = {}
    for seq in SEQS:
        d = build_data(seq, "_headroom/" + tag)
        if d is not None:
            per[seq] = metric.eval_sequence(d)
    comb = metric.combine_sequences(per)
    return 100 * float(np.mean(comb["HOTA"])), {
        s: 100 * float(np.mean(per[s]["HOTA"])) for s in per}


def main():
    print("=" * 88)
    print("실험 6 [관문] -- 매칭 임계값의 여유. 기준선(맨 IoU)만 본다")
    print("=" * 88)
    print("이건 여유 측정이다. **최적값을 골라 '이겼다' 고 하지 않는다.**")
    print()

    rows = {}
    hdr = "%-8s%9s   " % ("thresh", "HOTA") + "".join(
        "%9s" % s.replace("-FRCNN", "").replace("MOT17-", "") for s in SEQS)
    print(hdr)
    print("-" * 88)
    for th in GRID:
        h, per = run(th)
        rows[th] = (h, per)
        line = "%-8.2f%9.3f   " % (th, h) + "".join(
            "%9.2f" % per.get(s, float("nan")) for s in SEQS)
        print(line + ("   <- 기본값" if abs(th - 0.8) < 1e-9 else ""))

    print()
    print("=" * 88)
    print("판정")
    print("=" * 88)
    base = rows[0.8][0]
    best_th = max(rows, key=lambda t: rows[t][0])
    gain = rows[best_th][0] - base
    print("  [A] 기준선 0.8 = %.3f,  전역 최적 %.2f = %.3f  ->  여유 %+.3f HOTA"
          % (base, best_th, rows[best_th][0], gain))
    if gain < 0.3:
        print("      => **여유 없음** (판정폭 0.3 미만). 레버 2 는 죽었다")
    else:
        print("      => 여유 있다. 다만 이건 검정집합 상한이지 성능 주장이 아니다")

    print()
    print("  [B] 시퀀스별 최적 임계값:")
    spread = []
    for s in SEQS:
        cand = {t: rows[t][1][s] for t in rows if s in rows[t][1]}
        if not cand:
            continue
        bt = max(cand, key=cand.get)
        spread.append(bt)
        print("      %-16s 최적 %.2f (HOTA %.2f)   0.8 일 때 %.2f   차이 %+.2f"
              % (s, bt, cand[bt], cand.get(0.8, float("nan")),
                 cand[bt] - cand.get(0.8, float("nan"))))
    if spread:
        uniq = sorted(set(spread))
        print()
        print("      최적값이 %d 가지: %s" % (len(uniq), uniq))
        if len(uniq) == 1:
            print("      => **장면마다 같다.** 적응 규칙이 존재할 자리가 없다")
        else:
            print("      => 장면마다 다르다. 적응 규칙이 **존재할 수** 있다")
            print("         (존재 가능성이지 존재 증명이 아니다. 규칙은 장면 고유")
            print("          관측량에서 나와야 하고 leave-one-sequence-out 이 필요하다)")


if __name__ == "__main__":
    main()
