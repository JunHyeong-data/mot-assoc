# -*- coding: utf-8 -*-
"""실험 6 LOSO [1단계] -- 그리드 표를 만들고 **JSON 으로 남긴다**.

사전 선언은 `PREREG-loso.md`. 자료보다 먼저 커밋했다.

`headroom.py` 는 표를 print 만 하고 버렸다. `predictors.py` 는 그 값을
**손으로 옮겨 적었다**(`BEST`, `GAIN`). 재현 경로가 사람 손을 거친다.
그래서 여기서 표를 다시 만들고 JSON 으로 남긴다.

## 이 스크립트가 하는 검산 둘

**[0] 라벨 대조 (사전 선언한 관문).** 기존 8개 격자에 한정해 argmax 를
`predictors.BEST` 와 맞춰본다. **7/7 이 아니면 배관이 틀린 것이니 멈춘다.**
0.98 은 새 값이라 이 대조에서 뺀다.

**[0c] 재생 결정성.** 기존 `_headroom/` 트랙 파일의 해시를 먼저 뜨고,
같은 임계값을 다시 재생해 비교한다. 어제 만든 파일과 오늘 만든 파일이 다르면
재생이 결정적이지 않다는 뜻이다 (철회 10번 -- `hash()` 씨앗이 실행마다 다른
세계를 만든 적이 있다).

사용법:
    python experiments/exp06_levers/grid.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from headroom import run, OUTROOT                            # noqa: E402
from replay import SEQS                                      # noqa: E402

# 관문 [0] 대조용 -- **커밋 5b92db6 시점의 기록을 그대로 박아둔다.**
# predictors.py 에서 import 하지 않는다. 그 파일은 라벨이 갱신되므로
# (0.98 추가로 MOT17-13 이 옮겨갔다) 대조 상대가 같이 움직이면 관문이 무의미해진다.
BEST_OLD = {"MOT17-02-FRCNN": 0.85, "MOT17-04-FRCNN": 0.90, "MOT17-05-FRCNN": 0.75,
            "MOT17-09-FRCNN": 0.75, "MOT17-10-FRCNN": 0.95, "MOT17-11-FRCNN": 0.70,
            "MOT17-13-FRCNN": 0.95}

# 사전 선언한 그리드. 0.98 을 추가한 이유는 PREREG-loso.md 참고
# (MOT17-10, -13 의 최적이 상단 경계 0.95 에 붙어 있었다).
GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
OLD_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
OUT = Path("data/exp06/grid.json")


def tag(th):
    return "th%03d" % int(round(th * 100))


def digests(th):
    """주어진 임계값 폴더의 트랙 파일 해시. 없으면 빈 dict."""
    d = OUTROOT / tag(th)
    out = {}
    for seq in SEQS:
        f = d / ("%s.txt" % seq)
        if f.exists():
            out[seq] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def main():
    print("=" * 92)
    print("실험 6 LOSO [1단계] -- 그리드 표 생성. 사전 선언 PREREG-loso.md")
    print("=" * 92)
    print("그리드 %d개: %s" % (len(GRID), GRID))
    print()

    before = {th: digests(th) for th in GRID}

    table = {}          # th -> {seq: HOTA}
    combined = {}       # th -> 결합 HOTA (검출 수 가중)
    hdr = "%-8s%9s   " % ("thresh", "결합") + "".join(
        "%9s" % s.replace("-FRCNN", "").replace("MOT17-", "") for s in SEQS)
    print(hdr)
    print("-" * 92)
    for th in GRID:
        comb, per = run(th)
        table[th] = per
        combined[th] = comb
        line = "%-8.2f%9.3f   " % (th, comb) + "".join(
            "%9.2f" % per.get(s, float("nan")) for s in SEQS)
        print(line + ("   <- 기본값" if abs(th - 0.8) < 1e-9 else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "grid": GRID,
        "seqs": SEQS,
        "combined": {("%.2f" % t): combined[t] for t in GRID},
        "per_seq": {s: {("%.2f" % t): table[t].get(s) for t in GRID} for s in SEQS},
    }, indent=1))
    print()
    print("표를 %s 에 저장했다" % OUT)

    # ---------------- [0c] 재생 결정성 ----------------
    print()
    print("=" * 92)
    print("[0c] 재생 결정성 -- 어제 만든 트랙 파일과 오늘 만든 것이 같은가")
    print("=" * 92)
    checked = same = 0
    for th in GRID:
        old, new = before[th], digests(th)
        for seq in old:
            if seq in new:
                checked += 1
                same += int(old[seq] == new[seq])
    if checked == 0:
        print("  비교할 기존 파일이 없다 (첫 실행). 건너뛴다")
    else:
        print("  %d/%d 파일이 비트 단위로 동일" % (same, checked))
        if same != checked:
            print("  ** 재생이 결정적이지 않다. 원인을 찾기 전에는 판정하지 말 것 **")

    # ---------------- [0] 라벨 대조 (사전 선언한 관문) ----------------
    print()
    print("=" * 92)
    print("[0] 관문 -- 기존 8개 격자에서 argmax 가 predictors.BEST 와 맞는가")
    print("=" * 92)
    ok = 0
    for s in SEQS:
        cand = {t: table[t][s] for t in OLD_GRID if s in table[t]}
        got = max(cand, key=cand.get)
        exp = BEST_OLD.get(s)
        hit = abs(got - exp) < 1e-9
        ok += int(hit)
        print("  %-18s argmax %.2f   기록 %.2f   %s"
              % (s, got, exp, "OK" if hit else "** 불일치 **"))
    print("  => %d/%d 일치" % (ok, len(SEQS)))
    if ok != len(SEQS):
        print("  ** 관문 [0] 실패. 재생 경로가 틀렸다. 여기서 멈춘다 **")
        return 1

    # ---------------- 0.98 이 라벨을 바꿨는가 ----------------
    print()
    print("=" * 92)
    print("0.98 추가가 라벨을 바꿨는가 (사전 선언에서 예상한 변화)")
    print("=" * 92)
    moved = 0
    for s in SEQS:
        cand = {t: table[t][s] for t in GRID if s in table[t]}
        new_best = max(cand, key=cand.get)
        old_best = BEST_OLD[s]
        gain = cand[new_best] - cand[0.8]
        flag = ""
        if abs(new_best - old_best) > 1e-9:
            moved += 1
            flag = "   <- 바뀜 (%.2f 였다)" % old_best
        edge = "  [그리드 끝]" if new_best in (GRID[0], GRID[-1]) else ""
        print("  %-18s 최적 %.2f  HOTA %.2f  0.8 대비 %+.2f%s%s"
              % (s, new_best, cand[new_best], gain, flag, edge))
    print("  => %d개 시퀀스의 라벨이 바뀌었다" % moved)

    print()
    print("다음: python experiments/exp06_levers/loso.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
