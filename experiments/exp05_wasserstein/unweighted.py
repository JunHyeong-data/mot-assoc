# -*- coding: utf-8 -*-
"""**가중 없는 시퀀스 평균과 부호검정.** 원고 한계 (9) 를 줄이려는 것이다.

## 왜

CLAUDE.md 규칙 5 는 "가중과 비가중을 둘 다 보고한다" 이고, 그 규칙이 있는 이유는
**이 연구에서 가중 여부로 부호가 두 번 뒤집혔기 때문**이다 (exp05, exp06).

그런데 원고 표 5.2 의 네 경로는 **가중 값만** 보고하고 있다. 한계 (9) 에
*"효과가 뒤집힐 만큼 작지 않다고 판단했으나 이는 측정이 아니다"* 라고 적어 뒀다.

**적어도 잴 수 있는 것은 재서 그 문장을 줄인다.**

## 무엇을 잴 수 있고 무엇은 못 재는가

`data/exp06/exp05_perseq.json` 에 exp05 조건들의 **시퀀스별 HOTA** 가 남아 있다.
따라서 **거리 함수 경로**는 비가중 평균과 부호검정을 지금 낼 수 있다.

칼만(exp02)·게이팅(exp03) 은 트랙 출력이 `data/` 에 없다. 다시 돌려야 하고
그건 이 스크립트의 범위 밖이다 -- **못 잰다고 적는다.**

## 이건 판정이 아니다

새 종말점이 아니라 **이미 보고한 수치를 다른 집계로 다시 보는 것**이다.
부호가 같으면 결론이 집계 방식에 의존하지 않는다는 확인이고, 다르면
**규칙 5 가 예고한 바로 그 상황**이므로 원고에 그대로 적는다.

사용법:
    python experiments/exp05_wasserstein/unweighted.py
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("data/exp06/exp05_perseq.json")

# (원고 이름, 조건 키, 원고에 적힌 가중 값)
ARMS = [
    ("거리 함수 (검출기 sigma, DFL)", "wn_dfl_rate", -8.90),
    ("거리 함수 (박스 크기)", "wn_size_rate", -4.98),
    ("(참고) 정규화 전, DFL", "w_dfl_rate", -7.96),
]


def sign_two_sided(d):
    """양측 부호검정. 동률은 뺀다 (exp10 verify_1f 와 같은 처리)."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    win = int((nz > 0).sum())
    k = max(win, n - win)
    p = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return win, n, min(p, 1.0)


def main():
    if not SRC.exists():
        print("없다: %s   먼저 exp05 evaluate.py 를 돌려라" % SRC)
        return 1
    d = json.loads(SRC.read_text())
    base = d["iou"]
    seqs = sorted(base)

    print("=" * 92)
    print("가중 없는 시퀀스 평균 -- 원고 한계 (9) 를 줄인다")
    print("=" * 92)
    print("판정이 아니라 **같은 수치를 다른 집계로 다시 보는 것**이다.")
    print("가중 = TrackEval combine_sequences (검출 수 가중). 비가중 = 시퀀스 단순 평균.")
    print()

    for name, key, weighted in ARMS:
        if key not in d:
            print("  %-30s (조건 없음)" % name)
            continue
        diffs = [d[key][s] - base[s] for s in seqs]
        unw = float(np.mean(diffs))
        w, n, p = sign_two_sided(diffs)
        same = "같다" if (unw < 0) == (weighted < 0) else "**뒤집힌다**"
        print("  %s" % name)
        print("    가중 %+7.2f   비가중 %+7.2f   부호 %s" % (weighted, unw, same))
        print("    시퀀스별: " + "  ".join(
            "%s %+.2f" % (s.replace("MOT17-", "").replace("-FRCNN", ""), v)
            for s, v in zip(seqs, diffs)))
        print("    기준선보다 나쁜 시퀀스 %d/%d,  양측 부호검정 p = %.4f"
              % (n - w, n, p))
        print()

    print("=" * 92)
    print("못 잰 것 -- 정직하게 적는다")
    print("=" * 92)
    print("  칼만 R (exp02), 게이팅 (exp03) 은 트랙 출력이 data/ 에 없다.")
    print("  비가중 값을 내려면 두 실험을 다시 돌려야 한다. **한계로 남는다.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
