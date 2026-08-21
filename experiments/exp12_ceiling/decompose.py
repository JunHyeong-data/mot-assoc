# -*- coding: utf-8 -*-
"""**여지 +3.122 중 얼마가 정말 '연관' 의 몫인가.**

## 왜 이걸 재는가 -- 원고가 스스로 제기한 우려다

5.5 절이 이렇게 적었다:

> *"검출 캐시가 동일함에도 DetA 가 +1.233 변화한다. 오라클 연관은 GT id 가 다른
> 검출과의 매칭을 거부하므로 출력 행이 12.4% 줄고, 제거된 것의 상당수가 오탐이다."*

그렇다면 **+3.122 의 일부는 검출 쪽 효과**다. 우려를 적어만 놓고 크기를 안 쟀다.
**재고 나서 적는다.**

## 어떻게 정확히 가르는가

HOTA 는 알파별로 `HOTA_a = sqrt(DetA_a * AssA_a)` 이고 그것을 알파에 대해 평균한다.
TrackEval 지표 클래스가 **알파별 배열을 그대로 준다.** 그래서 성분을 갈아 끼운다:

    기준선         mean_a sqrt(DetA_a^base * AssA_a^base)
    AssA 만 오라클  mean_a sqrt(DetA_a^base * AssA_a^orc)   <- **연관의 몫**
    둘 다 오라클    mean_a sqrt(DetA_a^orc  * AssA_a^orc)   <- 보고한 여지

**근사가 아니라 정확한 분해다.**

## 이건 판정이 아니다

새 평가지표가 아니라 **이미 보고한 수치의 감도 분석**이다. 방향도 안전하다 --
연관의 몫은 +3.122 보다 **작아질 수만 있고**, 그러면 우리 결론(여지가 있는데
우리 방법이 못 갔다)이 **더 보수적**이 된다.

사용법:
    python experiments/exp12_ceiling/decompose.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import evaluate as EV                                           # noqa: E402
from tracker.eval.collections.hota import HOTA                  # noqa: E402


def combined(root, arm):
    """`evaluate.py` 의 적재·전처리를 **그대로** 쓰되 트랙 뿌리만 갈아 끼운다."""
    old = EV.TRACKS
    EV.TRACKS = Path(root)
    try:
        metric = HOTA()
        per = {}
        for seq in EV.SEQS:
            d = EV.build_data(seq, arm)
            if d is None:
                return None
            per[seq] = metric.eval_sequence(d)
        return metric.combine_sequences(per)
    finally:
        EV.TRACKS = old


def main():
    print("=" * 92)
    print("여지 +3.122 중 **연관의 몫**은 얼마인가 -- 알파별 정확 분해")
    print("=" * 92)
    print("판정이 아니라 이미 보고한 수치의 감도 분석이다.")
    print("HOTA_a = sqrt(DetA_a * AssA_a) 이므로 성분을 갈아 끼워 볼 수 있다.")
    print()

    base = combined("data/exp12/tracks", "base")
    orc = combined("data/exp12/tracks", "oracle")
    if base is None or orc is None:
        print("트랙 파일이 없다. 먼저 돌려라: python experiments/exp12_ceiling/run.py")
        return 1

    def h(dete, assa):
        return 100.0 * float(np.mean(np.sqrt(np.asarray(dete) * np.asarray(assa))))

    # [사전 점검] 재현: 기준선과 오라클의 HOTA 가 기록과 맞는가
    hb_rep = 100.0 * float(np.mean(base["HOTA"]))
    ho_rep = 100.0 * float(np.mean(orc["HOTA"]))
    print("[사전 점검] 기록 재현")
    print("  기준선 HOTA %.3f  (기록 61.002)   오라클 %.3f  (기록 64.124)"
          % (hb_rep, ho_rep))
    if abs(hb_rep - 61.002) > 0.01 or abs(ho_rep - 64.124) > 0.01:
        print("  !! 기록과 어긋난다. **분해를 믿지 말 것.**")
        return 1
    print("  일치. 분해로 넘어간다.")
    print()

    hb = h(base["DetA"], base["AssA"])
    ha = h(base["DetA"], orc["AssA"])
    hf = h(orc["DetA"], orc["AssA"])

    # [사전 점검] 갈아 끼우기가 항등식을 지키는가
    if abs(hb - hb_rep) > 0.01 or abs(hf - ho_rep) > 0.01:
        print("  !! sqrt(DetA*AssA) 재조립이 HOTA 와 안 맞는다 (%.3f/%.3f).**중단**"
              % (hb - hb_rep, hf - ho_rep))
        return 1

    print("  %-30s %8s %10s" % ("조건", "HOTA", "기준선 대비"))
    print("  " + "-" * 52)
    print("  %-30s %8.3f" % ("기준선", hb))
    print("  %-30s %8.3f %+10.3f" % ("AssA 만 오라클 (연관의 몫)", ha, ha - hb))
    print("  %-30s %8.3f %+10.3f" % ("둘 다 오라클 (보고한 여지)", hf, hf - hb))
    print()
    share = 100.0 * (ha - hb) / (hf - hb)
    print("  **연관의 몫 = %+.3f HOTA (보고한 여지의 %.1f%%)**" % (ha - hb, share))
    print("  나머지 %.1f%% 는 오라클이 오탐 행을 거부해 DetA 가 오른 몫이다."
          % (100 - share))
    print()
    print("  원고의 +3.122 는 **연관만의 상한이 아니다.** 연관만으로 얻을 수 있는")
    print("  것은 %+.3f 이고, 우리 네 경로(-0.21 ~ -8.90)는 전부 그 아래다."
          % (ha - hb))
    print("  **결론은 그대로이고 상한이 더 보수적이 된다.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
