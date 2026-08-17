# -*- coding: utf-8 -*-
"""실험 8 -- **2단계를 살리면 기준선이 얼마나 움직이는가.**

사전 선언은 `PREREG.md`. 자료보다 먼저 커밋했다 (`0bc1362`).
물음은 자체 심사 M3 이 못박았다 (`notes/self_review.md`).

## 고치는 것 -- 한 줄

`byte_tracker.py:341-344` 의 2단계는 `fuse_score` 를 걸고 `thresh=0.5` 로 자른다.
2단계 검출은 `score < track_high_thresh(0.25)` 이므로
`cost >= 1 - IoU*s >= 0.75 > 0.5`. **어떤 쌍도 통과 못 한다.**
참조 구현은 2단계에서 `fuse_score` 를 안 건다. 그것만 맞춘다.

**`fuse_score=False` 를 전역으로 주는 것과 다르다** -- 그건 1단계까지 바꾸므로
개입이 둘이 되어 교란된다. 우리는 2단계만 고친다.

## 어떻게 -- site-packages 를 안 고치고, update() 를 베끼지도 않는다

`update()` 를 통째로 베끼면 전사 오류 위험이 있다. 대신 `matching` 의 두 함수를
감싼다.

  fuse_score      점수 구간으로 단계를 판별한다 (1/3단계 >= 0.25, 2단계 < 0.25)
  linear_assignment  thresh 로 단계를 판별한다 (1단계 0.8, 2단계 0.5, 3단계 0.7)

**둘 다 가정하지 않고 검사한다.** 점수 구간이 섞인 호출이 한 번이라도 있으면
예외를 던지고, `match_thresh` 가 0.5 면(2단계와 충돌) 시작 전에 멈춘다.

사용법:
    python experiments/exp08_stage2/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                # noqa: E402
from replay import WTracker, Det, load, SEQS, BASE             # noqa: E402
from evaluate import build_data, TRACKS                        # noqa: E402
from tracker.eval.collections.hota import HOTA                 # noqa: E402
from tracker.eval.collections.clear import CLEAR               # noqa: E402

ARM = "iou"
OUTROOT = TRACKS / "_stage2"
STAGE2_THRESH = 0.5                     # byte_tracker.py:344 하드코딩
HIGH = BASE["track_high_thresh"]        # 0.25

ORIG_FUSE = matching.fuse_score
ORIG_LSA = matching.linear_assignment

# 계측기. run() 마다 초기화한다
ST = {}


def reset(fix):
    ST.clear()
    ST.update(fix=fix, mixed=0, s2_calls=0, s2_pairs=0, s2_matches=0, s1_calls=0)


def fuse_patched(cost_matrix, detections):
    """2단계면 fuse 를 걸지 않는다. 단계 판별은 **검사한다.**"""
    if len(detections):
        s = np.asarray([d.score for d in detections], dtype=float)
        low, high = bool((s < HIGH).all()), bool((s >= HIGH).all())
        if not (low or high):
            ST["mixed"] += 1
            raise RuntimeError(
                "점수 구간이 섞인 fuse_score 호출 -- 단계 판별이 불가능하다. "
                "이 방법은 무효이므로 판정하지 말 것 (사전 선언 관문 [0c])")
        if low:
            ST["s2_calls"] += 1
            if ST["fix"]:
                return cost_matrix              # <- 참조 구현과 같게
        else:
            ST["s1_calls"] += 1
    return ORIG_FUSE(cost_matrix, detections)


def lsa_patched(cost_matrix, thresh, use_lap=True):
    """2단계 채택 건수를 센다. 계산은 손대지 않는다."""
    out = ORIG_LSA(cost_matrix, thresh, use_lap)
    if abs(thresh - STAGE2_THRESH) < 1e-9:
        ST["s2_pairs"] += int(np.asarray(cost_matrix).size)
        ST["s2_matches"] += len(out[0])
    return out


def run(fix):
    """갈래 하나를 재생하고 HOTA/CLEAR 를 낸다."""
    reset(fix)
    tag = "alive" if fix else "dead"
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    per_seq_s2 = {}

    for seq in SEQS:
        c = load(seq, ARM)
        if c is None:
            continue
        before = ST["s2_matches"]
        tr = WTracker(SimpleNamespace(**BASE), ARM, 1.0, frame_rate=30)
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
        per_seq_s2[seq] = ST["s2_matches"] - before

    hota, clear = HOTA(), CLEAR()
    ph, pc = {}, {}
    for seq in SEQS:
        d = build_data(seq, "_stage2/" + tag)
        if d is None:
            raise SystemExit("트랙 파일이 없다: %s (%s)" % (seq, tag))
        ph[seq] = hota.eval_sequence(d)
        pc[seq] = clear.eval_sequence(d)
    ch = hota.combine_sequences(ph)
    cc = clear.combine_sequences(pc)
    return dict(
        tag=tag,
        HOTA=100 * float(np.mean(ch["HOTA"])),
        DetA=100 * float(np.mean(ch["DetA"])),
        AssA=100 * float(np.mean(ch["AssA"])),
        IDSW=float(cc["IDSW"]),
        MOTA=100 * float(cc["MOTA"]),
        per_hota={s: 100 * float(np.mean(ph[s]["HOTA"])) for s in ph},
        per_s2=per_seq_s2,
        s2_matches=ST["s2_matches"], s2_calls=ST["s2_calls"],
        s1_calls=ST["s1_calls"], mixed=ST["mixed"])


def main():
    print("=" * 92)
    print("실험 8 -- 2단계를 살리면 기준선이 얼마나 움직이는가")
    print("=" * 92)
    print("사전 선언 PREREG.md (커밋 0bc1362, 자료보다 먼저)")
    print()

    if abs(BASE["match_thresh"] - STAGE2_THRESH) < 1e-9:
        print("** match_thresh 가 2단계 임계값(0.5)과 같다. 단계 판별 불가. 멈춘다 **")
        return 1
    print("  1단계 thresh = %.2f,  2단계 = %.2f,  3단계 = 0.70  ->  판별 가능"
          % (BASE["match_thresh"], STAGE2_THRESH))
    print("  2단계 검출 구간: score < %.2f" % HIGH)
    print()

    matching.fuse_score = fuse_patched
    matching.linear_assignment = lsa_patched
    try:
        dead = run(False)
        alive = run(True)
    finally:
        matching.fuse_score = ORIG_FUSE
        matching.linear_assignment = ORIG_LSA

    # ---------------- 관문 ----------------
    print("=" * 92)
    print("사전 선언한 관문")
    print("=" * 92)
    ok = True

    g0a = dead["s2_matches"] == 0
    ok &= g0a
    print("  [0a] 결함 재현 -- 고치기 **전** 2단계 채택 = %d  %s"
          % (dead["s2_matches"], "OK (정확히 0)" if g0a else "** 실패 **"))

    g0b = alive["s2_matches"] > 0
    ok &= g0b
    print("  [0b] 수정 확인 -- 고친 **후** 2단계 채택 = %d  %s"
          % (alive["s2_matches"], "OK" if g0b else "** 실패: 수정이 안 먹었다 **"))

    g0c = dead["mixed"] == 0 and alive["mixed"] == 0
    ok &= g0c
    print("  [0c] 단계 판별 -- 점수 구간이 섞인 호출 = %d  %s"
          % (dead["mixed"] + alive["mixed"], "OK" if g0c else "** 실패 **"))

    g0d = abs(dead["HOTA"] - 61.002) < 0.01
    ok &= g0d
    print("  [0d] 재현 -- 고치기 전 HOTA %.3f vs 기록 61.002  %s"
          % (dead["HOTA"], "OK" if g0d else "** 불일치 **"))
    # 이 수는 **같지 않은 것이 정상이다.** 1단계 호출은 매 프레임 같은 검출로
    # 도는데, 3단계(unconfirmed) 호출은 1단계에서 남은 검출을 받으므로 트랙
    # 상태가 갈리면 함께 갈린다. 2단계가 트랙을 되살리면 그 뒤 프레임의 풀이
    # 달라진다. 즉 이 차이는 **개입이 실제로 퍼졌다는 표시**이지 결함이 아니다.
    print("       (고신뢰 fuse 호출 %d / %d 건 -- 갈라지는 것이 정상. 2단계가"
          % (dead["s1_calls"], alive["s1_calls"]))
    print("        트랙을 되살리면 뒤 프레임의 3단계 잔여 검출이 달라진다)")

    if not ok:
        print()
        print("  ** 관문 실패. 판정하지 않는다 (사전 선언) **")
        return 1

    # ---------------- 종말점 ----------------
    print()
    print("=" * 92)
    print("사전 선언한 종말점")
    print("=" * 92)
    print("%-22s %8s %8s %8s %8s %8s" % ("갈래", "HOTA", "DetA", "AssA", "IDSW", "MOTA"))
    print("-" * 92)
    for r in (dead, alive):
        name = "2단계 죽음 (기존)" if r["tag"] == "dead" else "2단계 **살림**"
        print("%-22s %8.3f %8.3f %8.3f %8.0f %8.3f"
              % (name, r["HOTA"], r["DetA"], r["AssA"], r["IDSW"], r["MOTA"]))

    d1 = alive["HOTA"] - dead["HOTA"]
    unw = float(np.mean([alive["per_hota"][s] - dead["per_hota"][s]
                         for s in dead["per_hota"]]))
    print()
    print("  [1] 주 종말점  HOTA 차이 = %+.3f  (가중 없는 시퀀스 평균 %+.3f)" % (d1, unw))
    print("  [2] AssA %+.3f,  DetA %+.3f,  IDSW %+.0f,  MOTA %+.3f"
          % (alive["AssA"] - dead["AssA"], alive["DetA"] - dead["DetA"],
             alive["IDSW"] - dead["IDSW"], alive["MOTA"] - dead["MOTA"]))

    print()
    print("  [3] 시퀀스별 -- 2단계 채택 건수와 HOTA 변화")
    print("      %-18s %10s %10s %10s %9s" % ("시퀀스", "2단계채택", "죽음", "살림", "차이"))
    print("      " + "-" * 62)
    for s in SEQS:
        if s not in dead["per_hota"]:
            continue
        print("      %-18s %10d %10.2f %10.2f %+9.2f"
              % (s.replace("-FRCNN", ""), alive["per_s2"][s],
                 dead["per_hota"][s], alive["per_hota"][s],
                 alive["per_hota"][s] - dead["per_hota"][s]))

    # ---------------- 읽는 법 (사전 선언) ----------------
    print()
    print("=" * 92)
    print("사전 선언한 읽는 법을 그대로 적용한다")
    print("=" * 92)
    a = abs(d1)
    if a < 0.3:
        print("  |차이| < 0.3 => 2단계가 거의 일을 안 한다.")
        print("     **기존 결과의 한계 서술이 가벼워진다.**")
    elif a <= 2.0:
        print("  0.3 ~ 2.0 => 기준선이 움직인다.")
        print("     **한계로 명시하고 주요 대조를 재실행해야 한다.**")
    else:
        print("  > 2.0 => **기존 추적 수준 결과를 전부 다시 재야 한다.**")
    print()
    print("  어느 쪽이든 exp05/exp06 의 **갈래 간 비교 자체는 무효가 아니다** --")
    print("  같은 트래커 안의 비교였다. 바뀌는 것은 '**ByteTrack 기준선**' 이라는")
    print("  이름을 쓸 수 있는가이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
