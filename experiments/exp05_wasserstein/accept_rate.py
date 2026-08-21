# -*- coding: utf-8 -*-
"""실험 5 -- 사전 등록 함정 3 의 직접 검증: **채택률**.

## 왜 별도 스크립트인가

README 함정 3 은 **"채택률(매칭 수 / 최적쌍 수)을 A 와 맞추도록 임계값을 보정"**
이라고 선언했다. 그런데 `wcost.solve_C` 가 실제로 맞추는 것은 **비용 중앙값**이다.
**같은 게 아니다.** 중앙값을 맞춰도 비용 분포의 모양이 다르면 `match_thresh=0.8`
아래로 들어오는 쌍의 비율이 달라질 수 있고, 그러면 조건 간 HOTA 차이가
**경로 효과가 아니라 임계값 효과**가 된다.

그래서 보정은 중앙값으로 하되, **채택률이 실제로 맞았는지를 사후에 직접 잰다.**
맞으면 함정 3 의 의도가 충족된 것이고, 안 맞으면 결과를 임계값 효과로 읽어야 한다.

## 무엇을 세는가

1단계 연관(`byte_tracker.py:326`, `thresh=match_thresh`)만 본다.
2단계는 `get_dists` 를 안 쓰고 `iou_distance` 를 직접 부르므로 조건과 무관하고,
3단계(unconfirmed, `:362`, `thresh=0.7`)는 조건이 갈리지만 규모가 작고
임계값도 다르다. **개입 지점과 판정 지점을 일치시킨다.**

    채택률 = (1단계 매칭 수) / (sum over frames of min(M, N))

분모는 **모든 쌍이 임계값을 통과했을 때의 매칭 수 상한**이다.

사용법:
    python experiments/exp05_wasserstein/accept_rate.py iou w_dfl w_size
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ultralytics.trackers.utils import matching          # noqa: E402

from replay import WTracker, Det, load, SEQS, ARMS, BASE  # noqa: E402


class RateTracker(WTracker):
    """1단계 비용행렬만 가로채 채택률을 센다. 트래킹 자체는 건드리지 않는다."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_match = 0
        self.n_cap = 0
        self._stage1_done = True     # update() 진입 때 False 로 내린다

    def update(self, results, img=None, feats=None):
        self._stage1_done = False
        return super().update(results, img, feats)

    def get_dists(self, tracks, detections):
        d = super().get_dists(tracks, detections)
        if not self._stage1_done:
            self._stage1_done = True                    # 첫 호출 = 1단계
            if len(tracks) and len(detections):
                m, _, _ = matching.linear_assignment(d, thresh=self.args.match_thresh)
                self.n_match += len(m)
                self.n_cap += min(len(tracks), len(detections))
        return d


def calibrate_C(arm):
    """replay.main 과 **같은 절차**로 C 를 푼다. 다르면 진단이 무의미하다."""
    if arm == "iou":
        return 1.0
    tmp = WTracker(SimpleNamespace(**BASE), arm, 1.0)
    c1 = load(SEQS[0], arm)
    if c1 is None:
        return None
    for f in range(1, min(c1["n_frames"], 80) + 1):
        m = c1["frame"] == f
        tmp.update(Det(c1["xyxy"][m], c1["conf"][m], np.zeros(int(m.sum())),
                       c1["sxx"][m], c1["syy"][m]))
    if not tmp.w2_log:
        return None
    from wcost import solve_C
    return solve_C(np.concatenate(tmp.w2_log), 0.5)


def measure(arm, C):
    """캐시된 모든 시퀀스에서 1단계 채택률을 잰다."""
    nm = ncap = 0
    for seq in SEQS:
        c = load(seq, arm)
        if c is None:
            continue
        tr = RateTracker(SimpleNamespace(**BASE), arm, C, frame_rate=30)
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            tr.update(Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                          c["sxx"][m], c["syy"][m]))
        nm += tr.n_match
        ncap += tr.n_cap
    return nm, ncap, 100.0 * nm / max(ncap, 1)


def solve_C_by_rate(arm, target_rate, C_hi, iters=12):
    """채택률을 target 에 맞추는 C 를 이분법으로 찾는다 (**강건성 확인용**).

    사전 등록한 보정 절차(비용 중앙값)를 **대체하지 않는다.** 중앙값 보정으로
    난 채택률이 기준선과 몇 %p 어긋나므로, "임계값 효과가 아니다" 라는 주장을
    한 번 더 받치기 위한 보조 조건다.

    C 가 커지면 cost = 1-exp(-r/C) 가 작아져 채택률이 오른다 -> 단조 증가.
    다만 매칭이 바뀌면 이후 트랙 구성도 바뀌므로 완전한 단조는 아니다.
    계단 함수라 이분법이 정확히 수렴하지 않을 수 있다 -- 근사면 충분하다.
    """
    lo, hi = C_hi * 1e-3, C_hi
    best = (None, None)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                      # 규모가 커서 로그 이분법
        _, _, r = measure(arm, mid)
        if best[0] is None or abs(r - target_rate) < abs(best[1] - target_rate):
            best = (mid, r)
        if r > target_rate:
            hi = mid
        else:
            lo = mid
    return best


def main():
    if "--solve" in sys.argv:
        arms = [a for a in sys.argv[1:] if a in ARMS] or ["w_dfl", "w_size"]
        print("=" * 74)
        print("채택률을 기준선(iou)에 정확히 맞추는 C 를 찾는다 -- 강건성 확인용")
        print("=" * 74)
        _, _, tgt = measure("iou", 1.0)
        print("목표 채택률 (iou) = %.2f%%" % tgt)
        for arm in arms:
            C0 = calibrate_C(arm)
            if C0 is None:
                print("%-10s (캐시 없음)" % arm)
                continue
            C, r = solve_C_by_rate(arm, tgt, C0 * 4.0)
            print("%-10s 중앙값보정 C = %.1f  ->  채택률맞춤 C = %.2f (채택률 %.2f%%)"
                  % (arm, C0, C, r))
        return

    arms = [a for a in sys.argv[1:] if a in ARMS] or ["iou", "w_dfl", "w_size"]
    print("=" * 74)
    print("실험 5 -- 사전 등록 함정 3 검증: 1단계 채택률")
    print("=" * 74)
    print("보정은 비용 중앙값으로 했다. 채택률이 실제로 맞았는지를 여기서 잰다.")
    print("조건 간 채택률이 비슷하면 HOTA 차이는 임계값 효과가 아니다.")
    print()
    hdr = "%-14s%10s%12s%14s%10s"
    print(hdr % ("조건", "C", "매칭 수", "최적쌍 상한", "채택률"))
    print("-" * 74)

    for arm in arms:
        C = calibrate_C(arm)
        if C is None:
            print("%-14s (캐시 없음)" % arm)
            continue
        nm, ncap, rate = measure(arm, C)
        print(hdr % (arm, "%.1f" % C, "%d" % nm, "%d" % ncap, "%.1f%%" % rate))


if __name__ == "__main__":
    main()
