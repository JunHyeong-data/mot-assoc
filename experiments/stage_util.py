# -*- coding: utf-8 -*-
"""**연관 단계를 판별하는 정본 함수.** 여기 하나에서만 나오게 한다.

## 왜 있는가 -- 감사에서 같은 실수가 세 번 나왔다 (2026-08-18)

`BYTETracker.get_dists` 는 **두 번** 불린다:

    byte_tracker.py:326   1단계  get_dists(strack_pool, detections)   thresh = match_thresh (0.8)
    byte_tracker.py:362   3단계  get_dists(unconfirmed, detections)   thresh = 0.7

exp11·exp12·exp15 가 셋 다 **1단계만 건드린다고 적어 놓고 3단계까지 건드렸다.**
exp11 의 기록으로 확인된다 -- val_half 2,652 프레임에서 **3,344 회 호출**
(프레임당 1.26 회).

## 판별 근거 (소스 확인)

`byte_tracker.py:306-313` 이 트랙을 이렇게 가른다:

    unconfirmed      = [t for t in self.tracked_stracks if not t.is_activated]
    tracked_stracks  = [t for t in self.tracked_stracks if     t.is_activated]
    strack_pool      = joint(tracked_stracks, self.lost_stracks)

**즉 1단계 트랙은 전부 `is_activated=True`, 3단계 트랙은 전부 `False` 다.**
겹치지 않으므로 판별이 정확하다. **그리고 가정하지 않고 검사한다** --
섞인 호출이 오면 예외를 던진다.
"""

STAGE1_THRESH_ATTR = "match_thresh"      # args 에서 읽는다
STAGE3_THRESH = 0.7                      # byte_tracker.py:363 하드코딩
STAGE2_THRESH = 0.5                      # byte_tracker.py:344 (get_dists 를 안 쓴다)


def which_stage(tracks):
    """1 (첫 연관) 또는 3 (unconfirmed). 섞이면 예외.

    빈 목록은 판별 불가이므로 None 을 준다 -- 호출자가 건너뛰면 된다.
    """
    if not len(tracks):
        return None
    act = [bool(getattr(t, "is_activated", False)) for t in tracks]
    if all(act):
        return 1
    if not any(act):
        return 3
    raise RuntimeError(
        "활성/비활성 트랙이 섞인 get_dists 호출 -- 단계 판별이 불가능하다. "
        "이 방법은 무효이므로 판정하지 말 것 (experiments/stage_util.py)")


def stage_thresh(args, stage):
    """그 단계에서 트래커가 실제로 쓰는 임계값."""
    if stage == 1:
        return float(getattr(args, STAGE1_THRESH_ATTR))
    if stage == 3:
        return STAGE3_THRESH
    raise ValueError("stage 는 1 또는 3 이다: %r" % (stage,))
