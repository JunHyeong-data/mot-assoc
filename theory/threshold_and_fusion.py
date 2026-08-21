# -*- coding: utf-8 -*-
"""
실제 트래커는 순수 헝가리안을 돌리지 않는다. 그래서 불변성 논증이 어디까지 가나.

두 가지를 시험한다.

(A) ultralytics 는 헝가리안을 푼 뒤 cost <= match_thresh 인 쌍만 남긴다.
    임계값이 붙으면 행상수 a_i 는 argmin 을 안 바꿔도 '임계 통과 여부' 를 바꾼다.
    => 분리되는 항의 기여가 0 이라는 명제가 M<=N 에서도 깨진다.

(B) ByteTrack 은 이미 검출 신뢰도를 비용에 넣고 있다 (fuse_score, 기본값 True).
    단 곱으로 넣는다: cost = 1 - IoU_ij * s_j.
    더하기로 넣으면 (cost = (1-IoU_ij) + k(1-s_j)) 순수 열상수가 되어
    잔차에 아무것도 남기지 못한다. 이 차이가 설계 제약이 된다.

대조한 구현 (ultralytics 8.4.46):
  trackers/utils/matching.py  linear_assignment(), fuse_score()
  cfg/trackers/bytetrack.yaml  match_thresh: 0.8, fuse_score: True
  * 이 구현에는 마할라노비스 게이팅이 없다. kalman_filter.gating_distance 는
    정의만 되어 있고 byte_tracker/bot_sort 어디서도 호출되지 않는다.
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(0)

THRESH = 0.8       # bytetrack.yaml match_thresh
LAM = 0.3          # 더하기 방식의 신뢰도 가중치
RUNS = 500


def iou_matrix(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aA = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aA[:, None] + aB[None, :] - inter + 1e-12)


def scene(M, N, jitter=12.0):
    """트랙 M 개. 검출은 트랙을 흔든 것 + 여분(신규/오검출)."""
    cx, cy = rng.uniform(100, 1800, M), rng.uniform(100, 980, M)
    w = rng.uniform(40, 110, M)
    h = w * rng.uniform(2.0, 3.0)
    T = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    k = min(M, N)
    D = T[:k] + rng.normal(0, jitter, (k, 4))
    if N > k:
        ex, ey = rng.uniform(100, 1800, N - k), rng.uniform(100, 980, N - k)
        ew = rng.uniform(40, 110, N - k)
        eh = ew * 2.5
        D = np.vstack([D, np.stack([ex - ew / 2, ey - eh / 2,
                                    ex + ew / 2, ey + eh / 2], 1)])
    return T, D


def ul_match(cost, thresh):
    """ultralytics matching.linear_assignment 의 scipy 경로를 그대로 옮긴 것."""
    x, y = linear_sum_assignment(cost)
    return frozenset((int(x[i]), int(y[i])) for i in range(len(x))
                     if cost[x[i], y[i]] <= thresh)


def residual(C):
    return C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()


print("=" * 72)
print("[A] 임계값이 붙으면 행상수 불변성이 M<=N 에서도 깨진다")
print("=" * 72)
print(f"  비용 = 1 - IoU. 행상수 a_i ~ U(0, 0.15) 를 더하고 할당 쌍 집합을 비교한다.")
print(f"  {RUNS} 회 중 불변인 시행 수.")
print()
print(f"  {'M x N':>9}{'임계 없음':>16}{'임계 0.8':>16}")
print("  " + "-" * 42)
for M, N in [(10, 15), (10, 10), (15, 10)]:
    no_th = th = 0
    for _ in range(RUNS):
        T, D = scene(M, N)
        C = 1 - iou_matrix(T, D)
        Ca = C + rng.uniform(0, 0.15, M)[:, None]
        no_th += ul_match(C, np.inf) == ul_match(Ca, np.inf)
        th += ul_match(C, THRESH) == ul_match(Ca, THRESH)
    print(f"  {f'{M} x {N}':>9}{f'{no_th}/{RUNS}':>16}{f'{th}/{RUNS}':>16}")

print()
print("  -> 임계 없음 열은 assignment_invariance.py 결과를 재확인한다")
print("     (M>N 인 15x10 만 깨진다).")
print("  -> 임계값을 붙이면 세 모양 모두 깨진다. argmin 은 그대로인데")
print("     '이 쌍을 채택할 것인가' 가 행상수만큼 밀린 비용으로 판정되기 때문이다.")
print()
print("  => '분리되면 기여 0' 은 임계값 없는 순수 헝가리안에서만 참이다.")
print("     실제 트래커에는 항상 임계값이 있다. 이 연구의 이론적 뼈대는")
print("     그대로 쓰면 안 되고, 임계값을 포함한 형태로 다시 세워야 한다.")

print()
print("=" * 72)
print("[B] 신뢰도를 곱으로 넣나 합으로 넣나 -- ByteTrack 은 이미 곱으로 넣는다")
print("=" * 72)
print("  곱 (fuse_score, 기본값 True) : cost = 1 - IoU_ij * s_j")
print("  합 (흔한 대안)               : cost = (1 - IoU_ij) + k(1 - s_j)")
print("  신뢰도 미사용 비용 대비 잔차가 얼마나 변하는가.")
print()
print("  **정방(M=N)에서 잰다.** 합 형태는 순수 '열'상수인데 열상수 불변성은")
print("  M >= N 조건부다 ([A], assignment_invariance.py). M<N 에서 재면 할당 변경이")
print("  임계값 때문인지 모양 때문인지 섞인다. 실제로 12x18 에서는 임계값 없이도")
print("  20/500 이 바뀌었다. 그래서 모양을 M=N 으로 두고 임계값만 남긴다.")
print("  (2026-08-17 정정. 그전에는 12x18 로 재고 전부 임계값 탓으로 적었다.)")
print()
print(f"  {'':20}{'|dR| 평균':>14}{'변경(임계 없음)':>18}{'변경(임계 0.8)':>18}")
print("  " + "-" * 72)

M, N = 15, 15
dr_mul, dr_add = [], []
ch_mul, ch_add, nt_mul, nt_add = 0, 0, 0, 0
for _ in range(RUNS):
    T, D = scene(M, N)
    iou = iou_matrix(T, D)
    s = rng.uniform(0.3, 1.0, N)
    base = 1 - iou
    mul = 1 - iou * s[None, :]
    add = base + LAM * (1 - s)[None, :]
    dr_mul.append(np.abs(residual(mul) - residual(base)).mean())
    dr_add.append(np.abs(residual(add) - residual(base)).mean())
    b_inf, b_th = ul_match(base, np.inf), ul_match(base, THRESH)
    nt_mul += ul_match(mul, np.inf) != b_inf
    nt_add += ul_match(add, np.inf) != b_inf
    ch_mul += ul_match(mul, THRESH) != b_th
    ch_add += ul_match(add, THRESH) != b_th

print(f"  {'곱 (fuse_score)':20}{np.mean(dr_mul):>14.2e}"
      f"{f'{nt_mul}/{RUNS}':>18}{f'{ch_mul}/{RUNS}':>18}")
print(f"  {'합 (순수 열상수)':20}{np.mean(dr_add):>14.2e}"
      f"{f'{nt_add}/{RUNS}':>18}{f'{ch_add}/{RUNS}':>18}")

print()
print("  -> 합으로 넣으면 잔차 변화가 2e-16, 기계정밀도다. 임계값이 없으면")
print("     할당이 단 한 건도 안 바뀐다 -- 할당 논리에 아무것도 전달하지 못한다.")
print("     그런데 임계값을 붙이면 바뀐다. **오직 임계값 경로로만** 바뀐다.")
print("     즉 '효과가 있었다' 를 보고 '정보가 전달됐다' 고 결론내면 안 된다.")
print("  -> 곱으로 넣으면 잔차 자체가 변한다 (2e-16 대 3e-2, 열네 자릿수 차이).")
print("     신뢰도가 IoU 와 상호작용해 비분리 성분을 만든다.")
print("     단 잔차가 변한다고 할당이 자주 뒤집히지는 않는다 -- 임계값 없이는")
print("     1/500 뿐이다. 도달 여부와 뒤집힘 빈도는 다른 문제다. 여기서 보이는")
print("     것은 '경로가 열려 있다' 이지 '많이 바뀐다' 가 아니다.")
print()
print("설계 제약: 검출별 불확실성을 스칼라로 '더하는' 형태는 원리적으로 못 쓴다.")
print("          트랙과 상호작용하는 형태여야 한다. ByteTrack 이 곱을 쓴 것은")
print("          (논문에 근거는 없지만) 이 제약과 맞는 선택이었다.")
