# -*- coding: utf-8 -*-
"""
할당에 실제로 도달하는 성분만 남기고 다 걷어낸다.

비용행렬 C 에서 행상수와 열상수를 빼면 남는 것이 이중중심화 잔차

    R = C - (행평균) - (열평균) + (전체평균)

이다. R 은 C 에 어떤 a_i + b_j 를 더해도 변하지 않는다. 그리고 정방행렬에서는
R 만으로 푼 할당이 C 로 푼 할당과 같다. 즉 R 이 '할당이 보는 전부'다.

주장: 가림이 길어질수록 트랙 공분산이 커지고, R 이 0 으로 줄어든다.
      비용의 절대 크기는 오히려 커지는데도 그렇다. 할당을 가르는 신호만
      사라지는 것이다. 이것이 공분산 역설의 행렬판이다.

covariance_paradox.py 는 트랙 하나의 d^2 을 봤고, 여기서는 행렬 전체를 본다.
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(0)
N_DIM = 4
SEP = 1.2          # 후보 간격 / 트랙 표준편차
P0, Q = 4.0, 1.0   # covariance_paradox.py 와 같은 칼만 설정


def spd(scale=1.0, aniso=3.0):
    Qm, _ = np.linalg.qr(rng.normal(size=(N_DIM, N_DIM)))
    e = scale * np.exp(rng.uniform(-np.log(aniso), np.log(aniso), size=N_DIM))
    return Qm @ np.diag(e) @ Qm.T


def bhatta(eps, St, Sd):
    Sb = (St + Sd) / 2.0
    mahal = float(eps @ np.linalg.solve(Sb, eps)) / 8.0
    _, lb = np.linalg.slogdet(Sb)
    _, lt = np.linalg.slogdet(St)
    _, ld = np.linalg.slogdet(Sd)
    return mahal + 0.5 * lb - 0.25 * lt - 0.25 * ld


def residual(C):
    """이중중심화 잔차. 행상수/열상수를 모두 제거한 나머지."""
    return C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()


def cost_matrix(M, N, track_scale):
    tracks = [spd(track_scale) for _ in range(M)]
    dets = [spd(0.05, aniso=8.0) for _ in range(N)]
    mu_t = rng.normal(size=(M, N_DIM)) * SEP
    mu_d = rng.normal(size=(N, N_DIM)) * SEP
    return np.array([[bhatta(mu_t[i] - mu_d[j], tracks[i], dets[j])
                      for j in range(N)] for i in range(M)])


print("=" * 72)
print("[1] R 이 정말 '할당이 보는 전부' 인가")
print("=" * 72)
print("  정방 8x8 을 200회. C 로 푼 할당과 R 로 푼 할당을 비교한다.")
print()

same_assign, same_resid = 0, 0
for _ in range(200):
    C = cost_matrix(8, 8, 1.0)
    a, b = rng.normal(size=8) * 3, rng.normal(size=8) * 3
    same_assign += np.array_equal(linear_sum_assignment(C)[1],
                                  linear_sum_assignment(residual(C))[1])
    same_resid += np.allclose(residual(C), residual(C + a[:, None] + b[None, :]))

print(f"  C 로 푼 할당 == R 로 푼 할당      : {same_assign}/200")
print(f"  R(C) == R(C + a_i + b_j)         : {same_resid}/200")
print()
print("  -> 행상수/열상수를 아무리 더해도 R 은 그대로고, 할당도 그대로다.")
print("     그러니 '비용에 무엇을 넣었는가' 가 아니라 'R 에 무엇이 남는가' 를 봐야 한다.")

print()
print("=" * 72)
print("[2] 가림이 길어질수록 R 이 사라진다")
print("=" * 72)
print("  가려진 프레임 t 에서 트랙 공분산 배율 = P0 + t*Q  (P0=4, Q=1).")
print("  검출 공분산은 개체별로 다르게 유지한다. 30회 평균.")
print()
print(f"  {'가려진 프레임':>14}{'트랙배율':>10}{'|R|_max':>11}{'|R|_mean':>11}"
      f"{'|C|_mean':>11}{'R 비중':>10}")
print("  " + "-" * 68)

M, N, REP = 10, 15, 30
first_mean = None
for t in (0, 2, 5, 10, 30, 60, 100, 200, 500):
    scale = P0 + t * Q
    acc = []
    for _ in range(REP):
        C = cost_matrix(M, N, scale)
        R = residual(C)
        acc.append((np.abs(R).max(), np.abs(R).mean(), np.abs(C).mean()))
    rmax, rmean, cmean = np.array(acc).mean(0)
    first_mean = rmean if first_mean is None else first_mean
    print(f"  {t:>14}{scale:>10.0f}{rmax:>11.4f}{rmean:>11.4f}"
          f"{cmean:>11.4f}{rmean / cmean:>10.4f}")

print()
print(f"  -> |R|_mean 이 {first_mean:.4f} 에서 {rmean:.4f} 로 줄었다. "
      f"약 {first_mean / rmean:.0f}배.")
print("     같은 구간에서 |C|_mean 은 오히려 커진다. 비용은 커지는데")
print("     할당을 가르는 성분만 사라진다.")
print()
print("  해석: 가림이 길어지면 트랙 공분산이 검출 공분산을 압도한다.")
print("        그러면 Sb = (St+Sd)/2 가 St/2 로 수렴하고, 비용의 두 항이")
print("        모두 '트랙만의 함수 + 검출만의 함수' 로 분해되어 버린다.")
print("        분해된 것은 할당에 도달하지 못한다 ([1]).")
print()
print("  결론: 오래 가려진 트랙일수록 어떤 검출과 짝지어도 비용차가 없다.")
print("        d^2 이 작아지는 것(covariance_paradox.py)에 더해, 애초에")
print("        고를 근거 자체가 소멸한다. 이 구간에서는 검출 불확실성을")
print("        비용에 넣어도 효과가 없다 -- 넣을 자리가 남아있지 않다.")
