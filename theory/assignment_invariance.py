# -*- coding: utf-8 -*-
"""
할당(Hungarian)이 실제로 사용하는 정보는 무엇인가.

주장: 비용행렬 c_ij 에 '행에만 의존하는 항' a_i 와 '열에만 의존하는 항' b_j 를
      더해도 최적 할당은 바뀌지 않는다. 따라서 트랙별/검출별로 분리되는(separable)
      정보는 할당 결과에 도달하지 못한다.

따름결과: Bhattacharyya 거리의 공분산항에서 검출 공분산 Σ_d 가 모든 검출에 대해
          동일한 상수이면, 그 항은 순수한 행상수가 되어 기여가 정확히 0 이다.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)
N_DIM = 4


def spd(scale=1.0, aniso=3.0):
    Q, _ = np.linalg.qr(rng.normal(size=(N_DIM, N_DIM)))
    e = scale * np.exp(rng.uniform(-np.log(aniso), np.log(aniso), size=N_DIM))
    return Q @ np.diag(e) @ Q.T


def bhatta(eps, St, Sd):
    """Bhattacharyya 거리를 두 항으로 분해해 반환."""
    Sb = (St + Sd) / 2.0
    mahal = float(eps @ np.linalg.solve(Sb, eps)) / 8.0
    _, lb = np.linalg.slogdet(Sb)
    _, lt = np.linalg.slogdet(St)
    _, ld = np.linalg.slogdet(Sd)
    return mahal, 0.5 * lb - 0.25 * lt - 0.25 * ld


print("=" * 72)
print("[1] 행상수 / 열상수는 할당을 바꾸지 않는다")
print("=" * 72)
for M, N in [(8, 8), (8, 14)]:
    C = rng.normal(size=(M, N)) * 5
    a, b = rng.normal(size=M) * 3, rng.normal(size=N) * 3
    base = linear_sum_assignment(C)
    same = lambda X: np.array_equal(linear_sum_assignment(X)[1], base[1])
    print(f"  {M}x{N}  행상수 불변={same(C + a[:, None])}   "
          f"열상수 불변={same(C + b[None, :])}   "
          f"둘다={same(C + a[:, None] + b[None, :])}")
print("  -> 정방이면 둘 다 불변. M<N(검출이 더 많은 실제 상황)이면 열상수는 깨진다.")

print()
print("=" * 72)
print("[2] 상수 Sigma_d 이면 공분산항의 기여가 정확히 0 이다")
print("=" * 72)
print("  게이트 안(트랙 불확실성 대비 후보 간격이 작은 상황)에서 100회 반복.")
print()
M, N, TRIALS = 10, 15, 100
SEP = 1.2   # 후보 분리 / 트랙 표준편차. 가림 중에는 이 값이 작아진다.

rows = {"상수 Sigma_d  ": [], "개체별 Sigma_d": []}
for _ in range(TRIALS):
    tracks = [spd(1.0) for _ in range(M)]
    mu_t = rng.normal(size=(M, N_DIM)) * SEP
    mu_d = rng.normal(size=(N, N_DIM)) * SEP
    Sd_const = spd(0.05)
    variants = {"상수 Sigma_d  ": [Sd_const] * N,
                "개체별 Sigma_d": [spd(0.05, aniso=8.0) for _ in range(N)]}
    for label, Sds in variants.items():
        A = np.zeros((M, N)); B = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                A[i, j], B[i, j] = bhatta(mu_t[i] - mu_d[j], tracks[i], Sds[j])
        ref = linear_sum_assignment(A)[1]
        rows[label].append((
            np.abs(B - B.mean(axis=1, keepdims=True)).max(),          # 행내 편차
            int((linear_sum_assignment(A + B)[1] != ref).sum()),      # 바뀐 할당 수
        ))

print(f"  {'':16}{'행내 최대편차':>16}{'할당 변경/시행':>18}{'변경된 트랙 총수':>18}")
for label, v in rows.items():
    v = np.array(v, dtype=float)
    print(f"  {label:16}{v[:, 0].max():>16.2e}"
          f"{f'{int((v[:, 1] > 0).sum())}/{TRIALS}':>18}{int(v[:, 1].sum()):>18}")

print()
print("  -> 상수 Sigma_d  : 행내 편차가 기계정밀도(0). 할당 변경이 단 한 건도 없다.")
print("     이것은 우연이 아니라 증명된 사실이다 (순수 행상수 -> Hungarian 불변).")
print("  -> 개체별 Sigma_d: 편차가 살아있고, 실제로 할당이 바뀐다.")
print()
print("결론: 검출 불확실성이 할당에 전달되기 위한 필요조건은")
print("      '트랙 x 검출로 분리되지 않는 성분이 존재하는 것'이다.")
