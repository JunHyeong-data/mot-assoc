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


def pairs(X):
    """최적 할당을 (트랙, 검출) 쌍의 집합으로. M>N 이면 어느 행이 빠지는지가 여기 담긴다."""
    return set(zip(*(v.tolist() for v in linear_sum_assignment(X))))


print("=" * 72)
print("[1] 행상수 / 열상수는 할당을 바꾸지 않는다 -- 단, 조건이 붙는다")
print("=" * 72)
print("  무작위 비용행렬 1000회. 할당 쌍 집합이 원래와 같은 시행의 비율.")
print()
print(f"  {'M x N':>10}{'행상수 불변':>16}{'열상수 불변':>16}{'둘다':>14}")
print("  " + "-" * 56)

rng1 = np.random.default_rng(1)
T1 = 1000
for M, N in [(8, 8), (8, 14), (14, 8)]:
    hit = np.zeros(3, dtype=int)
    for _ in range(T1):
        C = rng1.normal(size=(M, N)) * 5
        a, b = rng1.normal(size=M) * 3, rng1.normal(size=N) * 3
        base = pairs(C)
        hit += [pairs(C + a[:, None]) == base,
                pairs(C + b[None, :]) == base,
                pairs(C + a[:, None] + b[None, :]) == base]
    print(f"  {f'{M} x {N}':>10}{f'{hit[0]}/{T1}':>16}"
          f"{f'{hit[1]}/{T1}':>16}{f'{hit[2]}/{T1}':>14}")

print()
print("  정방 (M=N)              : 둘 다 불변.")
print("  M < N (검출이 더 많다)  : 모든 트랙이 배정된다 -> 행상수 불변, 열상수 깨짐.")
print("  M > N (가림으로 검출 줄음): 트랙 중 N 개만 배정된다. 어느 트랙을 버릴지가")
print("                            행상수에 달려 있다 -> 행상수가 깨진다.")
print()
print("  즉 '행상수 불변' 은 모든 행이 배정될 때(M <= N)만 성립하는 조건부 명제다.")

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
print("=" * 72)
print("[3] 그런데 M > N 이면 상수 Sigma_d 여도 할당이 바뀐다")
print("=" * 72)
print("  [2] 는 M<N 에서 쟀다. 가림 중에는 검출이 사라져 M>N 이 되는 것이 정상이다.")
print("  같은 실험을 모양만 바꿔 다시 돌린다. Sigma_d 는 내내 상수(= 순수 행상수).")
print()
print(f"  {'M x N':>10}{'행내 최대편차':>18}{'할당 변경/시행':>18}{'바뀐 쌍 총수':>16}")
print("  " + "-" * 60)

for M, N in [(10, 15), (10, 10), (15, 10), (20, 10)]:
    dev, changed, ndiff = 0.0, 0, 0
    for _ in range(TRIALS):
        tracks = [spd(1.0) for _ in range(M)]
        mu_t = rng.normal(size=(M, N_DIM)) * SEP
        mu_d = rng.normal(size=(N, N_DIM)) * SEP
        Sd_const = spd(0.05)
        A = np.zeros((M, N)); B = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                A[i, j], B[i, j] = bhatta(mu_t[i] - mu_d[j], tracks[i], Sd_const)
        lost = len(pairs(A) - pairs(A + B))
        dev = max(dev, np.abs(B - B.mean(axis=1, keepdims=True)).max())
        ndiff += lost
        changed += lost > 0
    print(f"  {f'{M} x {N}':>10}{dev:>18.2e}{f'{changed}/{TRIALS}':>18}{ndiff:>16}")

print()
print("  행내 편차는 모양과 무관하게 기계정밀도다. B 는 어느 경우에도 순수 행상수다.")
print("  그런데 M > N 에서만 할당이 바뀐다. 항이 분리되느냐가 아니라,")
print("  행상수 불변성 자체가 M <= N 조건부이기 때문이다.")
print()
print("결론: M <= N 이면 검출 불확실성이 할당에 전달되기 위한 필요조건은")
print("      '트랙 x 검출로 분리되지 않는 성분이 존재하는 것'이다.")
print("      M > N 이면 이 필요조건이 느슨해진다. 분리되는 항도")
print("      '어느 트랙을 버릴지'를 통해 할당에 도달한다.")
print()
print("      => 실데이터에서는 프레임마다 M,N 이 바뀐다. 어느 쪽 체제인지 먼저 세야 한다.")
