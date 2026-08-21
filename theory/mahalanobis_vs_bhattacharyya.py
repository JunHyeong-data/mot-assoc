# -*- coding: utf-8 -*-
"""
마할라노비스 vs 바타차야 -- 검출 불확실성이 지나갈 경로가 있는가.

세 가지 비용을 같은 장면에 대고 비교한다.

  (1) Maha-DeepSORT : d^2 = eps^T (S_t)^-1 eps,  S_t = H P H^T + R(h_i)
      ultralytics KalmanFilterXYAH.project() 는 측정잡음 R 을 '트랙의 높이 mean[3]'
      에서 만든다. 즉 R 은 순수 행 양이고, Sigma_d 는 식에 아예 등장하지 않는다.
      => 검출 불확실성 경로가 구조적으로 없다. 이것이 현재 표준이다.

  (2) Maha-perdet   : d^2 = eps^T (S_t + Sigma_d_j)^-1 eps
      가장 단순한 개조. 검출 공분산을 역행렬 안에 넣는다.

  (3) Bhattacharyya : (1/8) eps^T Sb^-1 eps + (1/2)ln|Sb| - (1/4)ln|St| - (1/4)ln|Sd|
      Sb = (St + Sd)/2. 마할라노비스형 항 + 로그행렬식 항.

측정하는 것: '검출 불확실성 채널의 세기'.
  Sigma_d 를 상수에서 개체별로 바꿨을 때 이중중심화 잔차 R 이 얼마나 변하는가.
  R 은 할당이 실제로 보는 전부이므로 (separability_residual.py), 이 변화량이
  곧 '검출 불확실성이 짝짓기에 도달한 양' 이다.

주의: theory/ 의 다른 스크립트들은 이미 바타차야를 쓴다. 이 파일은 둘의 대조다.
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(0)
N_DIM = 4          # (x, y, a, h)
M, N = 10, 15
TRIALS = 60
SEP = 1.2


def spd(scale=1.0, aniso=3.0):
    Q, _ = np.linalg.qr(rng.normal(size=(N_DIM, N_DIM)))
    e = scale * np.exp(rng.uniform(-np.log(aniso), np.log(aniso), size=N_DIM))
    return Q @ np.diag(e) @ Q.T


def residual(C):
    return C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()


def pairs(C):
    return set(zip(*(v.tolist() for v in linear_sum_assignment(C))))


def cost_maha_deepsort(eps, St, Sd, Ri):
    """Sigma_d 가 등장하지 않는다. 인자로 받되 쓰지 않는 것이 요점이다."""
    S = St + Ri
    return float(eps @ np.linalg.solve(S, eps))


def cost_maha_perdet(eps, St, Sd, Ri):
    S = St + Sd
    return float(eps @ np.linalg.solve(S, eps))


def cost_bhatta(eps, St, Sd, Ri):
    Sb = (St + Sd) / 2.0
    m = float(eps @ np.linalg.solve(Sb, eps)) / 8.0
    _, lb = np.linalg.slogdet(Sb)
    _, lt = np.linalg.slogdet(St)
    _, ld = np.linalg.slogdet(Sd)
    return m + 0.5 * lb - 0.25 * lt - 0.25 * ld


FORMS = [("Maha-DeepSORT (현행)", cost_maha_deepsort),
         ("Maha-perdet", cost_maha_perdet),
         ("Bhattacharyya", cost_bhatta)]


def build(fn, mu_t, mu_d, Sts, Sds, Ris):
    return np.array([[fn(mu_t[i] - mu_d[j], Sts[i], Sds[j], Ris[i])
                      for j in range(N)] for i in range(M)])


print("=" * 72)
print("[1] 검출 불확실성 채널의 세기")
print("=" * 72)
print("  Sigma_d 를 상수 -> 개체별로 바꿨을 때 잔차 R 이 변한 정도.")
print("  채널세기 = mean|R_perdet - R_const| / mean|R_const|")
print("  가림 = 트랙 공분산 배율. 검출 공분산 배율은 0.05 로 고정.")
print()
print(f"  {'비용':>22}" + "".join(f"{f'가림x{s:g}':>13}" for s in (1, 4, 16, 64)))
print("  " + "-" * 74)

for label, fn in FORMS:
    row = ""
    for occl in (1.0, 4.0, 16.0, 64.0):
        ch = []
        for _ in range(TRIALS):
            Sts = [spd(occl) for _ in range(M)]
            Ris = [spd(0.05) for _ in range(M)]          # 트랙 높이에서 나오는 R
            mu_t = rng.normal(size=(M, N_DIM)) * SEP
            mu_d = rng.normal(size=(N, N_DIM)) * SEP
            Sd_c = spd(0.05)
            C0 = build(fn, mu_t, mu_d, Sts, [Sd_c] * N, Ris)
            C1 = build(fn, mu_t, mu_d, Sts,
                       [spd(0.05, aniso=8.0) for _ in range(N)], Ris)
            R0, R1 = residual(C0), residual(C1)
            ch.append(np.abs(R1 - R0).mean() / max(np.abs(R0).mean(), 1e-15))
        row += f"{np.mean(ch):>13.3f}"
    print(f"  {label:>22}{row}")

print()
print("  -> Maha-DeepSORT 는 정확히 0. Sigma_d 가 식에 없으니 당연하다.")
print("     '검출 불확실성을 비용에 넣는다' 는 말이 성립하려면 식부터 바꿔야 한다.")
print("  -> 나머지 둘은 채널이 열려 있고, 가림이 길수록 채널이 좁아진다.")

print()
print("=" * 72)
print("[2] 같은 조건에서 할당이 실제로 바뀌는 비율")
print("=" * 72)
print("  Sigma_d 상수 vs 개체별. 두 할당의 쌍 집합이 다른 시행의 비율.")
print()
print(f"  {'비용':>22}" + "".join(f"{f'가림x{s:g}':>13}" for s in (1, 4, 16, 64)))
print("  " + "-" * 74)

for label, fn in FORMS:
    row = ""
    for occl in (1.0, 4.0, 16.0, 64.0):
        chg = 0
        for _ in range(TRIALS):
            Sts = [spd(occl) for _ in range(M)]
            Ris = [spd(0.05) for _ in range(M)]
            mu_t = rng.normal(size=(M, N_DIM)) * SEP
            mu_d = rng.normal(size=(N, N_DIM)) * SEP
            Sd_c = spd(0.05)
            C0 = build(fn, mu_t, mu_d, Sts, [Sd_c] * N, Ris)
            C1 = build(fn, mu_t, mu_d, Sts,
                       [spd(0.05, aniso=8.0) for _ in range(N)], Ris)
            chg += pairs(C0) != pairs(C1)
        row += f"{f'{chg / TRIALS:.0%}':>13}"
    print(f"  {label:>22}{row}")

print()
print("=" * 72)
print("[3] 바타차야 로그행렬식 항의 분해 -- 어디가 비분리인가")
print("=" * 72)
print("  B_ij = 0.5 ln|Sb_ij|  -  0.25 ln|St_i|  -  0.25 ln|Sd_j|")
print("           비분리(결합)      순수 행         순수 열")
print()
print("  뒤 두 항은 행/열 상수라 잔차에 기여가 0 이어야 한다. 확인한다.")
print()

Sts = [spd(1.0) for _ in range(M)]
Sds = [spd(0.05, aniso=8.0) for _ in range(N)]
lt = np.array([np.linalg.slogdet(S)[1] for S in Sts])
ld = np.array([np.linalg.slogdet(S)[1] for S in Sds])
coup = np.array([[0.5 * np.linalg.slogdet((Sts[i] + Sds[j]) / 2)[1]
                  for j in range(N)] for i in range(M)])
full = coup - 0.25 * lt[:, None] - 0.25 * ld[None, :]

print(f"  {'전체 로그행렬식 항의 잔차 |R|':<40}{np.abs(residual(full)).mean():>12.6f}")
print(f"  {'결합항 0.5 ln|Sb| 만의 잔차 |R|':<40}{np.abs(residual(coup)).mean():>12.6f}")
print(f"  {'둘의 차이':<40}{np.abs(residual(full) - residual(coup)).mean():>12.3e}")
print()
print("  -> 정확히 일치한다. 로그행렬식 항에서 할당에 도달하는 것은")
print("     오직 0.5 ln|(St_i + Sd_j)/2| 뿐이다.")
print()
print("  이 항은 St >> Sd 이면 0.5 ln|St_i/2| 로 수렴한다 -- 순수 행상수다.")
print("  즉 가림이 길어지면 바타차야의 검출 불확실성 경로도 닫힌다.")
print("  [1] 에서 가림이 커질수록 채널세기가 줄어든 것이 이 때문이다.")
print()
print("결론: 바타차야로 바꾸면 검출 불확실성 경로가 '생기기는 한다'.")
print("      마할라노비스(현행)에서는 경로 자체가 없다.")
print("      단 그 경로는 St 와 Sd 가 비슷한 크기일 때만 열려 있고,")
print("      가림이 길어질수록 닫힌다. 정작 필요한 구간에서 약해진다.")
