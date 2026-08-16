# -*- coding: utf-8 -*-
"""
잔차가 작아지는 것이 문제인가, 아니면 잡음 대비로 작아지는 것이 문제인가.

배경 정정
  separability_residual.py 에서 가림이 길어질수록 이중중심화 잔차 |R| 이
  0.342 -> 0.003 으로 줄어드는 것을 보였고, 거기에 "고를 근거가 소멸한다" 는
  해석을 붙였다. 그 해석은 과했다.
  **헝가리안은 비용 전체에 양수 배율을 곱해도 최적 할당이 불변이다.**
  따라서 |R| 이 균일하게 작아지는 것만으로는 할당이 나빠지지 않는다.
  나빠지려면 잡음이 신호보다 덜 줄어야 한다. 즉 SNR 이 관건이다.

이 스크립트가 판정하는 것
  (1) 잔차의 신호/잡음 분해. 가림이 길어질 때 SNR 이 실제로 떨어지는가.
  (2) 정답 할당 복원율. 떨어진다면 어디서부터 무너지는가.
  (3) **행 단위 정규화(z-점수/순위)가 그것을 되살리는가.**
      이것이 'REM 식 순위 보존을 연관 비용에 이식' 아이디어의 생사를 가른다.
      잔차가 신호 지배적이면 정규화가 살릴 여지가 있다.
      잡음 지배적이면 정규화는 잡음을 증폭할 뿐이다.

설정
  물체 M 개가 참 위치 x_i 에 있다. 트랙은 x_i 를 Sigma_t 만큼 빗나가게 예측하고
  (가림이 길수록 Sigma_t 가 커진다), 검출은 x_i 를 Sigma_d 만큼 빗나가게 관측한다.
  정답은 트랙 i <-> 검출 i 다. 이걸 얼마나 되찾는지 본다.
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(0)
N_DIM = 4
P0, Q = 4.0, 1.0
SD = 0.05          # 검출 공분산 배율
# 물체 간 간격은 장면 기하가 정하는 것이므로 가림 길이와 무관하게 **고정**한다.
# 여기를 scale 에 비례시키면 간격/불확실성 비가 상수로 묶여 아무것도 나빠지지
# 않는다 (실제로 처음에 그렇게 짜서 잘못된 null 이 나왔다).
SEP = 1.2
SEP_ABS = SEP * P0 ** 0.5
OCCL = (0, 2, 5, 10, 30, 60, 120, 240)


def spd(scale=1.0, aniso=3.0):
    A, _ = np.linalg.qr(rng.normal(size=(N_DIM, N_DIM)))
    e = scale * np.exp(rng.uniform(-np.log(aniso), np.log(aniso), size=N_DIM))
    return A @ np.diag(e) @ A.T


def bhatta(eps, St, Sd):
    Sb = (St + Sd) / 2.0
    m = float(eps @ np.linalg.solve(Sb, eps)) / 8.0
    _, lb = np.linalg.slogdet(Sb)
    _, lt = np.linalg.slogdet(St)
    _, ld = np.linalg.slogdet(Sd)
    return m + 0.5 * lb - 0.25 * lt - 0.25 * ld


def residual(C):
    return C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()


def build(M, N, scale, x, Sts, Sds, draw=True):
    """트랙 예측과 검출 관측을 뽑아 비용행렬을 만든다. 정답은 i <-> i."""
    mu_t = np.array([x[i] + rng.multivariate_normal(np.zeros(N_DIM), Sts[i])
                     for i in range(M)]) if draw else x[:M].copy()
    mu_d = np.empty((N, N_DIM))
    for j in range(N):
        base = x[j] if j < M else rng.normal(size=N_DIM) * SEP_ABS
        mu_d[j] = base + (rng.multivariate_normal(np.zeros(N_DIM), Sds[j])
                          if draw else 0)
    C = np.array([[bhatta(mu_t[i] - mu_d[j], Sts[i], Sds[j]) for j in range(N)]
                  for i in range(M)])
    return C


def row_z(C):
    s = C.std(1, keepdims=True)
    return (C - C.mean(1, keepdims=True)) / np.where(s < 1e-12, 1.0, s)


def row_rank(C):
    return np.apply_along_axis(rankdata, 1, C).astype(float)


def acc(C, M):
    r, c = linear_sum_assignment(C)
    return float(np.mean(c[:len(r)] == r)) if len(r) else 0.0


print("=" * 74)
print("[1] 잔차의 신호/잡음 분해")
print("=" * 74)
print("  같은 배치(참 위치·공분산 고정)에서 오차 실현만 바꿔 K 번 뽑는다.")
print("  신호 = 칸별 평균잔차의 분산   잡음 = 칸별 잔차의 실현간 분산 평균")
print()
print(f"  {'가려진 프레임':>14}{'|R| 평균':>12}{'신호':>12}{'잡음':>12}{'SNR':>10}")
print("  " + "-" * 62)

M = N = 8
K, CFG = 60, 12
snr_by_t = {}
for t in OCCL:
    scale = P0 + t * Q
    sig_l, noi_l = [], []
    for _ in range(CFG):
        x = rng.normal(size=(M, N_DIM)) * SEP_ABS
        Sts = [spd(scale) for _ in range(M)]
        Sds = [spd(SD, aniso=8.0) for _ in range(N)]
        Rs = np.array([residual(build(M, N, scale, x, Sts, Sds)) for _ in range(K)])
        sig_l.append(Rs.mean(0).var())
        noi_l.append(Rs.var(0).mean())
    sig, noi = np.mean(sig_l), np.mean(noi_l)
    snr_by_t[t] = sig / max(noi, 1e-300)
    print(f"  {t:>14}{np.sqrt(sig + noi):>12.4f}{sig:>12.3e}{noi:>12.3e}"
          f"{snr_by_t[t]:>10.2f}")

print()
print("  SNR 이 가림과 무관하게 유지되면, |R| 이 줄어드는 것은 단지 배율 문제다")
print("  (헝가리안은 배율에 불변). SNR 이 떨어져야 실제로 나빠지는 것이다.")

print()
print("=" * 74)
print("[2] 정답 할당 복원율, 그리고 행 정규화가 살리는가")
print("=" * 74)
print("  raw = 원비용, z = 행별 z-점수, rank = 행별 순위. 200 시행 평균.")
print()
print(f"  {'가려진 프레임':>14}{'raw':>10}{'행 z':>10}{'행 rank':>10}{'무작위':>10}")
print("  " + "-" * 56)

T = 200
for t in OCCL:
    scale = P0 + t * Q
    a = {"raw": [], "z": [], "rank": []}
    for _ in range(T):
        x = rng.normal(size=(M, N_DIM)) * SEP_ABS
        Sts = [spd(scale) for _ in range(M)]
        Sds = [spd(SD, aniso=8.0) for _ in range(N)]
        C = build(M, N, scale, x, Sts, Sds)
        a["raw"].append(acc(C, M))
        a["z"].append(acc(row_z(C), M))
        a["rank"].append(acc(row_rank(C), M))
    print(f"  {t:>14}{np.mean(a['raw']):>10.3f}{np.mean(a['z']):>10.3f}"
          f"{np.mean(a['rank']):>10.3f}{1.0 / M:>10.3f}")

print()
print("=" * 74)
print("판정")
print("=" * 74)
print(f"  SNR: 가림 0 에서 {snr_by_t[OCCL[0]]:.2f}  ->  가림 {OCCL[-1]} 에서 "
      f"{snr_by_t[OCCL[-1]]:.2f}")
print()
print("  * 복원율이 유지되고 SNR 도 유지되면: |R| 감소는 배율 현상이다.")
print("    'covariance paradox 의 원리적 해법' 으로 순위 보존을 내세울 근거가 약해진다.")
print("  * 복원율이 무너지는데 행 z/rank 가 되살리면: 정규화 아이디어가 산다.")
print("  * 복원율이 무너지고 정규화도 못 살리면: 정보가 소멸한 것이다.")
print("    비용을 어떻게 변환해도 못 되살린다. 다른 정보원을 넣어야 한다.")
