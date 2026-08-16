# -*- coding: utf-8 -*-
"""
공분산 역설을 1차원으로 확인한다.

주장: 검출이 안 들어오는 동안 칼만의 P 는 계속 커지고, 마할라노비스 거리는
      그 P 로 나누는 형태이므로, 오차가 똑같아도 거리가 작아진다.
      즉 오래 못 본 트랙일수록 아무 검출이나 '가깝다'고 판정한다.

손계산 답 맞추기용. 종이로 먼저 풀고 이 스크립트로 확인할 것.
"""
import sys

# 콘솔이 cp949 여도 죽지 않게. print 문의 특수문자는 ASCII 로 쓸 것.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# 1차원, 위치만 추적. F = H = 1
X0, P0, Q, R = 100.0, 4.0, 1.0, 1.0


def predict(x, P):
    return x, P + Q                      # x- = F x,  P- = F P F^T + Q


def update(x_pred, P_pred, z):
    S = P_pred + R                       # S  = H P- H^T + R
    K = P_pred / S                       # K  = P- H^T S^-1
    x = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred                 # P  = (I - K H) P-
    return x, P, S, K


print("=" * 72)
print("1-A. 측정이 들어올 때 -- P 는 수렴한다")
print("=" * 72)
print(f"{'사이클':>6}{'P-':>9}{'S':>9}{'K':>9}{'x':>10}{'P':>9}")
print("-" * 72)

x, P = X0, P0
for i, z in enumerate([103, 105, 106, 109, 110], start=1):
    xp, Pp = predict(x, P)
    x, P, S, K = update(xp, Pp, z)
    print(f"{i:>6}{Pp:>9.3f}{S:>9.3f}{K:>9.3f}{x:>10.3f}{P:>9.3f}")

print()
print("  -> 측정이 계속 들어오면 P 가 한 값으로 수렴한다.")
print("     검출이 트랙의 불확실성을 눌러주고 있다는 뜻.")
print()

print("=" * 72)
print("1-B. 측정이 없을 때 (가림) -- P 는 계속 커지고 d^2 은 작아진다")
print("=" * 72)
print("오차를 10 px 로 고정하고 d^2 = 10^2 / (P- + R) 만 다시 계산한다.")
print()
print(f"{'가려진 프레임':>14}{'P-':>9}{'sigma':>9}{'d^2':>9}")
print("-" * 72)

EPS = 10.0
P = P0
print(f"{0:>14}{P:>9.3f}{P ** 0.5:>9.3f}{EPS ** 2 / (P + R):>9.3f}")
for t in range(1, 11):
    _, P = predict(0.0, P)
    if t in (1, 2, 3, 4, 5, 10):
        print(f"{t:>14}{P:>9.3f}{P ** 0.5:>9.3f}{EPS ** 2 / (P + R):>9.3f}")

print()
print("  -> 오차는 내내 10 px 그대로인데 d^2 은 단조 감소한다. 이것이 공분산 역설.")
print()
print("  주의: 이 동작 자체는 베이즈적으로 틀리지 않다. 불확실하면 넓게 보는 게 맞다.")
print("        문제는 d^2 만 비용으로 쓸 때다. 정식 가우시안 비용은")
print("        (1/2)(d^2 + ln det S) 인데, ln det S 는 P 가 커질수록 함께 커져서")
print("        '불확실한 트랙에는 페널티' 역할을 한다. d^2 만 쓰면 그게 빠진다.")
print()

print("=" * 72)
print("확인: ln det S 를 같이 쓰면 얼마나 막아주나")
print("=" * 72)
import math

print(f"{'가려진 프레임':>14}{'S':>9}{'d^2':>10}{'ln S':>9}{'(d^2+lnS)/2':>15}")
print("-" * 72)
P = P0
for t in range(0, 200):
    if t:
        _, P = predict(0.0, P)
    if t in (0, 1, 2, 5, 10, 30, 60, 96, 130, 199):
        S = P + R
        d2 = EPS ** 2 / S
        print(f"{t:>14}{S:>9.1f}{d2:>10.3f}{math.log(S):>9.3f}"
              f"{(d2 + math.log(S)) / 2:>15.3f}")

print()
print(f"  d^2 + ln S 를 S 로 미분하면 (1/S)(1 - eps^2/S).")
print(f"  => S < eps^2 이면 아직 감소하고, S > eps^2 부터 증가로 바뀐다.")
print(f"  => 지금 eps = {EPS:.0f} 이므로 전환점은 S = {EPS**2:.0f}. Q=1 이니 약 96 프레임.")
print()
print("  -> ln det S 는 감소를 '늦출' 뿐, 현실적인 가림 길이에서는 뒤집지 못한다.")
print("     페널티가 실제로 작동하려면 불확실성이 오차 제곱에 필적해야 한다.")
print("     즉 정식 비용을 다 써도 공분산 역설이 완전히 사라지지는 않는다.")
