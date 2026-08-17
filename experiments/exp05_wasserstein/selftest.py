# -*- coding: utf-8 -*-
"""실험 5 [2단계] 자체시험 -- 비싼 실행 전에 합성으로 잡는다.

exp03 의 selftest 와 같은 역할이다. 콜랩/장시간 실행 전에 논리 오류를 여기서 잡는다.
출력은 ASCII 로만 쓴다 (Windows cp949 콘솔).

확인하는 것:
  [1] 대각 단순화가 일반 행렬형과 같은가          <- 수학 검산
  [2] St == Sd 이면 Bures 항이 0 인가             <- 경계 조건
  [3] Sigma 규모 맞추기가 실제로 E[tr] 을 맞추는가 <- 사전선언 함정 1
  [4] **공분산 역설이 없는가** (Sigma 커지면 비용이 커져야 한다)
  [5] C 보정이 목표 중앙값을 맞추는가             <- 사전선언 함정 3
  [6] 마할라노비스와 방향이 반대인가              <- theory/ 결과 재확인
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from wcost import (bures_diag, bures_full, w2_matrix, size_var,   # noqa: E402
                   match_scale, nwd_cost, solve_C)

rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail=""):
    global ok
    print("  [%s] %-44s %s" % ("OK" if cond else "FAIL", name, detail))
    if not cond:
        ok = False


print("=" * 74)
print("실험 5 자체시험 -- 와서스타인 비용함수")
print("=" * 74)

# ---- [1] 대각 단순화 == 일반형 ------------------------------------------
n = 500
st = rng.uniform(0.5, 40.0, (n, 2))
sd = rng.uniform(0.5, 40.0, (n, 2))
St = np.zeros((n, 2, 2)); St[:, 0, 0] = st[:, 0]; St[:, 1, 1] = st[:, 1]
Sd = np.zeros((n, 2, 2)); Sd[:, 0, 0] = sd[:, 0]; Sd[:, 1, 1] = sd[:, 1]
d1 = bures_diag(st, sd)
d2 = bures_full(St, Sd)
check("[1] 대각 단순화 == 일반 행렬형", np.abs(d1 - d2).max() < 1e-9,
      "max|diff| = %.2e" % np.abs(d1 - d2).max())

# ---- [2] St == Sd 이면 Bures = 0 ---------------------------------------
check("[2] St == Sd 이면 Bures 항 0", np.abs(bures_diag(st, st)).max() < 1e-12,
      "max = %.2e" % np.abs(bures_diag(st, st)).max())

# ---- [3] 규모 맞추기 ---------------------------------------------------
a = rng.uniform(0.01, 0.5, (300, 2))      # DFL 급 (작다)
b = rng.uniform(50, 500, (300, 2))        # NMS 급 (크다)
tgt = float(np.mean(a.sum(-1)))
b2 = match_scale(b, tgt)
check("[3] 규모 맞추기가 E[tr] 을 맞춘다",
      abs(float(np.mean(b2.sum(-1))) - tgt) < 1e-9 * max(tgt, 1),
      "E[tr] %.4f -> %.4f (목표 %.4f)" % (np.mean(b.sum(-1)), np.mean(b2.sum(-1)), tgt))

# ---- [4] 공분산 역설이 없는가 ------------------------------------------
# 같은 eps 를 두고 Sigma 만 키운다. **비용이 커져야 한다.**
# 마할라노비스는 여기서 작아진다 (theory/divergence_channels.py [3]).
t_box = np.array([[100.0, 100.0, 160.0, 280.0]])
d_box = np.array([[106.0, 104.0, 166.0, 284.0]])
base_t = np.array([[4.0, 9.0]])
costs = []
for s in (1, 5, 20, 100):
    w2 = w2_matrix(t_box, d_box, base_t * s, base_t * s * 0.5)
    costs.append(float(w2[0, 0]))
mono_up = all(x < y for x, y in zip(costs, costs[1:]))
check("[4] 공분산 역설 없음 (Sigma 키우면 비용 증가)", mono_up,
      " -> ".join("%.1f" % c for c in costs))

# 대조: 마할라노비스는 반대로 간다
mah = []
for s in (1, 5, 20, 100):
    S = base_t[0] * s + base_t[0] * s * 0.5
    eps = np.array([6.0, 4.0])
    mah.append(float((eps ** 2 / S).sum()))
check("[4b] 대조 - 마할라노비스는 감소한다 (역설 있음)",
      all(x > y for x, y in zip(mah, mah[1:])),
      " -> ".join("%.2f" % m for m in mah))

# ---- [5] C 보정 --------------------------------------------------------
w2s = rng.uniform(1, 5000, 2000)
for target in (0.3, 0.5, 0.7):
    C = solve_C(w2s, target)
    got = float(np.median(nwd_cost(w2s, C)))
    check("[5] C 보정: 목표 %.2f" % target, abs(got - target) < 1e-6,
          "얻은 중앙값 %.6f" % got)

# ---- [6] 상자크기 Sigma 가 NWD 정의와 맞는가 ---------------------------
bx = np.array([[0.0, 0.0, 40.0, 100.0]])
sv = size_var(bx)
check("[6] size_var = diag(w^2,h^2)/4",
      abs(sv[0, 0] - 400.0) < 1e-9 and abs(sv[0, 1] - 2500.0) < 1e-9,
      "w=40,h=100 -> %.1f, %.1f (기대 400, 2500)" % (sv[0, 0], sv[0, 1]))

# ---- [7] 완전히 같은 상자면 비용 0 -------------------------------------
same = w2_matrix(t_box, t_box, base_t, base_t)
check("[7] 같은 상자/같은 Sigma 이면 W^2 = 0", abs(float(same[0, 0])) < 1e-9,
      "W^2 = %.2e" % same[0, 0])

# ---- [8] 규모를 맞추면 갈래 간 비교가 규모에 안 흔들리는가 --------------
# 같은 '모양' 의 Sigma 를 100배 다르게 줘도, 규모를 맞춘 뒤에는 W^2 이 같아야 한다.
shape = rng.uniform(1, 10, (50, 2))
v_small, v_big = shape * 0.01, shape * 1.0
tgt2 = float(np.mean(v_small.sum(-1)))
tb = np.tile(t_box, (50, 1)); db = np.tile(d_box, (50, 1))
w_a = w2_matrix(tb, db, v_small, v_small)
w_b = w2_matrix(tb, db, match_scale(v_big, tgt2), match_scale(v_big, tgt2))
check("[8] 규모 맞춘 뒤 100배 차이가 사라진다",
      np.abs(w_a - w_b).max() < 1e-8, "max|diff| = %.2e" % np.abs(w_a - w_b).max())

print()
print("RESULT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
