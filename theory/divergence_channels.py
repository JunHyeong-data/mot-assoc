# -*- coding: utf-8 -*-
"""통로 후보들을 같은 자로 잰다 -- 검출 공분산이 할당에 도달하는가.

지금까지 이 저장소가 확인한 것:
  - 마할라노비스(트랙 공분산만) : Sigma_d 가 아예 안 들어간다. 통로 없음
  - 바타차야                    : 통로를 열지만 로그행렬식 항 하나뿐이고
                                  Sigma_t >> Sigma_d (긴 가림) 에서 닫힌다
  - 공분산 역설                 : Sigma 가 커지면 d^2 이 작아진다. 가려진 트랙이
                                  아무 검출과나 싸지는 병리

여기서 후보를 넷으로 넓혀 **같은 세 가지 진단**을 건다.

  [1] 도달   : Sigma_d 를 개체별로 두면 이중중심화 잔차가 상수일 때보다 커지는가
  [2] 가림   : Sigma_t 를 키워도 그 도달이 유지되는가
  [3] 역설   : 불확실성이 커질 때 정답쌍 비용이 **내려가는가** (내려가면 병리)

**와서스타인이 왜 다른가.** 2-Wasserstein 은
    W^2 = ||eps||^2 + tr(St + Sd - 2 (St^1/2 Sd St^1/2)^1/2)
평균항에 **Sigma^-1 이 안 붙는다.** 그래서 [3] 공분산 역설이 원리적으로 없다.
공분산항은 tr(St)(행상수) + tr(Sd)(열상수) + 교차항으로 갈라지는데,
**교차항만 비분리**다. 그 교차항이 [1][2] 를 통과하는지가 이 스크립트의 질문이다.

사용법:
    python theory/divergence_channels.py
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(0)
DIM = 2
SEP = 1.2
M, N, REP = 10, 15, 40


def spd(scale=1.0, aniso=3.0, r=None):
    r = r or rng
    Q, _ = np.linalg.qr(r.normal(size=(DIM, DIM)))
    e = scale * np.exp(r.uniform(-np.log(aniso), np.log(aniso), size=DIM))
    return Q @ np.diag(e) @ Q.T


def sqrtm_spd(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.maximum(w, 0))) @ V.T


# ---- 통로 후보 ----------------------------------------------------------
def d_mahal_track(eps, St, Sd):
    """현행. Sigma_d 가 안 들어간다."""
    return float(eps @ np.linalg.solve(St, eps))


def d_mahal_comb(eps, St, Sd):
    """'살짝 바꾼' 마할라노비스. 결합 공분산을 쓴다."""
    return float(eps @ np.linalg.solve(St + Sd, eps))


def d_bhatta(eps, St, Sd):
    Sb = (St + Sd) / 2.0
    _, lb = np.linalg.slogdet(Sb)
    _, lt = np.linalg.slogdet(St)
    _, ld = np.linalg.slogdet(Sd)
    return float(eps @ np.linalg.solve(Sb, eps)) / 8.0 + 0.5 * lb - 0.25 * lt - 0.25 * ld


def d_wasser(eps, St, Sd):
    """2-Wasserstein^2. 평균항에 Sigma^-1 이 없다."""
    rt = sqrtm_spd(St)
    cross = sqrtm_spd(rt @ Sd @ rt)
    return float(eps @ eps) + float(np.trace(St + Sd - 2.0 * cross))


CHANNELS = [("마할라노비스(트랙만)", d_mahal_track),
            ("마할라노비스(결합)", d_mahal_comb),
            ("바타차야", d_bhatta),
            ("와서스타인", d_wasser)]


def residual(C):
    return C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()


def build(fn, mu_t, mu_d, tracks, dets):
    return np.array([[fn(mu_t[i] - mu_d[j], tracks[i], dets[j])
                      for j in range(len(mu_d))] for i in range(len(mu_t))])


print("=" * 78)
print("[1] 도달 -- 개체별 Sigma_d 가 이중중심화 잔차에 무언가를 남기는가")
print("=" * 78)
print("  같은 장면에서 Sigma_d 를 (a) 개체별 (b) 전부 같은 상수 로 두고")
print("  잔차 R 의 차이를 잰다. 0 이면 그 통로로는 검출 정보가 안 간다.")
print()
print("  %-22s %14s %14s %10s" % ("통로", "|R(개체별)|", "|dR| 개체-상수", "할당변경"))
print("  " + "-" * 66)

base_rng = np.random.default_rng(7)
scenes = []
for _ in range(REP):
    tracks = [spd(1.0, r=base_rng) for _ in range(M)]
    dets = [spd(0.05, aniso=8.0, r=base_rng) for _ in range(N)]
    const = spd(0.05, r=base_rng)
    scenes.append((tracks, dets, const,
                   base_rng.normal(size=(M, DIM)) * SEP,
                   base_rng.normal(size=(N, DIM)) * SEP))

for name, fn in CHANNELS:
    rmean, dmean, changed = [], [], 0
    for tracks, dets, const, mu_t, mu_d in scenes:
        Ci = build(fn, mu_t, mu_d, tracks, dets)
        Cc = build(fn, mu_t, mu_d, tracks, [const] * N)
        Ri, Rc = residual(Ci), residual(Cc)
        rmean.append(np.abs(Ri).mean())
        dmean.append(np.abs(Ri - Rc).mean())
        changed += int((linear_sum_assignment(Ci)[1]
                        != linear_sum_assignment(Cc)[1]).any())
    print("  %-22s %14.4e %14.2e %8d/%d"
          % (name, np.mean(rmean), np.mean(dmean), changed, REP))

print()
print("  -> |dR| 이 기계정밀도(1e-16)면 그 통로는 Sigma_d 를 못 나른다.")

print()
print("=" * 78)
print("[2] 가림 -- 트랙 공분산이 커져도 도달이 유지되는가")
print("=" * 78)
print("  Sigma_t 배율을 키우며 [1] 의 |dR| 을 다시 잰다. t=0 대비 비율로 본다.")
print()
print("  %-22s %s" % ("통로", "".join("%11s" % ("x%d" % s) for s in (1, 5, 20, 100))))
print("  " + "-" * 70)

for name, fn in CHANNELS:
    row, first = [], None
    for scale in (1, 5, 20, 100):
        vals = []
        for tracks, dets, const, mu_t, mu_d in scenes[:20]:
            tr = [T * scale for T in tracks]
            Ci = build(fn, mu_t, mu_d, tr, dets)
            Cc = build(fn, mu_t, mu_d, tr, [const] * N)
            vals.append(np.abs(residual(Ci) - residual(Cc)).mean())
        v = np.mean(vals)
        first = v if first is None else first
        row.append(v / first if first > 0 else 0.0)
    print("  %-22s %s" % (name, "".join("%11.3f" % r for r in row)))

print()
print("  -> 1.0 을 유지하면 가림에서도 통로가 열려 있다. 0 으로 가면 닫힌다.")

print()
print("=" * 78)
print("[3] 공분산 역설 -- 불확실해질수록 정답쌍이 싸지는가")
print("=" * 78)
print("  같은 오차 eps 를 두고 Sigma 만 키운다. 비용이 **내려가면** 병리다.")
print("  (가려진 트랙이 아무 검출과나 싸져 엉뚱한 짝이 붙는다)")
print()
print("  %-22s %s" % ("통로", "".join("%11s" % ("x%d" % s) for s in (1, 5, 20, 100))))
print("  " + "-" * 70)

eps0 = np.array([3.0, 2.0])
St0, Sd0 = spd(1.0, r=np.random.default_rng(3)), spd(0.05, r=np.random.default_rng(4))
for name, fn in CHANNELS:
    vals = [fn(eps0, St0 * s, Sd0) for s in (1, 5, 20, 100)]
    print("  %-22s %s" % (name, "".join("%11.3f" % v for v in vals)))

print()
print("  -> 감소하면 공분산 역설이 있다. 와서스타인은 평균항에 Sigma^-1 이")
print("     없으므로 원리적으로 증가한다.")

print()
print("=" * 78)
print("정리")
print("=" * 78)
print("  세 진단을 다 통과하는 통로만 검출 불확실성을 실제로 쓸 수 있다.")
print("  [1] 도달  [2] 가림에서 유지  [3] 역설 없음")
print()
print("  * 이건 합성 수치다. 실데이터에서 이득이 난다는 말이 아니라,")
print("    '어느 통로가 원리적으로 가능한가' 만 가린다.")
