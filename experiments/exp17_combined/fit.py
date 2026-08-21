# -*- coding: utf-8 -*-
"""실험 17 — **합친 모형(D)은 A 와 C 를 둘 다 이기는가.**

사전 등록은 `PREREG.md` (자료보다 먼저 커밋, 읽는 법 포함).

원고 5.1 은 σ 가 **높이를 통제하고도** 정보를 가진다고 했고(+0.322),
5.2 는 σ-만 모형이 높이-만 모형에 7/7 로 진다고 했다. 둘을 합치면
**"높이와 σ 를 함께 쓴 모형이 둘 다 이겨야 한다"** 는 예측이 나오는데
**그 모형을 적합해 본 적이 없다.**

    A  Sigma = k * Sigma_d                      (k, nu)
    C  Sigma = k * diag(h^2, h^2)               (k, nu)
    D  Sigma = k * h^2 * (s/median(s))^g        (k, nu, g)   <- 새로 넣는 것

`s = sqrt(tr(Sigma_d)/2)` 는 σ 의 스칼라 요약. `median(s)` 는 **훈련 자료에서만**
잡는다(누출 방지). `g=0` 이면 D 는 C 와 정확히 같다.

모수가 하나 많으므로 **held-out 에서 못 이기면 그것만으로 결론**이다.

사용법:
    python experiments/exp17_combined/fit.py        # NMS 소스
    python experiments/exp17_combined/fit.py -dfl   # DFL 소스
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp01_nms_variance"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# student_t.py 는 sys.argv[1] 을 TAG 로 읽는다. -dfl 을 그 규약에 맞춘다.
TAG = "-dfl" if "-dfl" in sys.argv else ""
sys.argv = [sys.argv[0]] + ([TAG] if TAG else [])

import student_t as ST                                          # noqa: E402

SEQS = ST.SEQS
GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]       # 사전 등록한 격자


def sigma_scalar(d):
    """검출별 sigma 의 스칼라 요약. tr(Sigma_d)/2 의 제곱근."""
    return np.sqrt(np.maximum((d["sxx"] + d["syy"]) / 2.0, 1e-12))


def S_D(d, g, med):
    """합친 모형. g=0 이면 C 와 **정확히 같다** (아래에서 검산한다)."""
    r = (sigma_scalar(d) / med) ** g
    v = d["h"] ** 2 * r
    return (v, np.zeros(len(v)), v)


def fit_D(tr):
    """훈련 자료로 (g, nu, k) 를 적합. med 는 훈련에서만 잡는다."""
    med = float(np.median(sigma_scalar(tr)))
    best = (np.inf, None, None, med)
    for g in GRID:
        z, ld = ST.z2_and_logdet(S_D(tr, g, med), tr["ex"], tr["ey"])
        nu, k = ST.fit_nu_k(z, ld)
        f = float(np.mean(ST.nll_t(z, ld, nu, k)))
        if f < best[0]:
            best = (f, g, (nu, k), med)
    return best[1], best[2], best[3]


def eval_D(te, g, nuk, med):
    z, ld = ST.z2_and_logdet(S_D(te, g, med), te["ex"], te["ey"])
    return float(np.mean(ST.nll_t(z, ld, nuk[0], nuk[1])))


def eval_arm(te, sfun, nu, k):
    z, ld = ST.z2_and_logdet(sfun(te), te["ex"], te["ey"])
    return float(np.mean(ST.nll_t(z, ld, nu, k)))


def fit_arm(tr, sfun):
    z, ld = ST.z2_and_logdet(sfun(tr), tr["ex"], tr["ey"])
    return ST.fit_nu_k(z, ld)


def sign_two_sided(d):
    """양측 부호검정. 동률은 뺀다 (exp10 verify_1f 와 같은 처리)."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    win = int((nz > 0).sum())
    k = max(win, n - win)
    return win, n, min(2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n, 1.0)


def main():
    src = "DFL 분포 분산" if TAG else "NMS 후보 분산"
    print("=" * 92)
    print("실험 17 -- 합친 모형 D 는 A 와 C 를 둘 다 이기는가   [소스 %s]" % src)
    print("=" * 92)
    print("D: Sigma = k * h^2 * (s/median(s))^g   **모수가 C 보다 하나 많다.**")
    print("따라서 held-out 에서 못 이기면 그것만으로 결론이다.")
    print("NLL 이 낮을수록 좋다. d = NLL_D - NLL_C 이므로 **d < 0 이면 D 승**.")
    print()

    data = {s: ST.load(s) for s in SEQS}

    # [사전 점검] g=0 에서 D 가 C 와 같은가 -- 구현 검산
    tr0 = {k: np.concatenate([data[s][k] for s in SEQS]) for k in data[SEQS[0]]}
    med0 = float(np.median(sigma_scalar(tr0)))
    zc, ldc = ST.z2_and_logdet(ST.S_C(tr0), tr0["ex"], tr0["ey"])
    zd, ldd = ST.z2_and_logdet(S_D(tr0, 0.0, med0), tr0["ex"], tr0["ey"])
    gap = float(np.max(np.abs(zc - zd)) + np.max(np.abs(ldc - ldd)))
    print("[사전 점검] g=0 에서 D 와 C 가 같은가:  최대 차이 %.3g  %s"
          % (gap, "OK" if gap < 1e-9 else "!! 구현이 틀렸다"))
    if gap >= 1e-9:
        return 1
    print()

    # ---------------- [1] LOSO ----------------
    print("=" * 92)
    print("[1] LOSO -- 주 평가지표")
    print("=" * 92)
    print("%-12s %9s %9s %9s %9s   %6s" % ("held-out", "NLL_A", "NLL_C", "NLL_D",
                                           "D - C", "g"))
    print("-" * 92)
    dl, gs = [], []
    for held in SEQS:
        parts = [data[s] for s in SEQS if s != held]
        tr = {k: np.concatenate([p[k] for p in parts]) for k in data[held]}
        te = data[held]
        nuA, kA = fit_arm(tr, ST.S_A)
        nuC, kC = fit_arm(tr, ST.S_C)
        g, nuk, med = fit_D(tr)
        a = eval_arm(te, ST.S_A, nuA, kA)
        c = eval_arm(te, ST.S_C, nuC, kC)
        d = eval_D(te, g, nuk, med)
        dl.append(d - c)
        gs.append(g)
        print("%-12s %9.3f %9.3f %9.3f %+9.3f   %6.2f" % (held, a, c, d, d - c, g))

    w, n, p = sign_two_sided([-x for x in dl])     # D 승 = d<0
    print("-" * 92)
    print("  **D 승 %d/%d**,  짝지은 차이 중앙 %+.3f nats,  양측 부호검정 p = %.4f"
          % (w, n, float(np.median(dl)), p))
    print("  적합된 g: 중앙 %.2f  (범위 %.2f ~ %.2f)"
          % (float(np.median(gs)), min(gs), max(gs)))

    # ---------------- [2] 전역 적합 ----------------
    print()
    print("=" * 92)
    print("[2] 전역 적합 -- fold 의존성이 없다")
    print("=" * 92)
    nuC, kC = fit_arm(tr0, ST.S_C)
    g0, nuk0, med0f = fit_D(tr0)
    dg = []
    for s in SEQS:
        c = eval_arm(data[s], ST.S_C, nuC, kC)
        d = eval_D(data[s], g0, nuk0, med0f)
        dg.append(d - c)
    wg, ng, pg = sign_two_sided([-x for x in dg])
    print("  전역 적합값: g = %.2f,  nu = %.1f,  k = %.4g" % (g0, nuk0[0], nuk0[1]))
    print("  **D 승 %d/%d**,  짝지은 차이 중앙 %+.3f nats,  양측 p = %.4f"
          % (wg, ng, float(np.median(dg)), pg))

    # ---------------- 판정 ----------------
    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md)")
    print("=" * 92)
    gmed = float(np.median(gs))
    if abs(gmed) < 0.1:
        print("  [3] 적합된 g 의 중앙값이 %.2f -- **사전 등록대로 sigma 를 사실상 "
              "버렸다.**" % gmed)
        print("      [1] 이 어떻게 나오든 **sigma 는 기여하지 않는다** 로 읽는다.")
    if w == 7:
        print("  [1] D 가 7/7 승 => **원고 결론 2 를 고쳐야 한다.**")
        print("      sigma 는 크기와 *함께* 쓰면 쓸모가 있다.")
    elif w >= 5:
        print("  [1] D 가 %d/7 승 => 방향은 sigma 쪽이나 유의성 미달." % w)
        print("      **결론 2 에 단서를 단다.**")
    else:
        print("  [1] D 가 %d/7 승 (4/7 이하) => **원고 결론 2 가 선다.**" % w)
        print("      합쳐도 크기 모형을 못 이긴다.")
    print()
    print("  단, D 가 이겨도 그것만으로 'sigma 를 쓰라' 가 되지 않는다 --")
    print("  5.3 절의 네 경로는 여전히 음성이고, **밀도 모형에서 이기는 것과**")
    print("  **연관 비용에서 이득을 내는 것은 다른 문제다** (사전 등록에 적어 둔 것).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
