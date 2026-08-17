# -*- coding: utf-8 -*-
"""실험 10 -- **척추 검산.** 실험 1f 를 원고의 척추로 쓸 수 있는가.

## 왜 이걸 먼저 하는가

자체 심사 M6 이 "실험 1f 의 만장일치가 n=7 에서 유의성을 확보한 유일한 결과"
라고 했다. 원고를 거기 세우려면 **그 숫자부터 다른 경로로 다시 재야 한다**
(CLAUDE.md 규칙 3).

그리고 실제로 두 가지가 걸린다.

**(가) 검정 방향.** `student_t.py:182` 는 `P(X >= win)` 을 낸다. `win` 은 A 가
이긴 fold 수이고 0 이므로 p=1.0 이다. 이건 **"A 가 낫다" 는 가설의 한쪽꼬리**로
정확하다 -- A 는 낫지 않으니까. 그런데 **원고가 쓰려는 주장은 반대 방향**
(C 가 낫다)이고, 이 저장소의 다른 스크립트(`evaluate.py`, `loso.py`)는 전부
**양측**을 쓴다. **검정 관례가 파일마다 다르다.**

**(나) LOSO fold 는 독립이 아니다.** 각 fold 가 7개 중 6개로 적합하므로 이웃
fold 가 훈련자료를 5개 공유한다. 7/7 을 `1/128` 로 세면 **반보수적**이다.

## 그래서 무엇으로 가르는가 -- **전역 적합**

**A 와 C 는 모수 개수가 같다 (`k`, `nu` 둘 다).** 그러므로 두 모형을
**전체 자료로 한 번씩 적합**하고 시퀀스별로 평가하면, 과적합 비대칭 없이
**fold 의존성이 아예 없는** 비교가 된다. 거기서도 7/7 이면 (나) 는 무해하다.

  [1] LOSO (기존)     -- fold 의존성 있음
  [2] **전역 적합**   -- fold 의존성 없음. 모수 개수 같아서 공정
  [3] 적합값 안정성   -- fold 마다 (k, nu) 가 얼마나 흔들리는가

사용법:
    python experiments/exp10_spine/verify_1f.py          # NMS 소스 (실험 1f)
    python experiments/exp10_spine/verify_1f.py -dfl     # DFL 소스 (실험 1g)
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp01_nms_variance"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import student_t as ST                                          # noqa: E402

SEQS = ST.SEQS


def fit_arm(tr, sfun):
    """훈련자료로 (nu, k) 를 적합한다. student_t.py 의 절차를 그대로 쓴다."""
    z, ld = ST.z2_and_logdet(sfun(tr), tr["ex"], tr["ey"])
    return ST.fit_nu_k(z, ld)


def eval_arm(te, sfun, nu, k):
    z, ld = ST.z2_and_logdet(sfun(te), te["ex"], te["ey"])
    return float(np.mean(ST.nll_t(z, ld, nu, k)))


def sign_two_sided(d):
    """양측 부호검정. 동률은 뺀다 (exp06 감사에서 고친 것과 같은 처리)."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    win = int((nz > 0).sum())
    k = max(win, n - win)
    p = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return win, n, min(p, 1.0)


def main():
    tag = ST.TAG
    src = "DFL" if tag else "NMS"
    print("=" * 92)
    print("실험 10 -- 척추 검산. 실험 1f 를 원고 척추로 쓸 수 있는가  [소스 %s]" % src)
    print("=" * 92)
    print("A = 개체별 Sigma_d,  C = 상자크기 h^2 모형.  **모수 개수가 같다 (k, nu).**")
    print("NLL 이 낮을수록 좋다. d = NLL_A - NLL_C 이므로 **d > 0 이면 C 승**.")
    print()

    data = {s: ST.load(s) for s in SEQS}

    # ---------------- [1] LOSO ----------------
    print("=" * 92)
    print("[1] LOSO -- 기존 절차. fold 의존성이 있다")
    print("=" * 92)
    print("%-12s %9s %9s %9s   %9s %9s %9s %9s"
          % ("held-out", "NLL_A", "NLL_C", "d", "nu_A", "k_A", "nu_C", "k_C"))
    print("-" * 92)
    dl, pars = [], {"nu_A": [], "k_A": [], "nu_C": [], "k_C": []}
    for held in SEQS:
        parts = [data[s] for s in SEQS if s != held]
        tr = {k: np.concatenate([p[k] for p in parts]) for k in data[held]}
        te = data[held]
        nuA, kA = fit_arm(tr, ST.S_A)
        nuC, kC = fit_arm(tr, ST.S_C)
        a, c = eval_arm(te, ST.S_A, nuA, kA), eval_arm(te, ST.S_C, nuC, kC)
        dl.append(a - c)
        for nm, v in (("nu_A", nuA), ("k_A", kA), ("nu_C", nuC), ("k_C", kC)):
            pars[nm].append(v)
        print("%-12s %9.3f %9.3f %+9.3f   %9.1f %9.4g %9.1f %9.4g"
              % (held, a, c, a - c, nuA, kA, nuC, kC))

    w, n, p = sign_two_sided(dl)
    print("-" * 92)
    print("  C 승 %d/%d,  **양측** 부호검정 p = %.4f" % (w, n, p))
    print("  (student_t.py 는 'A 가 낫다' 한쪽꼬리를 내므로 p=1.0 이 나온다.")
    print("   그건 A 에 대한 답이지 C 에 대한 답이 아니다. 관례를 통일해야 한다)")

    # ---------------- [3] 적합값 안정성 ----------------
    print()
    print("=" * 92)
    print("[3] 적합값 안정성 -- fold 마다 모형이 얼마나 달라지는가")
    print("=" * 92)
    for nm in ("nu_A", "k_A", "nu_C", "k_C"):
        v = np.array(pars[nm], float)
        fin = v[np.isfinite(v)]
        cv = float(np.std(fin) / max(abs(np.mean(fin)), 1e-12)) if len(fin) else np.nan
        print("  %-6s 값 %s   CV = %.4f"
              % (nm, " ".join("%.4g" % x for x in v), cv))
    print()
    print("  **CV 가 0 에 가까우면 fold 마다 사실상 같은 모형을 쓴 것**이고,")
    print("  그러면 fold 간 차이는 held-out 자료(서로 겹치지 않는다)에서만 온다.")
    print("  즉 LOSO 의존성 걱정이 작아진다. 그래도 [2] 로 직접 확인한다.")

    # ---------------- [2] 전역 적합 ----------------
    print()
    print("=" * 92)
    print("[2] **전역 적합** -- fold 의존성이 아예 없는 비교")
    print("=" * 92)
    print("  두 모형을 전체 자료로 한 번씩 적합하고 시퀀스별로 평가한다.")
    print("  모수 개수가 같으므로 과적합 비대칭이 없다.")
    print()
    allt = {k: np.concatenate([data[s][k] for s in SEQS]) for k in data[SEQS[0]]}
    nuA, kA = fit_arm(allt, ST.S_A)
    nuC, kC = fit_arm(allt, ST.S_C)
    print("  전역 적합값:  A (nu=%.1f, k=%.4g)   C (nu=%.1f, k=%.4g)"
          % (nuA, kA, nuC, kC))
    print()
    print("%-12s %9s %9s %9s" % ("시퀀스", "NLL_A", "NLL_C", "d"))
    print("-" * 92)
    dg = []
    for s in SEQS:
        a, c = eval_arm(data[s], ST.S_A, nuA, kA), eval_arm(data[s], ST.S_C, nuC, kC)
        dg.append(a - c)
        print("%-12s %9.3f %9.3f %+9.3f" % (s, a, c, a - c))
    w2, n2, p2 = sign_two_sided(dg)
    print("-" * 92)
    print("  C 승 %d/%d,  **양측** 부호검정 p = %.4f" % (w2, n2, p2))

    # ---------------- 종합 ----------------
    print()
    print("=" * 92)
    print("척추로 쓸 수 있는가")
    print("=" * 92)
    print("  LOSO      : C 승 %d/7, 짝지은 차이 중앙 %+.3f nats, 양측 p = %.4f"
          % (w, float(np.median(dl)), p))
    print("  전역 적합 : C 승 %d/7, 짝지은 차이 중앙 %+.3f nats, 양측 p = %.4f"
          % (w2, float(np.median(dg)), p2))
    # 두 가지를 따로 본다. 섞으면 안 된다 --
    #   (가) 두 절차가 **서로 같은 답**을 내는가  -> LOSO 의존성 걱정의 해소 여부
    #   (나) 그 답이 **만장일치**인가            -> 척추로 쓸 만큼 센가
    agree = (w == w2)
    unanimous = (w == n) and (w2 == n2)
    print()
    print("  (가) 두 절차의 일치: %s (LOSO %d/7, 전역 %d/7)"
          % ("**같다**" if agree else "**다르다**", w, w2))
    print("  (나) 만장일치 여부: %s" % ("**그렇다**" if unanimous else "아니다"))
    print()
    if agree and unanimous:
        print("  => **척추로 쓸 수 있다.** 전역 적합은 fold 를 안 쓰므로 7개 평가가")
        print("     서로 겹치지 않는 자료 위에서 이뤄진다. LOSO 의존성 걱정이 해소됐다.")
    elif agree:
        print("  => 절차에 무관하게 재현되지만 **만장일치가 아니다.**")
        print("     방향은 믿을 만하고 **유의성은 못 얻는다.** 척추가 아니라 보조다.")
    else:
        print("  => **두 절차가 다른 답을 낸다.** 결과가 절차에 달렸다는 뜻이므로")
        print("     **척추로 쓰면 안 된다.**")
    print()
    print("  남는 한계: 장면이 7개다. 만장일치의 p 는 이론상 0.0156 이 바닥이고,")
    print("  장면들이 완전히 독립이라는 가정 위에서만 그렇다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
