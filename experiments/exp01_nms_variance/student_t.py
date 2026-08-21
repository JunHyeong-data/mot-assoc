# -*- coding: utf-8 -*-
"""실험 1f -- 분포족을 Student-t 로 바꾸면 sigma 가 쓸모 있어지는가.

**사전 등록은 README 의 "실험 1f" 절. 자료보다 먼저 커밋했다.**

실험 1e 가 실패 원인을 꼬리로 지목했다 (꼬리비 12~213, 가우시안이면 1.0).
꼬리를 담는 분포족을 주면 고쳐져야 한다. 안 고쳐지면 원인이 꼬리가 아니다.

**가장 중요한 설계**: t 를 개체별 Sigma_d 에만 주면 t 가 이기는 게 당연하다
(꼬리 유연성 덕이지 Sigma_d 에 정보가 있어서가 아니다). **모든 모형에 t 를
똑같이 주고 nu, k 를 각자 적합시킨다.** 그래야 남는 차이가 "개체별 Sigma_d 가
박스크기 h^2 보다 나은가" 하나뿐이다.

2차원 t: eps ~ t_nu(0, S) 이면 z^2 = eps^T S^-1 eps 에 대해 z^2/2 ~ F(2, nu).
따라서 적용범위 임계값은 2*F_0.95(2, nu) 이고 **nu 마다 다르다.**

사용법:
    python experiments/exp01_nms_variance/student_t.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, f as fdist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]
# 소스 조건. "" = NMS 후보 산포(실험 1), "-dfl" = DFL 분포 분산(실험 1g).
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
DIM = 2
# nu 격자. inf 는 가우시안이다 (같은 코드로 두 분포족을 다룬다).
NUS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 50.0, 100.0, np.inf]


def load(seq):
    d = np.load(Path("data/exp01") / ("%s-FRCNN%s.npz" % (seq, TAG)))
    sxx, sxy, syy = d["sxx"], d["sxy"], d["syy"]
    det2 = sxx * syy - sxy ** 2
    ok = ~np.isnan(sxx) & (det2 > 1e-9) & (sxx > 1e-9) & (syy > 1e-9)
    return dict(sxx=sxx[ok], sxy=sxy[ok], syy=syy[ok],
                ex=d["dcx"][ok], ey=d["dcy"][ok], h=d["h"][ok])


def z2_and_logdet(S, e_x, e_y):
    sxx, sxy, syy = S
    dt = sxx * syy - sxy ** 2
    z2 = (syy * e_x ** 2 - 2 * sxy * e_x * e_y + sxx * e_y ** 2) / dt
    return z2, np.log(dt)


def nll_t(z2, logdet, nu, k):
    """2차원 t 의 **완전한** 음의 로그밀도. 상수항을 다 넣는다.

    nu 를 적합시키므로 logGamma 항이 nu 에 의존한다. 빼면 nu 비교가 틀린다.
    축척행렬은 k*Sigma 이므로 logdet -> logdet + DIM*log(k), z2 -> z2/k.
    """
    ld = logdet + DIM * np.log(k)
    zz = z2 / k
    if np.isinf(nu):                      # 가우시안 극한
        return 0.5 * (zz + ld + DIM * np.log(2 * np.pi))
    return (-gammaln((nu + DIM) / 2) + gammaln(nu / 2)
            + (DIM / 2) * np.log(nu * np.pi) + 0.5 * ld
            + ((nu + DIM) / 2) * np.log1p(zz / nu))


def fit_nu_k(z2, logdet):
    """train 에서 (nu, k) 를 함께 적합시킨다. nu 는 격자, k 는 1차원 최적화."""
    best = (np.inf, None, None)
    for nu in NUS:
        r = minimize_scalar(
            lambda lk: float(np.mean(nll_t(z2, logdet, nu, np.exp(lk)))),
            bounds=(np.log(1e-8), np.log(1e8)), method="bounded")
        if r.fun < best[0]:
            best = (r.fun, nu, float(np.exp(r.x)))
    return best[1], best[2]


def q95(nu):
    """z^2 의 95% 임계값. t 는 2*F_0.95(2, nu), 가우시안은 chi2_0.95(2)."""
    return chi2.ppf(0.95, DIM) if np.isinf(nu) else DIM * fdist.ppf(0.95, DIM, nu)


# ---- 축척행렬 만들기 (모형별) -------------------------------------------
def S_A(d):
    return (d["sxx"], d["sxy"], d["syy"])


def S_C(d):
    z = np.zeros(len(d["h"]))
    return (d["h"] ** 2, z, d["h"] ** 2)


def S_B_from(tr):
    C = np.cov(np.stack([tr["ex"], tr["ey"]]))
    return lambda d: (np.full(len(d["ex"]), C[0, 0]),
                      np.full(len(d["ex"]), C[0, 1]),
                      np.full(len(d["ex"]), C[1, 1]))


def main():
    data = {s: load(s) for s in SEQS}
    print("=" * 92)
    print("실험 1f/1g -- Student-t. leave-one-sequence-out  소스 TAG=%r" % (TAG or "NMS"))
    print("=" * 92)
    print("**t 를 모든 모형에 똑같이 준다.** 그래야 남는 차이가 Sigma_d 대 h^2 뿐이다.")
    print("NLL 에 상수항을 전부 넣었다 -> 실험 1e 수치와 log(2pi)=1.838 만큼 다르다.")
    print()

    fams = [("T", False), ("G", True)]     # G 는 nu=inf 로 고정
    arms = [("A", S_A), ("C", S_C), ("B", None)]
    res = {"%s-%s" % (f, a): {"nll": [], "cov": [], "nu": []}
           for f, _ in fams for a, _ in arms}

    hdr = "".join("%14s" % ("%s-%s" % (f, a)) for f, _ in fams for a, _ in arms)
    print("%-11s" % "held-out" + hdr)
    print("-" * 92)
    for held in SEQS:
        tr_parts = [data[s] for s in SEQS if s != held]
        tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in data[held]}
        te = data[held]
        cells = []
        for fam, gauss in fams:
            for aname, sfun in arms:
                sf = S_B_from(tr) if sfun is None else sfun
                z_tr, ld_tr = z2_and_logdet(sf(tr), tr["ex"], tr["ey"])
                if gauss:
                    nu = np.inf
                    r = minimize_scalar(
                        lambda lk: float(np.mean(nll_t(z_tr, ld_tr, nu, np.exp(lk)))),
                        bounds=(np.log(1e-8), np.log(1e8)), method="bounded")
                    k = float(np.exp(r.x))
                else:
                    nu, k = fit_nu_k(z_tr, ld_tr)
                z_te, ld_te = z2_and_logdet(sf(te), te["ex"], te["ey"])
                v = float(np.mean(nll_t(z_te, ld_te, nu, k)))
                c = float(np.mean(z_te / k <= q95(nu)))
                key = "%s-%s" % (fam, aname)
                res[key]["nll"].append(v)
                res[key]["cov"].append(c)
                res[key]["nu"].append(nu)
                cells.append("%6.2f %5.0f%%" % (v, 100 * c))
        print("%-11s" % held + "".join("%14s" % c for c in cells))

    print("-" * 92)
    print("%-11s" % "중앙값" + "".join(
        "%14s" % ("%6.2f %5.0f%%" % (np.median(res["%s-%s" % (f, a)]["nll"]),
                                     100 * np.median(res["%s-%s" % (f, a)]["cov"])))
        for f, _ in fams for a, _ in arms))
    print()
    print("각 칸은 'NLL  coverage'. NLL 낮을수록, coverage 는 95%에 가까울수록 좋다.")
    print()
    print("적합된 자유도 nu (fold 별, 함정 2):")
    for a, _ in arms:
        print("  T-%s : %s" % (a, ", ".join("%g" % v for v in res["T-%s" % a]["nu"])))

    print()
    print("=" * 92)
    print("사전 등록한 판정 -- T-A vs T-C 하나다")
    print("=" * 92)
    ta, tc = np.median(res["T-A"]["nll"]), np.median(res["T-C"]["nll"])
    ga, gc = np.median(res["G-A"]["nll"]), np.median(res["G-C"]["nll"])
    # **짝지어 본다.** fold 마다 난이도가 달라서 두 목록의 중앙값을 따로 내
    # 비교하면 짝이 깨지고 부호가 뒤집힐 수 있다. 실제로 DFL 소스에서
    # 중앙값끼리는 T-C 승인데 fold 별로는 T-A 가 6/7 이었다. (2026-08-17)
    d = np.array(res["T-A"]["nll"]) - np.array(res["T-C"]["nll"])
    win = int((d < 0).sum())
    print("  T-A %.3f   vs   T-C %.3f     <- 짝 안 지은 중앙값 (참고용)" % (ta, tc))
    print("  **짝지은 차이 T-A − T-C**: 중앙 %+.3f  평균 %+.3f  A 승 %d/7"
          % (np.median(d), d.mean(), win))
    print("     fold 별: %s" % " ".join("%+.3f" % v for v in d))
    print("  G-A %.3f   vs   G-C %.3f     <- 분포족 바꾸기 전" % (ga, gc))
    print()
    print("  분포족 효과: A 는 %+.3f, C 는 %+.3f (t - 가우시안, 음수면 t 가 낫다)"
          % (ta - ga, tc - gc))
    # n=7 짝 표본의 부호검정. 6/7 이면 한쪽꼬리 p=0.0625 로 0.05 를 못 넘는다.
    from math import comb
    p_one = sum(comb(7, i) for i in range(win, 8)) / 2 ** 7
    print("  부호검정(한쪽꼬리) p = %.4f   <- n=7 이라 6/7 로도 0.05 를 못 넘는다" % p_one)
    print()
    if np.median(d) < 0:
        print("  => **분포족이 문제였다.** 꼬리를 담으니 Sigma_d 가 크기모형을 이긴다.")
    else:
        print("  => **꼬리가 원인이 아니다.** t 를 줘도 Sigma_d 가 크기모형에 진다.")
        print("     sigma 에 '크기 이상의 정보' 가 없다는 것이 **분포족과 무관하게**")
        print("     확정된다. 실험 1e [2b] 와 같은 결론이고, 이제 변명거리가 없다.")


if __name__ == "__main__":
    main()
