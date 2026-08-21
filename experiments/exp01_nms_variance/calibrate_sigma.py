# -*- coding: utf-8 -*-
"""실험 1e -- sigma 를 보정한 뒤 넣으면 쓸 수 있는가.

**사전 등록은 README 의 "실험 1e" 절. 자료보다 먼저 커밋했다.**

핵심 설계는 **leave-one-sequence-out** 이다. 같은 시퀀스 안에서 보정하면 당연히
맞는다 -- 그건 아무 말도 안 해준다. 6개로 적합시키고 **본 적 없는 1개**에서
평가해야 "배포 가능한 보정인가" 를 묻는 것이 된다.

주 평가지표: **적용범위(coverage)**. 보정됐다면 z^2 <= chi2(2)_95% 인 비율이 95%.
진짜 사전 점검: **박스크기 모형(C)을 NLL 로 이기는가.** coverage 만 맞추는 건 배율
하나로 되지만, C 를 이기려면 sigma 에 크기 이상의 정보가 있어야 한다.

사용법:
    python experiments/exp01_nms_variance/calibrate_sigma.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]
TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = NMS, "-dfl" = DFL
Q95 = chi2.ppf(0.95, 2)
QMED = chi2.ppf(0.5, 2)
NBIN = 4                       # E2 의 구간 수. **미리 못박는다** (사전 등록 함정 3)


def load(seq):
    d = np.load(Path("data/exp01") / ("%s-FRCNN%s.npz" % (seq, TAG)))
    sxx, sxy, syy = d["sxx"], d["sxy"], d["syy"]
    det2 = sxx * syy - sxy ** 2
    ok = ~np.isnan(sxx) & (det2 > 1e-9) & (sxx > 1e-9) & (syy > 1e-9)
    return dict(sxx=sxx[ok], sxy=sxy[ok], syy=syy[ok],
                ex=d["dcx"][ok], ey=d["dcy"][ok], h=d["h"][ok])


def quad(S, e_x, e_y):
    """eps^T S^-1 eps. S = (sxx, sxy, syy) 튜플."""
    sxx, sxy, syy = S
    dt = sxx * syy - sxy ** 2
    return (syy * e_x ** 2 - 2 * sxy * e_x * e_y + sxx * e_y ** 2) / dt


def nll(S, e_x, e_y):
    """가우시안 음의 로그우도 (상수항 제외)."""
    sxx, sxy, syy = S
    return 0.5 * (quad(S, e_x, e_y) + np.log(sxx * syy - sxy ** 2))


def scaled(S, k):
    return (S[0] * k, S[1] * k, S[2] * k)


def cover(S, e_x, e_y):
    return float(np.mean(quad(S, e_x, e_y) <= Q95))


# ---- 모형 ---------------------------------------------------------------
# 각 모형은 (train 자료) -> (test 자료에 쓸 Sigma) 를 돌려주는 함수다.

def fit_A0(tr):
    return lambda te: (te["sxx"], te["sxy"], te["syy"])


def fit_A1(tr):
    """전역 상수배. k = mean(z^2)/2 (2차원 배율의 MLE)."""
    k = float(np.mean(quad((tr["sxx"], tr["sxy"], tr["syy"]),
                           tr["ex"], tr["ey"]))) / 2.0
    return lambda te: scaled((te["sxx"], te["sxy"], te["syy"]), max(k, 1e-12))


def fit_B(tr):
    """상수 Sigma -- 검출별 정보 0."""
    C = np.cov(np.stack([tr["ex"], tr["ey"]]))
    return lambda te: (np.full(len(te["ex"]), C[0, 0]),
                       np.full(len(te["ex"]), C[0, 1]),
                       np.full(len(te["ex"]), C[1, 1]))


def fit_C(tr):
    """박스크기만. Sigma = k h^2 I. **넘어야 할 기준선.**"""
    hs = tr["h"] ** 2
    z = quad((hs, np.zeros_like(hs), hs), tr["ex"], tr["ey"])
    k = float(np.mean(z)) / 2.0
    return lambda te: (te["h"] ** 2 * k, np.zeros(len(te["ex"])), te["h"] ** 2 * k)


def fit_E1(tr):
    """컨포멀 배율. 보정집합 z^2 의 95% 분위를 chi2 95% 에 맞춘다.

    분포가정을 안 쓴다 -- 평균이 아니라 **분위수**를 맞추므로 꼬리가 두꺼워도
    적용범위가 목표대로 나온다. A1(평균 기반)과 갈리는 지점이 여기다.
    """
    z = quad((tr["sxx"], tr["sxy"], tr["syy"]), tr["ex"], tr["ey"])
    k = float(np.quantile(z, 0.95)) / Q95
    return lambda te: scaled((te["sxx"], te["sxy"], te["syy"]), max(k, 1e-12))


def fit_E2(tr):
    """조건부 컨포멀. sigma/h 4분위마다 배율을 따로 맞춘다.

    E1 이 규모만 고친다면 E2 는 **모양의 일부**까지 고친다. 구간 경계는
    train 에서 정하고 test 에 그대로 적용한다 (누출 방지).
    """
    r_tr = (tr["sxx"] * tr["syy"] - tr["sxy"] ** 2) ** 0.25 / tr["h"]
    edges = np.quantile(r_tr, np.linspace(0, 1, NBIN + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    z = quad((tr["sxx"], tr["sxy"], tr["syy"]), tr["ex"], tr["ey"])
    ks = []
    for b in range(NBIN):
        m = (r_tr >= edges[b]) & (r_tr < edges[b + 1])
        ks.append(float(np.quantile(z[m], 0.95)) / Q95 if m.sum() >= 20 else np.nan)
    ks = np.array(ks)
    if np.isnan(ks).any():                       # 표본 부족 구간은 전역값으로
        ks[np.isnan(ks)] = float(np.quantile(z, 0.95)) / Q95

    def apply(te):
        r_te = (te["sxx"] * te["syy"] - te["sxy"] ** 2) ** 0.25 / te["h"]
        k = np.clip(np.digitize(r_te, edges[1:-1]), 0, NBIN - 1)
        kk = np.maximum(ks[k], 1e-12)
        return scaled((te["sxx"], te["sxy"], te["syy"]), kk)
    return apply


MODELS = [("A0 날것 Sigma_d", fit_A0), ("A1 전역 상수배", fit_A1),
          ("B  상수 Sigma", fit_B), ("C  박스크기만", fit_C),
          ("E1 컨포멀 배율", fit_E1), ("E2 조건부 컨포멀", fit_E2)]


def main():
    data = {s: load(s) for s in SEQS}
    print("=" * 88)
    print("실험 1e -- leave-one-sequence-out 보정 (검출기 yolov8m, 200~299프레임)")
    print("=" * 88)
    print("적합: 6시퀀스   평가: 나머지 1시퀀스 (본 적 없는 장면)")
    # '%(' 는 매핑 키 참조로 파싱된다. 반드시 '%%(' 로 이스케이프할 것.
    print("주 평가지표 coverage = z^2 <= chi2(2)_95%% (%.2f) 인 비율. 목표 95%%" % Q95)
    print()

    res = {name: {"cov": [], "nll": []} for name, _ in MODELS}
    print("%-12s" % "held-out" + "".join("%15s" % n.split()[0] for n, _ in MODELS))
    print("-" * 88)
    for held in SEQS:
        tr_parts = [data[s] for s in SEQS if s != held]
        tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in data[held]}
        te = data[held]
        cells = []
        for name, fit in MODELS:
            S = fit(tr)(te)
            c = cover(S, te["ex"], te["ey"])
            v = float(np.mean(nll(S, te["ex"], te["ey"])))
            res[name]["cov"].append(c)
            res[name]["nll"].append(v)
            cells.append("%7.1f%% %6.2f" % (100 * c, v))
        print("%-12s" % held + "".join("%15s" % c for c in cells))

    print("-" * 88)
    print("%-12s" % "중앙값" + "".join(
        "%15s" % ("%7.1f%% %6.2f" % (100 * np.median(res[n]["cov"]),
                                     np.median(res[n]["nll"])))
        for n, _ in MODELS))
    print()
    print("각 칸은 'coverage  NLL'. coverage 는 95% 에 가까울수록, NLL 은 낮을수록 좋다.")
    print()

    print("=" * 88)
    print("사전 등록한 판정")
    print("=" * 88)
    cov_e1 = np.median(res["E1 컨포멀 배율"]["cov"])
    cov_e2 = np.median(res["E2 조건부 컨포멀"]["cov"])
    nll_c = np.median(res["C  박스크기만"]["nll"])
    best_e = min(np.median(res["E1 컨포멀 배율"]["nll"]),
                 np.median(res["E2 조건부 컨포멀"]["nll"]))
    # **95% 에 가까운 쪽**을 고른다. max 를 쓰면 과잉적용범위를 상으로 주게 된다
    # (99% 가 96% 보다 낫다고 판정해 버린다). 목표는 크기가 아니라 일치다.
    best_cov = min((cov_e1, "E1"), (cov_e2, "E2"), key=lambda t: abs(t[0] - 0.95))[0]
    print("  E 의 coverage 중앙값  E1 %.1f%%  E2 %.1f%%   (목표 95%%)"
          % (100 * cov_e1, 100 * cov_e2))
    print("  NLL 중앙값            E 최선 %.3f   C(박스크기) %.3f" % (best_e, nll_c))
    print()
    ok_cov = 0.90 - 1e-9 <= best_cov <= 0.97 + 1e-9
    ok_nll = best_e < nll_c
    if ok_cov and ok_nll:
        print("  => **보정하면 쓸 수 있다.** coverage 를 맞추면서 크기모형도 이긴다.")
    elif ok_cov:
        print("  => 스케일은 고쳐지지만 **모양은 여전히 틀렸다.**")
        print("     coverage 는 맞는데 박스크기 모형을 못 이긴다.")
        print("     sigma 에 '크기 이상의 정보' 는 없다는 [2b] 결론과 같다.")
    else:
        print("  => **보정이 장면을 넘어가지 못한다.** held-out 에서 coverage 도 못 맞춘다.")
        print("     제약 1(시퀀스마다 스케일이 다르다)이 최종 결론이 된다.")
    print()
    print("  * coverage 와 NLL 을 같이 본다. 하나만 골라 보고하지 않는다 (함정 2).")

    # ---- 탐색적 (사전 등록 아님. 판정에 쓰지 않는다) ----------------------
    # coverage 는 95% 분위 **한 점**만 본다. 꼬리를 맞춰도 분포의 가운데가
    # 어긋나 있을 수 있다. 보정 후 z^2 중앙값이 chi2(2) 중앙값과 맞는지 같이 본다.
    print()
    print("=" * 88)
    print("[탐색적] 보정 뒤 분포의 가운데도 맞는가 -- z^2 중앙값 / chi2(2) 중앙값")
    print("=" * 88)
    print("  coverage 는 95%% 분위 한 점만 본다. 꼬리를 맞춰도 가운데가 틀릴 수 있다.")
    print("  1.0 이면 완벽. 판정에는 쓰지 않는다 (사전 등록에 없다).")
    print()
    print("%-12s" % "held-out" + "".join("%13s" % n.split()[0] for n, _ in MODELS))
    print("-" * 88)
    mid = {n: [] for n, _ in MODELS}
    for held in SEQS:
        tr_parts = [data[s] for s in SEQS if s != held]
        tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in data[held]}
        te = data[held]
        cells = []
        for name, fit in MODELS:
            S = fit(tr)(te)
            r = float(np.median(quad(S, te["ex"], te["ey"])) / QMED)
            mid[name].append(r)
            cells.append("%13.2f" % r)
        print("%-12s" % held + "".join(cells))
    print("-" * 88)
    print("%-12s" % "중앙값" + "".join("%13.2f" % np.median(mid[n]) for n, _ in MODELS))
    print()
    print("  -> 1.0 에서 멀면 꼬리만 맞추고 가운데는 못 맞춘 것이다.")


if __name__ == "__main__":
    main()
