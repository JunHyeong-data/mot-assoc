# -*- coding: utf-8 -*-
"""
스칼라 산포 대신 공분산 Sigma_d 로 묻는다.

왜 바꾸나
  run_sequence.py 의 주 종말점은 스칼라 s_c 를 상자높이 h 로 나눈 값이다.
  h 로 나누는 정규화가 크기 교란을 다 걷어내는지 확실하지 않다.

  공분산을 쓰면 그 걱정이 원리적으로 사라진다. Sigma_d 는 오차 eps 와 단위가
  같으므로 eps^T Sigma_d^-1 eps 가 무차원이고, 크기는 저절로 상쇄된다.

  * 정정 (2026-08-16): 원래 이 자리에 "h 통제 편상관이 0.044 라 원시 상관의
    대부분이 크기 교란이었다" 고 적혀 있었다. **그 0.044 는 NMS in-place 버그의
    산물이고 고친 값은 +0.336 이다.** 크기 교란은 0.119 뿐이다. 공분산으로 가는
    동기는 남지만("무차원이라 깨끗하다"), '스칼라가 실패했으니 갈아탄다' 는
    서술은 틀렸다. 스칼라 종말점은 관문을 통과했다.

  이론과도 이어진다. mahalanobis_vs_bhattacharyya.py 에서 검출 불확실성이
  할당에 도달하는 유일한 통로가 0.5 ln|(Sigma_t + Sigma_d)/2| 였다. 필요한 것은
  스칼라 산포가 아니라 Sigma_d 자체다.

결정적 질문
  개체별 Sigma_d 가 상수 Sigma 보다 실제 오차를 잘 설명하는가?
  상수 Sigma_d 는 순수 행상수라 통로 기여가 0 이다 (assignment_invariance.py).
  따라서 이 우도 비교가 곧 '통로에 실제 정보가 흐르는가' 이다.

  진다면: NMS 후보 공분산은 쓸 수 없는 불확실성 소스다. 다른 소스를 찾아야 한다.

사용법:
    python experiments/exp01_nms_variance/analyze_covariance.py [시퀀스명] [TAG]

    TAG 는 run_all.py 의 갈래 접미사다 (빈칸 | -m60 | -fork).
        analyze_covariance.py MOT17-02-FRCNN -fork
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SEQ = sys.argv[1] if len(sys.argv) > 1 else "MOT17-02-FRCNN"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
NPZ = Path("data/exp01") / f"{SEQ}{TAG}.npz"
if not NPZ.exists():
    sys.exit(f"{NPZ} 가 없다. 먼저 run_sequence.py 를 돌릴 것.")

d = np.load(NPZ)
# Sigma_d 가 어느 좌표계인지 확인한다. 원본 좌표가 아니면 [1] 의 보정 배율이
# letterbox gain^2 만큼 부풀어 나온다. 옛 npz 에는 이 필드가 없다.
_coords = str(d["coords"]) if "coords" in d else "unknown"
if _coords != "original":
    print("*** 경고: 이 npz 의 Sigma_d 좌표계가 '%s' 다. [1] 의 보정 배율을"
          " 믿지 말 것. run_all.py 로 다시 낼 것. ***" % _coords)
sxx, sxy, syy = d["sxx"], d["sxy"], d["syy"]
dcx, dcy, hh, vis = d["dcx"], d["dcy"], d["h"], d["vis"]
frame = d["frame"]

ok = ~np.isnan(sxx)
# 특이/퇴화 공분산 제외
det2 = sxx * syy - sxy ** 2
ok &= (det2 > 1e-9) & (sxx > 1e-9) & (syy > 1e-9)

sxx, sxy, syy = sxx[ok], sxy[ok], syy[ok]
e = np.stack([dcx[ok], dcy[ok]], 1)
det2 = sxx * syy - sxy ** 2
n = len(e)

print("=" * 72)
print(f"공분산 기반 분석 -- {SEQ}{TAG}   (좌표계 {_coords})")
print("=" * 72)
print(f"  사용 가능 표본 {n} (전체 {len(d['sxx'])}, 퇴화 공분산 제외)")
print(f"  오차 중앙값 |eps| = {np.median(np.hypot(*e.T)):.2f} px")
print(f"  Sigma_d 의 일반화표준편차 중앙값 = {np.median(det2 ** 0.25):.2f} px")
print()


def quad(S_xx, S_xy, S_yy, e):
    """eps^T S^-1 eps, 2x2 를 닫힌형으로."""
    dt = S_xx * S_yy - S_xy ** 2
    return (S_yy * e[:, 0] ** 2 - 2 * S_xy * e[:, 0] * e[:, 1]
            + S_xx * e[:, 1] ** 2) / dt


def nll(S_xx, S_xy, S_yy, e):
    """가우시안 음의 로그우도 (상수항 제외). 0.5*(z^2 + ln|S|)"""
    dt = S_xx * S_yy - S_xy ** 2
    return 0.5 * (quad(S_xx, S_xy, S_yy, e) + np.log(dt))


print("=" * 72)
print("[1] 보정(calibration) -- Sigma_d 의 크기가 맞나")
print("=" * 72)
z2 = quad(sxx, sxy, syy, e)
print(f"  z^2 = eps^T Sigma_d^-1 eps   중앙값 {np.median(z2):.1f}")
print(f"  잘 보정됐다면 chi2(2) 중앙값 {chi2.ppf(0.5, 2):.3f} 이어야 한다")
print(f"  비율 {np.median(z2) / chi2.ppf(0.5, 2):.1f}배")
print()
print(f"  z^2 <= chi2(2) 95% 분위({chi2.ppf(0.95, 2):.2f}) 인 비율: "
      f"{(z2 <= chi2.ppf(0.95, 2)).mean():.1%}  (이상적으로 95%)")
print()
print("  -> 비율이 1 에서 크게 벗어나면 Sigma_d 가 실제 오차 규모를 못 맞춘다는 뜻이다.")
print("     단순 배율 문제라면 상수배 보정으로 고칠 수 있다. 그건 [2] 에서 본다.")

print()
print("=" * 72)
print("[2] 결정적 비교 -- 개체별 Sigma_d 가 상수 Sigma 를 이기는가")
print("=" * 72)
print("  두 모형을 같은 오차 자료에 적합시키고 음의 로그우도를 비교한다.")
print("  공평하게, 두 모형 모두 전역 배율 k 를 최적화해 준다 (보정 후 비교).")
print()


def best_scale(S_xx, S_xy, S_yy, e):
    """k*Sigma 형태에서 NLL 을 최소화하는 k 의 닫힌해: k = mean(z^2)/2"""
    return max(float(np.mean(quad(S_xx, S_xy, S_yy, e))) / 2.0, 1e-12)


# 모형 A: 개체별 Sigma_d (NMS 후보에서 추정), 전역 배율 보정
kA = best_scale(sxx, sxy, syy, e)
nllA = nll(sxx * kA, sxy * kA, syy * kA, e).mean()

# 모형 B: 상수 Sigma (전체 오차의 표본공분산). 검출별 정보 없음
Sb = np.cov(e.T)
bxx = np.full(n, Sb[0, 0]); bxy = np.full(n, Sb[0, 1]); byy = np.full(n, Sb[1, 1])
kB = best_scale(bxx, bxy, byy, e)
nllB = nll(bxx * kB, bxy * kB, byy * kB, e).mean()

# 모형 C: 상자 크기만 쓰는 상수 (Sigma = c * h^2 * I). '크기만으로도 되는가'
hs = hh[ok] ** 2
cxx = hs.copy(); cxy = np.zeros(n); cyy = hs.copy()
kC = best_scale(cxx, cxy, cyy, e)
nllC = nll(cxx * kC, cxy * kC, cyy * kC, e).mean()

print(f"  {'모형':<34}{'평균 NLL':>12}{'상수 대비':>12}")
print("  " + "-" * 58)
print(f"  {'A. 개체별 Sigma_d (NMS 후보)':<34}{nllA:>12.4f}{nllA - nllB:>+12.4f}")
print(f"  {'B. 상수 Sigma (기준선)':<34}{nllB:>12.4f}{0.0:>+12.4f}")
print(f"  {'C. 상자크기만 (Sigma = k h^2 I)':<34}{nllC:>12.4f}{nllC - nllB:>+12.4f}")
print()
print("  NLL 이 낮을수록 좋다. A 가 B 보다 낮아야 '검출별 불확실성에 정보가 있다'.")
print("  A 가 C 보다도 낮아야 '상자 크기 이상의 정보가 있다'. 이게 진짜 관문이다.")
print()

# 프레임 블록 부트스트랩으로 NLL 차이의 CI
uf = np.unique(frame[ok])
byf = {f: np.where(frame[ok] == f)[0] for f in uf}
rng = np.random.default_rng(0)
dAB, dAC = [], []
la = nll(sxx * kA, sxy * kA, syy * kA, e)
lb = nll(bxx * kB, bxy * kB, byy * kB, e)
lc = nll(cxx * kC, cxy * kC, cyy * kC, e)
for _ in range(2000):
    idx = np.concatenate([byf[f] for f in rng.choice(uf, len(uf), replace=True)])
    dAB.append(la[idx].mean() - lb[idx].mean())
    dAC.append(la[idx].mean() - lc[idx].mean())
for lab, v in (("A - B (상수 대비)", dAB), ("A - C (크기모형 대비)", dAC)):
    lo, hi = np.percentile(v, [2.5, 97.5])
    win = "A 승" if hi < 0 else ("A 패" if lo > 0 else "판정 불가")
    print(f"  {lab:<26} 차이 {np.mean(v):+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  -> {win}")

print()
print("=" * 72)
print("[2b] A 에게 최선의 기회를 준다 -- 강건 배율 + 분산 바닥")
print("=" * 72)
print("  위의 배율 k=mean(z^2)/2 는 z^2 의 두꺼운 꼬리에 끌려간다. A 만 손해볼 수 있다.")
print("  (1) 중앙값 기반 강건 배율, (2) Sigma_d + lam*I 바닥을 격자탐색해 A 를 최적화한다.")
print("  그래도 지면 배율 탓이 아니다.")
print()


def rob_scale(S_xx, S_xy, S_yy, e):
    return max(float(np.median(quad(S_xx, S_xy, S_yy, e))) / chi2.ppf(0.5, 2), 1e-12)


rows_ = []
for lab, (Sx, Sy, Sxy_) in (("A. 개체별 Sigma_d", (sxx, syy, sxy)),
                            ("B. 상수 Sigma", (bxx, byy, bxy)),
                            ("C. 상자크기만", (cxx, cyy, cxy))):
    k = rob_scale(Sx, Sxy_, Sy, e)
    rows_.append((lab, nll(Sx * k, Sxy_ * k, Sy * k, e).mean()))

# A 에 분산 바닥 lam*median(h)^2 을 넣고 최적 lam 을 찾는다
best = (None, np.inf, None)
hmed2 = float(np.median(hh[ok])) ** 2
curve = []
for lam in np.concatenate([[0.0], np.logspace(-6, 4, 41)]):
    fx, fy = sxx + lam * hmed2, syy + lam * hmed2
    k = rob_scale(fx, sxy, fy, e)
    v = nll(fx * k, sxy * k, fy * k, e).mean()
    curve.append((lam, v))
    if v < best[1]:
        best = (lam, v, k)

print(f"  {'모형 (강건 배율)':<30}{'평균 NLL':>12}")
print("  " + "-" * 44)
for lab, v in rows_:
    print(f"  {lab:<30}{v:>12.4f}")
print(f"  {'A + 분산바닥 (최적 lam)':<30}{best[1]:>12.4f}   lam={best[0]:.2e}")
print()
print("  분산바닥 lam 을 키울수록 NLL 이 어떻게 되나 (lam -> 무한대면 Sigma_d 가 지워진다):")
for lam, v in curve:
    if lam in (0.0,) or (lam > 0 and abs(np.log10(lam) - round(np.log10(lam))) < 1e-9):
        print(f"    lam={lam:>9.3g}  NLL={v:>12.4f}")
print("  -> 바닥을 키울수록(= Sigma_d 를 지울수록) 좋아지면, 그 정보가 해롭다는 뜻이다.")
print()
bestA = min(best[1], rows_[0][1])
print(f"  A 최선 {bestA:.4f}  vs  B {rows_[1][1]:.4f}  vs  C {rows_[2][1]:.4f}")
if bestA < min(rows_[1][1], rows_[2][1]):
    print("  -> A 승. 배율 탓이었다. 검출별 정보가 있다.")
elif bestA < rows_[1][1]:
    print("  -> A 가 상수는 이기지만 크기모형은 못 이긴다. '크기 이상의 정보' 는 없다.")
else:
    print("  -> A 가 최선의 조건에서도 진다. NMS 후보 공분산은 쓸 수 없다.")

print()
print("=" * 72)
print("[3] 탐색적")
print("=" * 72)
r1 = spearmanr(det2 ** 0.25, np.hypot(*e.T))[0]
r2 = spearmanr(det2 ** 0.25 / hh[ok], np.hypot(*e.T) / hh[ok])[0]
print(f"  Spearman(일반화표준편차, |eps|)              = {r1:+.4f}")
print(f"  Spearman(둘 다 h 로 정규화)                  = {r2:+.4f}")
v_ = vis[ok]
for name, m in (("가시성>=0.7", v_ >= 0.7), ("가시성<0.7", v_ < 0.7)):
    if m.sum() > 50:
        print(f"  {name:12} n={m.sum():5d}  z^2 중앙값 {np.median(z2[m]):7.1f}  "
              f"NLL(A-B) {la[m].mean() - lb[m].mean():+.4f}")
