# -*- coding: utf-8 -*-
"""설계 계산 -- **장면이 몇 개 있어야 판정이 되는가.**

## 이건 실험이 아니다. 사전 선언도 없다

트래커에 대해 아무 주장도 하지 않는다. **이미 잰 숫자들 위의 산술**이다.
그래서 사전 선언 없이 돈다. 대신 이 계산이 답하는 것은 실험만큼 중요하다:

  진행 기록이 여섯 번 "표본이 병목이다, MOT20/DanceTrack 을 넣어야 한다" 고
  적었는데 **몇 개가 필요한지는 한 번도 계산하지 않았다.** 데이터셋 하나에
  검출 캐싱만 110분이다. 그걸 쓰기 전에 계산부터 한다.

## 왜 지금 하는가 -- 실험 6 이 나쁜 소식을 하나 더 붙였다

실험 6 의 신탁 상한이 결합 +0.892 다. **효과가 작으면 필요 표본은 폭발한다.**

## 분석 단위는 장면이다

fold 를 늘려도 소용없다 (실험 1g 에서 확인). 시퀀스별 HOTA 차이가 서로
독립인 표본이고, 그 **장면 간 산포**가 검정력을 정한다.

## 두 가지 물음을 구별한다 -- 이게 이 계산의 핵심이다

  [우월성] 0.3 만큼의 차이를 **찾아내려면** 몇 개가 필요한가
  [동등성] "차이 없다" 고 **말하려면** 몇 개가 필요한가

**우리는 지금까지 두 번째를 첫 번째 방법으로 주장해 왔다.** "판정폭 0.3 안이면
차이 없음" 은 동등성 주장인데, 동등성은 우월성보다 표본이 더 든다.
`|추정치| < 0.3` 이 아니라 **신뢰구간 전체가 (-0.3, +0.3) 안에 들어가야** 한다.

사용법:
    python experiments/exp07_power/power.py
"""
import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    # **UTF-8 을 명시한다.** 이 저장소의 다른 스크립트가 한글을 제대로 뱉는 것은
    # `ultralytics` 가 import 될 때 stdout 을 UTF-8 로 바꿔주기 때문이다.
    # 이 파일은 ultralytics 를 안 거치므로 그냥 두면 cp949 로 나가 깨진다.
    # 부작용에 기대지 않고 여기서 직접 정한다.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate import build_data, SEQS                        # noqa: E402
from tracker.eval.collections.hota import HOTA               # noqa: E402

BAND = 0.3                  # 이 저장소가 쓰는 판정폭
ALPHA = 0.05
POWER = 0.80
NSIM = 20000
SEED = 20260818

GRIDJSON = Path("data/exp06/grid.json")
EXP05 = Path("data/exp06/exp05_perseq.json")

# 기록된 결합 HOTA (notes/progress.md 의 채택률 맞춤 표). 검산용.
RECORDED = {"iou": 61.00, "wn_size_rate": 56.02, "w_size_rate": 53.64,
            "w_dfl_rate": 53.04, "wn_dfl_rate": 52.10}


# ---------------------------------------------------------------- 검산
def verify_combined():
    """결합 HOTA 를 다시 재서 기록과 맞는지 본다.

    exp04 가 exp01 의 letterbox 버그를 잡은 방식이다 -- **같은 양을 다른
    경로로 한 번 더 재라.** 여기서 안 맞으면 아래 산포도 못 믿는다.
    """
    m = HOTA()
    ok = True
    print("%-16s %8s %8s   %s" % ("갈래", "다시 잼", "기록", "차이"))
    print("-" * 56)
    for arm, ref in RECORDED.items():
        per = {}
        for s in SEQS:
            d = build_data(s, arm)
            if d is not None:
                per[s] = m.eval_sequence(d)
        if not per:
            print("%-16s (트랙 없음)" % arm)
            ok = False
            continue
        got = 100 * float(np.mean(m.combine_sequences(per)["HOTA"]))
        bad = abs(got - ref) > 0.02
        ok &= not bad
        print("%-16s %8.3f %8.2f   %+.3f %s"
              % (arm, got, ref, got - ref, "** 불일치 **" if bad else "OK"))
    return ok


# ---------------------------------------------------------------- 검정력
def sign_crit(n, alpha=ALPHA):
    """양측 부호검정에서 기각에 필요한 최소 승수. 불가능하면 None.

    `2.0 ** n` 으로 직접 세면 n > 1023 에서 OverflowError 가 난다.
    binom 분포로 푼다.
    """
    c = int(stats.binom.isf(alpha / 2, n, 0.5)) + 1
    while c > n // 2 + 1 and 2.0 * stats.binom.sf(c - 2, n, 0.5) <= alpha:
        c -= 1                                   # 더 작은 c 로 되는지 확인
    if c > n or 2.0 * stats.binom.sf(c - 1, n, 0.5) > alpha:
        return None
    return c


def sign_power(n, delta, sd, alpha=ALPHA):
    """부호검정의 검정력. 차이가 delta 를 중심으로 대칭이라 보고
    p = P(d_i > 0) = Phi(delta/sd) 로 놓는다."""
    c = sign_crit(n, alpha)
    if c is None:
        return 0.0
    p = float(stats.norm.cdf(delta / sd))
    return float(stats.binom.sf(c - 1, n, p) + stats.binom.cdf(n - c, n, p))


def t_power(n, delta, sd, alpha=ALPHA):
    """일표본(짝지은) t 검정의 검정력. 비중심 t.

    **scipy 의 nct 는 ncp 가 10 부근일 때 nan 을 낸다** (확인함: ncp=10 에서
    nan, 5 와 20 에서는 정상). 그대로 두면 nan >= 0.8 이 False 라 이분법이
    반대로 간다. 유한하지 않으면 정규 근사로 갈아탄다 -- 그 영역은 검정력이
    이미 1 에 붙어 있어서 근사로 충분하다.
    """
    if n < 2 or sd <= 0:
        return 0.0
    df = n - 1
    ncp = delta / (sd / sqrt(n))
    crit = stats.t.ppf(1 - alpha / 2, df)
    p = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    if not np.isfinite(p):
        p = stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp)
    return float(min(max(float(p), 0.0), 1.0))


def boot_power(n, delta, d_emp, rng, alpha=ALPHA, nsim=NSIM):
    """**실측 차이 분포를 그대로 재표집한** 검정력 (t 검정 기준).

    정규 가정을 안 쓴다. 이 프로젝트는 꼬리가 두껍다는 것을 이미 쟀다
    (꼬리비 12~213, 실험 1e). 그래서 정규 공식과 나란히 놓고 본다.

    **방향까지 맞아야 검출로 센다.** 양측 기각만 세면 치우친 분포에서 반대
    방향 기각이 '검정력' 으로 잡혀 n 에 대해 단조가 아니게 된다 (실제로 그랬다).
    """
    x = np.asarray(d_emp, float)
    x = x - x.mean() + delta                     # 참 평균을 delta 로 옮긴다
    s = rng.choice(x, size=(nsim, n), replace=True)
    mu = s.mean(1)
    sd = s.std(1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mu / (sd / np.sqrt(n))
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    hit = (np.abs(t) > crit) & (np.sign(mu) == np.sign(delta))
    return float(np.mean(hit))


def n_for(fn, delta, sd, target=POWER, nmax=4000):
    """target 검정력에 도달하는 최소 n."""
    for n in range(2, nmax + 1):
        if fn(n, delta, sd) >= target:
            return n
    return None


def tost_power(n, sd, bound=BAND, alpha=ALPHA, nsim=NSIM, rng=None):
    """참 차이가 0 일 때 '|차이| < bound' 를 세울 확률 (TOST).

    두 단측 검정을 모두 통과해야 한다. 즉 90% 신뢰구간이 (-bound, +bound)
    안에 들어가야 한다.
    """
    if n < 2:
        return 0.0
    rng = rng or np.random.default_rng(SEED)
    s = rng.normal(0.0, sd, size=(nsim, n))
    mu = s.mean(1)
    se = s.std(1, ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha, n - 1)
    return float(np.mean(((mu + crit * se) < bound) & ((mu - crit * se) > -bound)))


def n_for_tost(sd, bound=BAND, target=POWER, nmax=4000):
    rng = np.random.default_rng(SEED)
    lo, hi = 2, nmax
    if tost_power(nmax, sd, bound, rng=rng) < target:
        return None
    while lo < hi:                               # 검정력은 n 에 단조 증가
        mid = (lo + hi) // 2
        if tost_power(mid, sd, bound, rng=rng) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------- 본체
def main():
    rng = np.random.default_rng(SEED)
    print("=" * 96)
    print("설계 계산 -- 장면이 몇 개 있어야 판정이 되는가")
    print("=" * 96)
    print("실험이 아니다. 이미 잰 숫자들 위의 산술이다. 사전 선언 없음.")
    print()

    if not GRIDJSON.exists() or not EXP05.exists():
        print("먼저 grid.py 를 돌려라. exp05 시퀀스별 값도 필요하다.")
        return 1

    print("=" * 96)
    print("[0] 검산 -- 결합 HOTA 를 다시 재서 기록과 맞추다")
    print("=" * 96)
    if not verify_combined():
        print()
        print("** 검산 실패. 아래 산포를 믿을 수 없다. 여기서 멈춘다 **")
        return 1

    # ---- 실측 대조군들의 시퀀스별 차이 ----
    G = json.loads(GRIDJSON.read_text())
    H = {s: {t: v for t, v in G["per_seq"][s].items()} for s in G["seqs"]}
    E = json.loads(EXP05.read_text())
    seqs = [s for s in SEQS if s in H]

    def dif(a, b):
        return np.array([a[s] - b[s] for s in seqs], float)

    thr = {t: {s: H[s][t] for s in seqs} for t in G["combined"]}
    contrasts = {
        "exp05 [1] w_dfl - w_size": dif(E["w_dfl_rate"], E["w_size_rate"]),
        "exp05 [4] wn_dfl - wn_size": dif(E["wn_dfl_rate"], E["wn_size_rate"]),
        "exp05 [3] wn_size - IoU": dif(E["wn_size_rate"], E["iou"]),
        "exp06 [A] thr 0.85 - 0.80": dif(thr["0.85"], thr["0.80"]),
        "exp06 [A] thr 0.90 - 0.80": dif(thr["0.90"], thr["0.80"]),
        # 실험 6 LOSO 의 R2-R0 (loso.py 가 낸 fold 별 값)
        "exp06 [1] LOSO R2 - R0": np.array(
            [-3.52, -0.19, 0.55, 0.20, 0.06, 3.25, 1.39]),
    }

    print()
    print("=" * 96)
    print("[1] 실측 -- 시퀀스별 HOTA 차이의 산포 (n=%d)" % len(seqs))
    print("=" * 96)
    print("%-30s %8s %8s %8s %8s" % ("대조", "평균", "SD", "|평균|/SD", "부호"))
    print("-" * 96)
    sds = []
    for name, d in contrasts.items():
        sd = float(np.std(d, ddof=1))
        sds.append(sd)
        print("%-30s %8.3f %8.3f %8.2f %6d/%d"
              % (name, d.mean(), sd, abs(d.mean()) / sd, int((d > 0).sum()), len(d)))
    sd_med = float(np.median(sds))
    print("-" * 96)
    print("  SD 중앙값 = %.3f HOTA,  범위 %.3f ~ %.3f" % (sd_med, min(sds), max(sds)))
    print()
    print("  **장면 간 산포가 판정폭(%.1f)의 %.0f~%.0f 배다.**"
          % (BAND, min(sds) / BAND, max(sds) / BAND))

    # ---- n=7 에서 무엇이 보이는가 ----
    print()
    print("=" * 96)
    print("[2] 민감도 -- 지금 가진 7개 장면으로 **검출 가능한 효과**")
    print("=" * 96)
    print("%-10s %10s %12s %12s" % ("SD", "n", "t 검정", "부호검정"))
    print("-" * 96)
    for sd in (min(sds), sd_med, max(sds)):
        row = []
        for fn in (t_power, sign_power):
            lo, hi = 0.0, 200.0
            for _ in range(60):                  # 이분법으로 검출가능 효과
                mid = (lo + hi) / 2
                if fn(7, mid, sd) >= POWER:
                    hi = mid
                else:
                    lo = mid
            row.append(hi)
        print("%-10.3f %10d %12.2f %12.2f" % (sd, 7, row[0], row[1]))
    print()
    print("  읽는 법: 검정력 %.0f%% 로 잡아내려면 효과가 이만큼 커야 한다는 뜻이다." % (100 * POWER))
    print("  판정폭 %.1f 과 비교하라." % BAND)

    # ---- 0.3 을 검출하려면 ----
    print()
    print("=" * 96)
    print("[3] 우월성 -- 판정폭 %.1f HOTA 를 검출하려면 장면이 몇 개 필요한가" % BAND)
    print("=" * 96)
    print("%-10s %14s %14s" % ("SD", "t 검정 n", "부호검정 n"))
    print("-" * 96)
    for sd in (min(sds), sd_med, max(sds)):
        nt = n_for(t_power, BAND, sd)
        ns = n_for(sign_power, BAND, sd)
        print("%-10.3f %14s %14s"
              % (sd, nt if nt else ">4000", ns if ns else ">4000"))

    print()
    print("  실측 분포를 그대로 재표집한 검정력 (정규 가정 없이):")
    print("  %-30s %8s %8s %8s" % ("대조", "n=7", "n=30", "n=100"))
    print("  " + "-" * 60)
    for name, d in contrasts.items():
        row = [boot_power(n, BAND, d, rng) for n in (7, 30, 100)]
        print("  %-30s %8.2f %8.2f %8.2f" % (name, *row))

    # ---- 차이 없음을 말하려면 ----
    print()
    print("=" * 96)
    print("[4] 동등성 -- **'차이 없다' 고 말하려면** 몇 개 필요한가 (TOST, +-%.1f)" % BAND)
    print("=" * 96)
    print("  '판정폭 안이면 차이 없음' 은 동등성 주장이다. 점추정이 아니라")
    print("  **신뢰구간 전체가 (-%.1f, +%.1f) 안에 들어가야** 한다." % (BAND, BAND))
    print()
    print("%-10s %14s   %s" % ("SD", "TOST n", "n=7 일 때 성공률"))
    print("-" * 96)
    for sd in (min(sds), sd_med, max(sds)):
        n = n_for_tost(sd)
        print("%-10.3f %14s   %.3f"
              % (sd, n if n else ">4000", tost_power(7, sd, rng=rng)))

    # ---- 우리 판정들이 이 설계의 해상도를 넘는가 ----
    print()
    print("=" * 96)
    print("[5] **기존 판정들이 이 설계의 해상도를 넘는가**")
    print("=" * 96)
    print("  각 대조를 자기 SD 로 재서, n=7 에서 검정력 %.0f%% 로 잡히는 최소 효과와" % (100 * POWER))
    print("  실제 관측 효과를 나란히 놓는다. **넘지 못하면 그 판정은 이 7개 장면")
    print("  밖으로 일반화할 수 없다** (이 7개 안에서의 측정이 틀렸다는 말이 아니다).")
    print()
    print("  %-30s %8s %8s %10s   %s" % ("대조", "관측", "SD", "검출가능", "판정"))
    print("  " + "-" * 76)
    for name, d in contrasts.items():
        sd = float(np.std(d, ddof=1))
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if t_power(7, mid, sd) >= POWER:
                hi = mid
            else:
                lo = mid
        obs = abs(float(d.mean()))
        print("  %-30s %8.2f %8.2f %10.2f   %s"
              % (name, d.mean(), sd, hi,
                 "넘는다" if obs >= hi else "** 못 넘는다 **"))

    # ---- 현실 점검 ----
    print()
    print("=" * 96)
    print("[6] 현실 점검 -- 데이터셋을 다 넣으면 장면이 몇 개인가")
    print("=" * 96)
    pool = [("MOT17 val_half", 7, "지금 쓰는 것"),
            ("MOT20 train", 4, "장면 수가 적다"),
            ("DanceTrack val", 25, "확인 필요"),
            ("KITTI tracking train", 21, "확인 필요")]
    tot = 0
    for name, k, note in pool:
        tot += k
        print("  %-24s %4d 개   (%s)" % (name, k, note))
    print("  %-24s %4d 개" % ("합계", tot))
    print()
    print("  위 개수는 **확인이 필요하다.** 요점은 개수가 아니라 자릿수다.")
    print("  [3] 과 [4] 가 요구하는 n 과 이 합계를 비교하라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
