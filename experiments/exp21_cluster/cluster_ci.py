# -*- coding: utf-8 -*-
"""실험 21 — **군집을 반영한 구간.** 사전 등록은 `PREREG.md` (자료보다 먼저 커밋).

## 왜

원고의 추론적 방어가 하나뿐이다 — *"주 주장은 표본이 수만 건인 쌍 수준"*.
그런데 그 쌍들이 **7 개 장면 안에 군집**돼 있고, 현재 구간(DeLong·이항)은
전부 **쌍의 독립을 가정**한다. NMS AUC 상한 `0.5495` 는 판정선 `0.55` 까지
여유가 **0.0005** 다.

## 어떻게

**시퀀스를 복원추출**해 7 개를 뽑고, 뽑힌 시퀀스의 쌍을 전부 모아 통계량을
다시 계산한다. 이것이 군집 부트스트랩이다. `B=2000`, 씨앗 고정.

**주 판독은 시퀀스별 부호 개수다** (PREREG 에 박아 뒀다) — 군집이 7 개면
부트스트랩 구간 자체가 낙관적이기 때문이다. 규칙 6 의 직접 적용.

사용법:
    python experiments/exp21_cluster/cluster_ci.py
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("data/exp15")
WITHIN = Path("data/exp21/withinrow.npz")     # exp18 이 남기면 쓴다
THRESH = 0.55                                  # 사전 등록한 AUC 판정선
B = 2000
SEED = 20260821


def auc(pos, neg):
    """AUC. 동점 0.5. `exp15/auc_ci.py` 의 규약과 같다."""
    pos = np.sort(np.asarray(pos, float))
    neg = np.sort(np.asarray(neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(((lo + (hi - lo) / 2.0) / len(neg)).mean())


def auc_se_delong(pos, neg):
    """비교용 -- 쌍 독립을 가정한 해석적 SE (원고가 지금 쓰는 것)."""
    pos = np.sort(np.asarray(pos, float)); neg = np.sort(np.asarray(neg, float))
    n1, n0 = len(pos), len(neg)
    lo = np.searchsorted(neg, pos, "left"); hi = np.searchsorted(neg, pos, "right")
    v10 = (lo + (hi - lo) / 2.0) / n0
    lo2 = np.searchsorted(pos, neg, "left"); hi2 = np.searchsorted(pos, neg, "right")
    v01 = ((n1 - hi2) + (hi2 - lo2) / 2.0) / n1
    return float(np.sqrt(v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0))


def cluster_boot(stat, groups, seqs, rng, b=B):
    """시퀀스를 복원추출해 통계량 분포를 만든다.

    `groups[s]` 는 시퀀스 s 의 자료. `stat(list_of_groups) -> float`.
    """
    out = []
    for _ in range(b):
        pick = rng.choice(len(seqs), len(seqs), replace=True)
        v = stat([groups[seqs[i]] for i in pick])
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out)


def report(name, per_seq, boot, point, indep_se, thresh, side):
    """한 신호의 결과를 찍는다. side 는 판정선의 어느 쪽이어야 하는가."""
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ind_lo, ind_hi = point - 1.96 * indep_se, point + 1.96 * indep_se
    c_se = float(boot.std(ddof=1))
    deff = (c_se / indep_se) ** 2 if indep_se > 0 else np.nan
    print("  %s" % name)
    print("     점추정 %.4f" % point)
    print("     독립 가정 95%%  [%.4f, %.4f]   (SE %.4f)" % (ind_lo, ind_hi, indep_se))
    print("     **군집 95%%**   [%.4f, %.4f]   (SE %.4f, DEFF %.1f배)"
          % (lo, hi, c_se, deff))
    ok = [s for s, v in per_seq if (v > thresh if side == "above" else v < thresh)]
    print("     시퀀스별: %s" % "  ".join("%s %.3f" % (s.replace("-FRCNN", "")[6:], v)
                                          for s, v in per_seq))
    print("     판정선 %.2f 기준 %s 쪽: **%d/%d**"
          % (thresh, "위" if side == "above" else "아래", len(ok), len(per_seq)))
    return lo, hi, len(ok), len(per_seq)


def main():
    print("=" * 92)
    print("실험 21 -- 군집을 반영한 구간 (시퀀스 부트스트랩 B=%d, 씨앗 %d)" % (B, SEED))
    print("=" * 92)
    print("**주 판독은 시퀀스별 부호 개수다.** 군집이 7 개면 부트스트랩도")
    print("낙관적이므로 구간은 참고로만 쓴다 (PREREG 에 자료 전에 박아 둠).")
    rng = np.random.default_rng(SEED)

    # ---------- [1] AUC ----------
    print()
    print("-" * 92)
    print("[1] 주 평가지표 -- AUC (sigma -> 고칠 수 있는 오류). 판정선 %.2f 미만이어야 한다"
          % THRESH)
    print("-" * 92)
    verdicts = {}
    for tag, name in (("nms", "NMS 후보 분산"), ("dfl", "DFL 분포 분산"),
                      ("size", "박스 크기 sigma_C")):
        src = "dfl" if tag == "size" else tag
        d = np.load(SRC / ("pairs-%s.npz" % src), allow_pickle=True)
        lab, seq = d["lab"], d["seq"]
        val = (d["hgt"].astype(float) / 2.0) if tag == "size" else d["sig"].astype(float)
        ok_m, bad_m = lab == "옳음", lab == "틀림_고칠수있음"

        seqs = sorted(set(seq.tolist()))
        groups = {s: (val[bad_m & (seq == s)], val[ok_m & (seq == s)]) for s in seqs}
        point = auc(val[bad_m], val[ok_m])
        se_i = auc_se_delong(val[bad_m], val[ok_m])
        per = [(s, auc(*groups[s])) for s in seqs]

        def stat(gs):
            return auc(np.concatenate([g[0] for g in gs]),
                       np.concatenate([g[1] for g in gs]))

        boot = cluster_boot(stat, groups, seqs, rng)
        print()
        verdicts[tag] = report(name, per, boot, point, se_i, THRESH, "above")

    # ---------- [2] 행 안 정답률 ----------
    print()
    print("-" * 92)
    print("[2] 여섯째 결과 -- 행 안 정답률. 0.5 를 넘어야 신호다")
    print("-" * 92)
    if not WITHIN.exists():
        print("  %s 가 없다. `python experiments/exp18_withinrow/paired.py --save` 를" % WITHIN)
        print("  먼저 돌릴 것. **[2] 는 판정하지 않는다.**")
    else:
        w = np.load(WITHIN, allow_pickle=True)
        seq = w["seq"]
        seqs = sorted(set(seq.tolist()))
        for tag, name in (("sig", "검출기 sigma"), ("size", "박스 크기 sigma_C")):
            a, b_ = (w[tag + "_wrong"].astype(float), w[tag + "_right"].astype(float))
            keep = a != b_                        # 동점 제외 (paired.py 규약)
            groups = {s: (a[keep & (seq == s)], b_[keep & (seq == s)]) for s in seqs}
            point = float((b_[keep] < a[keep]).mean())
            n = int(keep.sum())
            se_i = float(np.sqrt(point * (1 - point) / n))
            per = [(s, float((groups[s][1] < groups[s][0]).mean())
                    if len(groups[s][0]) else np.nan) for s in seqs]

            def stat(gs):
                x = np.concatenate([g[0] for g in gs])
                y = np.concatenate([g[1] for g in gs])
                return float((y < x).mean()) if len(x) else np.nan

            boot = cluster_boot(stat, groups, seqs, rng)
            print()
            verdicts["w_" + tag] = report(name, per, boot, point, se_i, 0.5, "above")

    # ---------- 판정 ----------
    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md)")
    print("=" * 92)
    lo, hi, k, n = verdicts["nms"]
    print("  [1] NMS AUC 군집 상한 %.4f  대 판정선 %.2f" % (hi, THRESH))
    if hi < THRESH:
        print("      => **판정선 아래를 유지한다.** 독립 가정이 결론을 만든 것이")
        print("         아니었음을 원고에 밝힌다.")
    else:
        print("      => **사전 등록한 판정선을 더 이상 충족하지 못한다.**")
        print("         다섯째 결과를 추론에서 **기술로 낮추고**, 쌍 독립을 가정한")
        print("         구간에서만 판정선 아래였다는 것을 원고에 적는다.")
    print("      시퀀스별로 판정선을 넘은 곳: %d/%d" % (k, n))
    if "w_size" in verdicts:
        lo2, hi2, k2, n2 = verdicts["w_size"]
        print()
        print("  [2] 박스 크기 행 안 군집 하한 %.4f  대 0.5" % lo2)
        if lo2 > 0.5 and k2 == n2:
            print("      => 여섯째 결과 **유지**. 군집을 반영해도 0.5 를 넘고 %d/%d 만장일치."
                  % (k2, n2))
        else:
            print("      => **탐색적 관측으로 낮춘다** (구간이 0.5 를 포함하거나 %d/%d)."
                  % (k2, n2))
            print("         이 값은 사전 등록상 주 평가지표가 아니라 기준 비교였다.")
    print()
    print("  **구간만으로 판정하지 않는다.** 군집 7 개의 부트스트랩은 낙관적이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
