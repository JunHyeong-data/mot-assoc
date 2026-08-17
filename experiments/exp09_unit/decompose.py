# -*- coding: utf-8 -*-
"""실험 9 -- **분석 단위를 바꾸면 천장이 열리는가.**

사전 선언은 `PREREG.md` (커밋 `b9561c3`, 자료보다 먼저).

설계 계산(exp07)이 갈래 2 를 "유망하다" 고 적었다. 그게 정말 되는지 따진다.

    Var(d_s) = Var_true(장면이 실제로 다르다) + Var_meas(재는 데 생긴 잡음)

Var_meas 가 지배적이면 결정 단위로 정밀하게 재서 천장이 열리고,
Var_true 가 지배적이면 장면 수가 진짜 천장이다.

단위는 (시퀀스, 프레임, GT id), 결과값은 **IDTP 지시자 0/1**.
전역 id 대응은 TrackEval `Identity` 의 헝가리안을 복제하고, 복제가 맞는지는
**IDF1 을 벤더링된 구현과 대조**해서 확인한다 (관문 [0a]).

사용법:
    python experiments/exp09_unit/decompose.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate import build_data, SEQS                          # noqa: E402
from tracker.eval.collections.identity import Identity         # noqa: E402

THR = 0.5                   # IoU 임계. Identity 기본값과 같다
BLOCK = 50                  # 블록 부트스트랩의 블록 길이 (프레임)
NBOOT = 2000
SEED = 20260818
MIN_DETS = 1000             # 관문 [0b]

PAIRS = [("iou", "wn_size_rate", "큰 효과 (기록 HOTA -4.98)"),
         ("_headroom/th080", "_headroom/th085", "작은 효과 (기록 HOTA +0.04)")]


def idtp_per_unit(data):
    """전역 id 대응을 풀고 **GT 검출마다 IDTP 지시자**를 낸다.

    `identity.py:48-80` 을 그대로 복제한다. 반환:
      keys  (t, gt_id) 튜플 배열 -- 갈래 간 짝짓기의 열쇠
      y     0/1 지시자
      idf1  복제가 맞는지 대조할 값
    """
    n_gt, n_tr = data["num_gt_ids"], data["num_tracker_ids"]
    if data["num_tracker_dets"] == 0 or data["num_gt_dets"] == 0:
        return None

    pot = np.zeros((n_gt, n_tr))
    gt_cnt, tr_cnt = np.zeros(n_gt), np.zeros(n_tr)
    for t, (g, k) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        mask = np.greater_equal(data["similarity_scores"][t], THR)
        ig, ik = np.nonzero(mask)
        pot[g[ig], k[ik]] += 1
        gt_cnt[g] += 1
        tr_cnt[k] += 1

    fp = np.zeros((n_gt + n_tr, n_gt + n_tr))
    fn = np.zeros((n_gt + n_tr, n_gt + n_tr))
    fp[n_gt:, :n_tr] = 1e10
    fn[:n_gt, n_tr:] = 1e10
    for g in range(n_gt):
        fn[g, :n_tr] = gt_cnt[g]
        fn[g, n_tr + g] = gt_cnt[g]
    for k in range(n_tr):
        fp[:n_gt, k] = tr_cnt[k]
        fp[k + n_gt, k] = tr_cnt[k]
    fn[:n_gt, :n_tr] -= pot
    fp[:n_gt, :n_tr] -= pot
    rows, cols = linear_sum_assignment(fn + fp)

    IDFN = fn[rows, cols].sum()
    IDFP = fp[rows, cols].sum()
    IDTP = gt_cnt.sum() - IDFN
    idf1 = IDTP / max(IDTP + 0.5 * IDFN + 0.5 * IDFP, 1e-10)

    # gt_id -> 대응된 tracker_id (더미면 대응 없음)
    mapped = {}
    for r, c in zip(rows, cols):
        if r < n_gt and c < n_tr:
            mapped[int(r)] = int(c)

    keys, y, frames = [], [], []
    for t, (g, k) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        if len(g) == 0:
            continue
        sim = data["similarity_scores"][t]
        pos = {int(v): i for i, v in enumerate(k)}
        for i, gid in enumerate(g):
            gid = int(gid)
            m = mapped.get(gid, -1)
            hit = 0
            if m >= 0 and m in pos and sim[i, pos[m]] >= THR:
                hit = 1
            keys.append((t, gid))
            y.append(hit)
            frames.append(t)
    return dict(keys=np.array(keys), y=np.array(y, dtype=float),
                frames=np.array(frames), idf1=float(idf1))


def block_boot_se(d, frames, rng, block=BLOCK, nboot=NBOOT):
    """프레임 블록 부트스트랩 SE. 같은 트랙의 인접 프레임은 독립이 아니다."""
    if len(d) == 0:
        return float("nan"), float("nan")
    iid = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    blk = frames // block
    ids = np.unique(blk)
    if len(ids) < 2:
        return float("nan"), iid
    groups = [d[blk == b] for b in ids]
    sums = np.array([g.sum() for g in groups], float)
    cnts = np.array([len(g) for g in groups], float)
    idx = rng.integers(0, len(ids), size=(nboot, len(ids)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-9)
    return float(np.std(means, ddof=1)), iid


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 96)
    print("실험 9 -- 분석 단위를 바꾸면 천장이 열리는가")
    print("=" * 96)
    print("사전 선언 PREREG.md (커밋 b9561c3, 자료보다 먼저)")
    print("이건 트래커가 아니라 **설계**에 대한 실험이다.")
    print()

    # ---------------- 관문 [0a] 대응 복제 검산 ----------------
    print("=" * 96)
    print("[0a] 관문 -- 복제한 Identity 대응이 벤더링된 구현과 맞는가")
    print("=" * 96)
    ident = Identity()
    cache, ok = {}, True
    for arm in {a for p in PAIRS for a in p[:2]}:
        for seq in SEQS:
            d = build_data(seq, arm)
            if d is None:
                print("  트랙 파일 없음: %s / %s" % (arm, seq))
                return 1
            mine = idtp_per_unit(d)
            ref = ident.eval_sequence(d)
            cache[(arm, seq)] = mine
            if arm == "iou":
                bad = abs(mine["idf1"] - float(ref["IDF1"])) > 1e-6
                ok &= not bad
                print("  %-18s IDF1 복제 %.8f  벤더링 %.8f  %s"
                      % (seq.replace("-FRCNN", ""), mine["idf1"],
                         float(ref["IDF1"]), "** 불일치 **" if bad else "OK"))
    if not ok:
        print("  ** 관문 [0a] 실패. 대응 복제가 틀렸다. 판정하지 않는다 **")
        return 1

    # ---------------- 관문 [0b] 표본 크기 ----------------
    print()
    print("=" * 96)
    print("[0b] 관문 -- 결정 단위가 정말 '수만' 인가")
    print("=" * 96)
    tot = 0
    for seq in SEQS:
        n = len(cache[("iou", seq)]["y"])
        tot += n
        flag = "" if n >= MIN_DETS else "  ** %d 미만 **" % MIN_DETS
        print("  %-18s GT 검출 %7d%s" % (seq.replace("-FRCNN", ""), n, flag))
    print("  %-18s        %7d" % ("합계", tot))
    if tot < MIN_DETS * len(SEQS):
        print("  ** 표본 전제가 틀렸다 **")

    # ---------------- 쌍마다 분해 ----------------
    for a, b, label in PAIRS:
        print()
        print("=" * 96)
        print("갈래 쌍: %s  vs  %s     [%s]" % (a, b, label))
        print("=" * 96)

        rows, dsl, ses, iids, ns = [], [], [], [], []
        mismatch = 0
        for seq in SEQS:
            A, B = cache[(a, seq)], cache[(b, seq)]
            # 관문 [0c]: 단위 집합이 같아야 한다
            if A["keys"].shape != B["keys"].shape or not np.array_equal(A["keys"], B["keys"]):
                mismatch += 1
                continue
            d = A["y"] - B["y"]
            se, iid = block_boot_se(d, A["frames"], rng)
            rows.append((seq, float(d.mean()), se, iid, len(d)))
            dsl.append(float(d.mean())); ses.append(se); iids.append(iid); ns.append(len(d))

        print("  [0c] 단위 집합 불일치 시퀀스 = %d  %s"
              % (mismatch, "OK" if mismatch == 0 else "** 짝짓기 실패 **"))
        if mismatch:
            print("  ** 판정하지 않는다 **")
            continue

        print()
        print("  [2] 시퀀스별 IDTP 비율 차이 (%s - %s)" % (a, b))
        print("      %-18s %10s %10s %10s %8s" % ("시퀀스", "차이", "블록SE", "iidSE", "n"))
        print("      " + "-" * 62)
        for seq, dm, se, iid, n in rows:
            print("      %-18s %+10.5f %10.5f %10.5f %8d"
                  % (seq.replace("-FRCNN", ""), dm, se, iid, n))

        ds = np.array(dsl); se_arr = np.array(ses)
        var_tot = float(np.var(ds, ddof=1))
        var_meas = float(np.mean(se_arr ** 2))
        var_true = var_tot - var_meas
        share = var_meas / var_tot if var_tot > 0 else float("nan")

        print()
        print("  [1] **주 종말점 -- 분산 분해**")
        print("      Var(d_s) 전체       = %.3e   (SD %.5f)" % (var_tot, np.sqrt(var_tot)))
        print("      Var_meas (측정잡음) = %.3e   (평균 블록SE %.5f)"
              % (var_meas, se_arr.mean()))
        print("      Var_true (진짜 차이)= %.3e   %s"
              % (var_true, "<- 음수. 측정잡음이 전체보다 크다" if var_true < 0 else ""))
        print("      **Var_meas / Var(d_s) = %.3f**" % share)

        print()
        print("  [3] 평균 차이의 SE -- 세 가지")
        naive = float(np.sqrt(np.sum(se_arr ** 2)) / len(ds))
        clustered = float(np.std(ds, ddof=1) / np.sqrt(len(ds)))
        floor = float(np.sqrt(max(var_true, 0.0)) / np.sqrt(len(ds)))
        print("      순진 (측정잡음만)        %.6f" % naive)
        print("      군집 보정 (시퀀스 간)    %.6f   <- 실제로 써야 하는 값" % clustered)
        print("      바닥 (측정잡음 = 0 이면) %.6f" % floor)
        print("      군집/순진 배율 = %.1f배" % (clustered / max(naive, 1e-12)))

        # ---- 민감도: SE 추정 방식에 결론이 얼마나 달렸는가 ----
        # 사전 선언이 "iid 값도 같이 적어 얼마나 다른지 보인다" 고 했다.
        # 블록 개수도 함께 낸다 -- 블록이 적으면 블록SE 자체가 불안정하다.
        print()
        print("  [민감도] SE 추정 방식에 따라 주 종말점이 어떻게 움직이는가")
        print("      %-14s %12s %12s %10s" % ("SE 추정", "Var_meas", "몫", "판정"))
        print("      " + "-" * 54)
        for name, blk in (("iid", None), ("블록 25", 25), ("블록 50", 50),
                          ("블록 100", 100)):
            vs = []
            for seq in SEQS:
                A, B = cache[(a, seq)], cache[(b, seq)]
                d = A["y"] - B["y"]
                se, iid = block_boot_se(d, A["frames"], rng,
                                        block=blk or BLOCK)
                vs.append(iid if blk is None else se)
            vm = float(np.nanmean(np.asarray(vs) ** 2))
            sh = vm / var_tot if var_tot > 0 else float("nan")
            v = ("열린다" if sh > 0.5 else ("부분적" if sh >= 0.1 else "천장"))
            print("      %-14s %12.3e %12.3f %10s" % (name, vm, sh, v))

        nblocks = [len(np.unique(cache[(a, s)]["frames"] // BLOCK)) for s in SEQS]
        print("      시퀀스당 블록 수 (길이 %d): %s" % (BLOCK, nblocks))
        if min(nblocks) < 15:
            print("      ** 블록이 %d개뿐인 시퀀스가 있다. 블록 부트스트랩 SE 자체가"
                  % min(nblocks))
            print("         불안정하므로 위 판정을 그대로 믿으면 안 된다 **")

        print()
        print("  판정 -- 사전 선언한 표를 그대로 적용한다")
        if share > 0.5:
            print("      Var_meas 몫 %.3f > 0.5 => **갈래 2 가 열린다.**" % share)
            print("      exp07 의 591 은 과대추정이므로 다시 계산해야 한다")
        elif share >= 0.1:
            print("      0.1 <= %.3f <= 0.5 => 부분적. 필요 장면 수가 줄지만" % share)
            print("      자릿수는 안 바뀐다")
        else:
            print("      Var_meas 몫 %.3f < 0.1 => **장면 수가 진짜 천장이다.**" % share)
            print("      결정 단위로 쪼개도 소용없다. **갈래 2 를 접는다**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
