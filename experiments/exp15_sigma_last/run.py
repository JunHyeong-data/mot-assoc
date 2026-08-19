# -*- coding: utf-8 -*-
"""실험 15 -- **σ 의 마지막 시험.** 통로와 무관하게, 연관 오류를 짚는가.

두 소스를 같은 절차로 잰다:
    python experiments/exp15_sigma_last/run.py         # DFL 분포 분산
    python experiments/exp15_sigma_last/run.py -nms    # NMS 후보 산포

사전 선언은 `PREREG.md` (자료보다 먼저 커밋).

exp01 은 σ 가 **위치 오차**를 예측한다고 했다. 우리는 그것이 **연관 오류**로
옮겨간다고 **가정하고** 통로 넷을 갈랐다. 그 가정을 이제 직접 잰다.

단위는 **1단계에서 채택된 (트랙, 검출) 쌍**이고 n 이 수만이라
**검정력 문제가 없다** (CLAUDE.md 규칙 6).

사용법:
    python experiments/exp15_sigma_last/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))                       # experiments/
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[0] / "exp12_ceiling"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                 # noqa: E402
from replay import WTracker, load, SEQS, BASE                   # noqa: E402
from run import det_gt_ids, gt_by_frame, GDet                   # noqa: E402
from stage_util import which_stage, stage_thresh                # noqa: E402

# 소스 갈래. 기본은 DFL(캐시 기본 npz), "-nms" 면 NMS 후보 산포.
SRC = "w_nms" if "-nms" in sys.argv else "iou"
SRC_NAME = "NMS 후보 산포" if SRC == "w_nms" else "DFL 분포 분산"

REC = []          # 채택된 쌍마다 한 줄
CALLS = {}        # 단계별 get_dists 호출 수 (진단)


class ProbeTracker(WTracker):
    """**1단계** 채택 쌍을 라벨한다. 비용은 손대지 않는다 (기준선 그대로).

    감사 정정 (2026-08-18): `get_dists` 는 1단계와 3단계에서 **둘 다** 불린다.
    `stage_util.which_stage` 로 갈라 **1단계만** 라벨한다. 3단계 건수도 세서
    얼마나 섞여 있었는지 보고한다.
    """

    seq_tag = "?"

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        for t in tracks:
            t.gt_id = int(results.gid[int(t.idx)])
        return tracks

    def get_dists(self, tracks, detections):
        d = super().get_dists(tracks, detections)
        if d.ndim != 2 or 0 in d.shape:
            return d
        stage = which_stage(tracks)
        CALLS[stage] = CALLS.get(stage, 0) + 1
        if stage != 1:                       # 3단계는 사전 선언 범위 밖이다
            return d
        m, _, _ = matching.linear_assignment(d, stage_thresh(self.args, stage))
        # **"고칠 수 있음" 은 그 단계가 실제로 본 검출로만 판단한다.**
        # 예전에는 프레임 전체 검출로 봐서, 1단계가 볼 수 없는 저신뢰 검출까지
        # "있었다" 로 세었다 (감사 지적).
        present = set(int(de.gt_id) for de in detections if int(de.gt_id) >= 0)
        for i, j in np.asarray(m).reshape(-1, 2):
            tr, de = tracks[int(i)], detections[int(j)]
            tg, dg = getattr(tr, "gt_id", -1), int(de.gt_id)
            h = float(de.xyxy[3] - de.xyxy[1])
            sig = float(np.sqrt(max(float(de.det_var[0]) + float(de.det_var[1]), 0.0)))
            if dg < 0 or tg < 0:
                lab = "미상"
            elif tg == dg:
                lab = "옳음"
            elif tg in present:
                lab = "틀림_고칠수있음"        # 올바른 검출이 그 프레임에 있었다
            else:
                lab = "틀림_못고침"            # 미검출이라 연관으로는 못 고친다
            REC.append((lab, sig, h, self.seq_tag))
        return d


def auc(pos, neg):
    """Mann-Whitney U 로 AUC. pos 가 클수록 1 에 가깝다."""
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1.0
    # 동점 보정
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def partial_corr(x, y, z):
    """z 를 통제한 x,y 의 편상관 (순위 기반). exp01 과 같은 자."""
    from scipy.stats import rankdata
    X, Y, Z = (rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    rx, ry = resid(X, Z), resid(Y, Z)
    s = np.std(rx) * np.std(ry)
    return float(np.mean(rx * ry) / s) if s > 0 else float("nan")


def main():
    print("=" * 92)
    print("실험 15 -- σ 의 마지막 시험. 통로와 무관하게 연관 오류를 짚는가")
    print("=" * 92)
    print("사전 선언 PREREG.md (자료보다 먼저)   **소스 = %s**" % SRC_NAME)
    print("exp01 은 σ 가 **위치 오차**를 예측한다고 했다. **연관 오류**는 안 쟀다.")
    print()

    for seq in SEQS:
        c = load(seq, SRC)
        if c is None:
            raise SystemExit(
                "소스 %r 캐시가 없다 (%s). 먼저 만들어라: "
                "python experiments/exp05_wasserstein/cache_nms.py"
                % (SRC, seq))
        gid = det_gt_ids(c, gt_by_frame(seq))
        tr = ProbeTracker(SimpleNamespace(**BASE), "iou", 1.0, frame_rate=30)
        tr.seq_tag = seq
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            tr.update(GDet(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                           c["sxx"][m], c["syy"][m], gid[m]))

    lab = np.array([r[0] for r in REC])
    sig = np.array([r[1] for r in REC], float)
    hgt = np.array([r[2] for r in REC], float)
    sqs = np.array([r[3] for r in REC])

    # **쌍별 기록을 그대로 저장한다** (2026-08-18 추가).
    # NMS 소스에서 합친 AUC 와 시퀀스별이 갈려 원인을 따로 보려면 원자료가 필요하다.
    # **판정 로직은 건드리지 않았다** -- 아래 [1]~[4] 는 그대로다.
    _raw = Path("data/exp15"); _raw.mkdir(parents=True, exist_ok=True)
    np.savez(_raw / ("pairs-%s.npz" % ("nms" if SRC == "w_nms" else "dfl")),
             lab=lab, sig=sig, hgt=hgt, seq=sqs)

    print("=" * 92)
    print("[진단] get_dists 호출의 단계 구성 -- 감사에서 나온 정정")
    print("=" * 92)
    tot = sum(CALLS.values())
    for s in sorted(k for k in CALLS if k is not None):
        print("  %d단계 %6d 회 (%.1f%%)%s"
              % (s, CALLS[s], 100.0 * CALLS[s] / tot,
                 "  <- 라벨에 쓴 것" if s == 1 else "  <- 사전 선언 범위 밖. 뺐다"))
    if None in CALLS:
        print("  판별불가(빈 목록) %d 회 -- 건너뛰었다" % CALLS[None])
    print("  **예전 판은 이 둘을 섞어서 세었다.**")

    print()
    print("=" * 92)
    print("[4] 채택된 1단계 쌍의 구성")
    print("=" * 92)
    for k in ("옳음", "틀림_고칠수있음", "틀림_못고침", "미상"):
        n = int((lab == k).sum())
        print("  %-18s %8d  (%5.2f%%)" % (k, n, 100.0 * n / len(lab)))
    print("  %-18s %8d" % ("합계", len(lab)))

    ok = lab == "옳음"
    bad = lab == "틀림_고칠수있음"
    print()
    print("=" * 92)
    print("[1] 주 종말점 -- AUC(σ -> 틀림(고칠 수 있음))")
    print("=" * 92)
    a_sig = auc(sig[bad], sig[ok])
    print("  옳음 %d 건 vs 틀림(고칠수있음) %d 건" % (ok.sum(), bad.sum()))
    print("  **AUC(σ) = %.4f**" % a_sig)

    # [3] 상자 크기 모형 sigma_C 로 같은 것
    sig_c = hgt / 2.0                      # C: 공분산이 상자 크기에 비례
    a_c = auc(sig_c[bad], sig_c[ok])
    print("  AUC(상자 크기 σ_C) = %.4f   <- [3] 기준 비교" % a_c)

    print()
    print("=" * 92)
    print("[2] 상자 높이를 통제한 편상관 (exp01 과 같은 자)")
    print("=" * 92)
    sel = ok | bad
    pc = partial_corr(sig[sel], bad[sel].astype(float), hgt[sel])
    print("  편상관(σ, 틀림 | 높이) = %+.4f    (exp01 의 위치오차 편상관은 +0.32)" % pc)

    print()
    print("=" * 92)
    print("판정 -- 사전 선언한 기준")
    print("=" * 92)
    if a_sig >= 0.60:
        print("  AUC %.4f >= 0.60 => **σ 가 연관 오류를 짚는다.**" % a_sig)
        print("     통로를 잘못 골랐던 것이고 **다시 볼 값어치가 있다**")
    elif a_sig >= 0.55:
        print("  0.55 <= AUC %.4f < 0.60 => 약하게 짚는다." % a_sig)
        print("     exp01 의 0.32 가 연관까지는 잘 안 옮겨간다")
    else:
        print("  AUC %.4f < 0.55 => **σ 는 연관 오류에 정보가 없다.**" % a_sig)
        print("     어느 통로에 넣어도 안 된다. **σ 질문이 완전히 닫힌다.**")
        print("     통로 넷의 음성이 '통로가 나빴다' 가 아니라")
        print("     **'애초에 옮길 정보가 없었다'** 로 설명된다")
    print()
    print("=" * 92)
    print("시퀀스별 AUC -- 그림 F 용. **전부 0.5 근처면 정보가 없다는 뜻이다**")
    print("=" * 92)
    per = {}
    for s in sorted(set(sqs)):
        m = sqs == s
        o, b = ok & m, bad & m
        v = auc(sig[b], sig[o]) if o.sum() and b.sum() else float("nan")
        per[s] = dict(auc=float(v), n_ok=int(o.sum()), n_bad=int(b.sum()))
        print("  %-18s AUC %.4f   (옳음 %5d / 틀림 %4d)"
              % (s.replace("-FRCNN", ""), v, o.sum(), b.sum()))
    import json
    out = Path("data/exp15"); out.mkdir(parents=True, exist_ok=True)
    fn = out / ("perseq-%s.json" % ("nms" if SRC == "w_nms" else "dfl"))
    fn.write_text(json.dumps(dict(source=SRC, overall=float(a_sig),
                                  overall_c=float(a_c), per=per), indent=1))
    print("  -> %s" % fn)

    if a_c > a_sig:
        print()
        print("  [3] 상자 크기가 σ 를 이긴다 (%.4f > %.4f)." % (a_c, a_sig))
        print("      exp1f 의 만장일치와 같은 이야기다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
