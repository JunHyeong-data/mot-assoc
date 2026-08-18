# -*- coding: utf-8 -*-
"""실험 14 -- **카메라 운동 보상을 넣으면 여지의 몇 %가 회복되는가.**

사전 선언은 `PREREG.md` (자료보다 먼저 커밋).

exp12 가 연관 여지를 +3.92 HOTA (MOT17-13 은 +14.44) 로 쟀고,
exp13 이 그 여지가 **카메라 운동** 때문임을 지목했다. 여기서 검증한다.

**바뀌는 것은 트랙 예측 상자의 위치 하나다.** 검출·비용함수·임계값·트랙
관리는 전부 그대로다.

사용법:
    python experiments/exp14_cmc/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                 # noqa: E402
from replay import WTracker, Det, load, SEQS, BASE              # noqa: E402
import evaluate as EV                                           # noqa: E402
from tracker.eval.collections.hota import HOTA                  # noqa: E402

OUT = Path("data/exp14/tracks")
PEDESTRIAN = 1

# exp12 가 낸 시퀀스별 여지 (천장 − 기준선)
ROOM = {"MOT17-02-FRCNN": 2.02, "MOT17-04-FRCNN": 0.91, "MOT17-05-FRCNN": 4.86,
        "MOT17-09-FRCNN": 2.75, "MOT17-10-FRCNN": 6.02, "MOT17-11-FRCNN": 8.87,
        "MOT17-13-FRCNN": 12.17}
# **정정 (2026-08-18 감사)**: 예전 값(2.44/1.44/6.50/2.92/6.42/11.21/14.44)은
# exp12 신탁이 3단계까지 풀던 판이다. 1단계로 한정하니 위 값이 됐다.
STATIC = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")     # exp13: 전역이동 0.00 px


def gt_shifts(seq):
    """프레임 t 의 전역 이동 (t−1 → t). exp13 과 같은 정의: 공통 id 중심이동 중앙값."""
    per = {}
    for line in open(EV.GT_ROOT / seq / "gt" / "gt.txt"):
        f = line.strip().split(",")
        if len(f) < 8 or int(f[6]) == 0 or int(f[7]) != PEDESTRIAN:
            continue
        t, i = int(f[0]), int(f[1])
        x, y, w, h = (float(v) for v in f[2:6])
        per.setdefault(t, {})[i] = (x + w / 2, y + h / 2)
    out = {}
    ts = sorted(per)
    for t0, t1 in zip(ts, ts[1:]):
        if t1 != t0 + 1:
            continue
        common = set(per[t0]) & set(per[t1])
        if not common:
            continue
        d = np.array([[per[t1][i][0] - per[t0][i][0],
                       per[t1][i][1] - per[t0][i][1]] for i in common])
        out[t1] = np.median(d, axis=0)
    return out


def gmc_shifts(seq, n_frames):
    """영상에서 추정한 전역 이동. ultralytics GMC(sparseOptFlow)."""
    import cv2
    from ultralytics.trackers.utils.gmc import GMC
    g = GMC(method="sparseOptFlow", downscale=2)
    img = EV.GT_ROOT / seq / "img1"
    out = {}
    for t in range(1, n_frames + 1):
        p = img / ("%06d.jpg" % t)
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        H = g.apply(frame)                     # 2x3 아핀
        out[t] = np.array([float(H[0, 2]), float(H[1, 2])])
    return out


class ShiftTracker(WTracker):
    """트랙 예측 상자를 전역 이동만큼 옮긴 뒤 IoU 를 잰다."""

    shift = np.zeros(2)

    def get_dists(self, tracks, detections):
        if not len(tracks) or not len(detections) or not np.any(self.shift):
            return super().get_dists(tracks, detections)
        t_box = np.asarray([t.xyxy for t in tracks], float).reshape(-1, 4)
        d_box = np.asarray([d.xyxy for d in detections], float).reshape(-1, 4)
        t_box = t_box + np.tile(self.shift, 2)[None, :]     # 예측을 현재 프레임으로
        d = 1.0 - _iou(t_box, d_box)
        if self.args.fuse_score:
            d = matching.fuse_score(d, detections)
        return d.astype(np.float32)


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def replay(tag, shifts_fn):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    mags = {}
    for seq in SEQS:
        c = load(seq, "iou")
        sh = shifts_fn(seq, c["n_frames"]) if shifts_fn else {}
        mags[seq] = float(np.median([np.hypot(*v) for v in sh.values()])) if sh else 0.0
        tr = ShiftTracker(SimpleNamespace(**BASE), "iou", 1.0, frame_rate=30)
        lines = []
        for f in range(1, c["n_frames"] + 1):
            tr.shift = np.asarray(sh.get(f, (0.0, 0.0)), float)
            m = c["frame"] == f
            det = Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                      c["sxx"][m], c["syy"][m])
            for row in tr.update(det):
                x1, y1, x2, y2 = row[:4]
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                             % (f, int(row[4]), x1, y1, x2 - x1, y2 - y1, float(row[5])))
        (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")
    return mags


def score(tag):
    keep = EV.TRACKS
    EV.TRACKS = OUT
    try:
        m, per = HOTA(), {}
        for seq in SEQS:
            d = EV.build_data(seq, tag)
            if d is None:
                raise SystemExit("트랙 없음 %s/%s" % (tag, seq))
            per[seq] = m.eval_sequence(d)
        comb = m.combine_sequences(per)
        return (100 * float(np.mean(comb["HOTA"])),
                {s: 100 * float(np.mean(per[s]["HOTA"])) for s in per})
    finally:
        EV.TRACKS = keep


def main():
    print("=" * 96)
    print("실험 14 -- 카메라 운동 보상을 넣으면 여지의 몇 %가 회복되는가")
    print("=" * 96)
    print("사전 선언 PREREG.md (자료보다 먼저). 바뀌는 것은 트랙 예측 상자의 위치 하나다.")
    print()

    replay("base", None)
    om = replay("ocmc", lambda s, n: gt_shifts(s))
    print("영상에서 GMC 를 추정한다 (sparseOptFlow)...")
    gm = replay("gmc", gmc_shifts)
    hb, pb = score("base")
    ho, po = score("ocmc")
    hg, pg = score("gmc")

    print()
    print("=" * 96)
    print("사전 선언한 관문")
    print("=" * 96)
    ok = True
    g0a = abs(hb - 61.002) < 0.01
    ok &= g0a
    print("  [0a] base 재현        %.3f vs 61.002  %s" % (hb, "OK" if g0a else "** 불일치 **"))
    worst = max(abs(po[s] - pb[s]) for s in STATIC)
    g0b = worst < 0.5
    ok &= g0b
    print("  [0b] 정지 카메라 통제  MOT17-02/04 최대 |Δ| = %.3f  %s"
          % (worst, "OK (<0.5)" if g0b else "** 실패: 개입이 다른 것을 건드렸다 **"))
    for s in STATIC:
        print("       %-18s ocmc %+.3f   gmc %+.3f"
              % (s.replace("-FRCNN", ""), po[s] - pb[s], pg[s] - pb[s]))
    g0c = all(gm[s] > 0 for s in SEQS)
    print("  [0c] GMC 가 이동을 낸다  %s" % ("OK" if g0c else "일부 0"))
    if not ok:
        print()
        print("  ** 관문 실패. 판정하지 않는다 **")
        return 1

    print()
    print("=" * 96)
    print("사전 선언한 종말점")
    print("=" * 96)
    print("  결합 HOTA   base %.3f   ocmc %+.3f   gmc %+.3f" % (hb, ho - hb, hg - hb))
    print()
    print("  %-14s %8s %9s %9s %9s %9s %9s" %
          ("시퀀스", "여지", "ocmcΔ", "회복률", "gmcΔ", "회복률", "GMC px"))
    print("  " + "-" * 76)
    for s in SEQS:
        room = ROOM[s]
        do, dg = po[s] - pb[s], pg[s] - pb[s]
        print("  %-14s %8.2f %+9.2f %8.0f%% %+9.2f %8.0f%% %9.2f"
              % (s.replace("-FRCNN", ""), room, do, 100 * do / room,
                 dg, 100 * dg / room, gm[s]))

    # 그림에 쓸 수 있게 저장한다. predictors.py 처럼 손으로 옮겨 적지 않는다.
    import json
    outd = Path("data/exp14"); outd.mkdir(parents=True, exist_ok=True)
    (outd / "recovery.json").write_text(json.dumps(dict(
        base_combined=hb, ocmc_combined=ho, gmc_combined=hg,
        room=ROOM,
        per={s: dict(base=pb[s], ocmc=po[s], gmc=pg[s],
                     d_ocmc=po[s] - pb[s], d_gmc=pg[s] - pb[s],
                     gmc_px=gm[s]) for s in SEQS}), indent=1))
    print("  -> data/exp14/recovery.json")

    s13 = "MOT17-13-FRCNN"
    rec = 100 * (po[s13] - pb[s13]) / ROOM[s13]
    recg = 100 * (pg[s13] - pb[s13]) / ROOM[s13]
    print()
    print("=" * 96)
    print("판정 -- 사전 선언한 읽는 법 (MOT17-13 의 신탁 CMC 회복률)")
    print("=" * 96)
    print("  MOT17-13: 여지 %.2f,  신탁 CMC %+.2f (**%.0f%%**),  실제 GMC %+.2f (%.0f%%)"
          % (ROOM[s13], po[s13] - pb[s13], rec, pg[s13] - pb[s13], recg))
    print()
    if rec > 60:
        print("  > 60%% => **여지의 정체가 카메라 운동으로 확정된다.** exp13 진단이 맞다")
    elif rec >= 30:
        print("  30~60%% => 카메라 운동이 **큰 몫이지만 전부는 아니다**")
    else:
        print("  < 30%% => **exp13 의 진단이 틀렸다.** 이동은 컸지만 여지의 원인이 아니다")
    # 회복률이 음수면 "절반" 비교가 뜻이 없다. 양수일 때만 본다 (감사 정정).
    if rec > 0 and recg < rec / 2:
        print("  실제 GMC 가 신탁의 절반도 못 낸다 -> **추정 정확도가 병목이다**")
    elif abs(hg - ho) < 0.05:
        print("  결합 HOTA 에서 실제 GMC 가 신탁과 사실상 같다 (%+.3f vs %+.3f)"
              % (hg - hb, ho - hb))
        print("  -> **추정 정확도는 병목이 아니다.** 평행이동 보상 자체의 한계다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
