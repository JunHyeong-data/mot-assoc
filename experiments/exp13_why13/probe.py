# -*- coding: utf-8 -*-
"""실험 13 [탐색] -- **MOT17-13 에 여지가 몰린 이유.**

## 이건 판정이 아니다

MOT17-13 이 **네 지표에서 튄다**:

  exp05  와서스타인이 IoU 를 이긴 **유일한** 시퀀스 (+5.2)
  exp06  최적 임계값이 격자 끝(0.98)이고 이득이 최대 (+4.66)
  exp09  결정 단위 IDTP 차이의 **부호가 유일하게 반대**
  exp12  **연관 여지가 최대** (+12.17, 다음이 +8.87) -- 감사 정정 후 값

**n=1 이다. 주장이 아니라 가설 고르기용이다** (predictors.py 와 같은 지위).

## 무엇을 재는가 -- 가장 직접적인 질문

> **진짜 짝이 IoU 임계값 안에 들어오기는 하는가?**

ByteTrack 은 `IoU < 0.2` 인 쌍을 거부한다 (비용 > 0.8 = `match_thresh`).
그러므로 **같은 GT id 의 연속 프레임 IoU 가 0.2 미만이면 그 짝은 IoU 로는
원리적으로 못 찾는다.** 신탁은 찾는다 -- **그게 여지의 정체일 수 있다.**

같이 재는 것: 프레임률, 전역 카메라 운동, 그것을 뺀 잔차 운동, 상자 크기, 밀도.
**전역 운동이 크면 카메라 탓이고, 잔차가 크면 물체 운동 탓이다.**

사용법:
    python experiments/exp13_why13/probe.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate import GT_ROOT, SEQS                              # noqa: E402

REJECT_IOU = 0.2          # ByteTrack: 이보다 작으면 매칭 거부
PEDESTRIAN = 1

# exp12 가 낸 시퀀스별 여지 (재현: python experiments/exp12_ceiling/run.py)
ROOM = {"MOT17-02-FRCNN": 2.02, "MOT17-04-FRCNN": 0.91, "MOT17-05-FRCNN": 4.86,
        "MOT17-09-FRCNN": 2.75, "MOT17-10-FRCNN": 6.02, "MOT17-11-FRCNN": 8.87,
        "MOT17-13-FRCNN": 12.17}
# **정정 (2026-08-18 감사)**: 예전 값(2.44/1.44/6.50/2.92/6.42/11.21/14.44)은
# exp12 신탁이 3단계까지 풀던 판이다. 1단계로 한정하니 위 값이 됐다.


def seqinfo(seq, key):
    for line in open(GT_ROOT / seq / "seqinfo.ini"):
        if line.lower().startswith(key.lower()):
            return line.strip().split("=")[1]
    return ""


def gt_frames(seq):
    """frame -> {id: xyxy}. zero_marked!=0, pedestrian 만."""
    per = {}
    for line in open(GT_ROOT / seq / "gt" / "gt.txt"):
        f = line.strip().split(",")
        if len(f) < 8 or int(f[6]) == 0 or int(f[7]) != PEDESTRIAN:
            continue
        t, i = int(f[0]), int(f[1])
        x, y, w, h = (float(v) for v in f[2:6])
        per.setdefault(t, {})[i] = np.array([x, y, x + w, y + h])
    return per


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def probe(seq):
    G = gt_frames(seq)
    ts = sorted(G)
    ious, disp, resid, glob, hs = [], [], [], [], []
    for t0, t1 in zip(ts, ts[1:]):
        if t1 != t0 + 1:
            continue
        a, b = G[t0], G[t1]
        common = set(a) & set(b)
        if not common:
            continue
        # 전역(카메라) 운동 대리: 공통 id 중심 이동의 중앙값
        d = np.array([[(b[i][0] + b[i][2] - a[i][0] - a[i][2]) / 2,
                       (b[i][1] + b[i][3] - a[i][1] - a[i][3]) / 2] for i in common])
        gm = np.median(d, axis=0)
        glob.append(float(np.hypot(*gm)))
        for k, i in enumerate(common):
            ious.append(iou(a[i], b[i]))
            h = a[i][3] - a[i][1]
            hs.append(h)
            disp.append(float(np.hypot(*d[k]) / max(h, 1e-9)))
            resid.append(float(np.hypot(*(d[k] - gm)) / max(h, 1e-9)))
    ious = np.array(ious)
    return dict(
        fps=float(seqinfo(seq, "frameRate")),
        n=len(ious),
        iou_med=float(np.median(ious)),
        below=100.0 * float(np.mean(ious < REJECT_IOU)),
        disp=float(np.median(disp)),            # 높이 대비 총 이동
        resid=float(np.median(resid)),          # 전역 뺀 잔차
        glob=float(np.median(glob)),            # 전역 이동 (px)
        h_med=float(np.median(hs)),
        dens=float(np.mean([len(v) for v in G.values()])))


def main():
    print("=" * 100)
    print("실험 13 [탐색] -- MOT17-13 에 여지가 몰린 이유.  **n=1. 판정이 아니다**")
    print("=" * 100)
    print("묻는 것: **진짜 짝(같은 GT id 연속 프레임)이 IoU 임계값 안에 들어오는가.**")
    print("ByteTrack 은 IoU < %.1f 인 쌍을 거부한다." % REJECT_IOU)
    print()

    R = {s: probe(s) for s in SEQS}
    cols = [("fps", "fps", "%6.0f"), ("h_med", "높이중앙", "%9.0f"),
            ("dens", "밀도", "%7.1f"), ("glob", "전역이동px", "%11.2f"),
            ("disp", "이동/높이", "%10.3f"), ("resid", "잔차/높이", "%10.3f"),
            ("iou_med", "연속IoU중앙", "%12.3f"), ("below", "IoU<0.2 %%", "%11.2f")]
    print("%-12s" % "시퀀스" + "".join("%s" % c[1].rjust(int(c[2][1:-1].split(".")[0]))
                                       for c in cols) + "%10s" % "여지")
    print("-" * 100)
    order = sorted(SEQS, key=lambda s: ROOM[s])
    for s in order:
        r = R[s]
        print("%-12s" % s.replace("-FRCNN", "").replace("MOT17-", "MOT17-")
              + "".join(c[2] % r[c[0]] for c in cols) + "%10.2f" % ROOM[s])

    print()
    print("=" * 100)
    print("여지와의 스피어만 상관 (n=7). **가설 고르기용이다**")
    print("=" * 100)
    from scipy.stats import spearmanr
    y = np.array([ROOM[s] for s in SEQS])
    out = []
    for key, name, _ in cols:
        x = np.array([R[s][key] for s in SEQS])
        if np.std(x) == 0:
            continue
        rho, p = spearmanr(x, y)
        out.append((abs(rho), name, rho, p))
    out.sort(reverse=True)
    print("%-14s%10s%10s" % ("관측량", "rho", "p"))
    print("-" * 40)
    for _, name, rho, p in out:
        print("%-14s%10.3f%10.3f" % (name, rho, p))

    print()
    print("=" * 100)
    print("MOT17-13 은 무엇이 다른가")
    print("=" * 100)
    t = R["MOT17-13-FRCNN"]
    oth = {k: np.median([R[s][k] for s in SEQS if s != "MOT17-13-FRCNN"])
           for k, _, _ in cols}
    for key, name, _ in cols:
        v, m = t[key], oth[key]
        ratio = v / m if m else float("nan")
        mark = "  <<<" if (ratio > 1.5 or ratio < 0.67) else ""
        print("  %-14s MOT17-13 %10.3f   나머지 중앙값 %10.3f   배율 %6.2f%s"
              % (name, v, m, ratio, mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
