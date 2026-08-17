# -*- coding: utf-8 -*-
"""실험 1c -- 입력 해상도가 sigma 의 눈금을 얼마나 정하는가.

MOT17-05 에서 imgsz 를 640 -> 1920 으로 올리니 chi2 배율이 649 -> 83 으로
떨어졌다 (README 의 "MOT17-05 를 갈랐다"). 한 시퀀스 두 점이다.
7시퀀스 x 4해상도로 곡선을 그린다.

**사전 선언은 README 의 "실험 1c" 절에 있다. 돌리기 전에 커밋했다.**
결과를 보고 기준을 바꾸지 않는다.

측정 설계:
  검출기·NMS 설정은 exp01 기본값 그대로 두고 imgsz 하나만 바꾼다.
  프레임 수는 60 으로 -m60 갈래와 맞춘다 (640 점은 -m60 을 그대로 쓴다).

사용법:
    python experiments/exp01_nms_variance/imgsz_sweep.py          # 전부
    python experiments/exp01_nms_variance/imgsz_sweep.py 960 1280 # 해상도 골라서
    python experiments/exp01_nms_variance/imgsz_sweep.py --table  # 실행 없이 표만
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]
SIZES = [640, 960, 1280, 1920]
FRAMES = 60
CHI2_MED = 1.386

RUNNER = Path(__file__).with_name("run_sequence.py")


def tag_for(isz):
    # 640 은 이미 -m60 으로 나 있다. 같은 설정이므로 다시 돌리지 않는다.
    return "-m60" if isz == 640 else "-isz%d" % isz


def npz_path(seq, isz):
    return Path("data/exp01") / ("%s-FRCNN%s.npz" % (seq, tag_for(isz)))


def measure(p):
    """chi2 배율, sigma/h, eps/h, 편상관, n 을 낸다."""
    d = np.load(p)
    sxx, sxy, syy = d["sxx"], d["sxy"], d["syy"]
    det2 = sxx * syy - sxy ** 2
    ok = ~np.isnan(sxx) & (det2 > 1e-9) & (sxx > 1e-9) & (syy > 1e-9)
    ex, ey = d["dcx"][ok], d["dcy"][ok]
    Sxx, Sxy, Syy = sxx[ok], sxy[ok], syy[ok]
    z2 = (Syy * ex ** 2 - 2 * Sxy * ex * ey + Sxx * ey ** 2) / (Sxx * Syy - Sxy ** 2)
    h = d["h"][ok]
    sig = det2[ok] ** 0.25

    g = ~np.isnan(d["s_c"])
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float)
                  for v in (d["s_c"][g], d["err"][g], d["h"][g]))
    A = np.column_stack([np.ones_like(rz), rz])
    resx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    resy = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return dict(n=int(ok.sum()), gain=float(d["gain"]),
                chi=float(np.median(z2) / CHI2_MED),
                sig_h=float(np.median(sig / h)),
                eps_h=float(np.median(np.hypot(ex, ey) / h)),
                pcorr=float(np.corrcoef(resx, resy)[0, 1]))


def run(sizes):
    t0 = time.time()
    for isz in sizes:
        if isz == 640:
            print("imgsz 640 은 -m60 갈래를 그대로 쓴다 (같은 설정). 건너뛴다.")
            continue
        print("=" * 72)
        print("imgsz %d" % isz)
        print("=" * 72)
        for seq in SEQS:
            p = npz_path(seq, isz)
            if p.exists():
                print("  %-10s 이미 있다 -> %s" % (seq, p.name))
                continue
            env = dict(os.environ, EXP01_IMGSZ=str(isz), EXP01_TAG=tag_for(isz))
            t = time.time()
            r = subprocess.run(
                [sys.executable, str(RUNNER), seq + "-FRCNN", str(FRAMES)],
                env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print("  %-10s [실패]\n%s" % (seq, (r.stdout + r.stderr)[-1500:]))
                continue
            for line in r.stdout.splitlines():
                if "배율" in line or "경고" in line:
                    print("  %-10s %s" % (seq, line.strip()))
            print("  %-10s %.0f초" % (seq, time.time() - t))
    print("총 %.1f분" % ((time.time() - t0) / 60))


def table():
    print()
    print("=" * 96)
    print("실험 1c -- imgsz 별 보정 배율 (검출기 yolov8m, conf .10 / iou .45, 60프레임)")
    print("=" * 96)
    print("%-10s" % "시퀀스" + "".join("%14s" % ("imgsz %d" % s) for s in SIZES))
    print("-" * 96)

    grid = {}
    for seq in SEQS:
        cells = []
        for isz in SIZES:
            p = npz_path(seq, isz)
            if not p.exists():
                cells.append("%14s" % "-")
                continue
            m = measure(p)
            grid[(seq, isz)] = m
            cells.append("%10.0fx(%.1f)" % (m["chi"], m["gain"]))
        print("%-10s" % seq + "".join(cells))

    print("-" * 96)
    for label, key, fmt in (("중앙값", "chi", "%10.0fx     "),
                            ("  MOT17-05 제외", "chi", "%10.0fx     ")):
        cells = []
        for isz in SIZES:
            vals = [grid[(s, isz)][key] for s in SEQS
                    if (s, isz) in grid and not (label.strip().startswith("MOT17-05")
                                                 and s == "MOT17-05")]
            cells.append((fmt % np.median(vals)) if vals else "%14s" % "-")
        print("%-10s" % label + "".join(cells))

    print()
    print("%-10s" % "산포(최대/최소)" + "".join(
        "%14.1f" % (max(v) / min(v)) if (v := [grid[(s, i)]["chi"] for s in SEQS
                                               if (s, i) in grid]) else "%14s" % "-"
        for i in SIZES))
    print()
    print("사전 선언한 종말점 (README 실험 1c):")
    print("  [1] chi2 배율 중앙값이 imgsz 에 단조 감소하는가")
    print("  [2] 시퀀스 간 산포(최대/최소)가 줄어드는가  <- 제약 1 이 해상도 산물인지")
    print("  [3] 편상관(신호)이 유지되는가 -- 0.1 밑으로 내려가면 경고")
    print()
    print("%-10s" % "편상관 중앙" + "".join(
        "%14.3f" % np.median(v) if (v := [grid[(s, i)]["pcorr"] for s in SEQS
                                          if (s, i) in grid]) else "%14s" % "-"
        for i in SIZES))
    print("%-10s" % "sig/h 중앙" + "".join(
        "%14.5f" % np.median(v) if (v := [grid[(s, i)]["sig_h"] for s in SEQS
                                          if (s, i) in grid]) else "%14s" % "-"
        for i in SIZES))
    print("%-10s" % "eps/h 중앙" + "".join(
        "%14.5f" % np.median(v) if (v := [grid[(s, i)]["eps_h"] for s in SEQS
                                          if (s, i) in grid]) else "%14s" % "-"
        for i in SIZES))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--table"]
    if "--table" not in sys.argv:
        run([int(a) for a in args] if args else SIZES)
    table()
