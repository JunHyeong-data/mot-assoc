# -*- coding: utf-8 -*-
"""실험 1c -- 입력 해상도가 sigma 의 스케일을 얼마나 정하는가.

MOT17-05 에서 imgsz 를 640 -> 1920 으로 올리니 chi2 배율이 649 -> 83 으로
떨어졌다 (README 의 "MOT17-05 를 갈랐다"). 한 시퀀스 두 점이다.
7시퀀스 x 4해상도로 곡선을 그린다.

**사전 등록은 README 의 "실험 1c" 절에 있다. 돌리기 전에 커밋했다.**
결과를 보고 기준을 바꾸지 않는다.

측정 설계:
  검출기·NMS 설정은 exp01 기본값 그대로 두고 imgsz 하나만 바꾼다.
  프레임 수는 60 으로 -m60 조건과 맞춘다 (640 점은 -m60 을 그대로 쓴다).

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
    """chi2 배율, sigma/h, eps/h, 편상관, n, sigma/h 의 CV 를 낸다.

    CV 는 함정 [2] 용이다. sigma 가 상수로 수렴하면서 배율만 좋아지는 경우를
    가려내야 한다 -- 그건 보정이 아니라 정보 소멸이다.
    """
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
    sh = sig / h
    return dict(n=int(ok.sum()), gain=float(d["gain"]),
                n_det=int(d["n_det"]),
                chi=float(np.median(z2) / CHI2_MED),
                sig_h=float(np.median(sh)),
                cv=float(sh.std() / max(sh.mean(), 1e-12)),
                eps_h=float(np.median(np.hypot(ex, ey) / h)),
                pcorr=float(np.corrcoef(resx, resy)[0, 1]))


def run(sizes):
    t0 = time.time()
    for isz in sizes:
        if isz == 640:
            print("imgsz 640 은 -m60 조건을 그대로 쓴다 (같은 설정). 건너뛴다.")
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


def _row(label, vals, fmt="%12s"):
    print("%-16s" % label + "".join(fmt % v for v in vals))


def table():
    grid = {}
    for seq in SEQS:
        for isz in SIZES:
            p = npz_path(seq, isz)
            if p.exists():
                grid[(seq, isz)] = measure(p)

    def col(isz, key, skip05=False):
        return [grid[(s, isz)][key] for s in SEQS
                if (s, isz) in grid and not (skip05 and s == "MOT17-05")]

    have = [i for i in SIZES if col(i, "chi")]
    if not have:
        print("자료가 없다.")
        return

    # **완전한 열만 판정에 쓴다.** 한두 시퀀스만 있는 열에서 max/min 을 재면
    # 산포가 1.0 으로 나와 "산포가 사라졌다" 로 잘못 읽힌다. 실제로 부분 자료로
    # 돌려 보니 1280 열(1시퀀스)이 1.0배로 나왔다. 미완성 열은 표시만 하고
    # [1][2] 판정에서 뺀다.
    full = [i for i in have if len(col(i, "chi")) == len(SEQS)]

    def mark(isz):
        k = len(col(isz, "chi"))
        return "" if k == len(SEQS) else " (부분 %d/%d)" % (k, len(SEQS))

    print()
    print("=" * 100)
    print("실험 1c -- imgsz 별 보정 (yolov8m, conf .10 / iou .45, 60프레임)")
    print("=" * 100)
    _row("시퀀스", ["imgsz %d" % s for s in have])
    if len(full) < len(have):
        print("  * 부분 완성 열: %s -- 판정에서 제외한다"
              % ", ".join("%d(%d/7)" % (i, len(col(i, "chi")))
                          for i in have if i not in full))
    print("-" * 100)
    for seq in SEQS:
        cells = []
        for isz in have:
            m = grid.get((seq, isz))
            cells.append("%.0fx(g%.1f)" % (m["chi"], m["gain"]) if m else "-")
        _row(seq, cells)
    print("-" * 100)
    _row("중앙값", ["%.0fx" % np.median(col(i, "chi")) for i in have])
    _row(" 05 제외", ["%.0fx" % np.median(col(i, "chi", True)) for i in have])

    print()
    print("=" * 100)
    print("사전 등록한 평가지표 (README 실험 1c. 기준을 바꾸지 않는다)")
    print("=" * 100)

    # [1] 단조 감소 -- 눈으로 판정하지 않고 시퀀스별로 센다 (함정 3)
    print("[1] chi2 배율이 imgsz 에 단조 감소하는가")
    _row("    중앙값", ["%.0fx" % np.median(col(i, "chi")) for i in have])
    if len(full) < 2:
        print("    완전한 열이 %d개뿐이라 판정 보류 (열 %s)" % (len(full), full))
    else:
        med = [np.median(col(i, "chi")) for i in full]
        mono = sum(all(a >= b for a, b in zip(v, v[1:]))
                   for s in SEQS
                   if (v := [grid[(s, i)]["chi"] for i in full]))
        print("    중앙값 단조 감소: %s   (완전한 열 %s)"
              % ("예" if all(a >= b for a, b in zip(med, med[1:])) else "**아니오**",
                 full))
        print("    시퀀스별 단조 감소: %d/%d" % (mono, len(SEQS)))

    # [2] 핵심 -- 시퀀스 간 산포. **완전한 열만** 쓴다.
    print()
    print("[2] **핵심** 시퀀스 간 산포(최대/최소)가 줄어드는가")
    _row("    전체 7개", ["%.1f배" % (max(col(i, "chi")) / min(col(i, "chi")))
                     for i in have])
    _row("    05 제외", ["%.1f배" % (max(col(i, "chi", True))
                                    / min(col(i, "chi", True))) for i in have])
    if len(full) < 2:
        print("    **판정 보류** -- 완전한 열이 %d개뿐이다. 한두 시퀀스짜리 열은"
              % len(full))
        print("    max/min 이 1.0 으로 나와 '산포가 사라졌다' 로 잘못 읽힌다.")
    else:
        sp = [max(col(i, "chi")) / min(col(i, "chi")) for i in full]
        print("    완전한 열끼리: imgsz %d 에서 %.1f배 -> imgsz %d 에서 %.1f배"
              % (full[0], sp[0], full[-1], sp[-1]))
        if sp[-1] <= 3.0:
            print("    => 3배 이하. **제약 1 의 일부는 해상도 산물이다.**")
        elif sp[-1] < sp[0] * 0.6:
            print("    => 크게 줄지만 3배 밑은 아니다. 제약 1 이 약해지되 남는다.")
        else:
            print("    => 안 줄어든다. **제약 1 은 그대로 살아남는다.**")

    # [3] 신호가 유지되는가 + 함정 2 (sigma 가 상수로 수렴하는 경우)
    print()
    print("[3] 편상관(신호)이 유지되는가 -- 0.1 밑이면 경고")
    _row("    편상관 중앙", ["%.3f" % np.median(col(i, "pcorr")) for i in have])
    _row("    sig/h 중앙", ["%.5f" % np.median(col(i, "sig_h")) for i in have])
    _row("    sig/h CV", ["%.3f" % np.median(col(i, "cv")) for i in have])
    _row("    eps/h 중앙", ["%.5f" % np.median(col(i, "eps_h")) for i in have])
    pc = [np.median(col(i, "pcorr")) for i in have]
    if min(pc) < 0.1:
        print("    !! 편상관이 0.1 밑으로 내려간 점이 있다. [1] 의 개선을 믿지 말 것.")
    cv = [np.median(col(i, "cv")) for i in have]
    if len(cv) >= 2 and cv[-1] < 0.5 * cv[0]:
        print("    !! sig/h 의 CV 가 절반 밑으로 줄었다 -- sigma 가 상수로 수렴하는 중이다.")
        print("       배율이 좋아진 것이 보정이 아니라 정보 소멸일 수 있다 (함정 2).")

    # 함정 1 -- 모집단이 바뀌었는가
    print()
    print("[함정 1] imgsz 를 키우면 검출 수가 는다. 모집단이 바뀌면 배율 비교가 오염된다.")
    _row("    매칭 n 중앙", ["%.0f" % np.median(col(i, "n")) for i in have])
    _row("    검출 n_det 중앙", ["%.0f" % np.median(col(i, "n_det")) for i in have])
    base = have[0]
    bad = [(s, i, grid[(s, i)]["n"] / max(grid[(s, base)]["n"], 1))
           for s in SEQS for i in have
           if (s, i) in grid and (s, base) in grid
           and not 0.5 <= grid[(s, i)]["n"] / max(grid[(s, base)]["n"], 1) <= 2.0]
    if bad:
        print("    n 이 2배 넘게 변한 칸 %d개 (배율 비교에서 주의):" % len(bad))
        for s, i, r in bad[:10]:
            print("      %-10s imgsz %-5d  n 이 %.2f배" % (s, i, r))
    else:
        print("    n 이 2배 넘게 변한 칸 없음. 모집단 안정.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--table"]
    if "--table" not in sys.argv:
        run([int(a) for a in args] if args else SIZES)
    table()
