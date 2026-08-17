# -*- coding: utf-8 -*-
"""실험 1 -- 시퀀스별 결과를 하나의 표로 모은다.

`analyze_covariance.py` 의 [1] 절과 **같은 계산**을 npz 에서 직접 한다
(콘솔 인코딩에 기대지 않으려고). 두 검출기를 나란히 낸다.

    python experiments/exp01_nms_variance/aggregate.py            # 우리 검출기
    python experiments/exp01_nms_variance/aggregate.py -fork      # 저자 가중치
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
CHI2_MED = 1.386          # chi2(2) 중앙값
CHI2_95 = 5.991           # chi2(2) 95% 분위
SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"]


def partial_spearman(x, y, z):
    """z 를 통제한 x,y 의 스피어만 편상관 (순위 잔차 상관)."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


rows = []
for s in SEQS:
    p = Path("data/exp01") / f"{s}-FRCNN{TAG}.npz"
    if not p.exists():
        print("없음: %s" % p)
        continue
    d = np.load(p)
    sxx, sxy, syy = d["sxx"], d["sxy"], d["syy"]
    det2 = sxx * syy - sxy ** 2
    ok = ~np.isnan(sxx) & (det2 > 1e-9) & (sxx > 1e-9) & (syy > 1e-9)
    ex, ey = d["dcx"][ok], d["dcy"][ok]
    Sxx, Sxy, Syy = sxx[ok], sxy[ok], syy[ok]
    dt = Sxx * Syy - Sxy ** 2
    z2 = (Syy * ex ** 2 - 2 * Sxy * ex * ey + Sxx * ey ** 2) / dt

    g = ~np.isnan(d["s_c"])
    rho = spearmanr(d["s_c"][g], d["err"][g]).statistic
    pc = partial_spearman(d["s_c"][g], d["err"][g], d["h"][g])
    rows.append((s, len(z2), rho, pc, np.median(z2),
                 np.median(z2) / CHI2_MED, 100 * np.mean(z2 <= CHI2_95)))

print("검출기: %s" % ("저자 ablation_17 + 포크 NMS" if TAG else "yolov8m COCO + exp01 설정"))
print()
print("시퀀스      n     rho     편상관   z2중앙   chi2배율  std배  95%안")
for s, n, r, pc, z, ratio, in95 in rows:
    print("%-9s %5d  %+.3f  %+.3f  %8.1f  %7.0fx %5.1fx %5.1f%%"
          % (s, n, r, pc, z, ratio, np.sqrt(ratio), in95))

pc = np.array([r[3] for r in rows]); ra = np.array([r[5] for r in rows])
print()
print("편상관   중앙값 %+.3f   범위 %+.3f~%+.3f   0.1 미만 %d/%d"
      % (np.median(pc), pc.min(), pc.max(), (pc < 0.1).sum(), len(pc)))
print("chi2배율 중앙값 %.0f배    범위 %.0f~%.0f배" % (np.median(ra), ra.min(), ra.max()))
print("std 환산 중앙값 %.1f배    범위 %.1f~%.1f배"
      % (np.sqrt(np.median(ra)), np.sqrt(ra.min()), np.sqrt(ra.max())))
