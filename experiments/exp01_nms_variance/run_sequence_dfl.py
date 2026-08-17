# -*- coding: utf-8 -*-
"""실험 1g -- 소스를 DFL 분포 분산으로 바꾼다.

**사전 선언은 README 의 "실험 1g" 절. 자료보다 먼저 커밋했다.**

YOLOv8 헤드는 상자 네 변의 거리를 reg_max=16 빈의 **분포**로 예측하고
DFL 모듈이 그 **기댓값만** 취한다. 분산은 버려진다. 그 분산이 곧 위치
불확실성이고 추가 추론 비용이 0 이다.

가로채기:
    `Detect.dfl` 에 forward-pre-hook 을 걸어 입력 (b, 4*reg_max, A) 을 잡는다.
    DFL.forward 와 **똑같은 순서**로 확률을 만든다:
        p = x.view(b, 4, c1, A).transpose(2, 1).softmax(1)
    빈 인덱스 i 에 대해 E[d]=sum(i*p), Var[d]=sum(i^2*p)-E[d]^2 (stride 단위).

    **검산됨**: 이렇게 복원한 기대거리로 상자를 다시 만들면 NMS 출력과
    6e-05 px 안에서 일치한다 (`scratchpad/dfl_probe.py`).

중심 공분산 (변끼리 독립 가정 -- DFL 이 변마다 별도 softmax 라 구조적으로 그렇다):
    x1 = ax - l, x2 = ax + r  ->  cx = ax + (r-l)/2  ->  Var[cx] = (Var[l]+Var[r])/4
    y1 = ay - t, y2 = ay + b  ->  cy = ay + (b-t)/2  ->  Var[cy] = (Var[t]+Var[b])/4

npz 는 `run_sequence.py` 와 **같은 키**로 저장한다. 그래야 aggregate /
analyze_covariance / calibrate_sigma / student_t 가 그대로 돈다 (TAG 만 다르게).

사용법:
    python experiments/exp01_nms_variance/run_sequence_dfl.py [시퀀스] [최대프레임]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from scipy.optimize import linear_sum_assignment          # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402
from ultralytics import YOLO                               # noqa: E402
from ultralytics.utils import nms as ulnms                 # noqa: E402

SEQ = sys.argv[1] if len(sys.argv) > 1 else "MOT17-02-FRCNN"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ROOT = Path("data/MOT17_A/ablation") / SEQ

MODEL = os.environ.get("EXP01_MODEL", "../BSDsystem/yolov8m.pt")
CONF = float(os.environ.get("EXP01_CONF", 0.10))
IOU_NMS = float(os.environ.get("EXP01_IOU", 0.45))
_isz = os.environ.get("EXP01_IMGSZ", "640")
IMGSZ = [int(x) for x in _isz.split(",")] if "," in _isz else int(_isz)
TAG = os.environ.get("EXP01_TAG", "-dfl")
GT_IOU = 0.5

CAP = {}
_orig_nms = ulnms.non_max_suppression


def patched_nms(prediction, *a, **kw):
    """살아남은 상자의 앵커 인덱스를 잡는다. exp01 과 같은 방식."""
    res = _orig_nms(prediction, *a, **{**kw, "return_idxs": True})
    out, keepi = res
    CAP["keep"] = keepi[0].long().cpu().numpy()
    return res if kw.get("return_idxs", False) else out


ulnms.non_max_suppression = patched_nms
for mod in ("ultralytics.models.yolo.detect.predict", "ultralytics.engine.results"):
    m = sys.modules.get(mod)
    if m is not None and hasattr(m, "non_max_suppression"):
        m.non_max_suppression = patched_nms


def iou_mat(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(path):
    gt = {}
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 9:
            continue
        fr, x, y, w, h = int(f[0]), *map(float, f[2:6])
        if int(f[6]) != 1 or int(f[7]) != 1:
            continue
        gt.setdefault(fr, []).append((x, y, x + w, y + h, float(f[8])))
    return {k: np.array(v) for k, v in gt.items()}


print("=" * 72)
print("실험 1g -- DFL 분포 분산")
print("=" * 72)
print(f"  시퀀스 {SEQ}   모델 {MODEL}   imgsz {IMGSZ}   conf {CONF} iou {IOU_NMS}")

model = YOLO(MODEL)
det_head = model.model.model[-1]
C1 = det_head.dfl.c1
print(f"  reg_max(c1) = {C1}   stride = {det_head.stride.tolist()}")


def dfl_pre(module, args):
    CAP["logits"] = args[0].detach().clone()
    CAP["strides"] = det_head.strides.detach().clone()


det_head.dfl.register_forward_pre_hook(dfl_pre)

BINS = torch.arange(C1, dtype=torch.float32).view(1, C1, 1, 1)

imgs = sorted((ROOT / "img1").glob("*.jpg"))[:LIMIT]
gt = load_gt(ROOT / "gt" / "gt.txt")
print(f"  프레임 {len(imgs)}장")

rows = []
n_det = n_matched = 0
gains, sat_frac, stride_hist = [], [], {}
for k, ip in enumerate(imgs):
    CAP.clear()
    r = model.predict(source=str(ip), conf=CONF, iou=IOU_NMS, classes=[0],
                      imgsz=IMGSZ, verbose=False)[0]
    detb = r.boxes.xyxy.cpu().numpy()
    n_det += len(detb)
    g = gt.get(k + 1)
    if g is None or not len(detb) or "logits" not in CAP:
        continue
    keep = CAP["keep"]
    if len(keep) != len(detb):
        continue

    lg = CAP["logits"]
    b, _, A = lg.shape
    p = lg.view(b, 4, C1, A).transpose(2, 1).softmax(1)     # (b, C1, 4, A)
    mean = (p * BINS).sum(1)[0]                              # (4, A) stride 단위
    var = (p * BINS ** 2).sum(1)[0] - mean ** 2
    st = CAP["strides"].view(-1)

    pv = p[0, :, :, keep].cpu().numpy()                      # (C1, 4, n)
    sat_frac.append(float(np.mean(pv[-1] > 0.10)))           # 함정 2: 마지막 빈 포화
    sk = st[keep].cpu().numpy()
    for s_, c_ in zip(*np.unique(sk, return_counts=True)):
        stride_hist[int(s_)] = stride_hist.get(int(s_), 0) + int(c_)

    vk = var[:, keep].cpu().numpy()                          # (4, n)
    s2 = sk ** 2
    var_cx = (vk[0] + vk[2]) / 4.0 * s2                      # letterbox px^2
    var_cy = (vk[1] + vk[3]) / 4.0 * s2

    # letterbox -> 원본 배율 (run_sequence.py 와 같은 방식)
    hl = np.array([(mean[3, i] + mean[1, i]) * st[i] for i in keep], dtype=float)
    ho = detb[:, 3] - detb[:, 1]
    okg = hl > 1e-6
    gain = float(np.median(ho[okg] / hl[okg])) if okg.any() else 1.0
    gains.append(gain)

    M = iou_mat(detb, g[:, :4])
    ri, ci = linear_sum_assignment(-M)
    for i, j in zip(ri, ci):
        if M[i, j] < GT_IOU:
            continue
        n_matched += 1
        h_o = float(detb[i, 3] - detb[i, 1])
        sxx = float(var_cx[i]) * gain ** 2
        syy = float(var_cy[i]) * gain ** 2
        s_c = float(np.sqrt(sxx + syy) / max(h_o, 1e-6))
        dcx = (detb[i, 0] + detb[i, 2]) / 2 - (g[j, 0] + g[j, 2]) / 2
        dcy = (detb[i, 1] + detb[i, 3]) / 2 - (g[j, 1] + g[j, 3]) / 2
        rows.append((s_c, np.nan, C1, h_o, 1.0 - M[i, j], g[j, 4], k + 1,
                     sxx, 0.0, syy, dcx, dcy))

print(f"  검출 {n_det}개, GT 매칭 {n_matched}개 (매칭률 {n_matched/max(n_det,1):.1%})")

import cv2                                                   # noqa: E402
_h0, _w0 = cv2.imread(str(imgs[0])).shape[:2]
_gh, _gw = (IMGSZ, IMGSZ) if isinstance(IMGSZ, int) else IMGSZ
_expect = 1.0 / min(_gh / _h0, _gw / _w0)
g_med = float(np.median(gains)) if gains else float("nan")
print(f"  letterbox->원본 배율: 실측 {g_med:.4f}  해석해 {_expect:.4f}")
if not np.isfinite(g_med) or abs(g_med - _expect) > 0.02 * _expect:
    print("  *** 경고: 배율이 해석해와 어긋난다. ***")
print(f"  [함정 2] 마지막 빈 확률 >0.10 인 변의 비율: {np.mean(sat_frac):.2%}")
print(f"  [함정 3] stride 분포: {stride_hist}")

A = np.array(rows, dtype=float)
if len(A) < 20:
    sys.exit("표본이 너무 적다.")
sc, sh, nc, hh, err, vis, frame, sxx, sxy, syy, dcx, dcy = A.T

out_npz = Path("data/exp01") / f"{SEQ}{TAG}.npz"
out_npz.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_npz, s_c=sc, s_h=sh, ncand=nc, h=hh, err=err, vis=vis, frame=frame,
         sxx=sxx, sxy=sxy, syy=syy, dcx=dcx, dcy=dcy,
         n_det=n_det, conf=CONF, iou_nms=IOU_NMS, min_cand=0,
         gain=g_med, coords="original")
print(f"  저장: {out_npz}")

rho, _ = spearmanr(sc, err)
rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (sc, err, hh))
Am = np.column_stack([np.ones_like(rz), rz])
ex_ = rx - Am @ np.linalg.lstsq(Am, rx, rcond=None)[0]
ey_ = ry - Am @ np.linalg.lstsq(Am, ry, rcond=None)[0]
print()
print("  [1] rho_primary = %+.4f   h 통제 편상관 = %+.4f   (관문 0.3)"
      % (rho, np.corrcoef(ex_, ey_)[0, 1]))
print("      * [2] (C 를 NLL 로 이기는가) 는 student_t.py 로 따로 판정한다")
