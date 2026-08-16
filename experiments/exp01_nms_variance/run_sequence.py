# -*- coding: utf-8 -*-
"""
실험 1 파이프라인 -- 한 시퀀스로 먼저 검증한다.

묻는 것: NMS 가 억제한 후보들의 표본 산포가, 그 검출이 실제로 얼마나
         부정확한지를 알려주는가.

사전 선언된 주 종말점 (README 참고. 나중에 바꾸지 않는다):
    rho_primary = Spearman( s_c , 1 - IoU )
    s_c = 억제된 후보 중심의 표본 표준편차 / 검출 상자 높이 h
    전체 검출 풀에서 하나의 값으로 계산한다. 나머지는 전부 탐색적이다.

판정: rho >= 0.3 진행 / 0.1~0.3 보정 후 재시도 / < 0.1 방향 접음

억제 후보를 어떻게 얻나:
    ultralytics 는 NMS 결과만 돌려주고 무엇이 억제됐는지는 버린다.
    그래서 `non_max_suppression` 을 가로채 raw prediction 을 붙잡고,
    살아남은 상자마다 IoU >= iou_thres 인 raw 후보를 직접 모은다.
    NMS 가 억제하는 조건이 정확히 이것이다.

좌표계 주의:
    가로챈 시점의 상자는 letterbox 좌표(640 기준)다. 원본 좌표가 아니다.
    그러나 주 종말점은 s_c / h 로 무차원이고 letterbox 는 x,y 를 같은 배율로
    줄이므로 비율이 보존된다. 따라서 letterbox 좌표에서 계산해도 된다.

사용법:
    python experiments/exp01_nms_variance/run_sequence.py [시퀀스명] [최대프레임]
"""
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
MODEL = "../BSDsystem/yolov8m.pt"

CONF = 0.10          # 낮게 잡아야 억제 후보가 충분히 생긴다
IOU_NMS = 0.45       # 이 이상 겹치는 raw 후보가 '억제된 것'
MIN_CAND = 3         # 표본 산포를 믿을 최소 후보 수
GT_IOU = 0.5         # GT 매칭 기준

CAPTURED = []        # 프레임별 (kept_boxes, [후보통계...])


def xywh2xyxy(b):
    o = b.clone()
    o[..., 0] = b[..., 0] - b[..., 2] / 2
    o[..., 1] = b[..., 1] - b[..., 3] / 2
    o[..., 2] = b[..., 0] + b[..., 2] / 2
    o[..., 3] = b[..., 1] + b[..., 3] / 2
    return o


def iou_mat(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m)"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


_orig_nms = ulnms.non_max_suppression


def patched_nms(prediction, *a, **kw):
    """살아남은 상자의 '원본 anchor 인덱스' 를 받아 letterbox 좌표에서 군집을 만든다.

    NMS 출력 상자는 이미 원본 이미지 좌표로 스케일되어 있고 raw prediction 은
    letterbox 좌표다. 둘을 직접 IoU 로 맞추면 좌표계가 어긋나 후보가 0 개가 된다.
    return_idxs=True 가 주는 인덱스를 쓰면 그 문제가 사라진다.
    """
    want = kw.get("return_idxs", False)
    res = _orig_nms(prediction, *a, **{**kw, "return_idxs": True})
    out, keepi = res
    try:
        p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        x = p[0].T if p.ndim == 3 else p.T           # (N, 4+nc), letterbox 좌표
        boxes = xywh2xyxy(x[:, :4]).cpu().numpy()
        person = x[:, 4].cpu().numpy()               # COCO class 0 = person
        pool = person >= CONF                        # NMS 가 실제로 본 후보 풀
        pool_box, pool_sc = boxes[pool], person[pool]

        idx = keepi[0].long().cpu().numpy()          # 원본 anchor 인덱스, out 과 같은 순서
        kept = boxes[idx]
        det_o = out[0][:, :4].cpu().numpy() if len(out[0]) else np.zeros((0, 4))

        # letterbox -> 원본 배율. NMS 출력은 원본 좌표, kept 는 letterbox 좌표다.
        # 공분산은 평행이동에 불변이므로 배율만 알면 원본 좌표로 옮길 수 있다.
        r = 1.0
        if len(det_o) == len(kept) and len(kept):
            hl = kept[:, 3] - kept[:, 1]
            ho = det_o[:, 3] - det_o[:, 1]
            good = hl > 1e-6
            if good.any():
                r = float(np.median(ho[good] / hl[good]))

        stats = []
        if len(kept) and len(pool_box):
            M = iou_mat(kept, pool_box)
            for i in range(len(kept)):
                sel = M[i] >= IOU_NMS                # NMS 가 이 상자로 억제한 것들
                c = pool_box[sel]
                h = max(kept[i, 3] - kept[i, 1], 1e-6)
                if len(c) >= MIN_CAND:
                    cx = (c[:, 0] + c[:, 2]) / 2
                    cy = (c[:, 1] + c[:, 3]) / 2
                    s_c = float(np.hypot(cx.std(ddof=1), cy.std(ddof=1)) / h)
                    s_h = float((c[:, 3] - c[:, 1]).std(ddof=1) / h)
                    # 후보 중심의 2x2 공분산을 원본 좌표로 (배율^2 을 곱한다)
                    C2 = np.cov(np.vstack([cx, cy])) * (r ** 2)
                    sxx, sxy, syy = float(C2[0, 0]), float(C2[0, 1]), float(C2[1, 1])
                else:
                    s_c = s_h = sxx = sxy = syy = np.nan
                stats.append((s_c, s_h, int(len(c)), float(h),
                              float(pool_sc[sel].max() if sel.any() else 0),
                              sxx, sxy, syy))
        CAPTURED.append(stats)
    except Exception as e:                            # 계측 실패가 추론을 막지 않게
        print(f"  [경고] NMS 가로채기 실패: {type(e).__name__}: {e}")
        CAPTURED.append([])
    return res if want else out


ulnms.non_max_suppression = patched_nms
for mod in ("ultralytics.models.yolo.detect.predict", "ultralytics.engine.results"):
    m = sys.modules.get(mod)
    if m is not None and hasattr(m, "non_max_suppression"):
        m.non_max_suppression = patched_nms


def load_gt(path):
    """frame -> (n,4) xyxy. 보행자(class 1) 이고 ignore 플래그가 1 인 것만."""
    gt = {}
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 9:
            continue
        fr, x, y, w, h = int(f[0]), *map(float, f[2:6])
        flag, cls, vis = int(f[6]), int(f[7]), float(f[8])
        if flag != 1 or cls != 1:
            continue
        gt.setdefault(fr, []).append((x, y, x + w, y + h, vis))
    return {k: np.array(v) for k, v in gt.items()}


print("=" * 72)
print("실험 1 파이프라인 검증 -- 한 시퀀스")
print("=" * 72)
print(f"  시퀀스   : {SEQ}")
print(f"  모델     : {MODEL}")
print(f"  conf     : {CONF}   NMS IoU: {IOU_NMS}   최소 후보수: {MIN_CAND}")
print()

imgs = sorted((ROOT / "img1").glob("*.jpg"))[:LIMIT]
gt = load_gt(ROOT / "gt" / "gt.txt")
print(f"  프레임 {len(imgs)}장 처리 (시퀀스 전체 {len(sorted((ROOT / 'img1').glob('*.jpg')))}장)")

model = YOLO(MODEL)
rows = []
n_matched = n_det = 0
for k, ip in enumerate(imgs):
    CAPTURED.clear()
    r = model.predict(source=str(ip), conf=CONF, iou=IOU_NMS, classes=[0],
                      verbose=False)[0]
    stats = CAPTURED[0] if CAPTURED else []
    det = r.boxes.xyxy.cpu().numpy()
    n_det += len(det)
    g = gt.get(k + 1)
    if g is None or not len(det) or len(stats) != len(det):
        continue
    M = iou_mat(det, g[:, :4])
    ri, ci = linear_sum_assignment(-M)
    for i, j in zip(ri, ci):
        if M[i, j] < GT_IOU:
            continue
        n_matched += 1
        s_c, s_h, ncand, h, _, sxx, sxy, syy = stats[i]
        dcx = (det[i, 0] + det[i, 2]) / 2 - (g[j, 0] + g[j, 2]) / 2
        dcy = (det[i, 1] + det[i, 3]) / 2 - (g[j, 1] + g[j, 3]) / 2
        rows.append((s_c, s_h, ncand, h, 1.0 - M[i, j], g[j, 4], k + 1,
                     sxx, sxy, syy, dcx, dcy))

print(f"  검출 {n_det}개, GT 매칭 {n_matched}개 (매칭률 {n_matched / max(n_det,1):.1%})")
print()

A = np.array(rows, dtype=float)
if len(A) < 20:
    sys.exit("표본이 너무 적다. 프레임 수를 늘릴 것.")

sc, sh, ncand, hh, err, vis, frame, sxx, sxy, syy, dcx, dcy = A.T
ok = ~np.isnan(sc)

out_npz = Path("data/exp01") / f"{SEQ}.npz"
out_npz.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_npz, s_c=sc, s_h=sh, ncand=ncand, h=hh, err=err, vis=vis, frame=frame,
         sxx=sxx, sxy=sxy, syy=syy, dcx=dcx, dcy=dcy,
         n_det=n_det, conf=CONF, iou_nms=IOU_NMS, min_cand=MIN_CAND)
print(f"  원자료 저장: {out_npz}  (재분석에 검출기 재실행 불필요)")
print()
print("=" * 72)
print("[1] 표본 조건")
print("=" * 72)
print(f"  매칭된 검출 {len(A)}개 중 후보 {MIN_CAND}개 이상: {ok.sum()} ({ok.mean():.1%})")
print(f"  조건 미달로 버려짐: {(~ok).sum()} ({(~ok).mean():.1%})")
print(f"  후보 수 중앙값 {np.median(ncand):.0f}, 사분위 "
      f"{np.percentile(ncand,25):.0f}~{np.percentile(ncand,75):.0f}")
print()

x, y, h_ = sc[ok], err[ok], hh[ok]
rho, p = spearmanr(x, y)

print("=" * 72)
print("[2] 주 종말점")
print("=" * 72)
print(f"  rho_primary = Spearman(s_c, 1-IoU) = {rho:+.4f}   (p={p:.3g}, n={ok.sum()})")

# 프레임 단위 블록 부트스트랩.
# 같은 프레임의 검출들은 독립이 아니다 (같은 장면·조명·군중). 검출 단위로
# 재표집하면 불확실성을 심하게 과소평가한다. 실제로 40프레임 결과의 검출단위
# CI [+0.283,+0.437] 는 전체 시퀀스 값 +0.2415 를 포함하지 못했다.
fr = frame[ok]
uf = np.unique(fr)
by_frame = {f: np.where(fr == f)[0] for f in uf}
bs, bs_naive = [], []
rs = np.random.default_rng(0)
for _ in range(2000):
    pick = rs.choice(uf, len(uf), replace=True)
    idx = np.concatenate([by_frame[f] for f in pick])
    if len(np.unique(x[idx])) > 2:
        bs.append(spearmanr(x[idx], y[idx])[0])
    j = rs.integers(0, len(x), len(x))
    if len(np.unique(x[j])) > 2:
        bs_naive.append(spearmanr(x[j], y[j])[0])
if bs:
    lo, hi = np.percentile(bs, [2.5, 97.5])
    nlo, nhi = np.percentile(bs_naive, [2.5, 97.5])
    print(f"  95% CI (프레임 블록, 정직) = [{lo:+.4f}, {hi:+.4f}]   프레임 {len(uf)}개")
    print(f"  95% CI (검출 단위, 과소평가) = [{nlo:+.4f}, {nhi:+.4f}]  <- 쓰지 말 것")
else:
    lo = hi = np.nan
    print("  부트스트랩 불가 (표본 부족)")
print()

# 크기 교란 통제: 순위변환 후 h 를 회귀로 제거한 편상관
def rank(v):
    return spearmanr(v, v)[0] * 0 + np.argsort(np.argsort(v)).astype(float)


rx, ry, rh = rank(x), rank(y), rank(h_)
rx_ = rx - np.polyval(np.polyfit(rh, rx, 1), rh)
ry_ = ry - np.polyval(np.polyfit(rh, ry, 1), rh)
prho = np.corrcoef(rx_, ry_)[0, 1]
print(f"  상자높이 h 통제 편상관      = {prho:+.4f}")
print(f"  (원시 rho 와 차이 {abs(prho-rho):.4f} -- 이만큼이 크기 교란이다)")
print()

print("=" * 72)
print("[3] 판정 (사전 선언 기준)")
print("=" * 72)
v = abs(rho)
verdict = ("신호 있음. 다음 단계 진행" if v >= 0.3 else
           "약함. 보정 붙여 재시도" if v >= 0.1 else
           "방향 접는다. 다른 불확실성 소스를 찾는다")
print(f"  |rho| = {v:.4f}  ->  {verdict}")
if lo < 0.3 < hi:
    print("  주의: CI 가 0.3 을 걸친다. '애매하다'로 보고하고 표본을 늘릴 것.")
print()
print("  * 이것은 한 시퀀스 부분 프레임 결과다. 파이프라인이 도는지 보는 것이 목적이고")
print("    판정 자체는 전체 시퀀스를 돌린 뒤에 한다.")
print()
print("  탐색적 (판정에 쓰지 않음):")
er, _ = spearmanr(sh[ok], y)
print(f"    Spearman(s_h, 1-IoU) = {er:+.4f}")
for name, m in (("가시성>=0.7", vis[ok] >= 0.7), ("가시성<0.7", vis[ok] < 0.7)):
    if m.sum() > 20:
        print(f"    {name:12} n={m.sum():5d}  rho={spearmanr(x[m], y[m])[0]:+.4f}")
