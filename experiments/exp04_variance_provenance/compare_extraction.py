# -*- coding: utf-8 -*-
"""
실험 4 -- 포크의 분산과 우리 분산을 같은 자로 잰다.

**왜 필요한가.** 실험 3 의 measure 실행에서 포크 `nms_var` 의 sigma 가 상자 크기와
거의 무관했다 (corr(s_x, w) = +0.123). 그런데 실험 1 에서 우리가 뽑은 분산은
크기 교란이 지배적이었다. 둘이 갈리는 이유가

    (a) 검출기·가중치가 달라서 (yolov8m COCO vs ablation_17_best yolov8l-mix)
    (b) NMS 설정이 달라서 (conf .10 / iou .45 / 640  vs  conf .01 / iou .7 / 800x1440)
    (c) 추출 방식이 달라서 (우리 return_idxs 재구성 vs 포크의 CUDA 커널)

셋 중 무엇인지 안 갈렸다. 이 스크립트가 (a)(b) 를 국소적으로 가른다.

**포크의 분산 정의는 소스로 확인했다** (DLR-MI/nms_var, `src/cuda/nms_var_kernel.cu`):

  - 살아남은 상자마다 **IoU >= iou_thres 인 후보 전부**가 기여한다.
    **살아남은 상자 자신도 포함**된다.
  - 가중치 없음. 점수로도 IoU 로도 가중하지 않는다.
  - xywh(중심x, 중심y, 폭, 높이) 각각의 **표본분산**, 분모는 **(n-1)**.
    n=1 이면 0.
  - `nms_var.nms(boxes, scores, iou_thres, top_k=1000)` 로 호출된다.
  - 원본 좌표로 되돌릴 때 `var_boxes[..., :4] /= gain ** 2` 로 **제대로** 나눈다
    (`ops.scale_boxes`). 패딩은 빼지 않는다 -- 분산은 평행이동 불변이므로 옳다.

이 스크립트는 그 정의를 그대로 재현한다. 그래야 숫자가 비교 가능하다.

**실험 3 의 s 와 같은 양을 낸다**:  s_x = sqrt(var_x + var_w/4),
s_y = sqrt(var_y + var_h/4)  (tlbr 좌변이 x - w/2 이므로 모서리의 표준편차).

사용법:
    python compare_extraction.py --arm exp01      # 실험 1 설정 그대로
    python compare_extraction.py --arm forkparams # 포크의 NMS 설정, 우리 검출기
    python compare_extraction.py --arm fork --weights ablation_17_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from ultralytics import YOLO                                   # noqa: E402

# 갈래별 설정. exp01 은 실험 1 이 쓴 값, fork* 는 UTrack 의 track_ablation_17.yaml.
ARMS = {
    'exp01':      dict(conf=0.10, iou=0.45, imgsz=640,          weights=None),
    'forkparams': dict(conf=0.01, iou=0.70, imgsz=(800, 1440),  weights=None),
    'fork':       dict(conf=0.01, iou=0.70, imgsz=(800, 1440),  weights='FORK'),
}
TOP_K = 1000          # nms_var.nms 의 top_k
DEFAULT_W = '../BSDsystem/yolov8m.pt'

_CAP = []             # 프레임별 (raw_prediction, kept_idx)


def find_nms_module():
    """ultralytics 판마다 non_max_suppression 위치가 다르다."""
    for path in ('ultralytics.utils.nms', 'ultralytics.utils.ops'):
        try:
            mod = __import__(path, fromlist=['non_max_suppression'])
        except ImportError:
            continue
        if hasattr(mod, 'non_max_suppression'):
            return mod, path
    sys.exit('ERROR non_max_suppression 을 못 찾았다')


def patch(mod):
    orig = mod.non_max_suppression

    def wrapped(prediction, *a, **kw):
        # ultralytics 는 예측을 [tensor, aux] 리스트로 넘긴다. 첫 원소가
        # (batch, 4+nc, n) 이고 행은 **xywh** 다.
        #
        # **반드시 호출 전에 복사해야 한다.** non_max_suppression 은 상자를
        # 제자리에서 xyxy 로 바꾼다. 뷰를 들고 있으면 호출 뒤에는 이미 xyxy 이고,
        # 거기에 xywh2xyxy 를 또 걸면 상자가 3배로 부풀어 조용히 틀린 값이 나온다.
        # (실제로 그렇게 한 번 틀렸다. E[w] 가 1010 px 로 나왔다.)
        t = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        snap = t[0].detach().cpu().float().clone()
        res = orig(prediction, *a, **{**kw, 'return_idxs': True})
        out, keep = res
        idx = keep[0].long().cpu().numpy() if len(keep) else np.zeros(0, int)
        _CAP.append((snap, idx))
        return out if not kw.get('return_idxs', False) else res

    mod.non_max_suppression = wrapped
    return orig


def xywh2xyxy(b):
    o = np.empty_like(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def iou_matrix(a, b):
    """a: (M,4) xyxy, b: (N,4) xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def frame_sigma(pred, kept_idx, conf, iou_thres):
    """포크 커널과 같은 정의로 살아남은 상자별 xywh 표본분산을 낸다.

    pred: (4+nc, n) letterbox 좌표의 raw 예측.
    반환: (m,4) var_xywh, (m,4) xywh   -- letterbox 좌표
    """
    p = pred.numpy().T                                  # (n, 4+nc)
    boxes_xywh = p[:, :4]
    scores = p[:, 4:].max(axis=1)

    above = scores > conf
    if not above.any() or kept_idx.size == 0:
        return np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(0)

    # 포크는 conf 통과분을 점수순으로 정렬해 top_k 만 커널에 넘긴다
    cand = np.flatnonzero(above)
    cand = cand[np.argsort(-scores[cand])][:TOP_K]
    cand_xywh = boxes_xywh[cand]
    cand_xyxy = xywh2xyxy(cand_xywh)

    keep_xyxy = xywh2xyxy(boxes_xywh[kept_idx])
    ious = iou_matrix(keep_xyxy, cand_xyxy)

    var = np.zeros((len(kept_idx), 4))
    for i in range(len(kept_idx)):
        grp = cand_xywh[ious[i] >= iou_thres]           # 자기 자신 포함
        if grp.shape[0] > 1:
            var[i] = grp.var(axis=0, ddof=1)            # 분모 (n-1)
    return var, boxes_xywh[kept_idx], scores[kept_idx]


def summarize(name, var, xywh, gain):
    """calibrate.py 와 **같은 형식**으로 낸다. 그래야 바로 대조된다."""
    var = var / (gain ** 2)                             # 원본 좌표로
    xywh = xywh / gain
    sx = np.sqrt(np.maximum(var[:, 0] + 0.25 * var[:, 2], 0))
    sy = np.sqrt(np.maximum(var[:, 1] + 0.25 * var[:, 3], 0))
    w, h = xywh[:, 2], xywh[:, 3]
    ok = (w > 1) & (h > 1)
    sx, sy, w, h = sx[ok], sy[ok], w[ok], h[ok]

    print('== %s ==' % name)
    print('  boxes %d   var 가 0 인 상자 %.1f%%'
          % (sx.size, 100 * np.mean((sx <= 0) & (sy <= 0))))
    print('  E[s_x] %.4f px (sd %.4f)   E[s_y] %.4f px (sd %.4f)'
          % (sx.mean(), sx.std(), sy.mean(), sy.std()))
    print('  E[w]   %.2f px             E[h]   %.2f px' % (w.mean(), h.mean()))
    print('  s_x/w  mean %.4f   s_y/h mean %.4f'
          % (np.mean(sx / w), np.mean(sy / h)))
    print('  CV(s_x) = %.3f   CV(s_y) = %.3f'
          % (sx.std() / max(sx.mean(), 1e-9), sy.std() / max(sy.mean(), 1e-9)))
    print('  corr(s_x, w) = %+.3f   corr(s_y, h) = %+.3f'
          % (np.corrcoef(sx, w)[0, 1], np.corrcoef(sy, h)[0, 1]))
    return dict(n=sx.size, e_sx=sx.mean(), e_w=w.mean(),
                cv=sx.std() / max(sx.mean(), 1e-9),
                corr=np.corrcoef(sx, w)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), default='exp01')
    ap.add_argument('--seq', default='MOT17-02-FRCNN')
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--weights', default=DEFAULT_W)
    ap.add_argument('--root', default='data/MOT17_A/ablation')
    ap.add_argument('--min_score', type=float, default=0.0,
                    help='연관에 들어가는 검출만 보려면 0.1 (UTrack scores.low)')
    a = ap.parse_args()

    cfg = ARMS[a.arm]
    if a.seq == 'all':
        # 포크 통계는 7시퀀스 전체에서 나왔다. 모집단을 맞추려면 여기도 그래야 한다.
        seqs = sorted(d.name for d in Path(a.root).iterdir() if d.is_dir())
    else:
        seqs = [a.seq]
    imgs = []
    for sq in seqs:
        imgs += sorted((Path(a.root) / sq / 'img1').glob('*.jpg'))[:a.frames]
    if not imgs:
        sys.exit('ERROR 이미지가 없다: %s' % a.root)
    print('시퀀스 %d개: %s' % (len(seqs), ', '.join(seqs)))

    mod, where = find_nms_module()
    print('non_max_suppression 위치: %s' % where)
    print('갈래 %s  conf=%.2f iou=%.2f imgsz=%s  가중치=%s  프레임=%d'
          % (a.arm, cfg['conf'], cfg['iou'], cfg['imgsz'], a.weights, len(imgs)))
    patch(mod)

    model = YOLO(a.weights)
    imgsz = cfg['imgsz'] if isinstance(cfg['imgsz'], int) else list(cfg['imgsz'])

    all_var, all_box, all_sc = [], [], []
    gain = None
    for k, im in enumerate(imgs):
        _CAP.clear()
        model.predict(str(im), conf=cfg['conf'], iou=cfg['iou'], imgsz=imgsz,
                      verbose=False, device='cpu')
        if not _CAP:
            continue
        if gain is None:
            import cv2
            h0, w0 = cv2.imread(str(im)).shape[:2]
            gh, gw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
            gain = min(gh / h0, gw / w0)
            print('원본 %dx%d, letterbox %s -> gain %.4f' % (h0, w0, imgsz, gain))
        pred, idx = _CAP[0]
        v, b, sc = frame_sigma(pred, idx, cfg['conf'], cfg['iou'])
        if v.shape[0]:
            all_var.append(v)
            all_box.append(b)
            all_sc.append(sc)
        if (k + 1) % 20 == 0:
            print('  ... %d/%d 프레임' % (k + 1, len(imgs)))

    if not all_var:
        sys.exit('ERROR 상자를 하나도 못 모았다')
    var, box, sc = (np.concatenate(all_var), np.concatenate(all_box),
                    np.concatenate(all_sc))
    if a.min_score > 0:
        # 포크 통계는 **연관에 들어간 검출**만 담았다. UTrack 은 그 전에
        # thresholds.scores.low 로 자른다. 모집단을 맞추지 않으면 비교가 안 된다.
        keep = sc >= a.min_score
        print('점수 >= %.2f 로 %d -> %d 개' % (a.min_score, sc.size, keep.sum()))
        var, box = var[keep], box[keep]
    summarize('우리 추출 · 갈래 %s (min_score %.2f, %s)'
              % (a.arm, a.min_score, a.seq),
              var, box, gain)


if __name__ == '__main__':
    main()
