# -*- coding: utf-8 -*-
"""실험 5 [1단계] -- 검출을 한 번만 돌려 DFL sigma 와 함께 캐시한다.

**왜 캐시하나.** 조건마다 검출기를 다시 돌리면 (a) 시간이 조건 수만큼 들고
(b) 검출이 조건마다 미세하게 달라질 여지가 생긴다. 한 번 돌려 캐시하고
트래커만 재생하면 **조건 사이에서 검출이 비트 단위로 같다.** 갈리는 것이
비용함수 하나뿐이 된다. exp00 의 `sweep.py` 가 쓴 것과 같은 수법이다.

저장하는 것 (프레임당):
    xyxy   (n,4)  원본 좌표 검출 박스
    conf   (n,)   점수
    sxx    (n,)   DFL 중심 분산 x  (원본 px^2)
    syy    (n,)   DFL 중심 분산 y  (원본 px^2)

DFL 분산 추출은 `exp01/run_sequence_dfl.py` 와 **같은 코드 경로**다
(Detect.dfl forward-pre-hook -> 분포 복원 -> Var[cx]=(Var[l]+Var[r])/4).
그 가로채기는 박스 재구성이 NMS 출력과 6e-05 px 안에서 일치하는 것으로 검산됐다.

사용법:
    python experiments/exp05_wasserstein/cache_detections.py
    EXP05_MODEL=... EXP05_IMGSZ=800,1440 python .../cache_detections.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ultralytics import YOLO                               # noqa: E402
from ultralytics.utils import nms as ulnms                 # noqa: E402

ROOT = Path("data/MOT17_A/ablation")
OUT = Path("data/exp05")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

# 저자 가중치 + 포크 설정이 기본값 -- exp02/exp03 의 기준선(HOTA 64.494)과
# 같은 검출기를 써야 숫자가 이어진다.
MODEL = os.environ.get("EXP05_MODEL", "data/weights/ablation_17_best.pt")
CONF = float(os.environ.get("EXP05_CONF", 0.01))
IOU_NMS = float(os.environ.get("EXP05_IOU", 0.70))
_isz = os.environ.get("EXP05_IMGSZ", "800,1440")
IMGSZ = [int(x) for x in _isz.split(",")] if "," in _isz else int(_isz)
TAG = os.environ.get("EXP05_TAG", "")

CAP = {}
_orig_nms = ulnms.non_max_suppression


def patched_nms(prediction, *a, **kw):
    res = _orig_nms(prediction, *a, **{**kw, "return_idxs": True})
    out, keepi = res
    CAP["keep"] = keepi[0].long().cpu().numpy()
    return res if kw.get("return_idxs", False) else out


ulnms.non_max_suppression = patched_nms
for mod in ("ultralytics.models.yolo.detect.predict", "ultralytics.engine.results"):
    m = sys.modules.get(mod)
    if m is not None and hasattr(m, "non_max_suppression"):
        m.non_max_suppression = patched_nms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)
    det_head = model.model.model[-1]
    C1 = det_head.dfl.c1
    BINS = torch.arange(C1, dtype=torch.float32).view(1, C1, 1, 1)
    print("모델 %s   reg_max %d   imgsz %s   conf %.3f iou %.2f"
          % (MODEL, C1, IMGSZ, CONF, IOU_NMS))

    def dfl_pre(module, args):
        CAP["logits"] = args[0].detach().clone()
        CAP["strides"] = det_head.strides.detach().clone()

    det_head.dfl.register_forward_pre_hook(dfl_pre)

    t0 = time.time()
    for seq in SEQS:
        f = OUT / ("%s%s.npz" % (seq, TAG))
        if f.exists():
            print("  %-18s 이미 있다" % seq)
            continue
        imgs = sorted((ROOT / seq / "img1").glob("*.jpg"))
        frames, boxes, confs, vxx, vyy = [], [], [], [], []
        t = time.time()
        for k, ip in enumerate(imgs):
            CAP.clear()
            r = model.predict(source=str(ip), conf=CONF, iou=IOU_NMS, classes=[0],
                              imgsz=IMGSZ, verbose=False)[0]
            db = r.boxes.xyxy.cpu().numpy()
            dc = r.boxes.conf.cpu().numpy()
            n = len(db)
            if n == 0 or "logits" not in CAP or len(CAP["keep"]) != n:
                sx = sy = np.zeros(n)
            else:
                lg = CAP["logits"]
                b, _, A = lg.shape
                p = lg.view(b, 4, C1, A).transpose(2, 1).softmax(1)
                mean = (p * BINS).sum(1)[0]
                var = (p * BINS ** 2).sum(1)[0] - mean ** 2
                st = CAP["strides"].view(-1)
                keep = CAP["keep"]
                vk = var[:, keep].cpu().numpy()
                sk = st[keep].cpu().numpy()
                # letterbox -> 원본 배율 (run_sequence_dfl.py 와 같은 방식)
                hl = np.array([(mean[3, i] + mean[1, i]) * st[i] for i in keep],
                              dtype=float)
                ho = db[:, 3] - db[:, 1]
                ok = hl > 1e-6
                g = float(np.median(ho[ok] / hl[ok])) if ok.any() else 1.0
                s2 = (sk ** 2) * (g ** 2)
                sx = (vk[0] + vk[2]) / 4.0 * s2
                sy = (vk[1] + vk[3]) / 4.0 * s2
            frames.append(np.full(n, k + 1, dtype=np.int32))
            boxes.append(db.astype(np.float32))
            confs.append(dc.astype(np.float32))
            vxx.append(np.asarray(sx, dtype=np.float32))
            vyy.append(np.asarray(sy, dtype=np.float32))
            if (k + 1) % 100 == 0:
                print("    %s %d/%d  %.0f초" % (seq, k + 1, len(imgs), time.time() - t))
        np.savez_compressed(
            f, frame=np.concatenate(frames), xyxy=np.concatenate(boxes),
            conf=np.concatenate(confs), sxx=np.concatenate(vxx),
            syy=np.concatenate(vyy), n_frames=len(imgs),
            model=MODEL, imgsz=str(IMGSZ), conf_th=CONF, iou_th=IOU_NMS)
        print("  %-18s %d프레임 검출 %d개  %.0f초 -> %s"
              % (seq, len(imgs), sum(len(b) for b in boxes), time.time() - t, f.name))
    print("총 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
