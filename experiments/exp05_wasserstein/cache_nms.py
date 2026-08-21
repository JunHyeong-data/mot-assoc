# -*- coding: utf-8 -*-
"""**NMS 후보 산포**를 exp05 캐시와 같은 검출 위에서 뽑는다 -> `{seq}-nms.npz`.

## 왜 필요한가

`replay.py` 의 `w_nms` 조건과 실험 15 가 이 파일을 읽는데 **아무도 만든 적이 없다.**
그래서 실험 15 가 **DFL 소스만** 재고 "NMS 는 미측정" 을 한계로 적었다.

exp01 은 **NMS 가 위치오차를 더 잘 짚는다**고 했다 (편상관 +0.322 대 DFL +0.151).
**순위를 더 잘 매기는 쪽을 안 재고 "σ 는 연관에 정보가 없다" 고 쓰면 약하다.**

## 무엇을 하는가

`cache_detections.py` 와 **완전히 같은 검출기·설정**으로 다시 돌리되, DFL 대신
**NMS 후보 군집의 중심 공분산**을 뽑는다. 로직은 `exp01/run_sequence.py` 의
검증된 것을 그대로 옮긴다.

## 이 코드가 조용히 틀리는 두 자리 -- 철회 8·9번이 여기서 났다

1. **NMS 가 입력을 in-place 로 바꾼다** (`xywh -> xyxy`). 호출 **뒤에** 읽으면
   이미 xyxy 인데 `xywh2xyxy` 를 또 걸어 박스가 3배로 부푼다.
   **-> 호출 전에 복사한다.**
2. **좌표계가 둘이다.** raw prediction 은 letterbox, `r.boxes.xyxy` 는 원본이다.
   ultralytics 는 NMS 가 **돌아온 뒤** `scale_boxes` 를 건다. 그래서 이 시점에
   배율을 재면 **언제나 정확히 1.0** 이 나온다 (오류도 경고도 없다).
   **-> 배율은 주 루프에서 `r.boxes` 와 대조해 잰다** (DFL 경로와 같은 방식).

## 정렬 검산 -- 가정하지 않는다

`replay.py` 는 이 파일의 `sxx/syy` 를 **base npz 의 검출 순서에 그대로 붙인다.**
그러므로 두 실행의 검출이 **같은 순서로 같은 개수** 나와야 한다.
`frame` 과 `xyxy` 를 같이 저장하고 **base 와 대조해서 확인한다.**
안 맞으면 **저장하지 않는다.**

사용법:
    python experiments/exp05_wasserstein/cache_nms.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ultralytics import YOLO                                # noqa: E402
from ultralytics.utils import nms as ulnms                  # noqa: E402

ROOT = Path("data/MOT17_A/ablation")
OUT = Path("data/exp05")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

# **cache_detections.py 와 같은 기본값을 쓴다.** 다르면 검출이 안 맞는다.
MODEL = os.environ.get("EXP05_MODEL", "data/weights/ablation_17_best.pt")
CONF = float(os.environ.get("EXP05_CONF", 0.01))
IOU_NMS = float(os.environ.get("EXP05_IOU", 0.70))
_isz = os.environ.get("EXP05_IMGSZ", "800,1440")
IMGSZ = [int(x) for x in _isz.split(",")] if "," in _isz else int(_isz)

MIN_CAND = 3          # exp01 과 같은 값. 표본 산포를 믿을 최소 후보 수

CAP = {}
_orig_nms = ulnms.non_max_suppression


def xywh2xyxy(b):
    o = b.clone()
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def iou_mat(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m). exp01 과 같은 함수."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def patched_nms(prediction, *a, **kw):
    # **[함정 1] 호출 전에 복사한다.** 아래 원본 호출이 prediction 을 in-place 로
    # xywh -> xyxy 로 바꾼다. 뒤에 읽으면 이미 바뀐 것에 또 변환을 건다.
    _p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    _snap = (_p[0] if _p.ndim == 3 else _p).detach().clone()

    res = _orig_nms(prediction, *a, **{**kw, "return_idxs": True})
    out, keepi = res
    try:
        x = _snap.T                                   # (N, 4+nc) letterbox xywh
        boxes = xywh2xyxy(x[:, :4]).cpu().numpy()
        person = x[:, 4].cpu().numpy()                # class 0 = person
        pool = person >= CONF                         # NMS 가 실제로 본 후보 풀
        pool_box = boxes[pool]
        idx = keepi[0].long().cpu().numpy()
        kept = boxes[idx]

        sxx = np.zeros(len(kept)); syy = np.zeros(len(kept))
        ncand = np.zeros(len(kept), dtype=np.int32)
        if len(kept) and len(pool_box):
            M = iou_mat(kept, pool_box)
            for i in range(len(kept)):
                c = pool_box[M[i] >= IOU_NMS]         # 이 박스가 억제한 후보들
                ncand[i] = len(c)
                if len(c) >= MIN_CAND:
                    cx = (c[:, 0] + c[:, 2]) / 2
                    cy = (c[:, 1] + c[:, 3]) / 2
                    C2 = np.cov(np.vstack([cx, cy]))  # **letterbox px^2 그대로**
                    sxx[i] = float(C2[0, 0]); syy[i] = float(C2[1, 1])
        # 배율 환산용으로 letterbox 높이도 남긴다 ([함정 2])
        CAP["sxx"], CAP["syy"] = sxx, syy
        CAP["ncand"] = ncand
        CAP["hl"] = (kept[:, 3] - kept[:, 1]).astype(float)
    except Exception as e:                            # 조용히 넘기지 않는다
        CAP["error"] = repr(e)
    return res if kw.get("return_idxs", False) else out


ulnms.non_max_suppression = patched_nms
for mod in ("ultralytics.models.yolo.detect.predict", "ultralytics.engine.results"):
    m = sys.modules.get(mod)
    if m is not None and hasattr(m, "non_max_suppression"):
        m.non_max_suppression = patched_nms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("NMS 후보 산포 캐싱 -- exp05 검출과 같은 설정")
    print("=" * 78)
    print("모델 %s  imgsz %s  conf %.2f  iou %.2f" % (MODEL, IMGSZ, CONF, IOU_NMS))
    print("**cache_detections.py 와 같은 값이어야 검출이 맞는다.**")
    print()
    model = YOLO(MODEL)
    t0 = time.time()
    for seq in SEQS:
        base_f = OUT / ("%s.npz" % seq)
        if not base_f.exists():
            print("  %s: base npz 가 없다. 건너뛴다" % seq)
            continue
        base = np.load(base_f)
        imgs = sorted((ROOT / seq / "img1").glob("*.jpg"))
        frames, boxes, vxx, vyy, ncs = [], [], [], [], []
        t = time.time()
        for k, ip in enumerate(imgs):
            CAP.clear()
            r = model.predict(source=str(ip), conf=CONF, iou=IOU_NMS, classes=[0],
                              imgsz=IMGSZ, verbose=False)[0]
            db = r.boxes.xyxy.cpu().numpy()
            n = len(db)
            sx = sy = np.zeros(n); nc = np.zeros(n, dtype=np.int32)
            if n and "sxx" in CAP and len(CAP["sxx"]) == n:
                # [함정 2] 배율은 여기서 잰다 -- letterbox 높이 대 원본 높이
                hl, ho = CAP["hl"], db[:, 3] - db[:, 1]
                ok = hl > 1e-6
                g = float(np.median(ho[ok] / hl[ok])) if ok.any() else 1.0
                sx = CAP["sxx"] * (g ** 2)            # px^2 -> 원본 px^2
                sy = CAP["syy"] * (g ** 2)
                nc = CAP["ncand"]
            frames.append(np.full(n, k + 1, dtype=np.int32))
            boxes.append(db.astype(np.float32))
            vxx.append(np.asarray(sx, dtype=np.float32))
            vyy.append(np.asarray(sy, dtype=np.float32))
            ncs.append(nc)
            if (k + 1) % 100 == 0:
                print("    %s %d/%d  %.0f초" % (seq, k + 1, len(imgs), time.time() - t))

        fr = np.concatenate(frames); bx = np.concatenate(boxes)
        # ---- 정렬 검산. 안 맞으면 저장하지 않는다 ----
        okf = fr.shape == base["frame"].shape and np.array_equal(fr, base["frame"])
        okb = bx.shape == base["xyxy"].shape and np.allclose(bx, base["xyxy"], atol=1e-3)
        if not (okf and okb):
            print("  ** %s: base npz 와 검출이 안 맞는다 (frame %s, box %s). 저장 안 함 **"
                  % (seq, okf, okb))
            print("     n = %d vs %d" % (len(fr), len(base["frame"])))
            continue
        f = OUT / ("%s-nms.npz" % seq)
        np.savez_compressed(
            f, frame=fr, xyxy=bx, sxx=np.concatenate(vxx), syy=np.concatenate(vyy),
            ncand=np.concatenate(ncs), n_frames=len(imgs),
            model=MODEL, imgsz=str(IMGSZ), conf_th=CONF, iou_th=IOU_NMS,
            min_cand=MIN_CAND)
        z = np.concatenate(vxx)
        print("  %-18s %d프레임 검출 %d개  분산0 %.1f%%  %.0f초 -> %s"
              % (seq, len(imgs), len(fr), 100.0 * np.mean(z == 0),
                 time.time() - t, f.name))
    print("총 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
