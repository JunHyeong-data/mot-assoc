# -*- coding: utf-8 -*-
"""실험 6 [탐색] -- 최적 매칭 임계값을 **무엇이 예측하는가**.

`headroom.py` 가 보인 것:
  [A] 전역 최적(0.85)은 기본값(0.80)보다 +0.037 -- **여유 없음**
  [B] 시퀀스별 최적은 0.70~0.95 로 갈리고, 신탁 이득이 **평균 +1.07 HOTA**

즉 이득은 전부 "장면마다 다르다" 에 들어 있다. 그러면 질문은 하나다:
**장면의 무엇을 보면 최적 임계값을 알 수 있는가.**

## 왜 이게 우리 연구와 맞물리나

임계값은 **장면당 스칼라 하나**다. 절대 눈금이 필요 없고 **단조 관계만** 있으면 된다.
그게 우리가 실험 1 에서 확정한 sigma 의 성질이다 -- **신호 O(편상관 0.32,
검출기 무관), 눈금 X(chi^2 80~562배)**. 지금까지 우리는 눈금이 필요한 자리
(거리, 칼만 R, 상자 확장)에 눈금이 틀린 양을 넣어 여덟 번 졌다.

## 이 스크립트가 답하지 않는 것 -- 미리 못박는다

**n=7 이다.** 상관계수가 얼마가 나오든 그건 가설이지 결과가 아니다.
게다가 최적 임계값을 **같은 자료에서 argmax 로** 뽑았으므로 이중으로 낙관적이다.
이 스크립트의 유일한 용도는 **"어느 예측변수를 KITTI 에서 시험할지 고르는 것"** 이다.

**여기서 나온 최고 상관을 그대로 논문에 쓰지 않는다.** 그건 7점 과적합이다.

사용법:
    python experiments/exp06_levers/predictors.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from replay import load, SEQS                               # noqa: E402

# headroom.py 가 낸 시퀀스별 최적 임계값 (재현: python experiments/exp06_levers/headroom.py)
#
# **정정 (LOSO 사전 선언대로).** headroom.py 의 그리드는 상단이 0.95 였는데
# MOT17-10 과 -13 의 최적이 거기 붙어 있었다. 0.98 을 넣으니 **MOT17-13 이
# 0.95 -> 0.98 로 옮겨가고 이득이 1.73 -> 4.66 으로 커졌다** (여전히 그리드 끝이다).
# 나머지 여섯은 그대로다. 손으로 옮겨 적는 대신 `grid.py` 가 표를 JSON 으로 남긴다:
#   data/exp06/grid.json  <- loso.py 는 이쪽을 읽는다
BEST = {"MOT17-02-FRCNN": 0.85, "MOT17-04-FRCNN": 0.90, "MOT17-05-FRCNN": 0.75,
        "MOT17-09-FRCNN": 0.75, "MOT17-10-FRCNN": 0.95, "MOT17-11-FRCNN": 0.70,
        "MOT17-13-FRCNN": 0.98}
GAIN = {"MOT17-02-FRCNN": 0.46, "MOT17-04-FRCNN": 0.35, "MOT17-05-FRCNN": 0.55,
        "MOT17-09-FRCNN": 0.20, "MOT17-10-FRCNN": 0.65, "MOT17-11-FRCNN": 3.57,
        "MOT17-13-FRCNN": 4.66}


def scene_stats(seq):
    """장면 고유 관측량. **GT 를 안 쓴다** -- 실전에서 쓸 수 있어야 하므로."""
    c = load(seq, "iou")
    if c is None:
        return None
    fr, box, conf = c["frame"], c["xyxy"], c["conf"]
    sxx, syy = c["sxx"], c["syy"]
    nf = c["n_frames"]

    hi = conf >= 0.25                       # track_high_thresh. 1단계에 들어가는 것
    w = np.maximum(box[:, 2] - box[:, 0], 1e-6)
    h = np.maximum(box[:, 3] - box[:, 1], 1e-6)
    area = w * h
    sig = np.sqrt(np.maximum(sxx + syy, 0.0))          # 중심 위치 표준편차 (px)

    # 프레임당 고신뢰 검출 수
    cnt = np.bincount(fr[hi], minlength=nf + 2)[1:nf + 1]

    # 카메라 운동 대리: 프레임 간 **검출 중심 중앙값**의 이동
    cx = (box[:, 0] + box[:, 2]) / 2
    cy = (box[:, 1] + box[:, 3]) / 2
    med = []
    for f in range(1, nf + 1):
        m = (fr == f) & hi
        med.append((np.median(cx[m]), np.median(cy[m])) if m.any() else (np.nan, np.nan))
    med = np.asarray(med, float)
    d = np.linalg.norm(np.diff(med, axis=0), axis=1)
    ego = float(np.nanmedian(d)) if len(d) else np.nan

    # 겹침 정도: 프레임당 고신뢰 상자들의 평균 최대 IoU (밀집·가림 대리)
    ov = []
    for f in range(1, nf + 1, 5):                       # 5프레임마다 -- 충분하다
        m = (fr == f) & hi
        b = box[m]
        if len(b) < 2:
            continue
        x1 = np.maximum(b[:, None, 0], b[None, :, 0])
        y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2])
        y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(a[:, None] + a[None, :] - inter, 1e-9)
        np.fill_diagonal(iou, 0.0)
        ov.append(float(iou.max(1).mean()))

    return {
        "밀도 (프레임당 고신뢰)": float(cnt.mean()),
        "겹침 (평균 최대 IoU)": float(np.mean(ov)) if ov else np.nan,
        "자차운동 (중앙 중심 이동 px)": ego,
        "상자 높이 중앙값": float(np.median(h[hi])),
        "높이 산포 CV": float(np.std(h[hi]) / max(np.mean(h[hi]), 1e-9)),
        "sigma 중앙값 (px)": float(np.median(sig[hi])),
        "**sigma/sqrt(area) 중앙값**": float(np.median(sig[hi] / np.sqrt(area[hi]))),
        "sigma 산포 CV": float(np.std(sig[hi]) / max(np.mean(sig[hi]), 1e-9)),
        "점수 중앙값": float(np.median(conf[hi])),
    }


def main():
    print("=" * 92)
    print("실험 6 [탐색] -- 최적 매칭 임계값의 예측변수. **n=7. 가설 고르기용이다**")
    print("=" * 92)
    print("최적 임계값을 같은 자료에서 argmax 로 뽑았다 -> 이중으로 낙관적이다.")
    print("여기 최고 상관을 논문에 쓰지 않는다. KITTI 에서 시험할 후보를 고를 뿐이다.")
    print()

    rows, keys = {}, None
    for s in SEQS:
        st = scene_stats(s)
        if st is None:
            continue
        rows[s] = st
        keys = list(st)

    print("%-30s" % "관측량" + "".join(
        "%9s" % s.replace("-FRCNN", "").replace("MOT17-", "") for s in rows))
    print("-" * 92)
    for k in keys:
        print("%-30s" % k + "".join("%9.3f" % rows[s][k] for s in rows))
    print("%-30s" % "최적 임계값" + "".join("%9.2f" % BEST[s] for s in rows))
    print("%-30s" % "신탁 이득 (HOTA)" + "".join("%9.2f" % GAIN[s] for s in rows))

    print()
    print("=" * 92)
    print("스피어만 상관 (n=%d). |rho| 가 커도 n=7 이면 p 는 크다" % len(rows))
    print("=" * 92)
    y = np.array([BEST[s] for s in rows])
    g = np.array([GAIN[s] for s in rows])
    out = []
    for k in keys:
        x = np.array([rows[s][k] for s in rows])
        if np.all(np.isnan(x)):
            continue
        r1, p1 = spearmanr(x, y)
        r2, _ = spearmanr(x, g)
        out.append((abs(r1), k, r1, p1, r2))
    out.sort(reverse=True)
    print("%-30s%10s%10s%14s" % ("관측량", "rho(최적)", "p", "rho(이득)"))
    print("-" * 92)
    for _, k, r1, p1, r2 in out:
        print("%-30s%10.3f%10.3f%14.3f" % (k, r1, p1, r2))

    print()
    print("읽는 법: rho(최적) 이 최적 임계값과의 단조 관계, rho(이득) 은")
    print("'그 장면에서 적응이 얼마나 남는가' 와의 관계다. **둘 다 큰 것이 좋은 후보다.**")


if __name__ == "__main__":
    main()
