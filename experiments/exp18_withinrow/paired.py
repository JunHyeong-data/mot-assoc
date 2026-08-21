# -*- coding: utf-8 -*-
"""실험 18 — **행 안에서 재면 σ 가 정답을 고르는가.**

사전 등록은 `PREREG.md` (자료보다 먼저 커밋, 읽는 법 포함).

원고 5.6 의 AUC 는 **채택된 쌍의 주변부 판별력**인데, 헝가리안이 푸는 것은
**한 행 안에서 후보들의 상대 순서**다. 두 질문이 다르다는 것이 심사의 지적이고,
원고 안에 반례가 있다 -- 판별력이 더 낮은 σ_C 가 같은 경로에서 3.93 HOTA 낫다.

그래서 헝가리안이 실제로 푸는 형태로 다시 잰다:

    각 수정 가능한 오류마다   j  = 실제로 채택된 **오답** 검출
                             j* = 같은 호출에 있었던 **정답** 검출
    -> sigma(j*) < sigma(j) 인가?

정답 후보가 여럿이면 **트랙 예측 박스와 IoU 가 가장 큰 것** 하나를 쓴다
(사전 등록에 자료 보기 전에 박았다).

사용법:
    python experiments/exp18_withinrow/paired.py        # NMS
    python experiments/exp18_withinrow/paired.py -dfl   # DFL
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[0] / "exp12_ceiling"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                 # noqa: E402
from replay import WTracker, load, SEQS, BASE                   # noqa: E402
from run import det_gt_ids, gt_by_frame, GDet                   # noqa: E402
from stage_util import which_stage, stage_thresh                # noqa: E402

SRC = "w_nms" if "-nms" in sys.argv or "-dfl" not in sys.argv else "iou"
SRC_NAME = "NMS 후보 산포" if SRC == "w_nms" else "DFL 분포 분산"

PAIRS = []          # (sig_wrong, sig_right, h_wrong, h_right, seq)
NOPAIR = [0]        # 정답 검출을 다시 못 찾은 경우
ZERO = [0]          # sigma == 0 (퇴화) 가 짝에 낀 경우


def iou_1toN(box, boxes):
    """xyxy 하나 대 여럿."""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


class PairTracker(WTracker):
    """1단계 채택 쌍마다 **오답과 그때 있었던 정답을 짝지어** 기록한다."""

    seq_tag = "?"

    def init_track(self, results, img=None):
        """exp15 와 **같은 방식**으로 STrack 에 gt_id 를 단다."""
        tracks = super().init_track(results, img)
        for t in tracks:
            t.gt_id = int(results.gid[int(t.idx)])
        return tracks

    def get_dists(self, tracks, detections):
        d = super().get_dists(tracks, detections)
        if d.ndim != 2 or 0 in d.shape:
            return d
        if which_stage(tracks) != 1:
            return d
        m, _, _ = matching.linear_assignment(d, stage_thresh(self.args, 1))

        def sig_of(de):
            return float(np.sqrt(max(float(de.det_var[0]) + float(de.det_var[1]), 0.0)))

        for i, j in np.asarray(m).reshape(-1, 2):
            tr, de = tracks[int(i)], detections[int(j)]
            tg, dg = getattr(tr, "gt_id", -1), int(de.gt_id)
            if dg < 0 or tg < 0 or tg == dg:
                continue                            # 미상 / 옳음 은 대상 아님
            # 같은 호출에 정답 검출이 있었는가
            cand = [k for k, x in enumerate(detections) if int(x.gt_id) == tg]
            if not cand:
                continue                            # 틀림_못고침
            if len(cand) == 1:
                kbest = cand[0]
            else:                                   # 사전 등록: 트랙과 IoU 최대
                boxes = np.array([detections[k].xyxy for k in cand], float)
                kbest = cand[int(np.argmax(iou_1toN(np.asarray(tr.xyxy, float),
                                                    boxes)))]
            good = detections[kbest]
            sw, sr = sig_of(de), sig_of(good)
            if sw <= 0 or sr <= 0:
                ZERO[0] += 1
            PAIRS.append((sw, sr,
                          float(de.xyxy[3] - de.xyxy[1]),
                          float(good.xyxy[3] - good.xyxy[1]),
                          self.seq_tag))
        return d


def report(name, wrong, right):
    """짝지은 비교. 동률은 분모에서 뺀다 (부호검정 관례와 같게)."""
    w, r = np.asarray(wrong, float), np.asarray(right, float)
    ties = int((w == r).sum())
    m = w != r
    n = int(m.sum())
    if n == 0:
        print("  %-22s (비교 가능한 짝 없음)" % name)
        return float("nan")
    frac = float((r[m] < w[m]).mean())
    lo = np.log(np.maximum(r[m], 1e-12)) - np.log(np.maximum(w[m], 1e-12))
    se = float(np.sqrt(frac * (1 - frac) / n))
    print("  %-22s **%.4f**   (n=%d, 동률 %d,  95%% CI %.3f~%.3f)"
          % (name, frac, n, ties, frac - 1.96 * se, frac + 1.96 * se))
    print("  %-22s log 비 중앙값 %+.4f" % ("", float(np.median(lo))))
    return frac


def main():
    print("=" * 92)
    print("실험 18 -- 행 안 짝 비교. **헝가리안이 푸는 형태로 다시 잰다**  [%s]"
          % SRC_NAME)
    print("=" * 92)
    print("각 수정 가능한 오류마다 (채택된 오답 j, 있었던 정답 j*) 를 짝짓는다.")
    print("sigma(j*) < sigma(j) 이면 **행 안에서 sigma 가 정답 쪽을 가리킨 것**이다.")
    print()

    for seq in SEQS:
        c = load(seq, SRC)
        if c is None:
            print("캐시 없음: %s" % seq)
            return 1
        gid = det_gt_ids(c, gt_by_frame(seq))
        tr = PairTracker(SimpleNamespace(**BASE), "iou", 1.0, frame_rate=30)
        tr.seq_tag = seq
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            tr.update(GDet(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                           c["sxx"][m], c["syy"][m], gid[m]))

    if not PAIRS:
        print("짝이 하나도 안 만들어졌다. **판정하지 말 것.**")
        return 1

    sw = np.array([p[0] for p in PAIRS])
    sr = np.array([p[1] for p in PAIRS])
    hw = np.array([p[2] for p in PAIRS])
    hr = np.array([p[3] for p in PAIRS])
    sq = np.array([p[4] for p in PAIRS])

    print("=" * 92)
    print("짝 %d 건  (sigma=0 이 낀 짝 %d 건 -- 심사 Q3 이 물은 것)"
          % (len(PAIRS), ZERO[0]))
    print("=" * 92)
    print()
    print("[1] 주 평가지표 -- sigma(정답) < sigma(오답) 인 비율")
    f_sig = report("sigma", sw, sr)
    print()
    print("[2] 기준 비교 -- 박스 크기로 같은 것")
    f_c = report("sigma_C (= h/2)", hw / 2.0, hr / 2.0)

    # **실험 21 을 위해 짝을 남긴다.** 판정 로직은 손대지 않았다 --
    # 군집(시퀀스) 부트스트랩으로 구간을 다시 내려면 시퀀스 라벨이 필요하다.
    out = Path("data/exp21"); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / ("withinrow-%s.npz" % ("nms" if SRC == "w_nms" else "dfl")),
             seq=sq, sig_wrong=sw, sig_right=sr,
             size_wrong=hw / 2.0, size_right=hr / 2.0)
    print("[저장] %s" % (out / ("withinrow-%s.npz"
                               % ("nms" if SRC == "w_nms" else "dfl"))))

    print()
    print("[4] 시퀀스별")
    for s in SEQS:
        m = sq == s
        if m.sum() < 10:
            continue
        a, b = sw[m], sr[m]
        k = a != b
        if k.sum():
            print("  %-18s %.4f  (n=%d)" % (s.replace("-FRCNN", ""),
                                            float((b[k] < a[k]).mean()), int(k.sum())))

    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md)")
    print("=" * 92)
    if f_sig >= 0.60:
        print("  [1] %.4f >= 0.60 => **행 안에서는 sigma 가 정답을 고른다.**" % f_sig)
        print("      5.6 의 AUC 가 잘못된 자였다. **원고의 중심 주장을 고쳐야 한다.**")
    elif f_sig >= 0.55:
        print("  [1] %.4f 는 0.55~0.60 => 약하게 고른다." % f_sig)
        print("      5.6 에 단서를 달고 7.2 의 사전 점검을 행 조건부로 바꾼다.")
    elif f_sig >= 0.45:
        print("  [1] %.4f 는 0.45~0.55 => **행 안에서도 못 고른다.**" % f_sig)
        print("      **5.6 의 결론이 더 단단해진다.** 주변부 AUC 가 자를 잘못 댄 것이")
        print("      아니라 신호 자체에 없는 것이다.")
    else:
        print("  [1] %.4f < 0.45 => **반대로 고른다** (오답 쪽 sigma 가 더 작다)." % f_sig)
        print("      그것도 결과이므로 그대로 적는다.")
    print()
    print("  **7.2 의 사전 점검 문장은 이 결과와 무관하게 고친다** (사전 등록).")
    print("  판별력이 더 낮은 sigma_C 가 같은 경로에서 3.93 HOTA 낫다는 반례만으로")
    print("  '무작위 수준이면 어떤 형태로도 이득이 없다' 는 일반 명제로 못 쓴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
