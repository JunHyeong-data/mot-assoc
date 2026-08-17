# -*- coding: utf-8 -*-
"""실험 5 [4단계] -- HOTA 평가.

UTrack 에 벤더링된 TrackEval 의 **HOTA 지표 클래스만** 쓰고, 데이터 적재는
직접 한다. 이유가 있다:

**exp02 에서 seqmap 헤더 누락으로 첫 시퀀스가 조용히 빠졌다.** 그 적재 경로를
쓰지 않으면 그 부류의 버그가 원천 차단된다. 대신 **MOTChallenge 전처리를
소스 그대로 복제**해야 한다 -- 거기가 재구현이 틀리는 자리다.

`mot_challenge_2d_box.py` 의 `get_preprocessed_seq_data` 를 그대로 옮겼다:

  1. 검출(트래커) 상자를 **모든 클래스의 GT** 와 헝가리안으로 맞춘다 (IoU>=0.5)
  2. **distractor 클래스**(person_on_vehicle 2, static_person 7, distractor 8,
     reflection 12)에 매칭된 트래커 상자를 **제거**한다
  3. GT 는 `zero_marked != 0` 이고 `class == 1`(pedestrian) 인 것만 남긴다
  4. id 를 0..N-1 로 연속 재부여한다

이 넷 중 2번을 빼먹는 재구현이 흔하고, 그러면 HOTA 가 몇 점씩 달라진다.

사용법:
    python experiments/exp05_wasserstein/evaluate.py iou w_dfl w_size
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "external" / "UTrack")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from tracker.eval.collections.hota import HOTA                # noqa: E402

GT_ROOT = Path("data/MOT17_A/ablation")
TRACKS = Path("data/exp05/tracks")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

# mot_challenge_2d_box.py 의 class_name_to_class_id 그대로
DISTRACTORS = [2, 7, 8, 12]      # person_on_vehicle, static_person, distractor, reflection
PEDESTRIAN = 1


def read_mot(path, is_gt):
    """frame -> (ids, boxes xywh, classes, zero_marked/conf)."""
    per = {}
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 7:
            continue
        t = int(f[0]); i = int(f[1])
        box = [float(f[2]), float(f[3]), float(f[4]), float(f[5])]
        if is_gt:
            zm, cls = int(f[6]), int(f[7])
        else:
            zm, cls = 1, PEDESTRIAN          # 트래커 출력은 전부 보행자
        per.setdefault(t, []).append((i, box, cls, zm))
    return per


def ious_xywh(a, b):
    """(n,4),(m,4) xywh -> (n,m). TrackEval 의 _calculate_box_ious 와 같은 규약."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(area > 0, inter / np.maximum(area, 1e-10), 0.0)


def build_data(seq, arm):
    """MOTChallenge 전처리를 그대로 복제해 HOTA 가 먹는 dict 를 만든다."""
    gt = read_mot(GT_ROOT / seq / "gt" / "gt.txt", True)
    tr_path = TRACKS / arm / ("%s.txt" % seq)
    if not tr_path.exists():
        return None
    tr = read_mot(tr_path, False)
    n_t = max(max(gt) if gt else 0, max(tr) if tr else 0)

    keys = ["gt_ids", "tracker_ids", "similarity_scores"]
    data = {k: [None] * n_t for k in keys}
    ug, ut, n_gd, n_td = [], [], 0, 0

    for idx in range(n_t):
        t = idx + 1
        g = gt.get(t, []); k = tr.get(t, [])
        g_ids = np.array([x[0] for x in g], int)
        g_box = np.array([x[1] for x in g], float).reshape(-1, 4)
        g_cls = np.array([x[2] for x in g], int)
        g_zm = np.array([x[3] for x in g], int)
        t_ids = np.array([x[0] for x in k], int)
        t_box = np.array([x[1] for x in k], float).reshape(-1, 4)

        sim = ious_xywh(g_box, t_box)

        # [2] distractor 에 매칭된 트래커 상자 제거
        to_remove = np.array([], int)
        if len(g_ids) and len(t_ids):
            ms = sim.copy()
            ms[ms < 0.5 - np.finfo("float").eps] = 0
            r, c = linear_sum_assignment(-ms)
            ok = ms[r, c] > 0 + np.finfo("float").eps
            r, c = r[ok], c[ok]
            to_remove = c[np.isin(g_cls[r], DISTRACTORS)]

        t_ids = np.delete(t_ids, to_remove, axis=0)
        sim = np.delete(sim, to_remove, axis=1)

        # [3] GT 는 zero_marked != 0 이고 pedestrian 인 것만
        keep = (g_zm != 0) & (g_cls == PEDESTRIAN)
        g_ids = g_ids[keep]
        sim = sim[keep]

        data["gt_ids"][idx] = g_ids
        data["tracker_ids"][idx] = t_ids
        data["similarity_scores"][idx] = sim
        ug += list(np.unique(g_ids)); ut += list(np.unique(t_ids))
        n_gd += len(g_ids); n_td += len(t_ids)

    # [4] id 연속 재부여
    for key, uniq in (("gt_ids", ug), ("tracker_ids", ut)):
        if not uniq:
            continue
        u = np.unique(uniq)
        m = np.nan * np.ones(int(np.max(u)) + 1)
        m[u] = np.arange(len(u))
        for i in range(n_t):
            if len(data[key][i]):
                data[key][i] = m[data[key][i]].astype(int)

    data["num_tracker_dets"] = n_td
    data["num_gt_dets"] = n_gd
    data["num_tracker_ids"] = len(np.unique(ut)) if ut else 0
    data["num_gt_ids"] = len(np.unique(ug)) if ug else 0
    data["num_timesteps"] = n_t
    data["seq"] = seq
    return data


def main():
    arms = sys.argv[1:] or ["iou", "w_dfl", "w_size"]
    metric = HOTA()
    print("=" * 84)
    print("실험 5 [4단계] HOTA -- UTrack 벤더링 TrackEval 의 지표 클래스, 적재는 직접")
    print("=" * 84)
    print("MOTChallenge 전처리 복제: distractor 매칭 제거 + zero_marked/pedestrian 필터")
    print()
    hdr = "%-16s" + "%9s" * 4
    print(hdr % ("갈래", "HOTA", "DetA", "AssA", "LocA"))
    print("-" * 84)

    results = {}
    for arm in arms:
        per_seq = {}
        for seq in SEQS:
            d = build_data(seq, arm)
            if d is None:
                continue
            per_seq[seq] = metric.eval_sequence(d)
        if not per_seq:
            print("%-16s (트랙 파일 없음)" % arm)
            continue
        comb = metric.combine_sequences(per_seq)
        results[arm] = (comb, per_seq)
        print(hdr % (arm, "%.3f" % (100 * np.mean(comb["HOTA"])),
                     "%.3f" % (100 * np.mean(comb["DetA"])),
                     "%.3f" % (100 * np.mean(comb["AssA"])),
                     "%.3f" % (100 * np.mean(comb["LocA"]))))

    if "w_dfl" in results and "w_size" in results:
        a = 100 * np.mean(results["w_dfl"][0]["HOTA"])
        b = 100 * np.mean(results["w_size"][0]["HOTA"])
        print()
        print("=" * 84)
        print("사전 선언한 판정")
        print("=" * 84)
        print("  [1] 주 종말점  W-DFL − W-size = %+.3f HOTA" % (a - b))
        if "iou" in results:
            c = 100 * np.mean(results["iou"][0]["HOTA"])
            print("  [2] 통로 효과  W-DFL − A(IoU)  = %+.3f HOTA" % (a - c))
        print("  판정폭 0.3 (exp03 과 같은 자). |차이| < 0.3 이면 '차이 없음'")
        v = "DFL 소스가 낫다" if a - b > 0.3 else (
            "크기 소스가 낫다" if a - b < -0.3 else "**차이 없음**")
        print("  => %s" % v)
        print()
        print("  시퀀스별 HOTA:")
        for seq in SEQS:
            if seq in results["w_dfl"][1] and seq in results["w_size"][1]:
                x = 100 * np.mean(results["w_dfl"][1][seq]["HOTA"])
                y = 100 * np.mean(results["w_size"][1][seq]["HOTA"])
                print("    %-18s W-DFL %6.2f   W-size %6.2f   차이 %+.2f"
                      % (seq, x, y, x - y))


if __name__ == "__main__":
    main()
