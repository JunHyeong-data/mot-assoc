# -*- coding: utf-8 -*-
"""실험 12 -- **연관 천장.** 이 벤치마크에 애초에 몇 점이 남아 있는가.

사전 선언은 `PREREG.md` (커밋 `c633eee`, 자료보다 먼저).

검출을 고정하고 **1단계 연관만 정답으로** 푼다.

    여지 = 연관 천장 − 기준선 = 어떤 연관 방법이든 딸 수 있는 전부

**방법이 아니라 측정이다.** GT 를 쓰므로 성능 주장이 아니고, 그래서
검정력 문제가 없다 (CLAUDE.md 규칙 6).

사용법:
    python experiments/exp12_ceiling/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from replay import WTracker, Det, load, SEQS, BASE             # noqa: E402
import evaluate as EV                                          # noqa: E402
from tracker.eval.collections.hota import HOTA                 # noqa: E402
from tracker.eval.collections.clear import CLEAR               # noqa: E402

OUT = Path("data/exp12/tracks")
DIAG = {"gid_hit": 0, "gid_tot": 0, "rows": {}}
IOU_MATCH = 0.5
BIG = 1e6
PEDESTRIAN = 1


def gt_by_frame(seq):
    """프레임별 GT (id, xyxy). zero_marked!=0 이고 pedestrian 만."""
    per = {}
    path = EV.GT_ROOT / seq / "gt" / "gt.txt"
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 8:
            continue
        t, i = int(f[0]), int(f[1])
        x, y, w, h = (float(v) for v in f[2:6])
        if int(f[6]) == 0 or int(f[7]) != PEDESTRIAN:
            continue
        per.setdefault(t, []).append((i, [x, y, x + w, y + h]))
    return per


def det_gt_ids(cache, gt):
    """캐시된 검출마다 GT id 를 붙인다 (IoU>=0.5 헝가리안). 없으면 -1."""
    ids = np.full(len(cache["conf"]), -1, dtype=np.int64)
    fr, box = cache["frame"], cache["xyxy"]
    for t, rows in gt.items():
        m = np.nonzero(fr == t)[0]
        if len(m) == 0:
            continue
        g = np.array([r[1] for r in rows], float)
        d = box[m].astype(float)
        x1 = np.maximum(g[:, None, 0], d[None, :, 0])
        y1 = np.maximum(g[:, None, 1], d[None, :, 1])
        x2 = np.minimum(g[:, None, 2], d[None, :, 2])
        y2 = np.minimum(g[:, None, 3], d[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
        iou = inter / np.maximum(ag[:, None] + ad[None, :] - inter, 1e-9)
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= IOU_MATCH:
                ids[m[b]] = rows[a][0]
    return ids


class OracleTracker(WTracker):
    """1단계 비용만 신탁으로 바꾼다. 나머지는 전부 그대로."""

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        for t in tracks:
            t.gt_id = int(results.gid[int(t.idx)])      # 생성 검출의 GT id 를 물려받는다
        return tracks

    def get_dists(self, tracks, detections):
        base = super().get_dists(tracks, detections)
        if base.ndim != 2 or 0 in base.shape:
            return base
        tg = np.array([getattr(t, "gt_id", -1) for t in tracks])[:, None]
        dg = np.array([getattr(d, "gt_id", -1) for d in detections])[None, :]
        same = (tg == dg) & (tg >= 0)
        return np.where(same, 0.0, BIG).astype(np.float32)


class GDet(Det):
    """검출에 GT id 를 실어 나른다."""

    def __init__(self, xyxy, conf, cls, sxx, syy, gid):
        super().__init__(xyxy, conf, cls, sxx, syy)
        self.gid = np.asarray(gid, dtype=np.int64)

    def __getitem__(self, m):
        return GDet(self.xyxy[m], self.conf[m], self.cls[m],
                    self.sxx[m], self.syy[m], self.gid[m])


def replay(oracle, tag):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        c = load(seq, "iou")
        gid = det_gt_ids(c, gt_by_frame(seq)) if oracle else np.full(len(c["conf"]), -1)
        if oracle:
            DIAG["gid_hit"] += int((gid >= 0).sum()); DIAG["gid_tot"] += len(gid)
        cls = OracleTracker if oracle else WTracker
        tr = cls(SimpleNamespace(**BASE), "iou", 1.0, frame_rate=30)
        lines = []
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            det = GDet(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                       c["sxx"][m], c["syy"][m], gid[m])
            for row in tr.update(det):
                x1, y1, x2, y2 = row[:4]
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                             % (f, int(row[4]), x1, y1, x2 - x1, y2 - y1, float(row[5])))
        (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")
        DIAG["rows"].setdefault(seq, {})[tag] = len(lines)


def score(tag):
    keep = EV.TRACKS
    EV.TRACKS = OUT
    try:
        h, c = HOTA(), CLEAR()
        ph, pc = {}, {}
        for seq in SEQS:
            d = EV.build_data(seq, tag)
            if d is None:
                raise SystemExit("트랙 파일 없음: %s / %s" % (tag, seq))
            ph[seq] = h.eval_sequence(d)
            pc[seq] = c.eval_sequence(d)
        ch, cc = h.combine_sequences(ph), c.combine_sequences(pc)
        return dict(
            HOTA=100 * float(np.mean(ch["HOTA"])),
            DetA=100 * float(np.mean(ch["DetA"])),
            AssA=100 * float(np.mean(ch["AssA"])),
            IDSW=float(cc["IDSW"]),
            per={s: 100 * float(np.mean(ph[s]["HOTA"])) for s in ph})
    finally:
        EV.TRACKS = keep


def main():
    print("=" * 92)
    print("실험 12 -- 연관 천장. 이 벤치마크에 애초에 몇 점이 남아 있는가")
    print("=" * 92)
    print("사전 선언 PREREG.md (커밋 c633eee, 자료보다 먼저)")
    print("검출을 고정하고 1단계 연관만 정답으로 푼다. **방법이 아니라 측정이다.**")
    print()

    replay(False, "base")
    replay(True, "oracle")
    b, o = score("base"), score("oracle")

    print("=" * 92)
    print("사전 선언한 관문")
    print("=" * 92)
    ok = True
    g0a = o["HOTA"] >= b["HOTA"]
    ok &= g0a
    print("  [0a] 신탁이 기준선보다 나쁘지 않다  %.3f vs %.3f  %s"
          % (o["HOTA"], b["HOTA"], "OK" if g0a else "** 실패: 배관이 틀렸다 **"))
    g0b = o["AssA"] - b["AssA"] > 0
    ok &= g0b
    print("  [0b] AssA 가 오른다                %+.3f  %s"
          % (o["AssA"] - b["AssA"], "OK" if g0b else "** 실패 **"))
    # [0c] 는 버렸다 -- PREREG-v2.md 참고. "캐시가 같으면 DetA 도 같다" 가 거짓이었다.
    g0c = o["IDSW"] < b["IDSW"]
    ok &= g0c
    print("  [0c'] 신탁의 IDSW 가 더 적다        %.0f vs %.0f  %s"
          % (o["IDSW"], b["IDSW"], "OK" if g0c else "** 실패: 신탁이 신탁이 아니다 **"))
    cov = 100.0 * DIAG["gid_hit"] / max(DIAG["gid_tot"], 1)
    g0d = cov >= 50.0
    ok &= g0d
    print("  [0d]  GT id 가 붙은 검출 비율       %.1f%%  %s"
          % (cov, "OK (>=50)" if g0d else "** 신탁이 약하다 **"))
    if not ok:
        print()
        print("  ** 관문 실패. 판정하지 않는다 **")
        return 1

    print()
    print("=" * 92)
    print("사전 선언한 종말점")
    print("=" * 92)
    print("%-22s %9s %9s %9s %9s" % ("", "HOTA", "DetA", "AssA", "IDSW"))
    print("-" * 92)
    print("%-22s %9.3f %9.3f %9.3f %9.0f" % ("기준선 (맨 IoU)", b["HOTA"], b["DetA"], b["AssA"], b["IDSW"]))
    print("%-22s %9.3f %9.3f %9.3f %9.0f" % ("**연관 천장**", o["HOTA"], o["DetA"], o["AssA"], o["IDSW"]))
    room = o["HOTA"] - b["HOTA"]
    unw = float(np.mean([o["per"][s] - b["per"][s] for s in b["per"]]))
    print()
    print("  [1] **여지 = %+.3f HOTA**   (가중 없는 시퀀스 평균 %+.3f)" % (room, unw))
    print("  [2] AssA %+.3f,  IDSW %+.0f" % (o["AssA"] - b["AssA"], o["IDSW"] - b["IDSW"]))
    print()
    print("  [진단] 출력 행 수 -- [0c] 를 버린 이유. **연관이 출력을 바꾼다**")
    tb = sum(v["base"] for v in DIAG["rows"].values())
    to = sum(v["oracle"] for v in DIAG["rows"].values())
    print("      기준선 %d 행 -> 신탁 %d 행   %+d (%.1f%%)"
          % (tb, to, to - tb, 100.0 * (to - tb) / max(tb, 1)))
    print("      신탁은 GT id 가 다른 검출과의 매칭을 거부한다. 빠진 것 상당수가")
    print("      거짓양성이라 DetA 가 오른다 -- 캐시가 같아도 정상이다")

    print()
    print("  [3] 시퀀스별 여지")
    for s in SEQS:
        if s in b["per"]:
            print("      %-18s %7.2f -> %7.2f   %+7.2f"
                  % (s.replace("-FRCNN", ""), b["per"][s], o["per"][s],
                     o["per"][s] - b["per"][s]))

    print()
    print("  [4] 우리가 시도한 것들과 나란히")
    print("      %-34s %+8.2f" % ("**연관 천장 (딸 수 있는 전부)**", room))
    for name, v in (("칼만 R (exp02)", -0.62), ("게이팅 (exp03)", -4.33),
                    ("거리/와서스타인 (exp05)", -4.98), ("임계값 LOSO (exp06)", -0.21),
                    ("임계값 신탁 상한 (exp06)", 0.892)):
        print("      %-34s %+8.2f" % (name, v))

    print()
    print("=" * 92)
    print("판정 -- **원본 PREREG.md(c633eee) 의 읽는 법 표를 그대로 적용한다**")
    print("=" * 92)
    if room < 3.0:
        print("  여지 %.2f < 3 => **이 벤치마크는 연관 포화다.**" % room)
        print("     우리 음성 결과의 상당 부분이 '통로가 나쁘다' 가 아니라")
        print("     **'딸 게 없었다'** 로 재해석된다. 성능 논문은 여기서 못 쓴다")
    elif room <= 10.0:
        print("  3 <= 여지 %.2f <= 10 => 여지가 있다. **우리 방법이 못 가져온 것이다**" % room)
    else:
        print("  여지 %.2f > 10 => **연관은 여전히 열린 문제다.**" % room)
        print("     우리 실패는 방법 탓이지 벤치마크 탓이 아니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
