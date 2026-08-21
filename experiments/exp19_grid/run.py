# -*- coding: utf-8 -*-
"""실험 19 — **2×2 격자.** 경로가 나쁜가, 소스가 나쁜가.

사전 등록은 `PREREG.md` (자료보다 먼저 커밋, 읽는 법 포함).

원고 한계 (1) 이 스스로 지목한 후속 실험이다. 네 경로가 서로 다른 추정 방식을
써서 **"경로가 나쁘다" 와 "그 경로에 쓴 소스가 나쁘다" 가 분리되지 않는다.**

경로 둘 × 추정 방식 셋을 **한 재생 틀 안에서** 돌린다. 검출·트래커·평가가 전부
같으므로 기존 exp03/exp05 값과 이어 붙이지 않고 **이 실험 안에서만** 비교한다.

    거리 함수   2-Wasserstein.  C 를 채택률 일치로 이분 탐색
    게이팅      박스 확장.      ALPHA 를 총 확장 면적 일치로 맞춤

사용법:
    python experiments/exp19_grid/run.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
sys.path.insert(0, str(HERE.parents[1] / "external" / "UTrack"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics.trackers.utils import matching                 # noqa: E402
import replay as RP                                             # noqa: E402
from replay import WTracker, Det, SEQS, BASE                    # noqa: E402
from wcost import w2_matrix_norm, size_var, match_scale, nwd_cost   # noqa: E402

OUT = Path("data/exp19/tracks")
CACHE = Path("data/exp05")

# (이름, 소스). 소스는 캐시에서 어떤 sigma 를 읽을지 정한다.
SOURCES = [("nms", "NMS 후보 분산"), ("dfl", "DFL 분포 분산"), ("size", "박스 크기")]
CHANNELS = [("dist", "거리 함수"), ("gate", "게이팅")]


def load(seq, source):
    """**세 소스가 같은 검출을 쓴다.** sigma 만 갈아 끼운다."""
    f0 = CACHE / ("%s.npz" % seq)
    if not f0.exists():
        return None
    d = np.load(f0)
    sxx, syy = d["sxx"], d["syy"]                 # dfl (본 캐시)
    if source == "nms":
        f = CACHE / ("%s-nms.npz" % seq)
        if not f.exists():
            return None
        n = np.load(f)
        # **행 정렬 사전 점검** -- 어긋나면 sigma 가 엉뚱한 검출에 붙는다
        if len(n["frame"]) != len(d["frame"]) or not np.array_equal(
                n["frame"], d["frame"]):
            raise RuntimeError("NMS 곁파일이 본 캐시와 행이 다르다: %s" % seq)
        sxx, syy = n["sxx"], n["syy"]
    return dict(frame=d["frame"], xyxy=d["xyxy"], conf=d["conf"],
                sxx=sxx, syy=syy, n_frames=int(d["n_frames"]))


class GridTracker(WTracker):
    """1단계 비용만 갈아끼운다. 경로와 소스를 함께 받는다."""

    def __init__(self, args, channel, source, scale, frame_rate=30):
        super().__init__(args, "iou", 1.0, frame_rate=frame_rate)
        self.channel = channel
        self.source = source
        self.scale = scale
        self.pad_area = []          # 게이팅 확장 면적 (검출+트랙)
        self.pad_lin = []           # (선형 확장량 합, 개수) -- 양쪽

    def _dvar(self, detections, d_box):
        if self.source == "size":
            return size_var(d_box)
        return np.stack([[d.det_var[0], d.det_var[1]] for d in detections])

    def get_dists(self, tracks, detections):
        if not len(tracks) or not len(detections):
            return matching.iou_distance(tracks, detections)

        t_box = np.asarray([t.xyxy for t in tracks], float).reshape(-1, 4)
        d_box = np.asarray([d.xyxy for d in detections], float).reshape(-1, 4)
        t_var = self._track_var(tracks)
        d_var = self._dvar(detections, d_box)

        if self.channel == "dist":
            d_var = match_scale(d_var, float(np.mean(t_var.sum(-1))))
            w2 = w2_matrix_norm(t_box, d_box, t_var, d_var)
            dists = nwd_cost(w2, self.scale)
        else:                                   # gate -- 박스를 키우고 IoU
            # **확장은 양쪽에 준다 (exp03 의 APPLY=both).** `box_relax.py:60` 이
            # 경고를 적어 뒀는데 첫 판이 검출만 키웠다 -- 검출만 키우면 트랙
            # 박스가 커진 검출 안으로 들어가 교집합은 그대로인데 합집합만 커져
            # **이미 잘 맞는 쌍의 IoU 가 떨어진다.** 문을 여는 게 아니라 닫는다.
            #
            # **트랙 쪽 sigma 는 검출 쪽과 *같은 규칙* 을 트랙에 적용해 얻는다**
            # (exp03 `box_relax.py:349` 의 `_var_of(tracks)` 와 같은 발상):
            #   크기 소스     -> size_var(t_box)   트랙 박스의 w,h. 트랙마다
            #   검출기 sigma  -> t.det_var         트랙을 만든 검출에서 물려받음
            # **프레임 평균을 쓰면 안 된다** -- 크기 신호를 정의하는 박스별
            # 정보를 트랙 쪽에서 지우는 셈이고, 크기 조건은 대조군의 앵커다.
            t_var_g = (size_var(t_box) if self.source == "size"
                       else np.asarray([getattr(t, "det_var", np.zeros(2))
                                        for t in tracks], float).reshape(-1, 2))
            sx = np.sqrt(np.maximum(d_var[:, 0], 0.0)) * self.scale
            sy = np.sqrt(np.maximum(d_var[:, 1], 0.0)) * self.scale
            tx = np.sqrt(np.maximum(t_var_g[:, 0], 0.0)) * self.scale
            ty = np.sqrt(np.maximum(t_var_g[:, 1], 0.0)) * self.scale

            # **확장량은 양쪽을 다 센다.** `box_relax.py:124` 가 2026-08-17 에
            # 열어 둔 항목이다 -- "검출 쪽 표본으로만 상수를 푸는데 both 라
            # pad 는 트랙에도 붙는다. 그러면 조건들이 확장량 일치가 아니게 된다."
            w = np.maximum(d_box[:, 2] - d_box[:, 0], 1e-6)
            h = np.maximum(d_box[:, 3] - d_box[:, 1], 1e-6)
            tw = np.maximum(t_box[:, 2] - t_box[:, 0], 1e-6)
            th = np.maximum(t_box[:, 3] - t_box[:, 1], 1e-6)
            self.pad_area.append(
                float(np.sum((w + 2 * sx) * (h + 2 * sy) - w * h))
                + float(np.sum((tw + 2 * tx) * (th + 2 * ty) - tw * th)))
            self.pad_lin.append((float(np.sum(sx) + np.sum(sy) + np.sum(tx)
                                       + np.sum(ty)),
                                 float(sx.size + sy.size + tx.size + ty.size)))

            big = d_box.copy()
            big[:, 0] -= sx; big[:, 2] += sx
            big[:, 1] -= sy; big[:, 3] += sy
            tb = t_box.copy()
            tb[:, 0] -= tx; tb[:, 2] += tx
            tb[:, 1] -= ty; tb[:, 3] += ty
            dists = _iou_dist(tb, big)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists


def _iou_dist(a, b):
    """1 - IoU. matching.iou_distance 와 같은 규약."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (1.0 - inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)
            ).astype(np.float32)


def run(channel, source, scale, tag):
    """전 시퀀스를 재생하고 MOT 형식으로 쓴다. 확장 면적 합도 돌려준다."""
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    area = 0.0
    for seq in SEQS:
        c = load(seq, source)
        if c is None:
            return None
        tr = GridTracker(SimpleNamespace(**BASE), channel, source, scale)
        lines = []
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            det = Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                      c["sxx"][m], c["syy"][m])
            for row in tr.update(det):
                x1, y1, x2, y2 = row[:4]
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                             % (f, int(row[4]), x1, y1, x2 - x1, y2 - y1,
                                float(row[5])))
        (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")
        area += float(np.sum(tr.pad_area))
    return area


def accept_rate(channel, source, scale):
    """1단계 채택률 = 매칭 수 / sum(min(M,N)). exp05 accept_rate.py 와 같은 정의."""
    num = den = 0
    for seq in SEQS:
        c = load(seq, source)
        tr = GridTracker(SimpleNamespace(**BASE), channel, source, scale)
        orig = tr.get_dists

        def counted(tracks, dets, _o=orig):
            d = _o(tracks, dets)
            nonlocal num, den
            if d.ndim == 2 and 0 not in d.shape and _stage1(tracks):
                m, _, _ = matching.linear_assignment(d, float(BASE["match_thresh"]))
                num += len(np.asarray(m).reshape(-1, 2))
                den += min(len(tracks), len(dets))
            return d

        tr.get_dists = counted
        for f in range(1, c["n_frames"] + 1):
            m = c["frame"] == f
            tr.update(Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                          c["sxx"][m], c["syy"][m]))
    return num / max(den, 1)


def _stage1(tracks):
    a = [bool(getattr(t, "is_activated", False)) for t in tracks]
    return bool(a) and all(a)


def solve_scale(channel, source, target, hi, lo=None, iters=11):
    """이분 탐색. 거리는 채택률을, 게이팅은 총 확장 면적을 target 에 맞춘다."""
    lo = hi * 1e-3 if lo is None else lo
    best = (None, None)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if channel == "dist":
            v = accept_rate(channel, source, mid)
        else:
            v = run(channel, source, mid, "_probe")
        if best[0] is None or abs(v - target) < abs(best[1] - target):
            best = (mid, v)
        if v > target:
            hi = mid
        else:
            lo = mid
    return best


def evaluate(tags):
    """exp05 의 적재·전처리를 그대로 쓰되 트랙 뿌리만 갈아 끼운다."""
    sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
    import evaluate as EV
    from tracker.eval.collections.hota import HOTA
    old = EV.TRACKS
    EV.TRACKS = OUT
    try:
        res = {}
        for tag in tags:
            metric = HOTA()
            per = {}
            for seq in EV.SEQS:
                d = EV.build_data(seq, tag)
                if d is None:
                    per = None
                    break
                per[seq] = metric.eval_sequence(d)
            if per is None:
                res[tag] = None
                continue
            comb = metric.combine_sequences(per)
            res[tag] = (100.0 * float(np.mean(comb["HOTA"])),
                        {s: 100.0 * float(np.mean(per[s]["HOTA"])) for s in per})
        return res
    finally:
        EV.TRACKS = old


def main():
    print("=" * 92)
    print("실험 19 -- 2x2 격자. **경로가 나쁜가, 소스가 나쁜가**")
    print("=" * 92)
    print("경로 둘 x 소스 셋을 **한 재생 틀 안에서** 돌린다.")
    print("기존 exp03/exp05 값과 이어 붙이지 않는다 -- 파이프라인이 다르다.")
    print()

    # ---- 사전 점검 [0a]: 기준선 재현 ----
    run("gate", "dfl", 0.0, "base")          # 확장량 0 => 순수 IoU
    ev = evaluate(["base"])
    h0 = ev["base"][0]
    print("[사전 점검 0a] 기준선 재현  HOTA %.3f  (기록 61.002)  %s"
          % (h0, "OK" if abs(h0 - 61.002) < 0.01 else "!! 어긋남"))
    if abs(h0 - 61.002) >= 0.01:
        print("  **멈춘다.** 확장량 0 인 게이팅이 IoU 와 같아야 한다.")
        return 1

    base_rate = accept_rate("dist", "dfl", 1e-9) if False else None
    # 기준선 채택률은 IoU 로 잰다
    print("[사전 점검 0b] 기준선 채택률 측정 중 ...")
    br = accept_rate("gate", "dfl", 0.0)
    print("               채택률 %.4f" % br)

    # ---- 게이팅 목표 면적: 박스 크기 소스 ALPHA=0.5 를 기준으로 삼는다 ----
    print()
    print("스케일 맞춤 (이분 탐색)")
    print("-" * 92)
    target_area = run("gate", "size", 0.5, "_probe")
    print("  게이팅 목표 확장 면적 = %.4g  (박스 크기 소스, ALPHA=0.5)" % target_area)

    scales = {}
    for src, nm in SOURCES:
        if src == "size":
            scales[("gate", src)] = 0.5
        else:
            # **상한 4.0 은 부족했다** -- 검출기 sigma 가 박스 크기의 수십분의
            # 일이라 이분 탐색이 경계에서 포화했다 (NMS 가 목표의 3.0%).
            # 규칙 2 로 그 판을 버리고 상한을 크게 잡는다.
            s, v = solve_scale("gate", src, target_area, hi=400.0, iters=16)
            scales[("gate", src)] = s
            print("  게이팅 %-6s ALPHA %.4g  -> 면적 %.4g  (목표의 %.1f%%)"
                  % (src, s, v, 100.0 * v / target_area))
            # **사전 점검**: 통제가 실제로 걸렸는가. 안 걸리면 판정하지 않는다.
            if abs(v - target_area) / target_area > 0.05:
                print("  !! 확장 면적이 목표에서 5%% 넘게 벗어났다. **통제 실패.**")
                print("     CLAUDE.md 규칙 2 -- 어긋난 절차로 판정하지 않는다. 멈춘다.")
                return 1
        s, v = solve_scale("dist", src, br, hi=40.0)
        scales[("dist", src)] = s
        print("  거리   %-6s C     %.4g  -> 채택률 %.4f" % (src, s, v))

    # ---- 본 실행 ----
    print()
    print("본 실행")
    print("-" * 92)
    tags = []
    for ch, _ in CHANNELS:
        for src, _ in SOURCES:
            tag = "%s_%s" % (ch, src)
            run(ch, src, scales[(ch, src)], tag)
            tags.append(tag)
            print("  %s 완료" % tag)

    ev = evaluate(["base"] + tags)
    print()
    print("=" * 92)
    print("[1] 격자 -- 기준선 %.3f 대비 dHOTA" % h0)
    print("=" * 92)
    print("  %-14s %14s %14s" % ("소스", "거리 함수", "게이팅"))
    print("  " + "-" * 46)
    grid = {}
    for src, nm in SOURCES:
        row = []
        for ch, _ in CHANNELS:
            v = ev["%s_%s" % (ch, src)]
            grid[(ch, src)] = v
            row.append("%+.3f" % (v[0] - h0) if v else "  --  ")
        print("  %-14s %14s %14s" % (nm, row[0], row[1]))

    # ---- 시퀀스별 부호 (규칙 6) ----
    print()
    print("[4] 시퀀스별 부호 -- n=7 이라 결합 값 하나로 판정하지 않는다")
    print("  %-14s %14s %14s" % ("소스", "거리 함수", "게이팅"))
    print("  " + "-" * 46)
    base_per = ev["base"][1]
    neg_all = True
    for src, nm in SOURCES:
        row = []
        for ch, _ in CHANNELS:
            v = grid[(ch, src)]
            if not v:
                row.append("  --  ")
                continue
            k = sum(1 for s in base_per if v[1][s] < base_per[s])
            row.append("%d/7 손해" % k)
            if v[0] - h0 >= 0:
                neg_all = False
        print("  %-14s %14s %14s" % (nm, row[0], row[1]))

    print()
    print("=" * 92)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG.md 보정 3)")
    print("=" * 92)
    # **NMS x 게이팅은 개입 불성립이라 판정에서 뺀다** (PREREG 보정 2·3).
    # 확장이 문을 여는 방향으로 작동하지 않는 조건의 dHOTA 는 "졌다" 의
    # 근거가 아니다. 유효한 **다섯 칸**으로 판정한다.
    INVALID = {("gate", "nms")}
    valid = {k: v for k, v in grid.items() if k not in INVALID and v}
    vals = [v[0] - h0 for v in valid.values()]
    print("  **NMS x 게이팅은 개입 불성립이라 제외한다.** 유효한 칸 %d개로 판정."
          % len(vals))
    print()
    if all(x < 0 for x in vals):
        print("  **다섯 칸 모두 음수 => 교란이 부분적으로 해소된다.**")
        print("  각 경로 안에서 추정 방식을 갈랐고(거리 3종, 게이팅 2종) 전부 음수다.")
        print("  **그러나 한계 (1) 을 승격하지 않는다** -- 사전 등록대로 고쳐 쓴다:")
        print("     '격자를 채운 범위가 두 경로이고, 그중 게이팅은 두 추정 방식뿐이다'")
    else:
        pos = [k for k, v in valid.items() if v[0] - h0 >= 0]
        print("  **양수인 칸이 있다: %s => 교란 미해소.**" % pos)
        print("  한계 (1) 을 그대로 둔다.")
    print()
    print("  원고에 적을 문장 (사전 등록에 박아 둔 것):")
    print("     'NMS 후보 분산 x 게이팅은 확장이 문을 여는 방향으로 작동하지")
    print("      않아 측정이 성립하지 않았다. 따라서 이 조합에 대해서는 소스와")
    print("      경로 중 무엇이 원인인지 말할 수 없다.'")
    print()
    print("  소스 간 차이(경로 안)와 경로 간 차이(소스 안)를 나란히 보라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
