# -*- coding: utf-8 -*-
"""실험 5 [3단계] -- 캐시된 검출을 트래커에 재생하고 MOT 결과를 쓴다.

**사전 등록은 README. 자료보다 먼저 커밋했다.**

조건마다 검출기를 다시 돌리지 않는다. `cache_detections.py` 가 만든 npz 를
읽어 **같은 검출**을 모든 조건에 먹인다. 갈리는 것은 `get_dists` 하나뿐이다.

## 훅 지점 (소스 확인)

`BYTETracker.get_dists(tracks, detections)` 가 1단계 연관 비용을 만든다.
`init_track` 이 STrack 을 만들 때 `xywh` 끝에 **검출 인덱스**를 붙여 주므로
(`STrack.idx`), 그 인덱스로 검출별 sigma 를 STrack 에 실을 수 있다.

**2단계(저신뢰)는 건드리지 않는다.** ultralytics 는 2단계에서 `get_dists` 를
안 쓰고 `matching.iou_distance` 를 직접 부른다. 게다가 exp00 이 기본 설정에서
2단계가 사실상 죽어 있음을 보였다. 개입을 1단계로 한정하는 것이 해석도 쉽다.

## 트랙 쪽 Sigma_t

`STrack.covariance` 는 칼만 상태 (x, y, a, h, ...) 의 공분산이다.
`covariance[0,0]`, `covariance[1,1]` 이 중심 x, y 의 분산(px^2)이라 검출 쪽과
단위가 같다. **조건과 무관하게 이것을 쓴다** (사전 등록 함정 2).

## 사전 점검 [0b] 를 여기서 잰다

수학 감사 (나): Bures 의 비분리항 `-2*sum sqrt(st*sd)` 는 **st 가 트랙마다
달라야** 산다. st 가 상수면 순수 열함수가 되어 할당에서 사라진다.
**매 연관에서 트랙 간 CV(Sigma_t) 를 기록**하고, 중앙값이 0.05 미만이면
이 설계는 검정력이 없다.

사용법:
    python experiments/exp05_wasserstein/replay.py iou
    python experiments/exp05_wasserstein/replay.py w_dfl w_size

**강건성 확인용 덮어쓰기** (기본 동작은 안 바뀐다):
    python experiments/exp05_wasserstein/replay.py w_dfl --C 210.4 --out w_dfl_rate

`--C` 를 주면 중앙값 보정을 건너뛰고 그 값을 쓴다. `accept_rate.py --solve` 가
채택률을 기준선에 정확히 맞추는 C 를 준다. `--out` 은 결과 폴더 이름이다.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ultralytics.trackers.byte_tracker import BYTETracker   # noqa: E402
from ultralytics.trackers.utils import matching             # noqa: E402

from wcost import (w2_matrix, w2_matrix_norm, size_var,       # noqa: E402
                   match_scale, nwd_cost, solve_C)

CACHE = Path("data/exp05")
OUTDIR = Path("data/exp05/tracks")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
# wn_* 는 실험 5b (크기 정규화). 사전 등록은 PREREG-norm.md
ARMS = ("iou", "w_dfl", "w_size", "w_nms", "wn_dfl", "wn_size")

# ByteTrack 기본값. 조건 사이에서 **바꾸지 않는다.**
BASE = dict(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
            new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
            fuse_score=True)


class Det:
    """BYTETracker 가 요구하는 최소 규약 + 검출별 sigma."""

    def __init__(self, xyxy, conf, cls, sxx, syy):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.sxx = np.asarray(sxx, dtype=np.float32)
        self.syy = np.asarray(syy, dtype=np.float32)

    @property
    def xywh(self):
        x = self.xyxy
        return np.stack([(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2,
                         x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]], 1)

    def __getitem__(self, m):
        return Det(self.xyxy[m], self.conf[m], self.cls[m], self.sxx[m], self.syy[m])

    def __len__(self):
        return len(self.conf)


class WTracker(BYTETracker):
    """1단계 비용만 갈아끼운다. 나머지는 손대지 않는다."""

    def __init__(self, args, arm, C, frame_rate=30):
        super().__init__(args, frame_rate=frame_rate)
        self.arm = arm
        self.C = C
        self.cv_log = []          # 사전 점검 [0b]
        self.w2_log = []          # C 보정용 표본

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        if tracks and not WTracker._patched:
            WTracker._patch_strack(type(tracks[0]))
        for t in tracks:
            i = int(t.idx)
            t.det_var = np.array([results.sxx[i], results.syy[i]], dtype=float)
        return tracks

    # **트랙 sigma 를 매 연관마다 갱신한다 -- 참조 구현과 같은 자리에 건다.**
    #
    # `external/UTrack/tracker/update.py:124,165` 가 `re_activate` 와 `update`
    # 양쪽에서 `self._var_xywh = new_track._var_xywh` 를 한다. exp03 은 그
    # 프로퍼티를 읽었으므로 **신선한 sigma 로 돌았다.**
    #
    # 여기서는 `init_track` 에서만 붙이고 있어서 **100 프레임 산 트랙이 1 프레임
    # 때 sigma 를 들고 있었다.** 양쪽 확장에서 트랙과 검출의 sigma 가 다른 시점
    # 것이라 pad 가 어긋난다 -- 철회 17 의 det-only 와 같은 증상이다.
    _patched = False

    @classmethod
    def _patch_strack(cls, STrack):
        """STrack.update / re_activate 가 det_var 도 물려받게 한다."""
        if cls._patched:
            return
        for name in ("update", "re_activate"):
            orig = getattr(STrack, name)

            def wrap(self, new_track, *a, _o=orig, **k):
                r = _o(self, new_track, *a, **k)
                dv = getattr(new_track, "det_var", None)
                if dv is not None:
                    self.det_var = np.array(dv, dtype=float)
                return r
            setattr(STrack, name, wrap)
        cls._patched = True

    @staticmethod
    def _track_var(tracks):
        """칼만 P 의 중심 x,y 분산. 아직 활성화 안 된 트랙은 검출 분산으로 대체."""
        out = []
        for t in tracks:
            c = getattr(t, "covariance", None)
            if c is None:
                out.append(getattr(t, "det_var", np.array([1.0, 1.0])))
            else:
                out.append([max(float(c[0, 0]), 1e-6), max(float(c[1, 1]), 1e-6)])
        return np.asarray(out, dtype=float).reshape(-1, 2)

    def get_dists(self, tracks, detections):
        if self.arm == "iou" or not len(tracks) or not len(detections):
            d = matching.iou_distance(tracks, detections)
            if self.args.fuse_score:
                d = matching.fuse_score(d, detections)
            return d

        # ultralytics STrack 의 박스 속성은 `.xyxy` 다 (`.tlbr` 은 UTrack 쪽 이름).
        # 트랙 쪽 .xyxy 는 칼만 예측이 반영된 현재 추정값이라 연관에 쓰는 것이 맞다.
        t_box = np.asarray([t.xyxy for t in tracks], dtype=float).reshape(-1, 4)
        d_box = np.asarray([d.xyxy for d in detections], dtype=float).reshape(-1, 4)
        t_var = self._track_var(tracks)

        # 사전 점검 [0b]: 트랙 간 Sigma_t 변동
        tr = t_var.sum(-1)
        if len(tr) >= 2 and tr.mean() > 0:
            self.cv_log.append(float(tr.std() / tr.mean()))

        if self.arm in ("w_size", "wn_size"):
            d_var = size_var(d_box)
        else:                                   # w_dfl / wn_dfl / w_nms
            d_var = np.stack([[d.det_var[0], d.det_var[1]] for d in detections])

        # 함정 1: Sigma_d 의 규모를 트랙 쪽에 맞춘다. 안 맞추면 조건 간 차이가
        # 정보가 아니라 규모가 된다. **정규화 조건도 원래 px^2 좌표에서 한다** --
        # 절차를 안 바꾼다 (PREREG-norm.md).
        d_var = match_scale(d_var, float(np.mean(t_var.sum(-1))))

        f = w2_matrix_norm if self.arm.startswith("wn_") else w2_matrix
        w2 = f(t_box, d_box, t_var, d_var)
        self.w2_log.append(w2.ravel())
        dists = nwd_cost(w2, self.C)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists


def load(seq, arm):
    f0 = CACHE / ("%s.npz" % seq)
    if not f0.exists():          # 캐싱이 아직 안 끝난 시퀀스는 건너뛴다
        return None
    d = np.load(f0)
    sxx, syy = d["sxx"], d["syy"]
    if arm == "w_nms":                          # NMS sigma 는 별도 파일에서
        f = CACHE / ("%s-nms.npz" % seq)
        if not f.exists():
            return None
        n = np.load(f)
        sxx, syy = n["sxx"], n["syy"]
    return dict(frame=d["frame"], xyxy=d["xyxy"], conf=d["conf"],
                sxx=sxx, syy=syy, n_frames=int(d["n_frames"]))


def run_seq(seq, arm, C):
    c = load(seq, arm)
    if c is None:
        return None, None
    args = SimpleNamespace(**BASE)
    tr = WTracker(args, arm, C, frame_rate=30)
    lines = []
    for f in range(1, c["n_frames"] + 1):
        m = c["frame"] == f
        det = Det(c["xyxy"][m], c["conf"][m], np.zeros(int(m.sum())),
                  c["sxx"][m], c["syy"][m])
        out = tr.update(det)
        for row in out:
            x1, y1, x2, y2 = row[:4]
            tid = int(row[4])
            lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1"
                         % (f, tid, x1, y1, x2 - x1, y2 - y1, float(row[5])))
    return lines, tr


def _flag(name, cast):
    """--name VALUE 를 읽는다. 없으면 None."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return None


def main():
    arms = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    C_override = _flag("--C", float)
    out_override = _flag("--out", str)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # C 보정 (함정 3): 기준선 비용의 중앙값에 맞춘다. 먼저 iou 로 목표를 잰다.
    print("C 보정용 기준선 비용 중앙값을 잰다...")
    probe = WTracker(SimpleNamespace(**BASE), "iou", 1.0)
    c0 = load(SEQS[0], "iou")
    for f in range(1, min(c0["n_frames"], 60) + 1):
        m = c0["frame"] == f
        det = Det(c0["xyxy"][m], c0["conf"][m], np.zeros(int(m.sum())),
                  c0["sxx"][m], c0["syy"][m])
        probe.update(det)
    print("  (기준선 비용 중앙값은 iou 조건 실행 중에 수집한다)")

    for arm in arms:
        print("=" * 66)
        print("조건 %s" % arm)
        print("=" * 66)
        # w2 표본을 먼저 모아 C 를 푼다 (한 시퀀스로 충분하다)
        C = 1.0
        if arm != "iou" and C_override is not None:
            C = C_override
            print("  C = %.3f  (**덮어쓰기** -- 채택률 맞춤. 중앙값 보정 건너뜀)" % C)
        elif arm != "iou":
            tmp = WTracker(SimpleNamespace(**BASE), arm, 1.0)
            c1 = load(SEQS[0], arm)
            if c1 is None:
                print("  캐시 없음. 건너뛴다."); continue
            for f in range(1, min(c1["n_frames"], 80) + 1):
                m = c1["frame"] == f
                tmp.update(Det(c1["xyxy"][m], c1["conf"][m], np.zeros(int(m.sum())),
                               c1["sxx"][m], c1["syy"][m]))
            if tmp.w2_log:
                C = solve_C(np.concatenate(tmp.w2_log), 0.5)
            print("  C = %.3f  (비용 중앙값 0.5 목표)" % C)

        cvs = []
        for seq in SEQS:
            lines, tr = run_seq(seq, arm, C)
            if lines is None:
                print("  %-18s 캐시 없음" % seq); continue
            out = OUTDIR / (out_override or arm)
            out.mkdir(parents=True, exist_ok=True)
            (out / ("%s.txt" % seq)).write_text("\n".join(lines) + "\n")
            if tr.cv_log:
                cvs.append(float(np.median(tr.cv_log)))
            print("  %-18s 트랙행 %d" % (seq, len(lines)))
        if cvs:
            print("  [0b] 트랙 간 CV(Sigma_t) 중앙값 = %.4f  %s"
                  % (np.median(cvs),
                     "<- 0.05 미만: 검정력 없음" if np.median(cvs) < 0.05 else "OK"))


if __name__ == "__main__":
    main()
