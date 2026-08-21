# -*- coding: utf-8 -*-
"""실험 3 보론 — **확장이 문을 여는가 닫는가.** (재생 단계)

사전 등록은 `PREREG-direction.md` (자료보다 먼저 커밋, 읽는 법 포함).

## 왜

본 실행의 α 는 **10** 인데(`README.md:240`) 방향을 검증한 `[3b]` 는
**α=2, 합성 장면**이다(`README.md:62`). **검증한 구간과 사용한 구간이
5 배 떨어져 있다.** exp19 에서 NMS 소스가 큰 α 에서 채택률을 떨어뜨리는
것이 나왔으므로 exp03 이 같은 자리인지 봐야 한다.

## 어떻게

`RELAX_MODE=measure`(확장 0) 로 한 번 돌며 연관 호출마다 **확장 전** 입력을
남긴다. 그 **같은 입력에** α 만 갈아끼워 exp03 의 `_edge_sigma` -> `_pads`
(CAP 포함) -> `_expand` -> IoU -> `linear_assignment` 를 다시 적용한다.
개입만 갈리고 트래커 상태는 고정되므로 **"확장이 문을 여는가" 에 곧바로
답한다.**

`fuse_score` 와 `linear_assignment` 는 **UTrack 것을 그대로 쓴다.**
재구현하면 조용히 어긋난다 (CLAUDE.md — 전처리 재구현이 HOTA 를 몇 점씩
틀리게 한 전례).

## 판정

문턱은 exp19 `[0b]` 규칙 그대로 **5 쌍**. 그 안이면 **중립**이고, 중립이면
채택 *집합* 의 대칭차를 세고 나서 판정한다 — **채택 수가 안 변했다고 개입이
안 돈 것이 아니다.**

    python direction.py /content/exp03_dumps
"""
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # **stderr 도 해야 한다.** sys.exit(메시지) 는 stderr 로 나가는데
    # 거기가 cp949 면 한글이 깨져 무슨 사전 점검에 걸렸는지 못 읽는다.
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(HERE))
UTRACK = os.environ.get("UTRACK_ROOT") or "/content/UTrack"
if os.path.isdir(UTRACK):
    sys.path.insert(0, UTRACK)

ALPHAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
USED = (10.0, 20.0, 40.0)          # 본 실행이 쓴 구간
VERIFIED = 2.0                     # [3b] 가 검증한 구간
GRAIN = 5                          # 채택 쌍 5 건 (exp19 [0b] 와 같은 문턱)
PRIMARY = 10.0                     # 주 판정 -- -4.33 이 나온 α


def load_relax(alpha, apply_to="both", cap=1.0):
    """box_relax 를 환경변수째로 다시 읽는다 (selftest.load 와 같은 방식)."""
    for k in list(os.environ):
        if k.startswith("RELAX_"):
            del os.environ[k]
    os.environ["RELAX_MODE"] = "measure" if alpha == 0 else "sigma"
    os.environ["RELAX_ALPHA"] = str(alpha)
    os.environ["RELAX_APPLY"] = apply_to
    os.environ["RELAX_CAP"] = str(cap)
    # **UTrack 패키지 안의 사본을 임포트한다.** 최상위로 임포트하면
    # `_get_ious` 의 `from .matching import ious` 가 ImportError 로 떨어져
    # **조용히 `_numpy_ious` 로 대체된다.** 그런데 두 함수는 값이 다르다 --
    # cython `bbox_overlaps` 는 Faster R-CNN 계열의 **+1 픽셀 규약**을 쓴다
    # (사전 점검이 max|diff| = 2.6e-02 로 잡았다. 잡음이 아니라 다른 함수다).
    # 본 실행은 패키지 안에서 돌았으므로 재생도 그래야 한다.
    sys.modules.pop("tracker.box_relax", None)
    return importlib.import_module("tracker.box_relax")


def _utrack_bits():
    """UTrack 의 fuse_score / linear_assignment 를 **그대로** 가져온다.

    재구현하지 않는다. 이 저장소는 전처리를 재구현했다가 HOTA 가 몇 점씩
    틀린 전례가 있다 (CLAUDE.md). **없으면 대체하지 않고 멈춘다.**
    """
    try:
        from tracker.matching import fuse_score, linear_assignment
    except ImportError:
        print("ERROR UTrack 의 tracker.matching 을 못 불러왔다.")
        print("      이 스크립트는 **UTrack 안에서** 돌아야 한다 --")
        print("        sys.path 에 /content/UTrack 이 있어야 하고")
        print("        nms_var / fuzzy_cython_bbox 가 설치돼 있어야 한다.")
        print("      **fuse_score 와 linear_assignment 를 재구현하지 않는다.**")
        print("      재생이 본 실행과 다른 계산을 하면 판정이 무의미하다.")
        sys.exit(2)
    return fuse_score, linear_assignment


def accepted(mod, calls, fuse_score, linear_assignment):
    """이 α 에서 채택된 쌍의 집합과 pad 비대칭을 낸다."""
    _iou_fn = mod._get_ious()          # 본 실행과 **같은** IoU
    acc, asym = set(), []
    for ci, (t_tlbr, t_var, d_tlbr, d_var, scores, thr, is_fuse) in enumerate(calls):
        if t_tlbr.shape[0] == 0 or d_tlbr.shape[0] == 0:
            continue
        dt, dd = t_tlbr.copy(), d_tlbr.copy()
        dsx, dsy, _ = mod._edge_sigma(d_var)
        dpx, dpy, _, _, _ = mod._pads(dd, dsx, dsy)
        dd = mod._expand(dd, dpx, dpy)
        tsx, tsy, _ = mod._edge_sigma(t_var)
        tpx, tpy, _, _, _ = mod._pads(dt, tsx, tsy)
        dt = mod._expand(dt, tpx, tpy)

        # 쌍별 pad 비대칭 -- [3b] 는 **평균 비** 만 쟀다 (0.940).
        # 평균이 1 이어도 쌍마다 크게 어긋날 수 있다.
        if dpx.size and tpx.size:
            m = min(tpx.size, dpx.size)
            den = np.maximum(dpx[:m], 1e-9)
            asym.append(np.abs(tpx[:m] - dpx[:m]) / den)

        cost = 1.0 - _iou_fn(dt, dd)
        if is_fuse:
            cost = fuse_score(cost, None, scores=np.repeat(
                scores[None, :], cost.shape[0], axis=0))
        m_, _, _ = linear_assignment(cost, thresh=(thr if thr > 0 else 0.8))
        for i, j in np.asarray(m_).reshape(-1, 2):
            acc.add((ci, int(i), int(j)))
    a = np.concatenate(asym) if asym else np.zeros(1)
    return acc, a


def gate_ious(mod):
    """**사전 점검** -- 재생이 본 실행과 **같은 IoU 함수**를 쓰는가.

    `box_relax._get_ious()` 는 패키지 안에서는 cython `bbox_overlaps` 를,
    밖에서는 `_numpy_ious` 를 돌려준다. **그 대체가 조용하다.** 그리고 둘은
    값이 다르다 -- cython 쪽은 Faster R-CNN 계열의 **+1 픽셀 규약**이라
    전형적 보행자 박스에서 IoU 가 2e-02 쯤 어긋난다. 첫 판 사전 점검이
    `max|diff| = 2.632e-02` 로 이것을 잡았다.

    **값을 비교하지 않고 객체 동일성을 본다.** 값 비교는 "얼마나 다르면
    통과인가" 라는 자유도를 남기는데, 여기서 옳은 답은 **같은 함수** 하나뿐이다.
    """
    try:
        from tracker.matching import ious as cy_ious
    except Exception as e:
        print("  !! UTrack 의 tracker.matching 을 못 불러왔다 (%s)."
              % type(e).__name__)
        print("     **대조 없이 통과시키지 않는다.** UTRACK_ROOT 를 주거나")
        print("     PYTHONPATH 에 /content/UTrack 을 넣을 것.")
        return False
    fn = mod._get_ious()
    ok = (fn is cy_ious)
    print("  [사전 점검] 재생의 IoU 가 본 실행의 것인가: %s"
          % ("OK (cython bbox_overlaps)" if ok
             else "!! **numpy 대체본이다 -- 멈춘다**"))
    if not ok:
        print("     box_relax 를 최상위로 임포트해 `from .matching import ious`")
        print("     가 떨어졌다. `tracker.box_relax` 로 임포트해야 한다.")
        print("     (두 함수는 +1 픽셀 규약 때문에 IoU 가 2e-02 쯤 다르다.)")
    return ok


def main(dump_dir):
    dumps = sorted(Path(dump_dir).glob("*.pkl"))
    if not dumps:
        sys.exit("ERROR %s 에 덤프가 없다. run_colab.py direction 을 먼저 돌릴 것"
                 % dump_dir)
    fuse_score, linear_assignment = _utrack_bits()

    calls = []
    for f in dumps:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        calls.extend(d["calls"])
    print("=" * 84)
    print("실험 3 보론 -- 확장이 문을 여는가 닫는가")
    print("=" * 84)
    print("시퀀스 %d 개, 연관 호출 %d 회" % (len(dumps), len(calls)))
    print()
    if not gate_ious(load_relax(0.0)):
        print()
        print("  **판정하지 않는다** (CLAUDE.md 규칙 2 -- 어긋난 절차로 앞서 가지 않는다).")
        return 1
    print()

    base_set, base_asym = None, None
    rows = []
    for al in ALPHAS:
        mod = load_relax(al)
        s, asym = accepted(mod, calls, fuse_score, linear_assignment)
        if al == 0.0:
            base_set, base_asym = s, asym
        rows.append((al, len(s), len(s ^ base_set), asym))

    print("  %6s %10s %12s %12s   %s"
          % ("alpha", "채택 쌍", "기준선 대비", "대칭차", ""))
    print("  " + "-" * 66)
    for al, n, sym, asym in rows:
        d = n - rows[0][1]
        tag = ("  <- 본 실행" if al in USED else
               "  <- [3b] 가 검증한 구간" if al == VERIFIED else "")
        print("  %6g %10d %+12d %12d   %s" % (al, n, d, sym, tag))

    print()
    print("  쌍별 pad 비대칭 |pad_t - pad_d| / pad_d  (α=%g)" % PRIMARY)
    a = [r[3] for r in rows if r[0] == PRIMARY][0]
    print("     중앙값 %.4f   90분위 %.4f   비율 %.2f"
          % (np.median(a), np.percentile(a, 90),
             np.percentile(a, 90) / max(np.median(a), 1e-9)))

    # ---- 판정 (PREREG-direction.md 에 자료 전에 박은 것) ----
    n0 = rows[0][1]
    n1, sym1 = [(r[1], r[2]) for r in rows if r[0] == PRIMARY][0]
    d = n1 - n0
    print()
    print("=" * 84)
    print("판정 -- 자료 보기 전에 정한 읽는 법 (PREREG-direction.md)")
    print("=" * 84)
    print("  α=%g 에서 채택 쌍 변화 %+d 건 (문턱 %d 건)" % (PRIMARY, d, GRAIN))
    if d > GRAIN:
        print("  => **개입 성립.** 확장이 문을 연다. -4.33 과 88% 는 그대로 유효하다.")
    elif d < -GRAIN:
        print("  => **개입 불성립.** exp19 의 NMS x 게이팅과 같은 자리다.")
        print("     PREREG 「파급」표대로 고친다 -- tab:channels 게이팅 행,")
        print("     '네 경로', 88%, 그리고 **실험 20 전체**.")
    else:
        pct = 100.0 * sym1 / max(n0, 1)
        print("  => **중립.** 개수로 판정하지 않는다. 대칭차 %d 건 (채택의 %.2f%%)"
              % (sym1, pct))
        if pct < 1.0:
            print("     대칭차 < 1% => **개입이 사실상 돌지 않았다.** -4.33 은")
            print("     확장이 아니라 다른 것에서 온다. 88% 는 뜻을 잃는다.")
        else:
            print("     대칭차 >= 1% => **개입은 돌았고 방향이 중립이다.**")
            print("     -4.33 은 유효하되 '문을 여는 개입' 이라는 서술은 못 쓴다.")
    print()
    print("  **단조인가** -- 보정 1 의 (1) 이 '이 자료에서 단조' 라고 적었는데")
    print("  안 쟀다. 위 표의 채택 쌍 열로 확인할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/content/exp03_dumps"))
