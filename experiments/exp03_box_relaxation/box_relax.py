"""
실험 3 - 박스 확장 통제군.

UncertaintyTrack (arXiv:2402.12303) 의 최대 기여 성분은 "bounding box relaxation"
(+2.3 mMOTA, IDSW -14.3%) 이다. 검출 공분산으로 박스를 키운 뒤 매칭한다.
그런데 **공분산 없이 그냥 키워도 같은 이득이 나는지** 는 논문에 통제군이 없다.

이 모듈은 UTrack 트래커에 확장 방식만 다른 네 조건을 붙인다. 확장량의 평균을
맞추므로, 갈라지는 것은 "얼마나 키웠나" 가 아니라 "무엇에 따라 키웠나" 뿐이다.

  A  measure       확장 없음. 기준선과 수치가 같아야 한다 (훅 검증) + 통계 수집
  R  sigma         검출 공분산으로 키운다.   pad = alpha * s
  K1 const         모두 같은 픽셀만큼.       pad = delta
  K2 prop          제 크기에 비례해.         pad = c * (w 또는 h)   <-- 판정 조건
  K3 shuffle       s 를 통째로 뒤섞는다.     pad = alpha * s[perm]  (진단용)
  K4 ratio_shuffle s/w 비율만 뒤섞는다.                             (진단용)

s 는 박스 모서리의 표준편차다. tlbr 의 좌변은 x - w/2 이므로
  s_x = sqrt(var_x + var_w / 4),  s_y = sqrt(var_y + var_h / 4).

**K2 가 판정 조건다.** 크기만 쓰는 자명한 규칙이면서 확장량은 R 과 같기 때문이다.
R 이 K2 를 못 이기면 "불확실성이 전달됐다" 는 해석이 무너진다.

> **설계 당시의 예상 근거는 철회됐다.** 처음에는 "실험 1 에서 h 를 통제하면
> 편상관이 +0.044 로 사라진다, 즉 s 는 거의 크기의 함수다" 를 근거로 K2 가 R 을
> 재현할 것이라 예상했다. **그 +0.044 는 NMS in-place 버그의 산물이고 고친 값은
> +0.32 다** (exp01 README). s 는 크기의 함수가 아니다.
> **예상의 근거는 무너졌지만 설계와 판정 기준은 그대로 유효하다** -- K2 는 여전히
> 크기만 쓰는 통제군이고, 오히려 s 에 크기 밖의 신호가 있는 지금이 R 에게
> 더 유리한 조건이다. 그런데도 R 이 졌다.

**뒤섞기 조건(K3, K4)를 판정에 쓰면 안 된다 — 검정력 확인에서 밝혀졌다.**
UTrack 규약상 트랙은 마지막으로 매칭된 검출의 `_var_xywh` 를 물고 있다. 그래서
정답쌍은 **같은 s 를 공유**하고, R 은 트랙과 검출을 같은 만큼 키워 기하를
유지한다. 뒤섞으면 그 짝 일관성이 깨진다. 검증 (`power_check.py`, sigma 가
순전한 잡음인 W3):

    짝 일관성 있음 :  R-K4 = +0.1726
    짝 일관성 제거 :  R-K4 = -0.0053      <- 이득이 통째로 사라진다

(2026-08-17 재실행 값. 그전에 적혀 있던 +0.1807 / -0.0038 은 `hash()` 씨앗으로
난 값이라 재현되지 않았고, '짝 일관성 제거' 는 코드에 있지도 않았다. 지금은
고정 씨앗 + `make_world(pair_consistent=False)` 로 둘 다 재현된다.)

즉 R 이 뒤섞기를 이기는 것은 **정보가 아니라 짝 일관성**이다. sigma 가 아무
정보도 없는 세계에서도 +0.18 이 나온다. K3·K4 는 진단으로만 쓴다.

설정은 전부 환경변수로 준다. UTrack 소스는 `collections.py` 에 클래스 하나를
덧붙이는 것 말고는 건드리지 않는다 (`patch_utrack.py`).

  RELAX_MODE    measure | sigma | const | prop | shuffle | ratio_shuffle | off
  RELAX_ALPHA   sigma 배율
  RELAX_DX/DY   const 확장량 (픽셀)
  RELAX_CW/CH   prop 확장 비율
  RELAX_APPLY   both | det   확장을 양쪽에 줄지 검출에만 줄지 (기본 both)
  RELAX_CAP     pad <= CAP * (w 또는 h) 상한 (기본 1.0)
  RELAX_SEED    뒤섞기 난수 씨앗
  RELAX_STATS   통계 덤프 경로 (시퀀스마다 다른 경로를 줄 것)

**`RELAX_APPLY=det` 를 기본으로 두면 안 된다.** 검출만 키우면 이미 잘 맞는
쌍에서 IoU 가 오히려 **떨어진다** — 트랙 박스가 커진 검출 안으로 들어가 교집합은
그대로인데 합집합만 커지기 때문이다. 자체 시험에서 확인했다 (정답쌍 비용이
alpha 에 대해 증가). 양쪽을 같이 키워야 확장이 임계값을 **여는** 쪽으로만 작동한다.
UncertaintyTrack 은 확장 뒤 GIoU 로 매칭해 이 문제를 피하지만, 그러면 확장과
GIoU 두 변화가 섞인다. 여기서는 모든 조건이 **같은 IoU** 를 쓰게 두고 확장 방식만
가른다. 이 차이는 결과 해석에 명시할 것.
"""

import os
import json
import atexit
import threading

import numpy as np


def _env_f(name, default):
    v = os.environ.get(name)
    if v is None or v == '':
        return default
    return float(v)


MODE = (os.environ.get('RELAX_MODE') or 'measure').lower()
ALPHA = _env_f('RELAX_ALPHA', 1.0)
DX = _env_f('RELAX_DX', 0.0)
DY = _env_f('RELAX_DY', 0.0)
CW = _env_f('RELAX_CW', 0.0)
CH = _env_f('RELAX_CH', 0.0)
APPLY = (os.environ.get('RELAX_APPLY') or 'both').lower()
CAP = _env_f('RELAX_CAP', 1.0)
STATS_PATH = os.environ.get('RELAX_STATS') or ''

_VALID = ('measure', 'off', 'sigma', 'const', 'prop', 'shuffle',
          'ratio_shuffle')
if MODE not in _VALID:
    raise ValueError('RELAX_MODE must be one of %s, got %r' % (_VALID, MODE))

# 누적 통계. s 를 alpha=1 기준으로 모으므로 **한 번의 measure 실행이 모든 alpha 를
# 감당한다** (pad 가 alpha 에 선형이므로).
#
# 합만으로는 부족하다. 상한(CAP)이 걸리면 pad = min(alpha*s, CAP*w) 라서
# 평균이 alpha 에 선형이 아니고, E[s] 만으로는 조건의 확장량을 맞출 수 없다.
# 그래서 (s_x, s_y, w, h) 를 저수지 표본으로 함께 남긴다. calibrate.py 가
# 그 표본으로 E[min(alpha*s, CAP*w)] 를 직접 계산한다.
RESERVOIR_N = int(_env_f('RELAX_RESERVOIR', 30000))

_lock = threading.Lock()
_rng = np.random.default_rng(0)
_shuffle_rng = np.random.default_rng(int(_env_f('RELAX_SEED', 0)))
_res = []            # [(sx, sy, w, h), ...]
_acc = {
    'n_boxes': 0.0,      # 연관에서 본 검출 박스 수 (호출마다 중복 계수)
    'n_calls': 0.0,
    'sum_sx': 0.0, 'sum_sy': 0.0,
    'sum_w': 0.0, 'sum_h': 0.0,
    'sum_sx2': 0.0, 'sum_sy2': 0.0,
    'n_zero_var': 0.0,   # var_xywh 가 전부 0 인 박스 (분산이 안 실린 경로)
    'n_clipped': 0.0,
}

# 트랙 쪽 누적. **저수지와 분리해서 담는다.**
#
# calibrate.py 는 검출 쪽 표본으로만 상수를 푼다. 그런데 RELAX_APPLY=both 라
# pad 는 트랙에도 붙는다. 트랙 sigma 분포가 검출과 다르면 **검출 쪽 평균만
# 맞고 트랙 쪽은 어긋난다** -- 그러면 조건들이 "확장량 일치" 가 아니게 된다.
# 이 실험의 전부가 확장량 일치에 걸려 있는데 그 절반이 검증된 적이 없었다.
#
# 저수지에 섞으면 calibrate 의 상수가 바뀌어 exp03 수치가 흔들린다. 그래서
# 합계만 따로 모으고, measure 실행이 트랙/검출 평균을 나란히 보고하게 한다.
# (2026-08-17 에 열어둔 항목)
_acc_t = {
    'n_boxes': 0.0, 'sum_sx': 0.0, 'sum_sy': 0.0,
    'sum_w': 0.0, 'sum_h': 0.0, 'sum_pad_x': 0.0, 'sum_pad_y': 0.0,
    'n_zero_var': 0.0,
}
_acc['sum_pad_x'] = 0.0   # 검출 쪽 실제 pad 합 (상한 적용 후)
_acc['sum_pad_y'] = 0.0


def _reserve(rows):
    """표준 저수지 표본(Algorithm R). 앞쪽 프레임에 치우치지 않게 한다.

    i 번째 항목(1-indexed)은 확률 k/i 로 채택돼야 한다. `seen` 은 이 항목
    **직전까지** 본 개수이므로 i = seen + 1 이고, 상한은 seen 이 아니라
    seen+1 이어야 한다. seen 을 쓰면 확률이 k/(i-1) 이 되어 앞쪽 항목이
    과소표집된다 (k 가 작을수록 심하다. k=2 에서 2배, k=100 에서 1%).
    RESERVOIR_N=30000 에서는 영향이 없지만 고쳐 둔다. (2026-08-17)
    """
    for row in rows:
        seen = _acc['n_boxes']
        if len(_res) < RESERVOIR_N:
            _res.append(row)
        else:
            j = _rng.integers(0, int(seen) + 1)
            if j < RESERVOIR_N:
                _res[int(j)] = row
        _acc['n_boxes'] += 1.0


def _record(sx, sy, w, h, n_clipped, n_zero, px=None, py=None):
    with _lock:
        _acc['n_calls'] += 1.0
        _acc['sum_sx'] += float(sx.sum())
        _acc['sum_sy'] += float(sy.sum())
        _acc['sum_w'] += float(w.sum())
        _acc['sum_h'] += float(h.sum())
        _acc['sum_sx2'] += float((sx ** 2).sum())
        _acc['sum_sy2'] += float((sy ** 2).sum())
        _acc['n_clipped'] += float(n_clipped)
        _acc['n_zero_var'] += float(n_zero)
        if px is not None:
            _acc['sum_pad_x'] += float(px.sum())
            _acc['sum_pad_y'] += float(py.sum())
        _reserve(zip(sx.tolist(), sy.tolist(), w.tolist(), h.tolist()))


def _record_track(sx, sy, w, h, px, py, n_zero):
    """트랙 쪽. 저수지에 넣지 않는다 (calibrate 의 상수를 바꾸면 안 된다)."""
    with _lock:
        _acc_t['n_boxes'] += float(sx.size)
        _acc_t['sum_sx'] += float(sx.sum())
        _acc_t['sum_sy'] += float(sy.sum())
        _acc_t['sum_w'] += float(w.sum())
        _acc_t['sum_h'] += float(h.sum())
        _acc_t['sum_pad_x'] += float(px.sum())
        _acc_t['sum_pad_y'] += float(py.sum())
        _acc_t['n_zero_var'] += float(n_zero)


def _dump_stats():
    """종료 시 저장. **합치지 않는다.**

    시퀀스마다 다른 경로를 주고 (`RELAX_STATS=stats/MOT17-02.json`),
    calibrate.py 에 전부 넘긴다. 합산 로직을 두면 저수지 표본이 시퀀스별로
    치우치는 것을 조용히 감춘다.
    """
    if not STATS_PATH or _acc['n_boxes'] == 0:
        return
    if os.path.exists(STATS_PATH):
        print('[relax] WARNING %s exists -- not overwriting. '
              '시퀀스마다 다른 경로를 줄 것.' % STATS_PATH)
        return
    out = dict(_acc)
    out['reservoir'] = [[round(v, 5) for v in row] for row in _res]
    out['reservoir_n'] = RESERVOIR_N
    out['track'] = dict(_acc_t)          # 트랙 쪽 (저수지 없음, 합계만)
    out['_meta'] = {'mode': MODE, 'apply': APPLY, 'cap': CAP, 'alpha': ALPHA}
    parent = os.path.dirname(os.path.abspath(STATS_PATH))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp = STATS_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    os.replace(tmp, STATS_PATH)
    print('[relax] stats -> %s (n_boxes=%d, reservoir=%d)'
          % (STATS_PATH, _acc['n_boxes'], len(_res)))


atexit.register(_dump_stats)


# ---- 방향 검사용 원자료 기록 (PREREG-direction.md, 2026-08-20) ------------
# **기본은 꺼져 있고 비용 경로를 건드리지 않는다.** RELAX_DUMP_CALLS 에 경로를
# 주면 연관 호출마다 *확장 전* 입력을 남긴다. 판정 로직은 손대지 않았다.
#
# 왜 필요한가: 본 실행은 alpha=10 인데 방향을 검증한 [3b] 는 alpha=2, 합성
# 장면이다. **검증한 구간과 사용한 구간이 5 배 떨어져 있다.** 같은 입력에
# alpha 만 갈아끼워 채택 쌍이 늘어나는지 줄어드는지를 봐야 한다.
DUMP_CALLS = os.environ.get('RELAX_DUMP_CALLS') or ''
DUMP_MAX = int(_env_f('RELAX_DUMP_MAX', 0))      # 0 이면 무제한
_calls = []


def _dump_calls():
    if not DUMP_CALLS or not _calls:
        return
    import pickle
    parent = os.path.dirname(os.path.abspath(DUMP_CALLS))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp = DUMP_CALLS + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'calls': _calls,
                     'meta': {'mode': MODE, 'apply': APPLY, 'cap': CAP,
                              'alpha': ALPHA}}, f, protocol=4)
    os.replace(tmp, DUMP_CALLS)
    print('[relax] calls -> %s (n=%d)' % (DUMP_CALLS, len(_calls)))


atexit.register(_dump_calls)


def _edge_sigma(var_xywh):
    """xywh 분산 -> 박스 모서리의 표준편차 (s_x, s_y)."""
    v = np.asarray(var_xywh, dtype=float).reshape(-1, 4)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(v, 0.0, out=v)
    sx = np.sqrt(v[:, 0] + 0.25 * v[:, 2])
    sy = np.sqrt(v[:, 1] + 0.25 * v[:, 3])
    n_zero = int(np.count_nonzero((sx <= 0.0) & (sy <= 0.0)))
    return sx, sy, n_zero


def _pads(tlbr, sx, sy):
    """조건별 확장량. 반환은 (pad_x, pad_y, w, h, n_clipped)."""
    w = np.maximum(tlbr[:, 2] - tlbr[:, 0], 1e-6)
    h = np.maximum(tlbr[:, 3] - tlbr[:, 1], 1e-6)

    if MODE in ('measure', 'off'):
        px = np.zeros_like(w)
        py = np.zeros_like(h)
    elif MODE == 'sigma':
        px = ALPHA * sx
        py = ALPHA * sy
    elif MODE == 'shuffle':
        # 같은 프레임 안에서 sigma 를 박스들 사이에 통째로 뒤섞는다.
        # 확장량의 분포는 R 과 같고 sigma-박스 짝이 전부 깨진다.
        # **크기와의 연결까지 깨지므로** 이것만으로는 "크기 이상의 정보" 를 못 잰다.
        perm = _shuffle_rng.permutation(sx.size)
        px = ALPHA * sx[perm]
        py = ALPHA * sy[perm]
    elif MODE == 'ratio_shuffle':
        # **이것이 결정적 통제군이다.**
        # s/w 비율만 뒤섞는다. 크기와의 연결(s 가 큰 박스에서 크다)은 그대로 두고
        # **크기로 설명되지 않는 부분의 짝만** 깬다.
        # R > ratio_shuffle 이라야 "크기 이상의 정보가 있었다" 고 말할 수 있다.
        rx = sx / w
        ry = sy / h
        perm = _shuffle_rng.permutation(sx.size)
        px = ALPHA * w * rx[perm]
        py = ALPHA * h * ry[perm]
    elif MODE == 'const':
        px = np.full_like(w, DX)
        py = np.full_like(h, DY)
    else:                                              # prop
        px = CW * w
        py = CH * h

    if MODE in ('shuffle', 'ratio_shuffle'):
        # 섞은 뒤 평균이 R 과 어긋날 수 있다 (ratio_shuffle 은 Cov(w, s/w) 만큼).
        # 호출마다 R 의 평균에 정확히 맞춰 되돌린다. 그래야 갈리는 것이
        # "확장량" 이 아니라 "짝" 뿐이다.
        for arr, ref in ((px, ALPHA * sx), (py, ALPHA * sy)):
            m = arr.mean()
            if m > 0:
                arr *= ref.mean() / m

    cap_x = CAP * w
    cap_y = CAP * h
    n_clipped = int(np.count_nonzero((px > cap_x) | (py > cap_y)))
    return np.minimum(px, cap_x), np.minimum(py, cap_y), w, h, n_clipped


def _expand(tlbr, px, py):
    out = tlbr.copy()
    out[:, 0] -= px
    out[:, 1] -= py
    out[:, 2] += px
    out[:, 3] += py
    return out


def _tlbr_of(objs):
    if len(objs) == 0:
        return np.zeros((0, 4), dtype=float)
    return np.asarray([o.tlbr for o in objs], dtype=float).reshape(-1, 4)


def _var_of(objs):
    if len(objs) == 0:
        return np.zeros((0, 4), dtype=float)
    return np.asarray([o.var_xywh for o in objs], dtype=float).reshape(-1, 4)


def _numpy_ious(a, b):
    """UTrack 밖(자체 시험)에서 쓰는 대체 IoU. `bbox_ious` 와 같은 규약."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _get_ious():
    """UTrack 안에서는 그쪽 cython IoU 를, 밖에서는 numpy 판을 쓴다."""
    try:
        from .matching import ious                      # 순환 임포트 회피
        return ious
    except (ImportError, ValueError):                    # 패키지 밖에서 임포트된 경우
        return _numpy_ious


def relaxed_iou_distance(tracks, detections, args=None,
                         match_thresh=None, is_fuse=None):
    """박스를 키운 뒤 IoU 비용을 낸다. `iou_distance` 와 반환 규약이 같다.

    MODE='measure' 이면 확장량이 0 이므로 `iou_distance` 와 **수치가 같아야 한다.**
    그것이 훅이 제대로 걸렸는지 확인하는 사전 점검이다.
    """
    ious = _get_ious()

    t_tlbr = _tlbr_of(tracks)
    d_tlbr = _tlbr_of(detections)

    # **확장 전** 입력을 남긴다 (기본 꺼짐). copy 는 보험이다 -- 이 저장소는
    # in-place 변경으로 두 번 데였다 (NMS in-place, letterbox).
    if DUMP_CALLS and (DUMP_MAX <= 0 or len(_calls) < DUMP_MAX):
        _calls.append((
            t_tlbr.copy(), _var_of(tracks).copy(),
            d_tlbr.copy(), _var_of(detections).copy(),
            np.asarray([getattr(d, 'score', 1.0) for d in detections],
                       dtype=float),
            (float(match_thresh) if match_thresh is not None else -1.0),
            bool(is_fuse)))

    if d_tlbr.shape[0] > 0:
        dsx, dsy, n_zero = _edge_sigma(_var_of(detections))
        dpx, dpy, dw, dh, n_clip = _pads(d_tlbr, dsx, dsy)
        _record(dsx, dsy, dw, dh, n_clip, n_zero, dpx, dpy)
        d_tlbr = _expand(d_tlbr, dpx, dpy)

    if APPLY == 'both' and t_tlbr.shape[0] > 0:
        tsx, tsy, t_zero = _edge_sigma(_var_of(tracks))
        tpx, tpy, tw, th, _ = _pads(t_tlbr, tsx, tsy)
        _record_track(tsx, tsy, tw, th, tpx, tpy, t_zero)
        t_tlbr = _expand(t_tlbr, tpx, tpy)

    cost_matrix = 1.0 - ious(t_tlbr, d_tlbr)
    return cost_matrix, None
