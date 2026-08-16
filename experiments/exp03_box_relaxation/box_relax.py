"""
실험 3 - 상자 확장 통제군.

UncertaintyTrack (arXiv:2402.12303) 의 최대 기여 성분은 "bounding box relaxation"
(+2.3 mMOTA, IDSW -14.3%) 이다. 검출 공분산으로 상자를 키운 뒤 매칭한다.
그런데 **공분산 없이 그냥 키워도 같은 이득이 나는지** 는 논문에 통제군이 없다.

이 모듈은 UTrack 트래커에 확장 방식만 다른 네 갈래를 붙인다. 확장량의 평균을
맞추므로, 갈라지는 것은 "얼마나 키웠나" 가 아니라 "무엇에 따라 키웠나" 뿐이다.

  A  measure  확장 없음. 기준선과 수치가 같아야 한다 (훅 검증용) + 통계 수집
  R  sigma    검출 공분산으로 키운다.   pad = alpha * s
  K1 const    모두 같은 픽셀만큼 키운다. pad = delta          (s 의 정보 제거)
  K2 prop     제 크기에 비례해 키운다.  pad = c * (w 또는 h)  (크기 교란만 남김)

s 는 상자 모서리의 표준편차다. tlbr 의 좌변은 x - w/2 이므로
  s_x = sqrt(var_x + var_w / 4),  s_y = sqrt(var_y + var_h / 4).

**K2 가 결정적이다.** 실험 1 에서 NMS 분산은 상자 높이를 통제하면 위치오차와의
편상관이 +0.044 로 사라졌다. 즉 s 는 거의 크기의 함수다. 그렇다면 K2 는 R 과
같은 성능을 내야 하고, 그것이 확인되면 "불확실성이 전달됐다" 는 해석이 무너진다.

설정은 전부 환경변수로 준다. UTrack 소스는 `collections.py` 에 클래스 하나를
덧붙이는 것 말고는 건드리지 않는다 (`patch_utrack.py`).

  RELAX_MODE    measure | sigma | const | prop | off
  RELAX_ALPHA   sigma 배율
  RELAX_DX/DY   const 확장량 (픽셀)
  RELAX_CW/CH   prop 확장 비율
  RELAX_APPLY   both | det   확장을 양쪽에 줄지 검출에만 줄지 (기본 both)

**`det` 를 기본으로 두면 안 된다.** 검출만 키우면 이미 잘 맞는 쌍에서 IoU 가
오히려 **떨어진다** — 트랙 상자가 커진 검출 안에 들어가 교집합은 그대로인데
합집합만 커지기 때문이다. 자체 시험에서 확인했다 (평균 비용이 alpha 에 증가).
양쪽을 같이 키워야 확장이 임계값을 **여는** 방향으로만 작동한다.
UncertaintyTrack 은 확장 뒤 GIoU 로 매칭해 이 문제를 피하지만, 그러면 확장과
GIoU 두 변화가 섞인다. 여기서는 네 갈래가 **같은 IoU** 를 쓰게 두고 확장 방식만
가른다. 이 차이는 결과 해석에 명시할 것.
  RELAX_CAP     pad <= CAP * (w 또는 h) 상한 (기본 1.0)
  RELAX_STATS   통계 덤프 경로. 있으면 종료 시 누적 저장(시퀀스별 실행을 합산)
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

_VALID = ('measure', 'off', 'sigma', 'const', 'prop')
if MODE not in _VALID:
    raise ValueError('RELAX_MODE must be one of %s, got %r' % (_VALID, MODE))

# 누적 통계. s 를 alpha=1 기준으로 모으므로 **한 번의 measure 실행이 모든 alpha 를
# 감당한다** (pad 가 alpha 에 선형이므로).
#
# 합만으로는 부족하다. 상한(CAP)이 걸리면 pad = min(alpha*s, CAP*w) 라서
# 평균이 alpha 에 선형이 아니고, E[s] 만으로는 갈래의 확장량을 맞출 수 없다.
# 그래서 (s_x, s_y, w, h) 를 저수지 표본으로 함께 남긴다. calibrate.py 가
# 그 표본으로 E[min(alpha*s, CAP*w)] 를 직접 계산한다.
RESERVOIR_N = int(_env_f('RELAX_RESERVOIR', 30000))

_lock = threading.Lock()
_rng = np.random.default_rng(0)
_res = []            # [(sx, sy, w, h), ...]
_acc = {
    'n_boxes': 0.0,      # 연관에서 본 검출 상자 수 (호출마다 중복 계수)
    'n_calls': 0.0,
    'sum_sx': 0.0, 'sum_sy': 0.0,
    'sum_w': 0.0, 'sum_h': 0.0,
    'sum_sx2': 0.0, 'sum_sy2': 0.0,
    'n_zero_var': 0.0,   # var_xywh 가 전부 0 인 상자 (분산이 안 실린 경로)
    'n_clipped': 0.0,
}


def _reserve(rows):
    """표준 저수지 표본. 앞쪽 시퀀스에 치우치지 않게 한다."""
    for row in rows:
        seen = _acc['n_boxes']
        if len(_res) < RESERVOIR_N:
            _res.append(row)
        else:
            j = _rng.integers(0, int(seen))
            if j < RESERVOIR_N:
                _res[int(j)] = row
        _acc['n_boxes'] += 1.0


def _record(sx, sy, w, h, n_clipped, n_zero):
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
        _reserve(zip(sx.tolist(), sy.tolist(), w.tolist(), h.tolist()))


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


def _edge_sigma(var_xywh):
    """xywh 분산 -> 상자 모서리의 표준편차 (s_x, s_y)."""
    v = np.asarray(var_xywh, dtype=float).reshape(-1, 4)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(v, 0.0, out=v)
    sx = np.sqrt(v[:, 0] + 0.25 * v[:, 2])
    sy = np.sqrt(v[:, 1] + 0.25 * v[:, 3])
    n_zero = int(np.count_nonzero((sx <= 0.0) & (sy <= 0.0)))
    return sx, sy, n_zero


def _pads(tlbr, sx, sy):
    """갈래별 확장량. 반환은 (pad_x, pad_y, w, h, n_clipped)."""
    w = np.maximum(tlbr[:, 2] - tlbr[:, 0], 1e-6)
    h = np.maximum(tlbr[:, 3] - tlbr[:, 1], 1e-6)

    if MODE in ('measure', 'off'):
        px = np.zeros_like(w)
        py = np.zeros_like(h)
    elif MODE == 'sigma':
        px = ALPHA * sx
        py = ALPHA * sy
    elif MODE == 'const':
        px = np.full_like(w, DX)
        py = np.full_like(h, DY)
    else:                                              # prop
        px = CW * w
        py = CH * h

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
    """상자를 키운 뒤 IoU 비용을 낸다. `iou_distance` 와 반환 규약이 같다.

    MODE='measure' 이면 확장량이 0 이므로 `iou_distance` 와 **수치가 같아야 한다.**
    그것이 훅이 제대로 걸렸는지 확인하는 관문이다.
    """
    ious = _get_ious()

    t_tlbr = _tlbr_of(tracks)
    d_tlbr = _tlbr_of(detections)

    if d_tlbr.shape[0] > 0:
        dsx, dsy, n_zero = _edge_sigma(_var_of(detections))
        dpx, dpy, dw, dh, n_clip = _pads(d_tlbr, dsx, dsy)
        _record(dsx, dsy, dw, dh, n_clip, n_zero)
        d_tlbr = _expand(d_tlbr, dpx, dpy)

    if APPLY == 'both' and t_tlbr.shape[0] > 0:
        tsx, tsy, _ = _edge_sigma(_var_of(tracks))
        tpx, tpy, _, _, _ = _pads(t_tlbr, tsx, tsy)
        t_tlbr = _expand(t_tlbr, tpx, tpy)

    cost_matrix = 1.0 - ious(t_tlbr, d_tlbr)
    return cost_matrix, None
