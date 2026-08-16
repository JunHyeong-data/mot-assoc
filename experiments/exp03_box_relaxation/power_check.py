"""
실험 3 - 사전 검정력 확인. **콜랩을 돌리기 전에 이걸 먼저 본다.**

설계가 아무리 깔끔해도, 갈래들이 원리적으로 안 갈리면 콜랩 시간은 낭비다.
그래서 **답을 아는 합성 세계**에서 먼저 시험한다.

세 세계를 만든다. 셋 다 검출기가 sigma 를 뱉지만 그 sigma 의 지위가 다르다.

  W1 size-only   sigma = k*h.               참오차도 그 sigma. 크기가 전부다
  W2 informative sigma = k*h*u.             참오차도 그 sigma. u 가 진짜 정보다
  W3 miscalib    sigma = k*h*u 라고 보고.   참오차는 k*h. u 는 잡음이다
                 (실험 1 이 실측한 세계. 편상관 +0.044)

각 세계에서 네 갈래를 돌려 **정답쌍 복원율**을 잰다.

  설계에 검정력이 있다면:  W2 에서 R > K2,  W1/W3 에서 R ~= K2
  검정력이 없다면:         어느 세계에서도 R == K2  -> 콜랩 돌릴 이유가 없다

이건 합성이다. MOT17 에서 무엇이 나올지는 말하지 않는다.
**"이 실험이 무언가를 구별할 수 있는가" 만 답한다.**

    python power_check.py
"""

import importlib
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibrate  # noqa: E402

GATE = 0.8          # UTrack ablation_17 의 1단계 matching distance
N_FRAMES = 400
N_OBJ = 40          # 프레임당 객체. MOT17-02 급 밀집 (exp00 실측 M 중앙값 33)
N_FALSE = 8         # 짝 없는 검출
K_SIGMA = 0.10      # 참오차 = K_SIGMA * h
PRED_ERR = 0.30     # 칼만 예측오차 (상자 크기 대비)
FIELD = (960, 540)  # 좁은 화면. 넓게 뿌리면 매칭이 자명해져 천장에 붙는다

# 천장에 붙으면(복원율 ~1.0) 어떤 갈래도 차이를 못 낸다. 첫 판에서 그렇게 나와
# 밀집도와 오차를 올렸다. 기준선이 0.7~0.9 사이여야 잴 것이 있다.


class Box:
    def __init__(self, tlbr, var_xywh):
        self.tlbr = np.asarray(tlbr, dtype=float)
        self.var_xywh = np.asarray(var_xywh, dtype=float)


def load(**env):
    for k in list(os.environ):
        if k.startswith('RELAX_'):
            del os.environ[k]
    for k, v in env.items():
        os.environ[k] = str(v)
    if 'box_relax' in sys.modules:
        del sys.modules['box_relax']
    return importlib.import_module('box_relax')


def make_world(world, seed=0):
    """프레임 목록을 만든다. 각 프레임은 (tracks, dets, n_true)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(N_FRAMES):
        h = rng.uniform(60, 260, size=N_OBJ)
        w = h * rng.uniform(0.32, 0.48, size=N_OBJ)
        cx = rng.uniform(40, FIELD[0] - 40, size=N_OBJ)
        cy = rng.uniform(40, FIELD[1] - 40, size=N_OBJ)

        # u: 가림/흐림 같은 개체별 요인. 크기와 무관하다.
        u = rng.lognormal(0.0, 0.75, size=N_OBJ)

        if world == 'W1':
            sig_true = K_SIGMA * h
            sig_rep = K_SIGMA * h
        elif world == 'W2':
            sig_true = K_SIGMA * h * u
            sig_rep = K_SIGMA * h * u
        elif world == 'W3':
            sig_true = K_SIGMA * h
            sig_rep = K_SIGMA * h * u
        else:
            raise ValueError(world)

        # 트랙: 칼만 예측. 모든 트랙에 같은 규모의 예측오차 (크기 비례)
        tpe = rng.normal(0.0, PRED_ERR, size=(N_OBJ, 2)) * np.stack([w, h], 1)
        # 검출: 참 sigma 로 흔들린다
        dpe = rng.normal(0.0, 1.0, size=(N_OBJ, 2)) * sig_true[:, None]

        tracks, dets = [], []
        for i in range(N_OBJ):
            # 트랙이 물고 있는 분산도 검출에서 온 것이다 (UTrack 규약)
            v = np.array([sig_rep[i] ** 2, sig_rep[i] ** 2,
                          0.5 * sig_rep[i] ** 2, 0.5 * sig_rep[i] ** 2])
            tx, ty = cx[i] + tpe[i, 0], cy[i] + tpe[i, 1]
            tracks.append(Box([tx - w[i] / 2, ty - h[i] / 2,
                               tx + w[i] / 2, ty + h[i] / 2], v))
            dx, dy = cx[i] + dpe[i, 0], cy[i] + dpe[i, 1]
            dets.append(Box([dx - w[i] / 2, dy - h[i] / 2,
                             dx + w[i] / 2, dy + h[i] / 2], v))
        for _ in range(N_FALSE):
            fh = rng.uniform(60, 260)
            fw = fh * rng.uniform(0.32, 0.48)
            fx = rng.uniform(40, FIELD[0] - 40)
            fy = rng.uniform(40, FIELD[1] - 40)
            fs = K_SIGMA * fh * (rng.lognormal(0.0, 0.75) if world != 'W1' else 1.0)
            v = np.array([fs ** 2, fs ** 2, 0.5 * fs ** 2, 0.5 * fs ** 2])
            dets.append(Box([fx - fw / 2, fy - fh / 2,
                             fx + fw / 2, fy + fh / 2], v))
        frames.append((tracks, dets, N_OBJ))
    return frames


def recovery(mod, frames):
    """정답쌍 복원율. 헝가리안 + 임계값 (실제 트래커와 같은 절차)."""
    hit = tot = 0
    for tracks, dets, n_true in frames:
        cost, _ = mod.relaxed_iou_distance(tracks, dets)
        r, c = linear_sum_assignment(cost)
        keep = cost[r, c] <= GATE
        hit += int(np.count_nonzero((r[keep] == c[keep]) & (r[keep] < n_true)))
        tot += n_true
    return hit / tot


def sigma_pool(frames):
    """measure 갈래가 남길 통계와 같은 것을 직접 만든다."""
    m = load(RELAX_MODE='measure', RELAX_RESERVOIR=10 ** 7)
    for tracks, dets, _ in frames:
        m.relaxed_iou_distance(tracks, dets)
    smp = np.asarray(m._res, dtype=float)
    return smp[:, 0], smp[:, 1], smp[:, 2], smp[:, 3]


def main():
    cap = 1.0
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    print('gate %.1f, %d frames, %d obj + %d false per frame'
          % (GATE, N_FRAMES, N_OBJ, N_FALSE))
    print('')

    verdicts = {}
    for world, label in [('W1', 'size-only  (sigma = k*h)'),
                         ('W2', 'informative(sigma = k*h*u, true too)'),
                         ('W3', 'miscalib   (sigma = k*h*u, true = k*h)')]:
        frames = make_world(world, seed=hash(world) % 10000)
        sx, sy, w, h = sigma_pool(frames)
        e_w, e_h = w.mean(), h.mean()

        base = recovery(load(RELAX_MODE='measure'), frames)
        print('== %s  %s' % (world, label))
        print('   corr(s_x, w) = %+.3f   CV(s_x) = %.3f   baseline %.4f'
              % (float(np.corrcoef(sx, w)[0, 1]), sx.std() / sx.mean(), base))
        print('   %-13s %-8s %-8s %-8s %-8s %s'
              % ('', 'R', 'K1 const', 'K2 prop', 'K3 shuf', 'K4 ratioshuf'))

        best = None
        for al in alphas:
            tx = calibrate.mean_pad_sigma(sx, w, al, cap)
            ty = calibrate.mean_pad_sigma(sy, h, al, cap)
            dx, _ = calibrate.solve_const(w, tx, cap)
            dy, _ = calibrate.solve_const(h, ty, cap)
            r = recovery(load(RELAX_MODE='sigma', RELAX_ALPHA=al), frames)
            k1 = recovery(load(RELAX_MODE='const', RELAX_DX=dx, RELAX_DY=dy), frames)
            k2 = recovery(load(RELAX_MODE='prop', RELAX_CW=min(tx / e_w, cap),
                               RELAX_CH=min(ty / e_h, cap)), frames)
            k3 = recovery(load(RELAX_MODE='shuffle', RELAX_ALPHA=al), frames)
            k4 = recovery(load(RELAX_MODE='ratio_shuffle', RELAX_ALPHA=al), frames)
            print('   alpha=%-7g %-8.4f %-8.4f %-8.4f %-8.4f %.4f'
                  % (al, r, k1, k2, k3, k4))
            if best is None or r > best[0][1]:
                best = ((al, r, k1, k2, k3, k4),)

        al, r, k1, k2, k3, k4 = best[0]
        verdicts[world] = {'alpha': al, 'K1': r - k1, 'K2': r - k2,
                           'K3': r - k3, 'K4': r - k4}
        print('   -> best alpha %g :  R-K1 %+.4f  R-K2 %+.4f  R-K3 %+.4f  R-K4 %+.4f'
              % (al, r - k1, r - k2, r - k3, r - k4))
        print('')

    print('== 검정력 판정 ==')
    print('        %-11s %-11s %-11s %s'
          % ('R-K1', 'R-K2', 'R-K3', 'R-K4 <- 판정'))
    for wname in ('W1', 'W2', 'W3'):
        v = verdicts[wname]
        print('   %s   %+-11.4f %+-11.4f %+-11.4f %+.4f'
              % (wname, v['K1'], v['K2'], v['K3'], v['K4']))

    inf4, size4, mis4 = (verdicts[w]['K4'] for w in ('W2', 'W1', 'W3'))
    sep = inf4 - max(size4, mis4)
    print('')
    print('   판정은 **R-K4 (비율 뒤섞기)** 로 한다.')
    print('   K1/K2 는 확장량의 분포까지 바꾸고, K3 는 크기와의 연결까지 깬다.')
    print('   K4 만이 크기는 남기고 **크기로 설명 안 되는 부분의 짝만** 깬다.')
    print('   W2(진짜 정보) R-K4 = %+.4f' % inf4)
    print('   W1/W3(정보 없음) R-K4 = %+.4f / %+.4f' % (size4, mis4))
    print('   분리폭 = %+.4f' % sep)
    if inf4 > 0.005 and sep > 0.005:
        print('   OK  설계에 구별력이 있다. 콜랩에서 R-K4 를 주 지표로 볼 것.')
    else:
        print('   NO  정보가 있는 세계에서도 R 이 K4 를 못 이긴다.')
        print('       이 설계로는 아무것도 못 가른다. 콜랩 전에 고칠 것.')
    print('')
    print('합성이다. MOT17 에서 무엇이 나올지는 말하지 않는다.')
    print('말하는 것은 "이 실험이 무언가를 구별할 수 있는가" 뿐이다.')


if __name__ == '__main__':
    main()
