"""
실험 3 - 사전 검정력 확인. **콜랩을 돌리기 전에 이걸 먼저 본다.**

설계가 아무리 깔끔해도, 조건들이 원리적으로 안 갈리면 콜랩 시간은 낭비다.
그래서 **답을 아는 합성 세계**에서 먼저 시험한다.

세 세계를 만든다. 셋 다 검출기가 sigma 를 뱉지만 그 sigma 의 지위가 다르다.

  W1 size-only   sigma = k*h.               참오차도 그 sigma. 크기가 전부다
  W2 informative sigma = k*h*u.             참오차도 그 sigma. u 가 진짜 정보다
  W3 miscalib    sigma = k*h*u 라고 보고.   참오차는 k*h. u 는 잡음이다

**세 세계 중 어디가 실측인지는 이제 다르게 답해야 한다.** 처음에는 실험 1 의
편상관이 +0.044 라 W3(u 는 순전한 잡음) 을 실측 세계로 적었다. **그 +0.044 는
NMS in-place 버그의 산물이고 고친 값은 +0.32 다.** 즉 u 에는 실제로 신호가 있고,
실측 세계는 W3 보다 **W2 에 가깝다.** 다만 스케일이 24배 어긋나 있으므로 (exp01)
W2 도 아니다 -- 굳이 말하면 'W2 의 방향에 W3 의 스케일'이다.
이 스크립트가 답하는 것은 여전히 **설계의 구별력** 하나뿐이라 그 정정에
영향받지 않지만, W3 을 '실측 세계' 라고 읽으면 안 된다.

각 세계에서 네 조건을 돌려 **정답쌍 복원율**을 잰다.

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
PRED_ERR = 0.30     # 칼만 예측오차 (박스 크기 대비)
FIELD = (960, 540)  # 좁은 화면. 넓게 뿌리면 매칭이 자명해져 천장에 붙는다

# 세계별 난수 씨앗. **hash(world) 를 쓰면 안 된다** -- 파이썬 문자열 해시는
# 프로세스마다 무작위화되므로(PYTHONHASHSEED) 돌릴 때마다 다른 세계가 만들어지고
# 보고한 숫자를 아무도 재현할 수 없다. 실제로 그렇게 되어 있었다 (2026-08-17 발견).
SEEDS = {'W1': 101, 'W2': 202, 'W3': 303}

# 천장에 붙으면(복원율 ~1.0) 어떤 조건도 차이를 못 낸다. 첫 판에서 그렇게 나와
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


def make_world(world, seed=0, pair_consistent=True):
    """프레임 목록을 만든다. 각 프레임은 (tracks, dets, n_true).

    pair_consistent=False 면 트랙이 물고 있는 sigma 를 다른 개체의 것으로
    바꾼다 (분포는 그대로, 짝만 깬다).

    **왜 이 스위치가 필요한가.** UTrack 규약상 트랙은 마지막으로 매칭된 검출의
    var 를 물려받으므로 정답쌍은 같은 sigma 를 공유한다. 그러면 R 은 트랙과
    검출을 **같은 만큼** 키워 기하를 보존하는데, 뒤섞기 조건(K3/K4)는 양쪽에
    다른 순열을 걸어 그 보존이 깨진다. 그래서 R > K3/K4 가 'sigma 에 정보가
    있어서' 인지 '짝이 일관돼서' 인지 갈리지 않는다.

    이 스위치를 끄면 R 도 짝 일관성을 잃는다. 그때 R 의 우위가 사라지면
    그 우위는 정보가 아니라 짝 일관성이었다는 뜻이다.

    이 통제군의 결과는 예전부터 인용돼 왔는데(box_relax.py 독스트링) **코드에는
    없었다.** 손으로 한 번 돌리고 커밋하지 않은 것이다. 여기 못박는다. (2026-08-17)
    """
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

        # 트랙이 물고 있는 분산도 검출에서 온 것이다 (UTrack 규약).
        # 짝을 깰 때는 분포는 그대로 두고 어느 트랙이 어느 sigma 를 무는지만 섞는다.
        sig_trk = sig_rep if pair_consistent else sig_rep[rng.permutation(N_OBJ)]

        tracks, dets = [], []
        for i in range(N_OBJ):
            def _v(s):
                return np.array([s ** 2, s ** 2, 0.5 * s ** 2, 0.5 * s ** 2])
            tx, ty = cx[i] + tpe[i, 0], cy[i] + tpe[i, 1]
            tracks.append(Box([tx - w[i] / 2, ty - h[i] / 2,
                               tx + w[i] / 2, ty + h[i] / 2], _v(sig_trk[i])))
            dx, dy = cx[i] + dpe[i, 0], cy[i] + dpe[i, 1]
            dets.append(Box([dx - w[i] / 2, dy - h[i] / 2,
                             dx + w[i] / 2, dy + h[i] / 2], _v(sig_rep[i])))
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
    """measure 조건이 남길 통계와 같은 것을 직접 만든다."""
    m = load(RELAX_MODE='measure', RELAX_RESERVOIR=10 ** 7)
    for tracks, dets, _ in frames:
        m.relaxed_iou_distance(tracks, dets)
    smp = np.asarray(m._res, dtype=float)
    return smp[:, 0], smp[:, 1], smp[:, 2], smp[:, 3]


def sweep_world(frames, alphas, cap, verbose=True):
    """한 세계에서 다섯 조건을 alpha 별로 돌리고 best alpha 의 차이를 낸다.

    best alpha 는 **R 의 복원율이 가장 높은** alpha 다. 검정력 확인이므로
    R 에게 최선을 주는 것이 맞다 -- 최선을 줘도 못 가르면 설계가 죽은 것이다.
    """
    sx, sy, w, h = sigma_pool(frames)
    e_w, e_h = w.mean(), h.mean()
    base = recovery(load(RELAX_MODE='measure'), frames)
    if verbose:
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
        if verbose:
            print('   alpha=%-7g %-8.4f %-8.4f %-8.4f %-8.4f %.4f'
                  % (al, r, k1, k2, k3, k4))
        if best is None or r > best[1]:
            best = (al, r, k1, k2, k3, k4)

    al, r, k1, k2, k3, k4 = best
    if verbose:
        print('   -> best alpha %g :  R-K1 %+.4f  R-K2 %+.4f  R-K3 %+.4f  R-K4 %+.4f'
              % (al, r - k1, r - k2, r - k3, r - k4))
    return {'alpha': al, 'base': base, 'R': r,
            'K1': r - k1, 'K2': r - k2, 'K3': r - k3, 'K4': r - k4}


def main():
    cap = 1.0
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    print('gate %.1f, %d frames, %d obj + %d false per frame'
          % (GATE, N_FRAMES, N_OBJ, N_FALSE))
    print('seeds %s (hash() 는 프로세스마다 달라서 쓰면 안 된다)' % SEEDS)
    print('')

    WORLDS = [('W1', 'size-only  (sigma = k*h)'),
              ('W2', 'informative(sigma = k*h*u, true too)'),
              ('W3', 'miscalib   (sigma = k*h*u, true = k*h)')]

    verdicts, broken = {}, {}
    for world, label in WORLDS:
        print('== %s  %s' % (world, label))
        verdicts[world] = sweep_world(
            make_world(world, seed=SEEDS[world]), alphas, cap)
        print('')

    # 짝 일관성을 깬 통제군. 트랙 sigma 만 다른 개체의 것으로 바꾼다.
    print('== 짝 일관성 제거 통제군 (트랙 sigma 를 개체 사이에서 섞는다) ==')
    for world, _ in WORLDS:
        broken[world] = sweep_world(
            make_world(world, seed=SEEDS[world], pair_consistent=False),
            alphas, cap, verbose=False)
        print('   %s  R-K1 %+.4f  R-K2 %+.4f  R-K3 %+.4f  R-K4 %+.4f'
              % (world, broken[world]['K1'], broken[world]['K2'],
                 broken[world]['K3'], broken[world]['K4']))
    print('')

    print('== 검정력 판정 ==')
    print('   분리폭 = W2(정보 있음) - max(W1, W3)(정보 없음).')
    print('   조건이 정보에만 반응하면 크고, 짝 일관성 같은 부산물에 반응하면 0 이다.')
    print('')
    print('   %-6s %-11s %-11s %-11s %-11s' % ('', 'W1', 'W2', 'W3', '분리폭'))
    print('   ' + '-' * 56)
    seps = {}
    for arm in ('K1', 'K2', 'K3', 'K4'):
        w1, w2, w3 = (verdicts[w][arm] for w in ('W1', 'W2', 'W3'))
        seps[arm] = w2 - max(w1, w3)
        print('   R-%-4s %+-11.4f %+-11.4f %+-11.4f %+-11.4f'
              % (arm, w1, w2, w3, seps[arm]))

    print('')
    print('   짝 일관성을 깨면 (같은 표, 트랙 sigma 만 섞음):')
    for arm in ('K1', 'K2', 'K3', 'K4'):
        w1, w2, w3 = (broken[w][arm] for w in ('W1', 'W2', 'W3'))
        print('   R-%-4s %+-11.4f %+-11.4f %+-11.4f %+-11.4f'
              % (arm, w1, w2, w3, w2 - max(w1, w3)))

    print('')
    print('   **판정 조건은 K2 다.** K3/K4 는 진단으로만 쓴다.')
    print('   뒤섞기 조건은 트랙과 검출에 서로 다른 순열을 걸어 정답쌍의 확장량이')
    print('   어긋나게 만든다. R 은 (UTrack 규약상 트랙이 검출의 var 를 물려받으므로)')
    print('   양쪽을 같은 만큼 키워 기하를 보존한다. 그래서 R > K3/K4 는 sigma 에')
    print('   정보가 없어도 나온다 -- 위 W3 열과 짝 일관성 제거 표가 그 증거다.')
    print('   K2 는 양쪽에 같은 규칙(크기 비례)을 걸어 그 부산물이 없다.')
    print('')
    print('   K2 분리폭 = %+.4f   K4 분리폭 = %+.4f' % (seps['K2'], seps['K4']))
    if seps['K2'] > 0.005:
        print('   OK  K2 로 정보 있는 세계와 없는 세계가 갈린다.')
    else:
        print('   NO  K2 로도 안 갈린다. 이 설계로는 못 가른다.')
    print('')
    print('   * 분리폭은 단일 실행 값이고 오차막대가 없다. 0.01 미만이면')
    print('     몬테카를로 잡음과 구별되지 않는다고 보는 것이 안전하다.')
    print('     여러 씨앗으로 돌려 산포를 보기 전에는 크기를 주장하지 말 것.')
    print('')
    print('합성이다. MOT17 에서 무엇이 나올지는 말하지 않는다.')
    print('말하는 것은 "이 실험이 무언가를 구별할 수 있는가" 뿐이다.')


if __name__ == '__main__':
    main()
