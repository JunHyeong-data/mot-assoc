"""
실험 3 - 콜랩에 올리기 전 자체 시험.

콜랩 시간이 아깝다. 논리 오류는 여기서 잡는다. numpy 만 있으면 돌아간다.

    python selftest.py

확인하는 것:
  [1] measure 조건이 맨 IoU 와 **완전히** 같은가 (훅이 조용히 바꾸지 않는가)
  [2] 확장이 정답쌍 비용을 낮추고 게이트를 여는가. APPLY=det 과 both 를 대조한다
  [3] calibrate 가 낸 상수가 세 조건의 평균 확장량을 실제로 맞추는가
  [4] 상한(CAP)이 걸리는가
  [5] 분산이 0 인 박스에서 R 이 기준선으로 되돌아가는가

출력은 ASCII 로만 쓴다 (Windows cp949 콘솔에서 죽지 않게).
"""

import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATE = 0.8          # UTrack ablation_17 의 1단계 matching distance


class Box:
    """box_relax 가 요구하는 최소 규약: .tlbr 과 .var_xywh."""

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


def scene(seed=0, n_t=40, n_extra=12):
    """실제 연관이 보는 것과 같은 배치를 만든다.

    검출은 트랙 예측 근처에 흩어져 있어야 한다. 무작위로 뿌리면 겹치는 칸이 없어
    확장의 효과를 못 본다. 앞의 n_t 개는 트랙 i 와 검출 i 가 정답쌍이다.
    크기가 다양해야 K2(크기 비례)가 의미를 갖는다.
    """
    def var_for(box_h):
        # 분산은 크기에 대체로 비례하고(실험 1) 그 위에 흩어짐이 있다
        base = (box_h / 40.0) ** 2
        return base * rng.lognormal(0.0, 0.6, size=4) * [1.0, 1.0, 0.5, 0.5]

    rng = np.random.default_rng(seed)
    tracks, dets = [], []
    for _ in range(n_t):
        w, h = rng.uniform(20, 90), rng.uniform(50, 220)
        x, y = rng.uniform(0, 1900), rng.uniform(0, 1000)
        # 트랙도 분산을 갖는다. UTrack 에서 트랙의 `_var_xywh` 는 마지막으로 매칭된
        # 검출의 것으로 갱신되므로 0 이 아니다. 여기서 0 을 주면 APPLY=both 가
        # 조용히 무력화되어 시험이 거짓 통과한다 (실제로 그렇게 한 번 틀렸다).
        tracks.append(Box([x, y, x + w, y + h], var_for(h)))
        # 대응 검출: 박스 크기의 몇 십 % 만큼 어긋나 있다 (칼만 예측 오차)
        off = rng.normal(0.0, 0.22, size=2) * np.array([w, h])
        dw, dh = w * rng.uniform(0.9, 1.1), h * rng.uniform(0.9, 1.1)
        dx, dy = x + off[0], y + off[1]
        dets.append(Box([dx, dy, dx + dw, dy + dh], var_for(dh)))
    for _ in range(n_extra):                        # 짝 없는 검출 (오검출/신규)
        w, h = rng.uniform(20, 90), rng.uniform(50, 220)
        x, y = rng.uniform(0, 1900), rng.uniform(0, 1000)
        dets.append(Box([x, y, x + w, y + h], var_for(h)))
    return tracks, dets


def mean_pad(mod, dets):
    tlbr = np.asarray([d.tlbr for d in dets], dtype=float)
    sx, sy, _ = mod._edge_sigma([d.var_xywh for d in dets])
    px, py, _, _, _ = mod._pads(tlbr, sx, sy)
    return px.mean(), py.mean()


def true_pair_cost(cost, n_t):
    return float(np.diagonal(cost[:n_t, :n_t]).mean())


def main():
    ok = True
    n_t = 40
    tracks, dets = scene(n_t=n_t)

    # ---- [1] measure == 맨 IoU -------------------------------------------
    m = load(RELAX_MODE='measure')
    cost, _ = m.relaxed_iou_distance(tracks, dets)
    plain = 1.0 - m._numpy_ious([t.tlbr for t in tracks], [d.tlbr for d in dets])
    d1 = np.abs(cost - plain).max()
    print('[1] measure vs plain IoU   max|diff| = %.3e' % d1)
    if d1 != 0.0:
        ok = False
        print('    FAIL measure changed the cost')

    # ---- [2] 확장이 실제로 완화인가 ---------------------------------------
    # 전체 평균 비용은 멀리 있는 쌍(비용 1)이 지배해서 지표가 안 된다.
    # 정답쌍 비용과 게이트 통과율로 본다.
    base_true = true_pair_cost(cost, n_t)
    base_gate = 100.0 * (cost <= GATE).mean()
    print('[2] relaxation check   (true-pair cost / gate rate at %.1f)' % GATE)
    print('    baseline          %.5f / %5.2f%%' % (base_true, base_gate))
    for apply_to in ('det', 'both'):
        prev = base_true
        worsened = False
        line = []
        for al in (0.5, 1.0, 2.0, 4.0):
            m = load(RELAX_MODE='sigma', RELAX_ALPHA=al, RELAX_APPLY=apply_to)
            c, _ = m.relaxed_iou_distance(tracks, dets)
            tp = true_pair_cost(c, n_t)
            gt = 100.0 * (c <= GATE).mean()
            line.append('a=%.1f %.5f/%5.2f%%' % (al, tp, gt))
            if tp > prev + 1e-12:
                worsened = True
            prev = tp
        print('    APPLY=%-5s %s' % (apply_to, '  '.join(line)))
        if apply_to == 'both' and worsened:
            ok = False
            print('    FAIL APPLY=both is not a relaxation')
        if apply_to == 'det' and not worsened:
            print('    NOTE APPLY=det happened to be monotone in this scene;')
            print('         it is not guaranteed to be. Default stays both.')

    # ---- [3] 확장량 일치 - 이 실험의 핵심 ---------------------------------
    # calibrate.py 와 **같은 함수**로 상수를 풀어야 의미가 있다. 여기서 따로
    # 계산하면 calibrate 의 버그를 못 잡는다.
    import calibrate

    m = load(RELAX_MODE='measure')
    m.relaxed_iou_distance(tracks, dets)
    smp = np.asarray(m._res, dtype=float)
    sx, sy, w, h = smp[:, 0], smp[:, 1], smp[:, 2], smp[:, 3]
    e_w, e_h = w.mean(), h.mean()
    cap = 1.0
    print('[3] matched mean expansion via calibrate.py (E[s_x]=%.3f px, E[w]=%.1f px)'
          % (sx.mean(), e_w))
    for al in (1.0, 3.0, 12.0):                 # 12 는 상한이 실제로 걸리는 영역
        tx = calibrate.mean_pad_sigma(sx, w, al, cap)
        ty = calibrate.mean_pad_sigma(sy, h, al, cap)
        dx, okx = calibrate.solve_const(w, tx, cap)
        dy, oky = calibrate.solve_const(h, ty, cap)
        mods = [
            load(RELAX_MODE='sigma', RELAX_ALPHA=al, RELAX_CAP=cap),
            load(RELAX_MODE='const', RELAX_DX=dx, RELAX_DY=dy, RELAX_CAP=cap),
            load(RELAX_MODE='prop', RELAX_CW=min(tx / e_w, cap),
                 RELAX_CH=min(ty / e_h, cap), RELAX_CAP=cap),
        ]
        pads = [mean_pad(mod, dets) for mod in mods]
        xs = [p[0] for p in pads]
        ys = [p[1] for p in pads]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        tol = 1e-3 * max(max(xs), 1.0)          # 이분법 오차 + 표본 오차
        flag = '' if spread < tol else '   FAIL expansion not matched'
        if flag or not (okx and oky):
            ok = False
        print('    alpha=%-4g R=(%.3f,%.3f)  K1=(%.3f,%.3f)  K2=(%.3f,%.3f)%s'
              % (al, xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], flag))

    # 조건이 실제로 다른 비용을 내는가. 같으면 실험 자체가 성립하지 않는다.
    al = 2.0
    tx = calibrate.mean_pad_sigma(sx, w, al, cap)
    ty = calibrate.mean_pad_sigma(sy, h, al, cap)
    dx, _ = calibrate.solve_const(w, tx, cap)
    dy, _ = calibrate.solve_const(h, ty, cap)
    cr = load(RELAX_MODE='sigma', RELAX_ALPHA=al
              ).relaxed_iou_distance(tracks, dets)[0]
    c1 = load(RELAX_MODE='const', RELAX_DX=dx, RELAX_DY=dy
              ).relaxed_iou_distance(tracks, dets)[0]
    c2 = load(RELAX_MODE='prop', RELAX_CW=tx / e_w, RELAX_CH=ty / e_h
              ).relaxed_iou_distance(tracks, dets)[0]
    print('    arm-to-arm cost gap  |R-K1|=%.4f  |R-K2|=%.4f  |K1-K2|=%.4f'
          % (np.abs(cr - c1).max(), np.abs(cr - c2).max(), np.abs(c1 - c2).max()))
    if np.abs(cr - c1).max() < 1e-9:
        ok = False
        print('    FAIL R and K1 are indistinguishable')

    # ---- [3b] 트랙 쪽도 맞는가 - 여태 검증 안 된 절반 ----------------------
    # calibrate.py 는 **검출 쪽** 표본으로만 상수를 푼다. 그런데 APPLY=both 라
    # pad 는 트랙에도 붙는다. 트랙 sigma/크기 분포가 검출과 다르면 검출 쪽만
    # 맞고 트랙 쪽은 어긋난다 -> 조건들이 '확장량 일치' 가 아니게 된다.
    # 이 실험의 전부가 확장량 일치에 걸려 있으므로 반드시 확인해야 한다.
    print('[3b] track-side expansion (calibrate 는 검출 쪽만 보고 상수를 푼다)')
    al = 2.0
    tx = calibrate.mean_pad_sigma(sx, w, al, cap)
    ty = calibrate.mean_pad_sigma(sy, h, al, cap)
    dx, _ = calibrate.solve_const(w, tx, cap)
    dy, _ = calibrate.solve_const(h, ty, cap)
    arms = (('R ', dict(RELAX_MODE='sigma', RELAX_ALPHA=al)),
            ('K1', dict(RELAX_MODE='const', RELAX_DX=dx, RELAX_DY=dy)),
            ('K2', dict(RELAX_MODE='prop', RELAX_CW=min(tx / e_w, cap),
                        RELAX_CH=min(ty / e_h, cap))))
    det_means, trk_means = [], []
    for name, env in arms:
        m = load(**env)
        m.relaxed_iou_distance(tracks, dets)
        a, at = m._acc, m._acc_t
        dpx = a['sum_pad_x'] / max(a['n_boxes'], 1)
        tpx = at['sum_pad_x'] / max(at['n_boxes'], 1)
        det_means.append(dpx)
        trk_means.append(tpx)
        print('    %s  검출 pad_x %.4f   트랙 pad_x %.4f   트랙/검출 %.3f'
              % (name, dpx, tpx, tpx / max(dpx, 1e-9)))
    d_spread = max(det_means) - min(det_means)
    t_spread = max(trk_means) - min(trk_means)
    # 검출 쪽 편차는 calibrate 가 0 으로 맞춰 주므로 비율을 찍으면 무의미하다.
    print('    조건 간 편차   검출 %.5f (calibrate 가 맞춘 쪽)   트랙 %.5f'
          % (d_spread, t_spread))
    print('    R 대 K2 트랙 pad 차이 %.2f%%  <- 판정 조건끼리의 실제 어긋남'
          % (100 * abs(trk_means[0] - trk_means[2]) / max(trk_means[0], 1e-9)))
    # 검출 쪽은 calibrate 가 맞춰 주므로 0 에 가깝다. 트랙 쪽이 그보다 크게
    # 벌어지면 '확장량 일치' 가 트랙에서는 성립하지 않는다는 뜻이다.
    if t_spread > 0.05 * max(trk_means):
        print('    WARN 트랙 쪽 확장량이 조건마다 %.1f%% 까지 어긋난다.'
              % (100 * t_spread / max(max(trk_means), 1e-9)))
        print('         이 크기를 결과 해석에 명시할 것 (실패는 아니다).')
    else:
        print('    OK 트랙 쪽도 조건 간 편차가 5%% 미만이다.')

    # ---- [4] 상한 --------------------------------------------------------
    m = load(RELAX_MODE='sigma', RELAX_ALPHA=50.0, RELAX_CAP=0.25)
    tlbr = np.asarray([d.tlbr for d in dets], dtype=float)
    sx, sy, _ = m._edge_sigma([d.var_xywh for d in dets])
    px, py, w, h, nclip = m._pads(tlbr, sx, sy)
    over = int(np.count_nonzero((px > 0.25 * w + 1e-9) | (py > 0.25 * h + 1e-9)))
    print('[4] CAP=0.25, alpha=50 -> clipped %d/%d, still over cap %d'
          % (nclip, len(dets), over))
    if over:
        ok = False
        print('    FAIL cap not applied')

    # ---- [5] 분산 0 이면 기준선으로 복귀 -----------------------------------
    # APPLY=both 이므로 트랙 쪽 분산도 0 으로 둬야 한다. 검출만 0 으로 두면
    # 트랙이 확장돼서 이 시험이 엉뚱한 것을 잰다.
    zero_t = [Box(t.tlbr, [0, 0, 0, 0]) for t in tracks]
    zero = [Box(d.tlbr, [0, 0, 0, 0]) for d in dets]
    m = load(RELAX_MODE='sigma', RELAX_ALPHA=3.0)
    c, _ = m.relaxed_iou_distance(zero_t, zero)
    base = 1.0 - m._numpy_ious([t.tlbr for t in zero_t], [d.tlbr for d in zero])
    d5 = np.abs(c - base).max()
    print('[5] R with var=0 vs baseline  max|diff| = %.3e' % d5)
    if d5 != 0.0:
        ok = False
        print('    FAIL expands with no variance')

    print('')
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
