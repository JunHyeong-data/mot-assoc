"""
실험 3 - 확장량을 맞춘다.

measure 실행이 남긴 통계에서, 각 alpha 마다 세 갈래의 **평균 확장량이 같아지도록**
상수를 낸다. 이걸 안 맞추면 갈래 사이의 차이가 "정보가 있었나" 가 아니라
"더 키웠나" 가 되어버린다. 이 실험의 전부가 여기에 걸려 있다.

    R  sigma : pad = min(alpha * s,  CAP * w)
    K1 const : pad = min(DX,         CAP * w)     DX 를 풀어서 평균을 맞춘다
    K2 prop  : pad = min(CW * w,     CAP * w)     CW = 목표평균 / E[w]

상한(CAP)이 걸리면 평균이 alpha 에 선형이 아니라서 E[s] 만으로는 못 맞춘다.
그래서 measure 가 남긴 저수지 표본으로 평균을 직접 계산하고, K1 은 이분법으로 푼다.

    python calibrate.py stats/*.json --alphas 0.5 1 2 3 5 --cap 1.0
"""

import argparse
import json
import sys

import numpy as np

# Windows 기본 콘솔(cp949)에서 죽지 않게. 실행은 주로 Colab(UTF-8)이지만
# 로컬에서 통계만 다시 볼 일이 있다.
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass


def load(paths):
    rows, n_total, sums = [], 0.0, {}
    caps = set()
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            s = json.load(f)
        n = s.get('n_boxes', 0.0)
        if not n:
            print('  skip %s (n_boxes=0)' % p)
            continue
        n_total += n
        for k, v in s.items():
            if k.startswith(('sum_', 'n_')) and isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
        res = np.asarray(s.get('reservoir', []), dtype=float).reshape(-1, 4)
        if res.shape[0]:
            # 시퀀스마다 표본 수가 다르다. 본 상자 수에 비례하도록 가중 재표집한다.
            rows.append((res, n))
        caps.add(s.get('_meta', {}).get('cap'))
    if not rows:
        sys.exit('ERROR no usable stats in %s' % list(paths))
    return rows, n_total, sums, caps


def pooled_sample(rows, n_total, size=200000, seed=0):
    """시퀀스별 표본을 '본 상자 수' 비율로 합쳐 하나의 표본을 만든다."""
    rng = np.random.default_rng(seed)
    out = []
    for res, n in rows:
        take = max(int(round(size * n / n_total)), 1)
        idx = rng.integers(0, res.shape[0], size=take)
        out.append(res[idx])
    return np.concatenate(out, axis=0)


def mean_pad_sigma(s, w, alpha, cap):
    return float(np.minimum(alpha * s, cap * w).mean())


def solve_const(w, target, cap):
    """E[min(D, cap*w)] = target 인 D. 단조라서 이분법으로 푼다."""
    hi = float(np.max(cap * w))
    if mean_pad_sigma(np.full_like(w, hi), w, 1.0, cap) < target - 1e-12:
        return hi, False                       # 상한 때문에 도달 불가
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.minimum(mid, cap * w).mean()) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stats', nargs='+', help='RELAX_STATS 로 만든 json 들')
    ap.add_argument('--alphas', nargs='+', type=float,
                    default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument('--cap', type=float, default=None,
                    help='실행에 쓸 RELAX_CAP. 생략하면 measure 때 값을 쓴다')
    a = ap.parse_args()

    rows, n_total, sums, caps = load(a.stats)
    cap = a.cap if a.cap is not None else (
        [c for c in caps if c is not None] or [1.0])[0]

    smp = pooled_sample(rows, n_total)
    sx, sy, w, h = smp[:, 0], smp[:, 1], smp[:, 2], smp[:, 3]
    e_sx, e_sy, e_w, e_h = sx.mean(), sy.mean(), w.mean(), h.mean()
    zero_frac = sums.get('n_zero_var', 0.0) / n_total

    print('== measure pass ==')
    print('  files %d   boxes seen in association %d   calls %d'
          % (len(a.stats), n_total, sums.get('n_calls', 0)))
    print('  pooled sample %d rows   CAP = %.3f' % (smp.shape[0], cap))
    print('  E[s_x] %.4f px (sd %.4f)   E[s_y] %.4f px (sd %.4f)'
          % (e_sx, sx.std(), e_sy, sy.std()))
    print('  E[w]   %.2f px             E[h]   %.2f px' % (e_w, e_h))
    print('  s_x/w  mean %.4f   s_y/h mean %.4f'
          % (float((sx / w).mean()), float((sy / h).mean())))
    print('  boxes with zero variance : %.1f%%' % (100.0 * zero_frac))

    if zero_frac > 0.5:
        print('')
        print('  !! 절반 넘는 상자에 분산이 안 실렸다. nms_var 가 안 붙었거나')
        print('     검출 경로가 분산을 안 넘긴다. 이대로면 R 은 의미가 없다.')
    if e_sx <= 0 or e_sy <= 0:
        sys.exit('ERROR E[s] is 0 -- 확장이 아예 안 일어난다. 위 경고부터 해결할 것')

    cv_x, cv_y = sx.std() / e_sx, sy.std() / e_sy
    print('')
    print('  CV(s_x) = %.3f   CV(s_y) = %.3f' % (cv_x, cv_y))
    print('  CV 가 0 에 가까우면 R 과 K1 은 원리적으로 구별되지 않는다.')
    print('  그 경우 HOTA 가 아니라 이 숫자를 보고해야 한다.')
    # s 가 크기로 얼마나 설명되는가. 실험 1 은 이것이 거의 전부라고 했다.
    r_sw = float(np.corrcoef(sx, w)[0, 1])
    r_sh = float(np.corrcoef(sy, h)[0, 1])
    print('  corr(s_x, w) = %+.3f   corr(s_y, h) = %+.3f' % (r_sw, r_sh))
    print('  이 상관이 높으면 K2 가 R 을 거의 재현할 것으로 예상된다 (실험 1 과 일치).')

    print('')
    print('== 갈래별 환경변수 (평균 확장량 일치, CAP=%.3f 반영) ==' % cap)
    for al in a.alphas:
        tx = mean_pad_sigma(sx, w, al, cap)
        ty = mean_pad_sigma(sy, h, al, cap)
        dx, ok_x = solve_const(w, tx, cap)
        dy, ok_y = solve_const(h, ty, cap)
        cw = min(tx / e_w, cap)
        ch = min(ty / e_h, cap)
        clip = 100.0 * float(np.mean((al * sx > cap * w) | (al * sy > cap * h)))

        print('')
        print('-- alpha = %g   target mean pad = %.3f px (x) / %.3f px (y)'
              '   R clipped %.1f%%' % (al, tx, ty, clip))
        if not (ok_x and ok_y):
            print('   !! CAP 때문에 K1 이 목표 평균에 도달 못 한다. CAP 를 올릴 것')
        print('   R  RELAX_MODE=sigma RELAX_ALPHA=%g' % al)
        print('   K1 RELAX_MODE=const RELAX_DX=%.6f RELAX_DY=%.6f' % (dx, dy))
        print('   K2 RELAX_MODE=prop  RELAX_CW=%.6f RELAX_CH=%.6f' % (cw, ch))
        # 확인: 세 갈래의 평균이 실제로 같은가
        m1 = float(np.minimum(dx, cap * w).mean())
        m2 = float(np.minimum(cw * w, cap * w).mean())
        print('   check mean pad x :  R %.4f   K1 %.4f   K2 %.4f' % (tx, m1, m2))

    print('')
    print('세 갈래 모두 RELAX_APPLY=both, RELAX_CAP=%.3f 로 같게 둘 것.' % cap)


if __name__ == '__main__':
    main()
