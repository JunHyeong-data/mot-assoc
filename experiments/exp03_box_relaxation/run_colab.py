"""
실험 3 - Colab 실행기. 셀에 긴 코드를 붙여넣지 않기 위한 것.

`track.py` 는 한 번 돌 때마다 (1) 결과 폴더를 새로 만들고 (2) 추적하고
(3) 평가까지 한다. 그런데 시퀀스를 통합해 돌리면 MOT17-11 구간에서
17초/프레임으로 붕괴한다(exp02 실측). 그래서 **시퀀스별로 돌리고 결과를
갈래별 폴더에 모은 뒤 마지막에 한 번만 평가**해야 한다. 그 잔손질을 여기서 한다.

단계:

    python run_colab.py measure     A 갈래. 확장 0. 기준선과 같아야 한다 + 통계 수집
    python run_colab.py calibrate   통계 -> 갈래별 상수 (판정 전에 눈으로 볼 것)
    python run_colab.py arms        R 격자 + K1 + K2 (+ 진단용 K3, K4)
    python run_colab.py table       모아서 표로

중간에 끊겨도 다시 돌리면 **이미 끝난 것은 건너뛴다.**

주의: 시퀀스별 실행을 위해 `annotations/val_half.json` 을 임시로 갈아끼운다.
원본은 `val_half.json.orig` 로 백업하고 끝나면 되돌린다. 중간에 죽었으면
`python run_colab.py restore` 로 복구할 것.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SEQS = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
        'MOT17-10', 'MOT17-11', 'MOT17-13']

# seqmap 헤더 버그 우회. TrackEval 이 첫 줄을 무조건 헤더로 버리므로
# videos 맨 앞에 더미가 있어야 MOT17-02 가 평가에 들어간다 (exp02 발견).
DUMMY_VIDEO = {'id': 0, 'file_name': 'name'}

ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]


def sh(cmd, cwd, env=None):
    print('$ ' + ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode:
        sys.exit('FAILED (exit %d): %s' % (r.returncode, ' '.join(map(str, cmd))))


class Paths:
    def __init__(self, a):
        self.utrack = Path(a.utrack)
        self.data_root = Path(a.data_root)
        self.exp = a.exp
        self.ann = self.data_root / 'annotations' / 'val_half.json'
        self.ann_orig = self.ann.with_suffix('.json.orig')
        self.results = self.utrack / 'track_results' / a.exp
        self.stats = Path(a.stats)


def backup_ann(p):
    if not p.ann.exists():
        sys.exit('ERROR %s 가 없다. exp02 의 COCO 변환을 먼저 끝낼 것' % p.ann)
    if not p.ann_orig.exists():
        shutil.copyfile(p.ann, p.ann_orig)
        print('백업 %s' % p.ann_orig)


def restore_ann(p):
    if p.ann_orig.exists():
        shutil.copyfile(p.ann_orig, p.ann)
        print('복구 %s' % p.ann)


def write_ann(p, seqs):
    """원본에서 지정 시퀀스만 남긴 val_half.json 을 쓴다."""
    with open(p.ann_orig, 'r', encoding='utf-8') as f:
        d = json.load(f)
    vids = [v for v in d['videos']
            if v.get('id') != 0 and any(s in str(v['file_name']) for s in seqs)]
    if not vids:
        sys.exit('ERROR 시퀀스 %s 를 val_half.json 에서 못 찾았다' % seqs)
    keep = {v['id'] for v in vids}
    imgs = [im for im in d['images'] if im['video_id'] in keep]
    img_ids = {im['id'] for im in imgs}
    anns = [an for an in d['annotations'] if an['image_id'] in img_ids]
    d['videos'] = [DUMMY_VIDEO] + vids       # 더미가 반드시 맨 앞
    d['images'] = imgs
    d['annotations'] = anns
    with open(p.ann, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return len(vids), len(imgs)


def newest_run_dir(p, assoc):
    """track.py 가 방금 만든 결과 폴더."""
    cands = sorted((p.results).glob(assoc + '_*'), key=lambda q: q.stat().st_mtime)
    return cands[-1] if cands else None


def run_one(p, arm, assoc, seq, env_extra, gpu):
    """한 시퀀스를 돌리고 결과 txt 를 갈래 폴더로 옮긴다."""
    dest = p.results / arm / 'data'
    dest.mkdir(parents=True, exist_ok=True)
    if list(dest.glob(seq + '*.txt')):
        print('  skip %s / %s (이미 있음)' % (arm, seq))
        return

    n_v, n_i = write_ann(p, [seq])
    print('  %s / %s  (videos %d, images %d)' % (arm, seq, n_v, n_i))

    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    sh([sys.executable, 'track.py',
        '--project', 'yolov8l-mix', '--exp', p.exp,
        '--data_root', str(p.data_root),
        '--association', assoc, '--gpu_id', str(gpu)],
       cwd=p.utrack, env=env)

    src = newest_run_dir(p, assoc)
    if src is None:
        sys.exit('ERROR %s 결과 폴더를 못 찾았다' % assoc)
    moved = 0
    for txt in (src / 'data').glob('*.txt'):
        shutil.move(str(txt), str(dest / txt.name))
        moved += 1
    if moved == 0:
        sys.exit('ERROR %s 에서 결과 txt 가 안 나왔다' % src)
    shutil.rmtree(src, ignore_errors=True)


def evaluate(p, arm):
    """갈래 폴더 전체를 7시퀀스 기준으로 한 번에 평가한다."""
    n_v, n_i = write_ann(p, SEQS)
    print('평가 %s  (videos %d)' % (arm, n_v))
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from multiprocessing import freeze_support\n"
        "from tracker.config import io\n"
        "from tracker.eval.metrics import TrackEvalMetrics\n"
        "from pathlib import Path\n"
        "freeze_support()\n"
        "io.set_seqmap(%r, Path('track_results')/%r, 'val_half')\n"
        "ec, dl, ml = io.track_eval_config_mot(\n"
        "    data_dir=%r, val_ann='val_half.json', experiment=%r,\n"
        "    benchmark=%r, split_to_eval='val_half', trackers_to_eval=[%r],\n"
        "    metrics=['HOTA','CLEAR','Identity'], use_parallel=False,\n"
        "    num_parallel_cores=8)\n"
        "TrackEvalMetrics(ec).evaluate(dl, ml)\n"
    ) % (str(p.data_root), p.exp, str(p.data_root), p.exp,
         p.data_root.name, arm)
    sh([sys.executable, '-c', code], cwd=p.utrack)


def read_hota(p, arm):
    """TrackEval 이 남긴 요약에서 COMBINED 행을 읽는다."""
    for name in ('pedestrian_summary.txt', 'pedestrian_detailed.csv'):
        f = p.results / arm / name
        if f.exists():
            lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            if len(lines) >= 2:
                sep = ',' if name.endswith('.csv') else None
                keys = lines[0].split(sep)
                vals = lines[-1].split(sep)
                return dict(zip(keys, vals))
    return None


# ----------------------------------------------------------------- 단계들

def stage_measure(p, a):
    backup_ann(p)
    p.stats.mkdir(parents=True, exist_ok=True)
    try:
        for seq in SEQS:
            run_one(p, 'A_measure', 'relax_botsort', seq,
                    {'RELAX_MODE': 'measure', 'RELAX_APPLY': 'both',
                     'RELAX_CAP': str(a.cap),
                     'RELAX_STATS': str(p.stats / (seq + '.json'))}, a.gpu)
        evaluate(p, 'A_measure')
    finally:
        restore_ann(p)
    print('')
    print('=' * 66)
    print('관문: 위 HOTA 가 exp02 기준선 botsort 64.494 와 같아야 한다.')
    print('다르면 훅이 뭔가 바꾸고 있다. 여기서 멈추고 원인을 찾을 것.')
    print('=' * 66)


def stage_calibrate(p, a):
    files = sorted(str(f) for f in p.stats.glob('*.json'))
    if not files:
        sys.exit('ERROR %s 에 통계가 없다. measure 를 먼저 돌릴 것' % p.stats)
    here = Path(__file__).resolve().parent
    sh([sys.executable, str(here / 'calibrate.py')] + files
       + ['--alphas'] + [str(x) for x in a.alphas] + ['--cap', str(a.cap)],
       cwd=here)


def stage_arms(p, a):
    common = {'RELAX_APPLY': 'both', 'RELAX_CAP': str(a.cap)}
    arms = {}
    for al in a.alphas:
        arms['R_sigma_a%g' % al] = dict(common, RELAX_MODE='sigma',
                                        RELAX_ALPHA=str(al))
    have_const = None not in (a.dx, a.dy, a.cw, a.ch)
    if have_const:
        # K1/K2 는 alpha 하나에만 맞춰진다. 이름에 그 alpha 를 박아 둬야
        # 나중에 어느 R 과 짝인지 헷갈리지 않는다.
        tag = '_a%g' % a.alpha
        arms['K1_const' + tag] = dict(common, RELAX_MODE='const',
                                      RELAX_DX=str(a.dx), RELAX_DY=str(a.dy))
        arms['K2_prop' + tag] = dict(common, RELAX_MODE='prop',
                                     RELAX_CW=str(a.cw), RELAX_CH=str(a.ch))
    else:
        print('참고: --dx/--dy/--cw/--ch 가 없어 R 격자만 돌린다.')
        print('      table 로 최고 alpha 를 고른 뒤 그 alpha 의 상수로 다시 부를 것.')
    if a.diagnostics:
        # 진단용. 짝 일관성 교란에 걸리므로 판정에 쓰지 않는다 (README 참고).
        arms['K3_shuffle'] = dict(common, RELAX_MODE='shuffle',
                                  RELAX_ALPHA=str(a.alpha), RELAX_SEED='0')
        arms['K4_ratioshuf'] = dict(common, RELAX_MODE='ratio_shuffle',
                                    RELAX_ALPHA=str(a.alpha), RELAX_SEED='0')
    backup_ann(p)
    try:
        for arm, env_extra in arms.items():
            for seq in SEQS:
                run_one(p, arm, 'relax_botsort', seq, env_extra, a.gpu)
            evaluate(p, arm)
    finally:
        restore_ann(p)


ARM_PREFIXES = ('A_measure', 'R_sigma', 'K1_const', 'K2_prop',
                'K3_shuffle', 'K4_ratioshuf')


def _arm_sort_key(name):
    order = {p: i for i, p in enumerate(ARM_PREFIXES)}
    head = next((p for p in ARM_PREFIXES if name.startswith(p)), '')
    tail = name[len(head):].lstrip('_a')
    try:
        num = float(tail)
    except ValueError:
        num = 0.0
    return (order.get(head, 99), num)


def stage_table(p, a):
    cols = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW']
    found = [d.name for d in p.results.iterdir()
             if d.is_dir() and (d / 'data').exists()]
    if a.all_runs:
        arms = sorted(found)
        print('경고: --all 은 다른 실험의 실행 폴더까지 보여준다.')
        print('      그 폴더들의 요약은 **그때의 seqmap 으로** 평가된 것이라')
        print('      시퀀스 수가 다르면 서로 비교할 수 없다.')
        print('      (exp02 에서 MOT17-02 누락 시 모든 갈래가 ~2.35 부풀었다)')
        print('')
    else:
        arms = sorted((n for n in found if n.startswith(ARM_PREFIXES)),
                      key=_arm_sort_key)
        skipped = len(found) - len(arms)
        if skipped:
            print('참고: 이 실험 것이 아닌 폴더 %d 개는 뺐다 (--all 로 보기).'
                  % skipped)
    if not arms:
        sys.exit('ERROR 이 실험의 결과 폴더가 없다. measure 부터 돌릴 것')
    base = None
    rows = []
    for arm in arms:
        m = read_hota(p, arm)
        if not m:
            rows.append((arm, None))
            continue
        vals = {c: m.get(c, '-') for c in cols}
        if arm.startswith('A_'):
            try:
                base = float(vals['HOTA'])
            except ValueError:
                pass
        rows.append((arm, vals))

    print('')
    print('| %-16s | %s | dHOTA  |' % ('arm', ' | '.join('%-7s' % c for c in cols)))
    print('|%s|%s%s|' % ('-' * 18, '---------|' * len(cols), '--------'))
    for arm, vals in rows:
        if vals is None:
            print('| %-16s | (결과 없음)' % arm)
            continue
        try:
            d = '%+.3f' % (float(vals['HOTA']) - base) if base else '-'
        except ValueError:
            d = '-'
        print('| %-16s | %s | %6s |'
              % (arm, ' | '.join('%-7s' % vals[c] for c in cols), d))
    print('')
    print('판정은 K2_prop 로만 한다. K1/K3/K4 는 진단이다 (README 의 사전 선언).')
    print('R > K2 이면 짝 일관성부터 의심할 것.')


def stage_direction(p, a):
    """**확장이 문을 여는가 닫는가** -- 연관 호출의 원자료를 남긴다.

    사전 등록은 `PREREG-direction.md`. 본 실행의 alpha 는 10 인데 방향을
    검증한 `[3b]` 는 alpha=2 의 합성 장면이다. **검증한 구간과 사용한
    구간이 5 배 떨어져 있다.**

    확장 0 (measure) 으로 한 번 돌며 호출마다 확장 전 입력을 남기고,
    `direction.py` 가 그 같은 입력에 alpha 만 갈아끼워 재생한다.
    """
    dumps = Path(a.dumps)
    dumps.mkdir(parents=True, exist_ok=True)
    backup_ann(p)
    try:
        for seq in SEQS:
            run_one(p, 'A_direction', 'relax_botsort', seq,
                    {'RELAX_MODE': 'measure', 'RELAX_APPLY': 'both',
                     'RELAX_CAP': str(a.cap),
                     'RELAX_DUMP_CALLS': str(dumps / (seq + '.pkl'))}, a.gpu)
    finally:
        restore_ann(p)
    here = Path(__file__).resolve().parent
    # **UTrack 을 경로에 넣어 넘긴다.** 안 넘기면 direction.py 가
    # box_relax 를 최상위로 임포트해 IoU 가 numpy 대체본으로 조용히
    # 떨어진다 (+1 픽셀 규약 차이로 2e-02 어긋남). 관문이 잡았다.
    env = dict(os.environ)
    env['UTRACK_ROOT'] = str(p.utrack)
    env['PYTHONPATH'] = str(p.utrack) + os.pathsep + env.get('PYTHONPATH', '')
    sh([sys.executable, str(here / 'direction.py'), str(dumps)],
       cwd=here, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['measure', 'calibrate', 'arms',
                                      'table', 'restore', 'direction'])
    ap.add_argument('--utrack', default='/content/UTrack')
    ap.add_argument('--data_root', default='/content/data/MOT17')
    ap.add_argument('--exp', default='ablation_17')
    ap.add_argument('--stats', default='/content/relax_stats')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cap', type=float, default=1.0)
    ap.add_argument('--dumps', default='/content/exp03_dumps',
                    help='direction: 연관 호출 원자료를 남길 곳')
    ap.add_argument('--alphas', nargs='+', type=float, default=ALPHAS,
                    help='R 격자. 기본은 사전 선언 격자 %s' % ALPHAS)
    ap.add_argument('--alpha', type=float, default=2.0,
                    help='K1/K2/K3/K4 를 맞출 alpha (R 격자에서 고른 값)')
    ap.add_argument('--dx', type=float)
    ap.add_argument('--dy', type=float)
    ap.add_argument('--cw', type=float)
    ap.add_argument('--ch', type=float)
    ap.add_argument('--diagnostics', action='store_true')
    ap.add_argument('--all', dest='all_runs', action='store_true',
                    help='table: 다른 실험의 실행 폴더까지 보여준다')
    a = ap.parse_args()
    p = Paths(a)

    {'measure': stage_measure, 'calibrate': stage_calibrate,
     'arms': stage_arms, 'table': stage_table,
     'restore': lambda p, a: restore_ann(p),
     'direction': stage_direction}[a.stage](p, a)


if __name__ == '__main__':
    main()
