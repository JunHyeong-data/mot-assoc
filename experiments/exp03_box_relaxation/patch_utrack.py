"""
실험 3 - UTrack 클론에 박스확장 조건을 설치한다.

건드리는 것은 두 가지뿐이다.
  1. `tracker/box_relax.py`      새 파일 복사
  2. `tracker/associations/collections.py`  끝에 클래스 하나 + 등록 한 줄 덧붙임

**소스를 고치지 않고 덧붙이기만 한다.** 재현 실패 시 원인 후보를 줄이기 위함이다.
같은 이유로 두 번 돌려도 안전하다 (표식으로 확인).

    python patch_utrack.py /content/UTrack
    python patch_utrack.py /content/UTrack --revert
"""

import argparse
import shutil
import sys
from pathlib import Path

MARK = '# ---- exp03 box relaxation (mot-assoc) ----'

SNIPPET = '''

''' + MARK + '''
from ..box_relax import relaxed_iou_distance


class RelaxBoTSORT(Associations):
    """BoTSORT 와 모든 것이 같고 IoU 를 확장된 박스로 잰다.

    확장 방식은 RELAX_MODE 환경변수로 고른다. 'measure' 면 확장이 0 이므로
    botsort 와 수치가 같아야 한다.
    """

    def __init__(self, distance=None, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(relaxed_iou_distance, args=args), args=args)


ASSOCIATIONS['relax_botsort'] = RelaxBoTSORT
# ---- exp03 end ----
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('utrack_root', help='UTrack 클론 경로 (track.py 가 있는 곳)')
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()

    root = Path(a.utrack_root).resolve()
    coll = root / 'tracker' / 'associations' / 'collections.py'
    dest = root / 'tracker' / 'box_relax.py'
    src = Path(__file__).resolve().parent / 'box_relax.py'

    if not coll.exists():
        sys.exit('ERROR not a UTrack clone: %s missing' % coll)

    text = coll.read_text(encoding='utf-8')

    if a.revert:
        if MARK in text:
            head = text.split('\n\n' + MARK)[0]
            coll.write_text(head + '\n', encoding='utf-8')
            print('reverted %s' % coll)
        else:
            print('nothing to revert in %s' % coll)
        if dest.exists():
            dest.unlink()
            print('removed %s' % dest)
        return

    if not src.exists():
        sys.exit('ERROR missing %s' % src)

    shutil.copyfile(src, dest)
    print('copied  %s' % dest)

    if MARK in text:
        print('already patched %s (snippet left as is)' % coll)
    else:
        coll.write_text(text.rstrip('\n') + SNIPPET, encoding='utf-8')
        print('patched %s' % coll)

    print('')
    print('sanity check:')
    print("  python -c \"import sys; sys.path.insert(0,'%s');"
          " from tracker.associations.collections import ASSOCIATIONS;"
          " print('relax_botsort' in ASSOCIATIONS)\"" % root)


if __name__ == '__main__':
    main()
