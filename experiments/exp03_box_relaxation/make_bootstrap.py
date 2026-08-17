"""
실험 3 - 콜랩 부트스트랩 셀을 만든다.

**언제 쓰나.** 평소에는 콜랩에서 `git clone` 을 쓴다 (저장소가 공개돼 있다).
이 스크립트는 **push 하지 않은 로컬 수정본을 그대로 콜랩에 올릴 때만** 쓰는
대안이다. 실행에 필요한 파일들을 gzip+base64 로 묶어 셀 하나로 만들므로
인증도 clone 도 필요 없다.

    python make_bootstrap.py

`bootstrap_cell.txt` 가 나온다. 그 내용을 콜랩 셀에 그대로 붙여넣는다.

**이 산출물은 커밋하지 않는다** (`.gitignore`). 소스를 통째로 박제하기 때문에
커밋해 두면 옛 코드가 조용히 콜랩에 설치된다. 실제로 한 번 그렇게 됐다 --
저장돼 있던 셀이 `box_relax.py` 의 옛 판(저수지 off-by-one 이 있는)을 담고
있었다. 그래서 **쓰기 직전에 이 스크립트를 돌린다.** 셀 머리에 생성 시각과
파일별 SHA 앞자리를 찍으므로 콜랩에서 무엇이 설치되는지 눈으로 확인할 수 있다.
"""

import base64
import gzip
import hashlib
import io
import tarfile
import time
from pathlib import Path

# 콜랩 실행에 실제로 필요한 것만. selftest/power_check 는 로컬 전용이다.
FILES = ['box_relax.py', 'calibrate.py', 'patch_utrack.py', 'run_colab.py']
DEST = '/content/exp03'

TEMPLATE = '''# 실험 3 부트스트랩 - push 안 한 로컬 수정본을 콜랩에 올릴 때 쓴다
# 생성: {stamp}
# 담긴 소스: {manifest}
#   ^ 로컬 `git rev-parse` 와 대조할 것. 다르면 make_bootstrap.py 를 다시 돌려라.
import base64, gzip, io, tarfile, pathlib, subprocess, sys

BLOB = "{blob}"

dest = pathlib.Path("{dest}")
dest.mkdir(parents=True, exist_ok=True)
raw = gzip.decompress(base64.b64decode(BLOB))
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    # filter='data' 는 3.12+ 에서 기본이 된다. 명시해 경고와 미래 변경을 피한다.
    try:
        tf.extractall(dest, filter='data')
    except TypeError:
        tf.extractall(dest)
print("설치:", sorted(p.name for p in dest.glob("*.py")))
print("생성 시각:", "{stamp}")

r = subprocess.run([sys.executable, str(dest / "patch_utrack.py"), "/content/UTrack"])
if r.returncode:
    raise SystemExit("patch_utrack 실패 - /content/UTrack 이 맞는지 확인할 것")

sys.path.insert(0, "/content/UTrack")
from tracker.associations.collections import ASSOCIATIONS
print("relax_botsort 등록:", "relax_botsort" in ASSOCIATIONS)
'''


def main():
    here = Path(__file__).resolve().parent
    buf = io.BytesIO()
    sigs = []
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name in FILES:
            p = here / name
            if not p.exists():
                raise SystemExit('ERROR missing %s' % p)
            sigs.append('%s@%s' % (name.replace('.py', ''),
                                   hashlib.sha256(p.read_bytes()).hexdigest()[:8]))
            tf.add(str(p), arcname=name)
    blob = base64.b64encode(gzip.compress(buf.getvalue(), 9)).decode()
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest = ' '.join(sigs)

    cell = TEMPLATE.format(blob=blob, dest=DEST, stamp=stamp, manifest=manifest)
    out = here / 'bootstrap_cell.txt'
    out.write_text(cell, encoding='utf-8')
    print('wrote %s' % out)
    print('  files %d, blob %d chars (%.1f KB)'
          % (len(FILES), len(blob), len(blob) / 1024))
    print('  %s' % manifest)
    print('')
    print('콜랩 셀에 붙여넣을 내용은 위 파일 전체다.')
    print('이 산출물은 커밋하지 않는다 -- 쓰기 직전에 다시 만들 것.')


if __name__ == '__main__':
    main()
