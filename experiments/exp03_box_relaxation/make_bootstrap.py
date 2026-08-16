"""
실험 3 - 콜랩 부트스트랩 셀을 만든다.

이 저장소는 GitHub 원격이 없다. 그래서 콜랩에서 `git clone` 을 못 한다.
대신 실행에 필요한 파일들을 gzip+base64 로 묶어 **셀 하나**로 만든다.
붙여넣기 한 번이면 끝이고 업로드 대화상자도 인증도 필요 없다.

    python make_bootstrap.py

`bootstrap_cell.txt` 가 나온다. 그 내용을 콜랩 셀에 그대로 붙여넣는다.
**파일을 고칠 때마다 다시 돌려야 한다.**
"""

import base64
import gzip
import io
import tarfile
from pathlib import Path

# 콜랩 실행에 실제로 필요한 것만. selftest/power_check 는 로컬 전용이다.
FILES = ['box_relax.py', 'calibrate.py', 'patch_utrack.py', 'run_colab.py']
DEST = '/content/exp03'

TEMPLATE = '''# 실험 3 부트스트랩 - 이 셀 하나면 설치가 끝난다 (원격 저장소 불필요)
import base64, gzip, io, tarfile, pathlib, subprocess, sys

BLOB = "{blob}"

dest = pathlib.Path("{dest}")
dest.mkdir(parents=True, exist_ok=True)
raw = gzip.decompress(base64.b64decode(BLOB))
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    tf.extractall(dest)
print("설치:", sorted(p.name for p in dest.glob("*.py")))

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
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name in FILES:
            p = here / name
            if not p.exists():
                raise SystemExit('ERROR missing %s' % p)
            tf.add(str(p), arcname=name)
    blob = base64.b64encode(gzip.compress(buf.getvalue(), 9)).decode()

    cell = TEMPLATE.format(blob=blob, dest=DEST)
    out = here / 'bootstrap_cell.txt'
    out.write_text(cell, encoding='utf-8')
    print('wrote %s' % out)
    print('  files %d, blob %d chars (%.1f KB)'
          % (len(FILES), len(blob), len(blob) / 1024))
    print('')
    print('콜랩 셀에 붙여넣을 내용은 위 파일 전체다.')


if __name__ == '__main__':
    main()
