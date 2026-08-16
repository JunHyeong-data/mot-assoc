# -*- coding: utf-8 -*-
"""
MOT17 미러 무결성 검증.

motchallenge.net 이 죽어 공식 체크섬을 못 쓴다. 대신 서로 독립적인 업로더 둘이
올린 사본을 바이트 단위로 대조한다. 두 미러가 같은 바이트를 갖고 있으면
어느 한쪽이 훼손됐을 가능성은 배제된다.

배제되지 않는 것: 둘이 같은 훼손본을 퍼뜨린 경우. 공식 서버 복구 시 재검증할 것.

주석파일(gt/det/seqinfo)은 전수 대조한다. 실험이 여기 직접 의존하기 때문이다.
이미지는 무작위 표본만 대조한다.

사용법:
    python experiments/exp01_nms_variance/verify_data.py
"""
import hashlib
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

REPO_A = "Morrison1025/MOT17"     # 이미 data/MOT17_A 에 받아둔 것
REPO_B = "Lekim89/MOT17"          # 대조용
LOCAL_A = Path("data/MOT17_A")
CACHE_B = Path("data/_verify_B")
IMG_SAMPLE = 120
SEED = 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rel_paths():
    """huggingface_hub 이 local_dir 안에 만드는 .cache 메타데이터는 제외한다.
    저장소에 없는 파일이라 대조하려 들면 전부 '받기 실패' 로 잡힌다."""
    ann, img = [], []
    for p in sorted(LOCAL_A.rglob("*")):
        if not p.is_file():
            continue
        r = p.relative_to(LOCAL_A).as_posix()
        if ".cache/" in r or r.startswith(".cache"):
            continue
        (img if r.endswith(".jpg") else ann).append(r)
    return ann, img


def compare(rels, label):
    ok = bad = err = 0
    bad_list = []
    for i, r in enumerate(rels, 1):
        try:
            b = hf_hub_download(repo_id=REPO_B, repo_type="dataset", filename=r,
                                local_dir=CACHE_B)
        except Exception as e:
            err += 1
            print(f"    [받기 실패] {r} :: {type(e).__name__}")
            continue
        if sha256(LOCAL_A / r) == sha256(b):
            ok += 1
        else:
            bad += 1
            bad_list.append(r)
        if i % 25 == 0 or i == len(rels):
            print(f"    {label} {i}/{len(rels)}  일치 {ok}  불일치 {bad}  오류 {err}")
    return ok, bad, err, bad_list


print("=" * 72)
print("MOT17 미러 무결성 검증 -- 독립 미러 2개 바이트 대조")
print("=" * 72)
print(f"  A (보유): {REPO_A}  ->  {LOCAL_A}")
print(f"  B (대조): {REPO_B}")
print()

if not LOCAL_A.exists():
    sys.exit(f"{LOCAL_A} 가 없다. 먼저 A 미러를 받을 것.")

ann, img = rel_paths()
random.Random(SEED).shuffle(img)
img_s = img[:IMG_SAMPLE]

print(f"  주석파일 {len(ann)}개 전수 대조, 이미지 {len(img)}개 중 {len(img_s)}개 표본 대조")
print()

print("  [1] 주석파일 (gt / det / seqinfo) -- 전수")
a_ok, a_bad, a_err, a_list = compare(ann, "주석")
print()
print("  [2] 이미지 -- 무작위 표본")
i_ok, i_bad, i_err, i_list = compare(img_s, "이미지")

print()
print("=" * 72)
print("결과")
print("=" * 72)
print(f"  주석파일  일치 {a_ok}/{len(ann)}   불일치 {a_bad}   오류 {a_err}")
print(f"  이미지    일치 {i_ok}/{len(img_s)}   불일치 {i_bad}   오류 {i_err}")
print()
if a_bad or i_bad:
    print("  *** 불일치 발견. 이 데이터를 쓰지 말 것. ***")
    for r in (a_list + i_list)[:20]:
        print("   ", r)
    sys.exit(1)
elif a_err or i_err:
    print("  일부 파일을 받지 못해 검증이 불완전하다. 다시 돌릴 것.")
    sys.exit(2)
else:
    print("  전부 일치. 두 독립 미러가 바이트 단위로 같다.")
    print("  주의: 둘이 같은 훼손본을 퍼뜨린 경우는 배제되지 않는다.")
    print("        motchallenge.net 복구 시 공식 체크섬으로 재검증할 것.")
