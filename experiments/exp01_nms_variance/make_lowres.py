# -*- coding: utf-8 -*-
"""실험 1d -- 저해상도 원본을 인위적으로 만든다.

**왜 필요한가.** 실험 1c 에서 imgsz 를 키울 때 MOT17-05(640x480)만 보정이
개선되고 1920x1080 여섯은 악화됐다. 원인이 **원본 해상도**인지 **그 장면**인지
못 가른다 -- 저해상도 원본이 하나뿐이라 n=1 이기 때문이다.

MOT17-06 을 넣으면 될 것 같지만 **MOT17-06 은 test 분할이라 GT 가 없다.**
공개된 MOT17 train 은 02/04/05/09/10/11/13 일곱뿐이고 640x480 은 05 하나다.

그래서 **다른 시퀀스를 줄여서 저해상도 조건을 만든다.** 이게 오히려 낫다 --
시퀀스를 하나 더 넣으면 해상도와 장면이 같이 바뀌지만, 줄이면 **장면을 고정한 채
해상도만** 바꾼다. 인과를 직접 겨눈다.

  1920x1080 을 1/3 로 줄여 640x360 으로 만든다 (가로가 MOT17-05 와 같다).
  종횡비를 보존하므로 사람 모양이 안 뭉개진다. GT 박스도 같은 배율로 줄인다.
  축소는 INTER_AREA -- 확대용 보간을 쓰면 없던 정보가 생긴 것처럼 보인다.

사용법:
    python experiments/exp01_nms_variance/make_lowres.py
    python experiments/exp01_nms_variance/make_lowres.py MOT17-02-FRCNN --frames 60
"""
import argparse
import sys
from pathlib import Path

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path("data/MOT17_A/ablation")
DEFAULT = ["MOT17-02-FRCNN", "MOT17-10-FRCNN"]   # 1c 에서 제자리/최악이던 둘
SCALE = 1.0 / 3.0                                 # 1920x1080 -> 640x360


def build(seq, frames, scale):
    src = ROOT / seq
    dst = ROOT / (seq.replace("-FRCNN", "") + "-LOWRES")
    if not src.exists():
        sys.exit("ERROR %s 가 없다" % src)

    (dst / "img1").mkdir(parents=True, exist_ok=True)
    (dst / "gt").mkdir(parents=True, exist_ok=True)

    imgs = sorted((src / "img1").glob("*.jpg"))[:frames]
    w0 = h0 = None
    for p in imgs:
        im = cv2.imread(str(p))
        if h0 is None:
            h0, w0 = im.shape[:2]
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        # 축소에는 INTER_AREA. 픽셀을 평균내므로 에일리어싱이 안 생긴다.
        small = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / "img1" / p.name), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    # GT 도 같은 배율로. 열 순서는 MOT 규약 그대로 두고 2~5번(x,y,w,h)만 줄인다.
    n = 0
    with open(dst / "gt" / "gt.txt", "w") as out:
        for line in open(src / "gt" / "gt.txt"):
            f = line.strip().split(",")
            if len(f) < 9 or int(f[0]) > len(imgs):
                continue
            for i in (2, 3, 4, 5):
                f[i] = "%.2f" % (float(f[i]) * scale)
            out.write(",".join(f) + "\n")
            n += 1

    print("%-18s -> %-22s  %dx%d -> %dx%d, %d프레임, GT %d줄"
          % (seq, dst.name, w0, h0, nw, nh, len(imgs), n))
    return dst.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=DEFAULT)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--scale", type=float, default=SCALE)
    a = ap.parse_args()
    made = [build(s, a.frames, a.scale) for s in (a.seqs or DEFAULT)]
    print()
    print("다음: 각 시퀀스를 imgsz 640(원해상도) 과 1920(3배 확대) 으로 돌린다.")
    print("MOT17-05 와 정확히 같은 두 조건이다.")
    for m in made:
        print("  EXP01_IMGSZ=640  EXP01_TAG=-isz640  run_sequence.py %s 60" % m)
        print("  EXP01_IMGSZ=1920 EXP01_TAG=-isz1920 run_sequence.py %s 60" % m)


if __name__ == "__main__":
    main()
