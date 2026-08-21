# -*- coding: utf-8 -*-
"""실험 1 의 세 조건을 전부 다시 낸다.

여태 조건별 실행 명령이 어디에도 적혀 있지 않았다. 환경변수 조합을 손으로
쳐서 돌렸고, 그래서 무엇이 어떤 설정으로 나온 숫자인지 npz 파일명에만 남았다.
**재현이 안 되는 상태였다.** 이 스크립트가 그 구멍을 메운다.

  (빈 TAG)  우리 검출기, 사전 등록 설정. 판정용 본 실행
  -m60      같은 설정, 60프레임. 저자 가중치와 프레임 수를 맞추기 위한 대조군
  -fork     저자 가중치 + 포크 NMS 설정, 60프레임. 실험 2/3 이 실제로 쓴 sigma

사용법:
    python experiments/exp01_nms_variance/run_all.py            # 전부
    python experiments/exp01_nms_variance/run_all.py base m60   # 조건 골라서
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

OURS = {"EXP01_MODEL": "../BSDsystem/yolov8m.pt", "EXP01_CONF": "0.10",
        "EXP01_IOU": "0.45", "EXP01_IMGSZ": "640"}
FORK = {"EXP01_MODEL": "data/weights/ablation_17_best.pt", "EXP01_CONF": "0.01",
        "EXP01_IOU": "0.70", "EXP01_IMGSZ": "800,1440"}

# 조건 -> (TAG, 환경변수, 시퀀스별 프레임 수)
# 본 실행만 MOT17-02 를 전체(299)로 돌린다. 사전 점검 판정을 처음 낸 실행이 그랬다.
ARMS = {
    "base": ("",      OURS, lambda s: 299 if s.startswith("MOT17-02") else 200),
    "m60":  ("-m60",  OURS, lambda s: 60),
    "fork": ("-fork", FORK, lambda s: 60),
}

RUNNER = Path(__file__).with_name("run_sequence.py")


def main():
    want = sys.argv[1:] or list(ARMS)
    bad = [a for a in want if a not in ARMS]
    if bad:
        sys.exit("모르는 조건: %s (가능: %s)" % (bad, list(ARMS)))

    t0 = time.time()
    for arm in want:
        tag, env_extra, nframe = ARMS[arm]
        print("=" * 72)
        print("조건 %s   TAG=%r   %s" % (arm, tag, env_extra["EXP01_MODEL"]))
        print("=" * 72)
        for seq in SEQS:
            env = dict(os.environ, EXP01_TAG=tag, **env_extra)
            n = nframe(seq)
            t = time.time()
            r = subprocess.run([sys.executable, str(RUNNER), seq, str(n)],
                               env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print("  [실패] %s\n%s" % (seq, r.stdout[-2000:] + r.stderr[-2000:]))
                continue
            # 배율 검산 줄만 뽑아 보여준다. 여기가 틀리면 Sigma_d 단위가 틀린다.
            for line in r.stdout.splitlines():
                if "배율" in line or "경고" in line or "매칭률" in line:
                    print("  %-16s %s" % (seq, line.strip()))
            print("  %-16s %d프레임 %.0f초" % (seq, n, time.time() - t))
    print("총 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
