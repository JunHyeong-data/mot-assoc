# -*- coding: utf-8 -*-
"""**먼저 이걸 돌려라.** 신선한 클론에서 무엇이 없는지 알려준다.

## 왜 있는가

2026-08-18 에 **깨끗한 디렉터리로 클론해서 재현성을 시험했다.** 결과:

- `theory/` 7개는 **전부 돈다** (numpy+scipy 만 있으면 된다는 서술이 맞았다)
- `experiments/` 는 **하나도 안 돈다.** `external/` 과 `data/` 가 git 제외라서다
- 그런데 오류가 `ModuleNotFoundError: No module named 'tracker'` 로만 나와서
  **무엇을 해야 하는지 알 수 없었다**
- `external/UTrack` 을 어떻게 구하는지는 `exp02_utrack_replication/colab_setup.md`
  **45번째 줄의 콜랩 셀 안에만** 있었다

이 스크립트가 그 간극을 메운다. **판정 스크립트가 관문을 먼저 두는 것과 같은 정신**이다.

사용법:
    python check_setup.py
"""
import importlib
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

UTRACK_URL = "https://github.com/DLR-MI/UTrack.git"


def head(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def main():
    print("=" * 78)
    print("mot-assoc 준비 점검 -- 신선한 클론에서 무엇이 없는지 본다")
    print("=" * 78)
    missing = []

    # ---------------- 1. 파이썬 꾸러미 ----------------
    head("[1] 파이썬 꾸러미")
    need = [("numpy", "theory/ 와 experiments/ 전부"),
            ("scipy", "theory/ 와 experiments/ 전부"),
            ("ultralytics", "experiments/ 의 트래커 재생"),
            ("torch", "검출기"),
            ("cv2", "검출기 (opencv-python)"),
            ("fitz", "논문 PDF 정독 (PyMuPDF). 없어도 실험은 돈다")]
    for mod, why in need:
        try:
            importlib.import_module(mod)
            print("  OK   %-14s %s" % (mod, why))
        except ImportError:
            opt = mod == "fitz"
            print("  %s %-14s %s" % ("--  " if opt else "없음", mod, why))
            if not opt:
                missing.append("pip install -r requirements.txt")

    # ---------------- 2. 벤더링된 TrackEval ----------------
    head("[2] external/UTrack -- HOTA 지표 클래스를 여기서 가져온다")
    ut = ROOT / "external" / "UTrack"
    hota = ut / "tracker" / "eval" / "collections" / "hota.py"
    if hota.exists():
        print("  OK   %s" % hota.relative_to(ROOT))
    else:
        print("  없음 %s" % hota.relative_to(ROOT))
        print()
        print("  이게 없으면 evaluate.py 를 쓰는 실험이 전부 다음으로 죽는다:")
        print("      ModuleNotFoundError: No module named 'tracker'")
        print()
        print("  고치는 법:")
        print("      git clone %s external/UTrack" % UTRACK_URL)
        missing.append("git clone %s external/UTrack" % UTRACK_URL)

    # ---------------- 3. 자료 ----------------
    head("[3] data/ -- git 제외다 (MOT17 은 CC BY-NC-SA 3.0, 재배포 불가)")
    checks = [
        ("data/MOT17_A/ablation", "MOT17 val_half GT. 직접 받아야 한다", None),
        ("data/exp05", "검출 캐시 (npz). exp05/06/08/11 이 쓴다",
         "python experiments/exp05_wasserstein/cache_detections.py   # 약 110분"),
        ("data/exp01", "실험 1 계열의 sigma/오차 npz. exp10 이 쓴다",
         "python experiments/exp01_nms_variance/run_all.py"),
        ("data/exp06", "임계값 그리드 표",
         "python experiments/exp06_levers/grid.py"),
    ]
    for rel, why, how in checks:
        p = ROOT / rel
        n = len(list(p.glob("*"))) if p.is_dir() else 0
        if n:
            print("  OK   %-26s %d개 항목" % (rel, n))
        else:
            print("  없음 %-26s %s" % (rel, why))
            if how:
                print("       -> %s" % how)
                missing.append(how)
            else:
                print("       -> notes/data_sources.md 참고. **재배포할 수 없다**")
                missing.append("MOT17 을 직접 받아 data/MOT17_A/ablation 에 둔다")

    # ---------------- 요약 ----------------
    head("요약")
    if not missing:
        print("  준비 끝. 전부 있다.")
        return 0
    print("  theory/ 7개는 **지금도 돈다** (numpy+scipy 만 필요하다):")
    print("      python theory/assignment_invariance.py   등")
    print()
    print("  experiments/ 를 돌리려면 아래가 필요하다:")
    seen = set()
    for m in missing:
        if m not in seen:
            seen.add(m)
            print("      %s" % m)
    print()
    print("  자세한 것은 README 의 '준비' 절.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
