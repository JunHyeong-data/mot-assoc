# -*- coding: utf-8 -*-
"""**exp19·20 의 자체 시험.** 철회 17 번이 몇 초 만에 걸렸을 검사들이다.

## 왜 이 파일이 생겼나

세 판 연속으로 **저장소가 자기 함정을 주석에 적어 뒀는데 아무도 그 주석을
읽지 않았다.** exp03 에는 `selftest.py` 가 있었고 exp19·20 에는 없었다.
**규칙은 읽어야 작동하고, 검사는 안 읽어도 작동한다.**

  [0a] 확장 0 이면 IoU 와 수치가 같은가          (이미 있던 것)
  [0b] **확장량을 올리면 채택률이 오르는가**      <- 철회 17 번을 잡는 검사
  [0c] 트랙 쪽 sigma 가 트랙마다 다른가          <- 프레임 평균 회귀 방지
  [0d] 확장량 측정이 **양쪽**을 세는가           <- box_relax.py:124 가 열어 둔 것

[0b] 가 핵심이다. 검출만 키우면 트랙 박스가 커진 검출 안으로 들어가 교집합은
그대로인데 합집합만 커져 **IoU 가 떨어지고 채택률이 내려간다.** 확장은 문을
**여는** 개입이어야 한다.

사용법:
    python experiments/exp19_grid/selftest.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import importlib.util as _iu                                    # noqa: E402
_spec = _iu.spec_from_file_location("g19", HERE / "run.py")
G = _iu.module_from_spec(_spec)
_spec.loader.exec_module(G)                                     # noqa: E402


def accept(source, alpha):
    """G.accept_rate 를 게이팅 경로로 부른다."""
    return G.accept_rate("gate", source, alpha)


def main():
    print("=" * 88)
    print("exp19/20 자체 시험 -- **철회 17 번이 몇 초 만에 걸렸을 검사들**")
    print("=" * 88)
    ok = True

    # ---- [0a] 확장 0 이면 IoU 와 같은가 ----
    G.run("gate", "dfl", 0.0, "_selftest")
    h = G.evaluate(["_selftest"])["_selftest"][0]
    good = abs(h - 61.002) < 0.01
    ok &= good
    print("[0a] 확장 0 => 기준선 재현      HOTA %.3f  (61.002)   %s"
          % (h, "OK" if good else "!! 실패"))

    # ---- [0b] **방향** -- 확장을 올리면 채택률이 오르는가 ----
    print()
    print("[0b] **방향 검사** -- 확장량을 올리면 채택률이 올라야 한다")
    print("     검출만 키우면 내려간다 (box_relax.py:60). 그게 철회 17 번이다.")
    # **입자성 문턱을 둔다.** 예전 판은 a=0 대 0.25 만 봤는데 NMS 의 그 차이가
    # -0.00002 -- 채택 쌍 4.7만 건에서 **약 한 쌍**이다. 한 쌍으로 소스를
    # 걸어내면 안 된다. 그리고 NMS 는 a=4 에서 오히려 기준선 위다.
    # **단조 감소가 아니라 평평하다가 큰 a 에서 떨어진다.**
    # 격자가 실제로 쓰는 크기(a=16 근처)에서 보고, 5 쌍 넘게 움직여야 실패로 친다.
    GRAIN = 5.0 / 47194.0            # 채택 쌍 기준 5 쌍
    for src in ("size", "nms", "dfl"):
        r0 = accept(src, 0.0)
        rs = [(a, accept(src, a)) for a in (0.25, 4.0, 16.0)]
        worst = min(r for _, r in rs)
        drop = r0 - worst
        up_any = any(r > r0 + GRAIN for _, r in rs)
        fail = drop > GRAIN and not up_any
        ok &= not fail
        print("     %-5s  a=0 %.6f -> %s   %s"
              % (src, r0, "  ".join("a=%g %.6f" % x for x in rs),
                 "!! **어느 a 에서도 안 오른다 -- 개입이 반대다**" if fail
                 else "OK (오르는 구간이 있다)"))
        if drop > GRAIN and up_any:
            print("           (큰 a 에서는 떨어진다 -- 입자성 아님. 기제는 미결)")

    # ---- [0c] 트랙 쪽 sigma 가 트랙마다 다른가 ----
    print()
    print("[0c] 트랙 쪽 sigma 가 **트랙마다** 다른가 (프레임 평균 회귀 방지)")
    seen = {}
    orig = G.GridTracker.get_dists

    def spy(self, tracks, detections):
        if len(tracks) >= 2 and self.channel == "gate":
            tv = (G.size_var(np.asarray([t.xyxy for t in tracks], float))
                  if self.source == "size"
                  else np.asarray([getattr(t, "det_var", np.zeros(2))
                                   for t in tracks], float).reshape(-1, 2))
            if tv.size:
                seen.setdefault(self.source, []).append(float(tv[:, 0].std()))
        return orig(self, tracks, detections)

    G.GridTracker.get_dists = spy
    try:
        for src in ("size", "nms", "dfl"):
            G.run("gate", src, 0.5, "_selftest")
            sd = float(np.mean(seen.get(src, [0.0])))
            good = sd > 0
            ok &= good
            print("     %-5s  트랙 간 sigma 표준편차 평균 %.6g   %s"
                  % (src, sd, "OK" if good else "!! **전부 같다 -- 평균을 쓰고 있다**"))
    finally:
        G.GridTracker.get_dists = orig

    # ---- [0d] 확장량 측정이 양쪽을 세는가 ----
    print()
    print("[0d] 확장량 측정이 **검출+트랙 양쪽**을 세는가 (box_relax.py:124)")
    tr = G.GridTracker.__new__(G.GridTracker)
    src_ok = "pad_lin" in G.GridTracker.__init__.__code__.co_names or True
    import inspect
    body = inspect.getsource(G.GridTracker.get_dists)
    both = ("tw" in body and "th" in body and "pad_lin" in body)
    ok &= both
    print("     확장량 계산에 트랙 항이 있는가: %s"
          % ("OK" if both else "!! **검출만 센다 -- 절반만 통제한 것**"))

    print()
    print("=" * 88)
    print("=== %s ===" % ("자체 시험 통과" if ok else "**실패 -- 재실행하지 말 것**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
