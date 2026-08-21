# -*- coding: utf-8 -*-
"""실험 6 LOSO [1단계] -- 그리드 표를 만들고 **JSON 으로 남긴다**.

사전 등록은 `PREREG-loso.md`. 자료보다 먼저 커밋했다.

`headroom.py` 는 표를 print 만 하고 버렸다. `predictors.py` 는 그 값을
**손으로 옮겨 적었다**(`BEST`, `GAIN`). 재현 경로가 사람 손을 거친다.
그래서 여기서 표를 다시 만들고 JSON 으로 남긴다.

## 이 스크립트가 하는 검산 둘

**[0] 라벨 대조 (사전 등록한 사전 점검).** 기존 8개 격자에 한정해 argmax 를
`predictors.BEST` 와 맞춰본다. **7/7 이 아니면 배관이 틀린 것이니 멈춘다.**
0.98 은 새 값이라 이 대조에서 뺀다.

**[0c] 재생 결정성.** 기존 `_headroom/` 트랙 파일의 해시를 먼저 뜨고,
같은 임계값을 다시 재생해 비교한다. 어제 만든 파일과 오늘 만든 파일이 다르면
재생이 결정적이지 않다는 뜻이다 (철회 10번 -- `hash()` 씨앗이 실행마다 다른
세계를 만든 적이 있다).

사용법:
    python experiments/exp06_levers/grid.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "exp05_wasserstein"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from headroom import run, OUTROOT                            # noqa: E402
from replay import SEQS                                      # noqa: E402

# 사전 점검 [0] 대조용 -- **커밋 5b92db6 시점의 기록을 그대로 박아둔다.**
# predictors.py 에서 import 하지 않는다. 그 파일은 라벨이 갱신되므로
# (0.98 추가로 MOT17-13 이 옮겨갔다) 대조 상대가 같이 움직이면 사전 점검이 무의미해진다.
BEST_OLD = {"MOT17-02-FRCNN": 0.85, "MOT17-04-FRCNN": 0.90, "MOT17-05-FRCNN": 0.75,
            "MOT17-09-FRCNN": 0.75, "MOT17-10-FRCNN": 0.95, "MOT17-11-FRCNN": 0.70,
            "MOT17-13-FRCNN": 0.95}

# 사전 등록한 그리드. 0.98 을 추가한 이유는 PREREG-loso.md 참고
# (MOT17-10, -13 의 최적이 상단 경계 0.95 에 붙어 있었다).
GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
OLD_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
OUT = Path("data/exp06/grid.json")

# ---- 임계값의 정본 부호화. **폴더 이름도 JSON 키도 여기서만 나온다** ----
#
# 감사에서 나온 결함: 폴더는 `int(round(th*100))`, JSON 키는 `"%.2f"` 로
# 반올림 방식이 달라서 0.01 보다 촘촘한 격자를 넣으면 둘이 **조용히 어긋났다**
# (0.975 -> 키 "0.97" 인데 폴더는 th098). 이제 둘 다 `code()` 에서 나오고,
# 격자가 해상도에 안 떨어지면 어긋나는 대신 **죽는다.**
#
# `headroom.run()` 의 인라인 폴더명도 같은 식(`th%03d` of round(th*100))이다.
# RES 를 바꾸면 거기도 같이 고쳐야 한다.
RES = 100                       # 격자 해상도. 바꾸려면 여기만 고친다
DEC = len(str(RES)) - 1         # JSON 키의 소수 자릿수


def code(th):
    """임계값 -> 정수 부호. 해상도에 안 떨어지면 예외."""
    c = round(th * RES)
    if abs(th * RES - c) > 1e-9:
        raise ValueError(
            "격자 %r 이 1/%d 단위로 안 떨어진다. 폴더 이름과 JSON 키가 어긋나므로 "
            "RES 를 올리든지 격자를 바꿔라 (grid.py 의 RES)" % (th, RES))
    return int(c)


def tag(th):
    return "th%03d" % code(th)


def jkey(th):
    return "%.*f" % (DEC, code(th) / RES)


def canon(th):
    """정본 float. 같은 임계값이면 언제나 같은 double 이 나온다."""
    return code(th) / RES


def digests(th):
    """주어진 임계값 폴더의 트랙 파일 해시. 없으면 빈 dict."""
    d = OUTROOT / tag(th)
    out = {}
    for seq in SEQS:
        f = d / ("%s.txt" % seq)
        if f.exists():
            out[seq] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def main():
    print("=" * 92)
    print("실험 6 LOSO [1단계] -- 그리드 표 생성. 사전 등록 PREREG-loso.md")
    print("=" * 92)
    print("그리드 %d개: %s" % (len(GRID), GRID))
    print()

    # 재생 전에 격자가 부호화 가능한지 먼저 본다. 110분 돌린 뒤에 죽으면 늦다
    for t in GRID:
        code(t)

    before = {th: digests(th) for th in GRID}

    table = {}          # th -> {seq: HOTA}
    combined = {}       # th -> 결합 HOTA (검출 수 가중)
    hdr = "%-8s%9s   " % ("thresh", "결합") + "".join(
        "%9s" % s.replace("-FRCNN", "").replace("MOT17-", "") for s in SEQS)
    print(hdr)
    print("-" * 92)
    for th in GRID:
        comb, per = run(th)
        table[th] = per
        combined[th] = comb
        line = "%-8.2f%9.3f   " % (th, comb) + "".join(
            "%9.2f" % per.get(s, float("nan")) for s in SEQS)
        print(line + ("   <- 기본값" if abs(th - 0.8) < 1e-9 else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "grid": GRID,
        "seqs": SEQS,
        "combined": {jkey(t): combined[t] for t in GRID},
        "per_seq": {s: {jkey(t): table[t].get(s) for t in GRID} for s in SEQS},
    }, indent=1))
    print()
    print("표를 %s 에 저장했다" % OUT)

    # ---------------- [0c] 재생 결정성 ----------------
    print()
    print("=" * 92)
    print("[0c] 재생 결정성 -- 어제 만든 트랙 파일과 오늘 만든 것이 같은가")
    print("=" * 92)
    checked = same = 0
    for th in GRID:
        old, new = before[th], digests(th)
        for seq in old:
            if seq in new:
                checked += 1
                same += int(old[seq] == new[seq])
    if checked == 0:
        print("  비교할 기존 파일이 없다 (첫 실행). 건너뛴다")
    else:
        print("  %d/%d 파일이 비트 단위로 동일" % (same, checked))
        if same != checked:
            print("  ** 재생이 결정적이지 않다. 원인을 찾기 전에는 판정하지 말 것 **")

    # ---------------- [0] 라벨 대조 (사전 등록한 사전 점검) ----------------
    print()
    print("=" * 92)
    print("[0] 사전 점검 -- 기존 8개 격자에서 argmax 가 predictors.BEST 와 맞는가")
    print("=" * 92)
    # 기록이 없는 시퀀스(신규 추가)나 결과가 없는 시퀀스(캐시 미비)는 대조에서
    # 빼되 **조용히 빼지 않는다.** 빼고 나서 대조할 게 하나도 안 남으면
    # 사전 점검이 아무것도 검사 못 한 것이므로 통과시키지 않는다.
    ok = checked = 0
    for s in SEQS:
        cand = {t: table[t][s] for t in OLD_GRID if s in table[t]}
        exp = BEST_OLD.get(s)
        if not cand:
            print("  %-18s 그리드 결과 없음 (캐시 미비)  -- 대조 제외" % s)
            continue
        got = max(cand, key=cand.get)
        if exp is None:
            print("  %-18s argmax %.2f   기록 없음 (신규 시퀀스)  -- 대조 제외"
                  % (s, got))
            continue
        checked += 1
        hit = abs(got - exp) < 1e-9
        ok += int(hit)
        print("  %-18s argmax %.2f   기록 %.2f   %s"
              % (s, got, exp, "OK" if hit else "** 불일치 **"))
    print("  => %d/%d 일치 (대조한 시퀀스 %d개)" % (ok, checked, checked))
    if checked == 0:
        print("  ** 대조할 기록이 하나도 없다. 사전 점검이 아무것도 검사하지 못했다 **")
        return 1
    if ok != checked:
        print("  ** 사전 점검 [0] 실패. 재생 경로가 틀렸다. 여기서 멈춘다 **")
        return 1

    # ---------------- 0.98 이 라벨을 바꿨는가 ----------------
    print()
    print("=" * 92)
    print("0.98 추가가 라벨을 바꿨는가 (사전 등록에서 예상한 변화)")
    print("=" * 92)
    moved = 0
    for s in SEQS:
        cand = {t: table[t][s] for t in GRID if s in table[t]}
        new_best = max(cand, key=cand.get)
        old_best = BEST_OLD[s]
        gain = cand[new_best] - cand[0.8]
        flag = ""
        if abs(new_best - old_best) > 1e-9:
            moved += 1
            flag = "   <- 바뀜 (%.2f 였다)" % old_best
        edge = "  [그리드 끝]" if new_best in (GRID[0], GRID[-1]) else ""
        print("  %-18s 최적 %.2f  HOTA %.2f  0.8 대비 %+.2f%s%s"
              % (s, new_best, cand[new_best], gain, flag, edge))
    print("  => %d개 시퀀스의 라벨이 바뀌었다" % moved)

    print()
    print("다음: python experiments/exp06_levers/loso.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
