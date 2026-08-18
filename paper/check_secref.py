# -*- coding: utf-8 -*-
"""절 상호참조가 **다시 평문으로 돌아가지 않았는지** 지킨다.

## 왜 이 파일이 이렇게 생겼는가 — 두 번 데였다

원고가 처음에는 `"5.2절"` 처럼 **숫자를 손으로** 적고 있었다. 그러면 절을 하나
끼워 넣을 때 뒤 번호가 전부 밀리는데 **아무도 안 알려준다.**

  * 6.4 에 절을 신설 -> **6곳**이 어긋났다
  * 7.2 에 절을 신설 -> **2곳**이 어긋났다

처음에는 "참조가 가리키는 절 제목을 찍어 주는" 검사기를 만들었는데, 밀린 번호도
**존재하는 절**을 가리키므로 존재 검사로는 안 잡히고 낱말 heuristic 은 오탐이
많았다. 그래서 **원인을 없앴다** -- 평문 참조 67곳을 전부 `\\ref` 로 바꾸고
각 절에 `\\label{sec:장-절}` 을 달았다. 이제 LaTeX 이 번호를 매긴다.

**이 파일은 그 상태를 지키는 역행 방지 장치다.**

사용법:
    python paper/check_secref.py paper/report.tex
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BS = chr(92)


def main(p):
    t = io.open(p, encoding="utf-8").read()
    bad = 0

    # --- 1. 평문 참조가 남았는가 (핵심) ---
    plain = []
    for m in re.finditer(r"(?<!\})(\d+)\.(\d+)절", t):
        ctx = t[max(0, m.start() - 40):m.end()].replace("\n", " ")
        plain.append((m.group(0), ctx[-46:]))
    if plain:
        print("!! **평문 절 참조가 %d곳 남았다.** ref 로 바꿔라:" % len(plain))
        for s, ctx in plain[:12]:
            print("     %-8s ...%s" % (s, ctx))
        bad += len(plain)
    else:
        print("평문 절 참조 0곳. **전부 ref 로 되어 있다.**")

    # --- 2. 절마다 라벨이 있는가 ---
    ch = sec = 0
    nolabel = []
    for m in re.finditer(BS * 2 + r"(chapter|section)\{([^}]*)\}(" + BS * 2
                         + r"label\{([^}]*)\})?", t):
        if m.group(1) == "chapter":
            ch += 1
            sec = 0
            continue
        sec += 1
        want = "sec:%d-%d" % (ch, sec)
        got = m.group(4)
        if got != want:
            nolabel.append(("%d.%d" % (ch, sec), m.group(2)[:34], got))
    if nolabel:
        print()
        print("!! 라벨이 없거나 번호와 어긋난 절 %d개:" % len(nolabel))
        for k, title, got in nolabel:
            print("     %-6s %-36s 라벨=%s" % (k, title, got))
        bad += len(nolabel)
    else:
        print("모든 절에 sec:장-절 라벨이 붙어 있다.")

    # --- 3. ref 가 있는 라벨을 가리키는가 ---
    labels = set(re.findall(BS * 2 + r"label\{(sec:[^}]*)\}", t))
    refs = set(re.findall(BS * 2 + r"ref\{(sec:[^}]*)\}", t))
    dangling = sorted(refs - labels)
    if dangling:
        print()
        print("!! 없는 절 라벨을 가리키는 ref: %s" % dangling)
        bad += len(dangling)

    print()
    print("=== 문제 %d 건 ===" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
