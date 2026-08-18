# -*- coding: utf-8 -*-
"""본문의 "N.M절" 평문 상호참조가 실제 절 번호와 맞는지 검사한다.

`\\ref` 를 안 쓰고 숫자를 손으로 적었기 때문에 **절을 하나 끼워 넣으면 조용히
어긋난다.** 실제로 6.4 에 절을 신설하면서 어긋났다.

각 참조 옆에 그 번호가 가리키는 절 제목을 찍어 준다. 사람이 읽고 판단한다.
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BS = chr(92)

t = io.open(sys.argv[1], encoding="utf-8").read()

# 장/절 번호를 실제 순서대로 매긴다
num = {}
ch = 0
sec = 0
for m in re.finditer(BS * 2 + r"(chapter|section)\{([^}]*)\}", t):
    kind, title = m.group(1), m.group(2)
    if kind == "chapter":
        ch += 1
        sec = 0
    else:
        sec += 1
        num["%d.%d" % (ch, sec)] = title

print("실제 절 번호")
print("-" * 70)
for k in sorted(num, key=lambda s: tuple(int(x) for x in s.split("."))):
    print("  %-6s %s" % (k, num[k]))

print()
print("본문의 평문 참조")
print("-" * 70)
bad = 0
seen = {}
for m in re.finditer(r"(\d+)\.(\d+)절", t):
    key = "%s.%s" % (m.group(1), m.group(2))
    seen[key] = seen.get(key, 0) + 1
for k in sorted(seen, key=lambda s: tuple(int(x) for x in s.split("."))):
    title = num.get(k)
    if title is None:
        print("  %-6s x%-3d !! **그런 절이 없다**" % (k, seen[k]))
        bad += 1
    else:
        print("  %-6s x%-3d -> %s" % (k, seen[k], title))

print()
print("=== 존재하지 않는 참조 %d 종 ===" % bad)
sys.exit(1 if bad else 0)
