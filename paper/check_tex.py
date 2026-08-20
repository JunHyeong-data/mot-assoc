# -*- coding: utf-8 -*-
"""report.tex 구조 검사. xelatex 이 없으니 컴파일 대신 이걸로 잡는다."""
import io
import re
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BS = chr(92)  # 백슬래시. 헤어독이 백슬래시를 먹는 환경이라 상수로 둔다
t = io.open(sys.argv[1], encoding="utf-8").read()
ok = True

b = collections.Counter(re.findall(BS * 2 + r"begin\{([\w\*]+)\}", t))
e = collections.Counter(re.findall(BS * 2 + r"end\{([\w\*]+)\}", t))
for k in sorted(set(b) | set(e)):
    if b[k] != e[k]:
        print("환경 불일치: %-16s begin %d / end %d" % (k, b[k], e[k]))
        ok = False
print("환경 %d종 확인" % len(set(b) | set(e)))

op = len(re.findall(r"(?<!" + BS*2 + r")\{", t))
cl = len(re.findall(r"(?<!" + BS*2 + r")\}", t))
print("중괄호  { %d   } %d   %s" % (op, cl, "OK" if op == cl else "불일치"))
ok &= op == cl

d = len(re.findall(r"(?<!" + BS*2 + r")\$", t))
print("수식 $  %d개  %s" % (d, "OK(짝수)" if d % 2 == 0 else "홀수 -- 문제"))
ok &= d % 2 == 0

c = set(x.strip() for s in re.findall(BS * 2 + r"cite[tp]?\{([^}]*)\}", t)
        for x in s.split(","))
bi = set(re.findall(BS * 2 + r"bibitem\[[^\]]*\]\{([^}]*)\}", t))
print("인용했으나 bibitem 없음:", sorted(c - bi) or "없음")
print("bibitem 있으나 미인용   :", sorted(bi - c) or "없음")
ok &= not (c - bi)

lab = set(re.findall(BS * 2 + r"label\{([^}]*)\}", t))
ref = set(re.findall(BS * 2 + r"ref\{([^}]*)\}", t))
print("ref 했으나 label 없음   :", sorted(ref - lab) or "없음")
# 절 라벨(sec:*)은 참조 안 될 수 있다 -- 잡음이라 뺀다.
# 표/그림/식 라벨이 미참조면 그건 진짜 문제다.
orphan = sorted(x for x in (lab - ref) if not x.startswith("sec:"))
print("label 있으나 미참조     :", orphan or "없음", "(sec:* 제외)")
# **판정에 반영한다.** 예전에는 찍기만 하고 통과시켰다 -- tab:withinrow 가
# 인용 0회인 채로 "구조 검사 통과" 를 받았다. 표/그림이 미참조면 진짜 문제다.
ok &= not orphan
ok &= not (ref - lab)

# 표 열 개수 -- LaTeX 에서 제일 흔한 컴파일 실패다
ROW_END = BS * 2
for m in re.finditer(BS * 2 + r"begin\{tabular\}\{([^}]*)\}(.*?)" + BS * 2
                     + r"end\{tabular\}", t, re.S):
    spec, body = m.group(1), m.group(2)
    bare = re.sub(r"\{[^}]*\}", "", spec)
    ncol = len(re.sub(r"[^lcr]", "", bare))
    for line in body.split(ROW_END):
        line = re.sub(BS * 2 + r"multirow\{\d+\}\{[^}]*\}", "", line)
        line = re.sub(BS * 2 + r"multicolumn\{(\d+)\}",
                      lambda x: "&" * (int(x.group(1)) - 1), line)
        line = re.sub(BS * 2 + r"(top|mid|bottom)rule", "", line)
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        n = s.count("&") + 1
        if n != ncol:
            print("표 열 수 이상: 기대 %d, 실제 %d  ->  %s" % (ncol, n, s[:70]))
            ok = False

# 캡션 없는 그림/표
for env in ("figure", "table"):
    for m in re.finditer(BS * 2 + r"begin\{" + env + r"\}(.*?)" + BS * 2
                         + r"end\{" + env + r"\}", t, re.S):
        if BS + "caption" not in m.group(1):
            print("%s 에 caption 없음" % env)
            ok = False

for f in re.findall(BS * 2 + r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
    import os
    print("그림 %-40s %s" % (f, "있음" if os.path.exists(f) else "!! 없음"))
    ok &= os.path.exists(f)

if chr(11) in t or chr(12) in t:
    print("!! 제어문자가 들어 있다")
    ok = False

print()
print("=== %s ===" % ("구조 검사 통과" if ok else "문제 있음"))
sys.exit(0 if ok else 1)
