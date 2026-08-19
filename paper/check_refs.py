# -*- coding: utf-8 -*-
"""참고문헌을 **Crossref 로 대조**한다.

인용 하나가 이미 틀려 있었다 -- `solano2024` 에 UTrack 이 아니라 같은 저자의
**다른 논문 제목**이 들어가 있었다. 사람 눈으로 잡은 것이라 나머지도 기계로 본다.

제목으로 Crossref 를 조회해 **학술지·권·쪽·연도**를 받아 원고와 대조한다.
학회 논문(CVPR/ECCV/ICIP)은 Crossref 등록이 들쭉날쭉하므로 **제목 일치만** 본다.

사용법:
    python paper/check_refs.py paper/report.tex
"""
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BS = chr(92)

UA = "mot-assoc-refcheck (mailto:parkjune0310@gmail.com)"

# **알려진 오탐.** arXiv 전용이거나 오래된 논문이라 Crossref 가 엉뚱한 것을
# 물어 온다. 셋 다 2026-08-19 에 **손으로 확인했다**:
#   kuhn1955   Naval Research Logistics Quarterly 2(1-2):83-97 -- 맞다
#   milan2016  MOT16 벤치마크. arXiv:1603.00831 전용 -- 맞다
#   wang2021   NWD. arXiv:2110.13389 전용 -- 맞다
#   somers2025 CAMELTrack. arXiv:2505.01257 전용 (2026-08-19 arXiv 로 확인:
#              제목·저자 일치, 학회 게재 없음) -- Crossref 가 엉뚱한 것을 문다
KNOWN_MISS = {"kuhn1955", "milan2016", "wang2021", "somers2025"}


def crossref(title):
    q = urllib.parse.urlencode({
        "query.bibliographic": title, "rows": 1,
        "select": "title,container-title,volume,page,issued,type"})
    req = urllib.request.Request(
        "https://api.crossref.org/works?" + q, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            items = json.load(r)["message"]["items"]
    except Exception as e:
        return {"err": str(e)}
    if not items:
        return {"err": "결과 없음"}
    it = items[0]
    return {
        "title": (it.get("title") or [""])[0],
        "venue": (it.get("container-title") or [""])[0],
        "volume": it.get("volume", ""),
        "page": it.get("page", ""),
        "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "type": it.get("type", ""),
    }


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main(p):
    t = io.open(p, encoding="utf-8").read()
    i = t.index(BS + "begin{thebibliography}")
    j = t.index(BS + "end{thebibliography}")
    body = t[i:j]

    entries = re.split(BS * 2 + r"bibitem", body)[1:]
    print("참고문헌 %d편을 Crossref 로 대조한다" % len(entries))
    print("=" * 78)
    bad = 0
    for e in entries:
        key = re.search(r"\{([^}]*)\}", e[e.index("]") + 1:]).group(1)
        text = re.sub(BS * 2 + r"textit\{([^}]*)\}", r"\1", e)
        text = re.sub(BS * 2 + r"[a-zA-Z]+", " ", text)
        text = text.replace("&", " ").replace("{", " ").replace("}", " ")
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        # 2번째 줄이 저자, 3번째가 제목인 형식이다
        title = lines[2] if len(lines) > 2 else ""
        title = title.rstrip(".")
        if not title:
            print("  %-16s !! 제목을 못 뽑았다" % key)
            bad += 1
            continue

        r = crossref(title)
        time.sleep(0.7)                       # Crossref 예의
        if "err" in r:
            print("  %-16s ?  조회 실패 (%s)" % (key, r["err"]))
            continue

        hit = norm(title)[:44] in norm(r["title"]) or norm(r["title"])[:44] in norm(title)
        if not hit and key in KNOWN_MISS:
            print("  %-16s ~  Crossref 오탐 (손으로 확인함)" % key)
            continue
        mark = "OK" if hit else "!!"
        bad += (not hit)
        print("  %-16s %s %s" % (key, mark, r["title"][:62]))
        if hit and r["venue"]:
            vp = []
            if r["volume"]:
                vp.append("vol " + str(r["volume"]))
            if r["page"]:
                vp.append("pp " + str(r["page"]))
            print("  %-16s    %s %s (%s)"
                  % ("", r["venue"][:46], " ".join(vp), r["year"]))
            # 권/쪽이 원고에 있는가
            for v in (str(r["volume"]), str(r["page"]).replace("-", "--")):
                if v and v not in e:
                    print("  %-16s    ?  원고에 '%s' 가 없다" % ("", v))

    print()
    print("=== 제목 불일치 %d 건 ===" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
