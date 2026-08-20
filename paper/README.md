# 원고

`report.tex` — 다중객체추적 데이터 연관에서 검출 불확실성은 활용 가능한가.

## 빌드

한글이 들어가므로 **xelatex + kotex** 이 필요하다. pdflatex 로는 안 된다.

```bash
cd paper && xelatex report.tex && xelatex report.tex
```

두 번 돌려야 목차와 상호참조(`\ref`)가 채워진다. `natbib` 을 쓰지만
`thebibliography` 를 직접 적었으므로 **bibtex 는 돌릴 필요가 없다.**

MiKTeX 이 없으면:

```bash
winget install MiKTeX.MiKTeX
```

## 그림

`figures/` 의 두 PDF 는 `experiments/figures/` 의 스크립트가 만든다.
다시 만들려면 저장소 뿌리에서:

```bash
python experiments/figures/fig_signal_transfer.py && python experiments/figures/fig_ceiling.py
```

산출물이 뿌리의 `figures/` 에 떨어지므로 `paper/figures/` 로 복사한다.

| 파일 | 논문 위치 | 무엇 |
|---|---|---|
| `fig_signal_transfer.pdf` | 그림 1 (5.1절) | 같은 σ, 같은 시퀀스, 다른 평가 대상. 위치 오차는 0 위, 연관 오류는 소스에 따라 갈린다 |
| `fig_ceiling.pdf` | 그림 2 (5.5절) | 여지는 +3.12 있는데 검출기 σ 경로 넷은 전부 0 아래 |

**`fig_source_channel.py` 는 더 이상 원고에 안 쓴다.** 7×5 격자에 글자를 채운
것이라 **사실 표였고**, 35칸 중 20칸이 비어 지면의 절반이 흰 칸이었다.
**표 2.1** 로 옮겼다 — 추정 방식으로 묶어 정렬하니 패턴이 세로로 읽힌다.
스크립트는 지우지 않는다 (CLAUDE.md 규칙 4).

공통 서식은 `experiments/figures/style.py` 에 있다. 그림은 **지면 크기(6.2in)로
직접 만들고 `\includegraphics` 에 width 옵션을 안 준다** — 배율이 없어야
8pt 로 설계한 글자가 지면에서도 8pt 다.

## 수치의 출처

원고의 모든 수치는 실험 스크립트를 **다시 돌려** 대조했다 (2026-08-18).
대조에서 네 건이 어긋나 고쳤다.

| 원고 위치 | 무엇이 틀렸나 | 고친 값 |
|---|---|---|
| 3.2절 | 검출 수를 GT 검출 수(53,890)로 적었다 | 캐시 실측 **63,137** |
| 5.2절 | **DFL 이 크기 모형을 6/7 이긴 것을 본문이 빠뜨렸다** (표에만 있었다) | 문단 추가 |
| 표 5.3 | 곱셈의 95.88% 는 전체이고 `N<=M` 만은 96.06% 다 | 조건을 갈라 적음 |
| 6.4절 | 53,890 을 "연관 결정" 이라 적었다 | GT 검출 |

재현 명령:

```bash
python experiments/exp10_spine/verify_1f.py && python experiments/exp11_add_vs_mul/run.py && python experiments/exp12_ceiling/run.py && python experiments/exp14_cmc/run.py && python experiments/exp15_sigma_last/run.py
```

## 아직 안 채워진 것

**한계 (5)** — 5.6절의 핵심 측정(AUC)은 DFL 소스로만 했다. NMS 소스 캐시가
`experiments/exp05_wasserstein/cache_nms.py` 로 생성 중이다. 끝나면:

```bash
python experiments/exp15_sigma_last/run.py -nms && python experiments/figures/fig_signal_transfer.py
```

그림 2 오른쪽 판에 NMS 계열이 추가되고, 한계 (5) 를 결과로 옮길 수 있다.

## 검사기 셋 — 컴파일 못 하는 대신 이걸 돌린다

이 기계에 xelatex 이 없어 컴파일로 잡을 수 없는 것들을 기계로 본다.
**셋 다 손으로 찾은 실제 오류에서 나왔다.**

```bash
python paper/check_tex.py paper/report.tex && python paper/check_secref.py paper/report.tex && python paper/check_numbers.py paper/report.tex
```

| 검사기 | 무엇을 잡나 | 왜 만들었나 |
|---|---|---|
| `check_tex.py` | 환경 짝, 중괄호, 수식 `$`, `cite`/`bibitem`, `ref`/`label`, **표의 열 개수**, 그림 파일 존재 | 표 열 수 불일치는 LaTeX 컴파일 실패 1위다 |
| `check_secref.py` | 절 참조가 **평문으로 되돌아가지 않았는지** + 모든 절에 라벨이 있는지 | 절을 끼울 때마다 번호가 밀렸다 — **6곳, 그리고 2곳.** 원인을 없앴다(아래) |
| `check_numbers.py` | 핵심 수치 47개가 원고에 있는가 + **원고 안에서 일관된가** | 거리 함수 값이 다른 실험 조건이었던 것을 손으로 찾았다 |

`check_numbers.py` 의 표에는 각 수치의 **출처 실험**이 적혀 있다.
숫자가 바뀌면 그 실험을 다시 돌려 확인한다.

### 절 참조는 `
ef` 로 바꿨다

처음에는 `"5.2절"` 처럼 숫자를 손으로 적었다. 절을 하나 끼우면 뒤 번호가 전부
밀리는데 **아무도 안 알려준다.** 실제로 두 번 났다 — 6.4 신설 때 6곳, 7.2 신설 때 2곳.

검사기로 잡는 것보다 **원인을 없애는 것**이 옳다고 판단해, 평문 참조 **67곳**을
`
ef{sec:장-절}` 로 바꾸고 각 절에 라벨을 달았다. 이제 LaTeX 이 번호를 매기므로
**이 부류의 버그가 생길 수 없다.** `check_secref.py` 는 그 상태를 지키는
역행 방지 장치로 바뀌었다.

### `check_refs.py` — 참고문헌을 Crossref 로 대조

```bash
python paper/check_refs.py paper/report.tex
```

인용 하나가 이미 틀려 있었다 — `solano2024` 에 UTrack 이 아니라 같은 저자의
**다른 논문 제목**이 들어가 있었다. 그건 사람 눈으로 잡았고, 나머지를 기계로 봤더니
**심사 논문이 있는데 arXiv 로만 인용한 것이 둘** 나왔다:

| 항목 | 원고에 있던 것 | 실제 게재 |
|---|---|---|
| `lee2024` UncertaintyTrack | arXiv:2402.12303 | **ICRA 2024, 4946–4953** |
| `meng2023` LG-Track | arXiv:2309.09765 | **IEEE Sensors Journal 25, 5282–5293 (2025)** |

나머지 다섯(OC-SORT, GFLv2, UTrack, UCMCTrack, ByteTrack)은 권·쪽을 보강했다.

`kuhn1955`·`milan2016`·`wang2021` 은 arXiv 전용이거나 1955년 논문이라 Crossref 가
엉뚱한 것을 물어 온다. **손으로 확인하고** 스크립트의 `KNOWN_MISS` 에 등록했다.
