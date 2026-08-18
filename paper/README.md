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
python experiments/figures/fig_source_channel.py && python experiments/figures/fig_signal_transfer.py
```

산출물이 뿌리의 `figures/` 에 떨어지므로 `paper/figures/` 로 복사한다.

| 파일 | 논문 위치 | 무엇 |
|---|---|---|
| `fig_signal_transfer.pdf` | 그림 1 (5.1절) | 같은 σ, 같은 시퀀스, 다른 평가 대상. 위치 오차는 0 위, 연관 오류는 소스에 따라 갈린다 |
| `fig_ceiling.pdf` | 그림 2 (5.5절) | 여지는 +3.12 있는데 검출기 σ 경로 넷은 전부 0 아래 |

**`fig_source_channel.py` 는 더 이상 원고에 안 쓴다.** 7×5 격자에 글자를 채운
것이라 **사실 표였고**, 35칸 중 20칸이 비어 지면의 절반이 흰 칸이었다.
**표 2.1** 로 옮겼다 — 산출 방식으로 묶어 정렬하니 패턴이 세로로 읽힌다.
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
