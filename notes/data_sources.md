# 데이터 출처와 라이선스

논문에 프레임을 싣거나 결과를 공개할 때 필요하다. 받을 때 바로 적는다.
실제 파일은 `data/` 에 있고 git 에서 제외된다.

---

## 보행자 밀집 (2026-08-16 취득)

| 파일 | 출처 | 라이선스 | 귀속표시 의무 |
|---|---|---|---|
| `data/yonge_dundas.webm` | Wikimedia Commons, `DiagonalCrosswalkYongeDundas.webm` | **CC0** | 없음 |
| `data/shibuya.webm` | Wikimedia Commons, `Shibuya Crossing, Tokyo, Japan (video).webm` | **CC BY-SA 4.0** | **있음** |

- 원본 페이지: `https://commons.wikimedia.org/wiki/File:<파일명>`
- 다운로드: `https://upload.wikimedia.org/wikipedia/commons/...`
- 사양: 둘 다 1920×1080. yonge_dundas 1356프레임 23.98fps, shibuya 1464프레임 25fps.

**CC BY-SA 4.0 주의** — shibuya 프레임을 논문·발표에 실으면 (1) 원저작자 표시,
(2) 라이선스 명시, (3) 개작물도 동일 라이선스. 그림으로 쓸 거면 CC0 인
yonge_dundas 를 쓰는 게 안전하다. 수치만 보고할 때는 해당 없다.

**개인정보** — 공공장소 보행자가 찍혀 있다. 수치 집계만 보고하고 얼굴이
식별되는 프레임은 논문에 싣지 않는다. 실을 거면 마스킹한다.

## 주행 (사용자 소유, 저장소 외부)

| 파일 | 위치 | 비고 |
|---|---|---|
| `sample_autobahn.mp4` | `../VLMDrivingAssis/assets/` | 551프레임 1080p 30fps |
| `hero.mp4` | `../VLMDrivingAssis/assets/` | 240프레임 720p 24fps. 객체 희소 |

출처·라이선스 **미확인**. 외부 공개 전에 확인할 것.

## 검출기

`../BSDsystem/yolov8m.pt` — 표준 COCO 80클래스 yolov8m (파인튜닝 아님).
보행자 영상은 `classes=0` (person) 으로 제한해 MOT17 의 보행자 전용 조건에 맞췄다.

## MOT17 (2026-08-16 취득) — 공식 아닌 미러에서 받았다

**논문에 반드시 명시할 것.** `motchallenge.net` 이 응답 없어(브라우저·curl 모두,
IPv4/IPv6 모두 443 연결 실패, DNS 는 정상 131.159.19.34 TUM) 공식 배포본을
받지 못했다. HuggingFace 의 **개인 계정 재업로드**에서 취득했다.

| 항목 | 값 |
|---|---|
| 취득처 | `Morrison1025/MOT17` (HuggingFace, 개인 계정) |
| 받은 범위 | `ablation/` — FRCNN 7개 시퀀스, 2673 파일, 0.41 GB |
| 위치 | `data/MOT17_A/ablation/` (git 제외) |
| 대조 미러 | `Lekim89/MOT17` (독립 업로더) |

`ablation/` 이 CenterTrack/ByteTrack 관례의 half-half split 이다.
MOT17-02-FRCNN 이 299프레임 = 원본 600프레임의 앞 절반으로 확인됐다.
GT 열 규약도 확인: `frame,id,x,y,w,h,ignore,class,visibility`.

### 무결성 검증

공식 체크섬을 못 쓰므로 **독립 미러 2개를 바이트 대조**한다
(`experiments/exp01_nms_variance/verify_data.py`).
사전 확인: 두 미러의 파일 목록 16660개가 이름·바이트크기 전부 일치.

**검증 결과 (2026-08-16) — 통과.**

| 대상 | 결과 |
|---|---|
| 주석파일 (gt/det/seqinfo) 전수 | **21/21 SHA256 일치**, 불일치 0 |
| 이미지 무작위 표본 | **120/120 SHA256 일치**, 불일치 0 |

**배제되지 않는 위험** — 두 업로더가 같은 훼손본을 퍼뜨린 경우.
`motchallenge.net` 복구 시 공식 체크섬으로 재검증해야 한다.
그전까지 논문에는 "공식이 아닌 미러에서 취득, 독립 미러 2개로 대조" 로 쓴다.

라이선스: MOT17 원본은 CC BY-NC-SA 3.0 (비상업). 미러의 `license` 필드는
`other` 로만 적혀 있어 원 라이선스를 따르는 것으로 간주한다.
