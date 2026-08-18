# 원문 정독 — 요약이 아니라 논문을 직접 읽었다 (2026-08-18)

> 지금까지 이 저장소의 문헌 서술은 **초록·절제표·웹검색 요약**에 기대고 있었다.
> 원고의 척추가 문헌 주장에 걸려 있으므로 **PDF 를 받아 본문을 직접 읽었다.**
> `notes/00_index.md` 의 세 질문 형식을 따른다 —
> **무엇이 비용에 들어가는가 / 어디서 오는가 / 안 들어가는 건 왜.**
>
> **결과: 뼈대가 두 군데 바뀌었다.** 하나는 강해지고 하나는 약해졌다.

---

## 1. ByteTrack (Zhang et al., ECCV 2022, arXiv 2110.06864) — **정독**

### 무엇이 비용에 들어가는가

- **Similarity#1** = *"IoU or the Re-ID feature distances"*
- **Similarity#2** = **IoU 단독.** 논문이 근거를 명시한다 —
  *"the low score detection boxes usually contains severe occlusion or motion blur
  and appearance features are not reliable"*
- 절제표(Table 1)로 확인: Similarity#2 를 Re-ID 대신 IoU 로 하면 MOTA 가 약 1.0 오른다
- 임계값: *"if the IoU between the detection box and the tracklet box is smaller than
  0.2, the matching will be rejected"* → 비용으로는 0.8. ultralytics `match_thresh=0.8` 과 일치
- 검출 점수 임계 **τ = 0.6** (기본값)

### **핵심 확인 — `fuse_score` 는 논문에 없다**

전문(64,480자)에서 검색했다:

| 낱말 | 출현 |
|---|---|
| `fuse` | **1회** — 그것도 관련 없는 문맥 (*detection-by-tracking* 절에서 예측 상자와 검출 상자를 합치는 이야기) |
| `fusing` / `multiply` / `weighted by` | **0회** |

**`cost = 1 − IoU·s` 라는 식이 논문 어디에도 없다.** 점수는 **높음/낮음을 가르는
문지기**로만 쓰이고 **비용 안으로는 안 들어간다.**

> ### 이것이 `00_index.md` 가 남긴 숙제의 답이다
>
> 거기 이렇게 적혀 있었다: *"논문에서 확인할 것: 위 `fuse_score` 식이 논문에 있나,
> 구현에만 있나. 구현에만 있다면 **근거 없이 표준이 된 설계**이고, 그 자체가
> 이 연구의 소재다."*
>
> **구현에만 있다. 확인됐다.**

ultralytics 는 여기에 더해 `track_high_thresh=0.25` 를 쓴다 — 논문의 τ=0.6 과도 다르다.

---

## 2. DeepSORT (Wojke et al., ICIP 2017, arXiv 1703.07402) — **정독**

### 무엇이 비용에 들어가는가

```
c(i,j) = λ · d⁽¹⁾(i,j)  +  (1−λ) · d⁽²⁾(i,j)          ← 가중 "합"
```

`d⁽¹⁾` 은 마할라노비스, `d⁽²⁾` 은 외형 코사인 거리. **λ 는 전역 상수**다.

> **주의 — 이건 우리 설계 제약의 반례가 아니다.** 제약이 금지하는 것은
> **검출 하나에만 의존하는 스칼라를 더하는 것**(순수 열상수)이다.
> `d⁽¹⁾`, `d⁽²⁾` 은 **둘 다 쌍(i,j)에 의존**하므로 그 합은 열상수가 아니다.
> DeepSORT 는 애초에 **검출별 스칼라를 안 넣는다.**

### 캐스케이드는 **트랙 나이** 순이다

*"Select tracks by age 𝒯ₙ ← {i ∈ 𝒯 | aᵢ = n}"*, n 을 1 부터 A_max 까지 올린다.
*"a matching cascade that gives priority to more frequently seen objects"*

**검출 쪽 불확실성은 순서에도 게이팅에도 안 쓰인다.** 검출 신뢰도는 0.3 문지기로만.

### **가장 값진 것 — 캐스케이드의 동기가 공분산 역설 그 자체다**

> *"when two tracks compete for the same detection, the Mahalanobis distance
> favors larger uncertainty ... This is an undesired behavior as it can lead to
> increased track fragmentations and unstable tracks."*

**DeepSORT 가 2017년에 이 역설을 정확히 지목했다.** 그리고 **비용을 고치는 대신
캐스케이드로 우회했다.** `theory/covariance_paradox.py` 가 형식화한 바로 그것이다.

> **원고 서론이 여기서 시작해야 한다.** "아무도 몰랐다" 가 아니라
> **"9년 전에 지목됐고, 고치는 대신 우회했고, 그 뒤로 아무도 다시 안 열었다."**

---

## 3. LG-Track (Meng et al., arXiv 2309.09765) — **정독. 여기서 뼈대가 바뀐다**

### 무엇이 비용에 들어가는가 — **곱이다**

```
C₁ = C_iou · d_l        (식 1)   d_l = localization confidence
C₃ = C_cos · d_s        (식 3)   d_s = detection confidence
```

**둘 다 곱이다.** 그리고 **왜 곱이어야 하는지에 대한 설명은 없다.**
*"a proper cost matrix is selected"* 수준의 설계 서술뿐이다.

### ⚠ **예상 못 한 것 — 이미 4단계 캐스케이드를 쓴다**

> *"The core of deep association is a four-level matching mechanism:
> (1) detection boxes with **high localization confidence and high classification
> confidence** are assigned the highest priority ... (2) high localization but low
> classification ... (3) low localization and high classification ... (4) low
> localization confidence and low classification confidence."*

**즉 LG-Track 은 연관을 검출 쪽 위치 품질 신호로 순서 지어 나눈다.**

> ### 이게 내 실험 제안 E2 를 직접 때린다
>
> 나는 `paper_skeleton.md` 에 *"검출 쪽 σ 순위로 캐스케이드를 나눈 사례가 없다"*
> 고 적었다. **틀렸다.** LG-Track 이 정확히 그 통로를 쓰고 있고, **이득도 보고한다.**
>
> **읽지 않았으면 이미 점유된 통로를 '빈 칸' 이라고 주장할 뻔했다.**

---

## 이 정독이 원고를 어떻게 바꾸는가

### 강해지는 것 ①  설계 제약 절이 훨씬 단단해진다

주장을 **정밀하게** 다시 쓴다:

> **문헌이 검출별 스칼라(신뢰도, 위치 품질)를 비용에 넣을 때는 예외 없이 곱이다.
> 덧셈으로 넣은 사례가 없다. 그리고 왜 곱이어야 하는지 말한 사례도 없다.**

| 방법 | 검출별 스칼라 | 형태 | 근거 제시 |
|---|---|---|---|
| ByteTrack **논문** | 없음 (점수는 문지기) | — | — |
| ByteTrack **구현** | `s_j` | **곱** `1−IoU·s` | **없음. 논문에 식 자체가 없다** |
| LG-Track | `d_l`, `d_s` | **곱** `C_iou·d_l` | **없음** |
| DeepSORT | 없음 | (쌍 항의 가중합) | 해당 없음 |

**DeepSORT 를 표에 넣는 것이 중요하다** — 반례처럼 보이지만 아니고,
**검출별 스칼라를 아예 안 넣는다**는 것이 제약의 범위를 정확히 보여 준다.

### 강해지는 것 ②  서론이 생겼다

**DeepSORT 가 2017년에 공분산 역설을 지목하고 우회했다.** 원고는
*"그 우회를 9년 만에 다시 열어 본다"* 로 시작할 수 있다.

### 약해지는 것 ③  **E2 를 다시 설계해야 한다**

캐스케이드 통로는 **비어 있지 않다.** 그러나 **더 나은 실험이 된다** —
이 프로젝트의 논지가 **소스 × 통로**이기 때문이다.

> **새 E2**: LG-Track 이 **이득을 보고한 그 통로**(위치 품질로 연관을 순서 지음)에
> **우리 소스(NMS σ, DFL σ)를 넣는다.** 통로는 고정, 소스만 간다.
>
> - **이득이 나오면** — 통로가 아니라 소스가 문제였다는 직접 증거
> - **이득이 안 나오면** — **작동하는 것이 확인된 통로에서도 σ 가 실패한다.**
>   이건 지금까지 중 **가장 강한 음성 결과**다. "통로를 잘못 골랐다" 는 반박이 죽는다
>
> **어느 쪽이든 이긴다. 그리고 원래 E2 보다 훨씬 강하다.**

한 가지 유의: LG-Track 의 `d_l`(예측 IoU 품질)과 우리 σ(산포)는 **다른 양**이다.
"같은 통로, 다른 소스" 라고 정확히 쓰고, **품질 ≠ 불확실성**임을 명시한다.

---

## 아직 안 읽은 것 (정직하게)

- **Deep LG-Track** (arXiv 2504.01457) — 후속. 캐스케이드가 바뀌었는지 확인 필요
- **Bae & Yoon TPAMI'18** — **CVPR'14 원판을 읽었다(5절).** TPAMI 판은 IEEE
  유료라 미독. 개념(tracklet confidence)은 원판에 다 있으나 **딥 외형학습 부분은
  TPAMI 판이 확장했으므로** 인용할 때 판본을 구분할 것
- ~~UCMCTrack~~ — **읽었다 (4절).** 여기가 이번 정독의 최대 수확
- **OC-SORT** (2203.14360) — observation-centric 보정이 역설과 무엇이 다른가
- **UncertaintyTrack** — 절제표만 봤다. 본문 미독

## 추출 방법 (재현용)

arXiv PDF 를 받아 `PyMuPDF` 로 본문을 뽑았다. **WebFetch 는 PDF 본문을 못 읽는다**
(이진 데이터로 온다). LG-Track 은 2단 조판이라 `get_text(sort=True)` 를 써야
수식이 순서대로 나온다.

---

## 4. UCMCTrack (Yi et al., AAAI 2024, arXiv 2312.08952) — **정독. 여기가 제일 값지다**

### 무엇이 비용에 들어가는가

```
D = ε^T S^{-1} ε + ln|S|              (식 8)  "normalized Mahalanobis distance"
S = H P H^T + R_k                     (식 7)
R_k = C R_uv,k C^T                    (식 4)  지면 평면으로 사상
```

### **어디서 오는가 — 이게 핵심이다**

```
R_uv,k = diag( (σ_m · w_k)² ,  (σ_m · h_k)² )        (식 2)
```

> *"σ_m represents the detection noise factor as a **hyper-parameter**,
> and w_k and h_k denote the **detected width and height** from the detector."*

**이건 상자 크기 모형 그 자체다. 우리 실험의 기준선 `C` 다.**
검출기가 내는 불확실성이 아니라 **상자 크기 × 전역 상수 하나**.

### ln|S| 는 공분산 역설을 막으려고 넣은 것이다

> *"This ensures that data association decisions are not solely based on the
> discrepancies between measurements and predictions, but also holistically
> consider the accuracy and uncertainty of measurements."*

`ε^T S^{-1} ε` 만 쓰면 S 가 클수록 거리가 작아진다(역설). `ln|S|` 가 그걸 벌한다.
**이게 사용자 브리프의 A-LL(association log-likelihood) 형태다.**

### ⚠ 이 발견이 원고를 바꾼다 — **음성 결과가 설명력을 얻는다**

지금까지 `C`(상자 크기 모형)는 "우리가 세운 기준선" 이었다. 이제 이렇게 된다:

> **AAAI 2024 의 SOTA 트래커가 검출 측정잡음으로 정확히 `C` 를 쓴다.**
> 그리고 우리 실험 1f 는 **`C` 가 NMS σ 를 7/7 로 이긴다**고 보였다.

**즉 우리 음성 결과는 UCMCTrack 의 설계 선택을 설명한다.**

| | 무엇을 R 에 넣었나 | 결과 |
|---|---|---|
| **UCMCTrack** (AAAI'24) | **상자 크기** (`σ_m·w`, `σ_m·h`) | SOTA |
| UTrack (ECCV'24 W) | **검출기 NMS σ** | 우리 재현 **−0.62** |
| UncertaintyTrack | **검출기 공분산** | 저자 보고 **−0.1** |

> **같은 통로(칼만 R), 다른 소스. 상자 크기를 쓴 쪽만 이긴다.**
> 우리 실험 1f 의 만장일치가 **왜 그런지**를 준다.

**이건 순수한 음성 결과가 아니라 건설적 진술이다** — 원고의 무게중심이 여기로
옮겨갈 수 있다. `C` 는 허수아비가 아니라 **배포된 SOTA 의 실제 모형**이다
(NWD 에 이어 두 번째 확인).

---

## 5. Bae & Yoon (CVPR 2014) — **정독.** TPAMI'18 의 원판

> TPAMI'18 은 IEEE 유료라 못 받았다. **같은 저자의 CVPR 2014 원판**을 읽었다
> (tracklet confidence 개념의 출처). CVF 공개판.

### tracklet confidence (식 2)

```
conf(T^i) = (1/L) · Σ_{k: v_i(k)=1} Λ(T^i, z_k^i)  ×  max(1 + β·log((L−w)/L), 0)
```

- `L` = tracklet 길이, `w` = 가림 등으로 **놓친 프레임 수**, `Λ` = affinity
- 앞항 = **검출가능성**(평균 affinity), 뒷항 = **연속성**(놓친 프레임 벌점)
- `conf ∈ [0,1]`, **0.5** 로 높음/낮음을 가른다

### **어떻게 쓰이는가 — 비용이 아니라 라우팅이다**

> *"the tracklets with **high confidence** are first considered to be **locally
> associated with detections** ... "*

- **높은 confidence 트랙** → 검출과 **지역(local) 연관**
- **낮은 confidence 트랙** → 다른 트랙·검출과 **전역(global) 연관**

비용 자체는 `c_ij = −log(Λ(T^i(lo), y_j^t))` 로 **affinity 의 음의 로그**이고,
**confidence 는 비용 안에 안 들어간다.** 어느 연관 문제에 들어갈지를 정할 뿐이다.

**즉 트랙 쪽 양이고, 순수한 라우팅이다.** `direction.md` 의 기록이 맞았다.

### 그래서 재설계한 E2 의 자리가 정확해진다

| 방법 | 신호 | 어느 쪽 | 쓰임 |
|---|---|---|---|
| DeepSORT (2017) | 트랙 나이 | 트랙 | 캐스케이드 순서 |
| **Bae** (2014/2018) | tracklet confidence (검출가능성+연속성) | **트랙** | **라우팅** (지역/전역) |
| **LG-Track** (2023) | 위치 품질 + 분류 신뢰도 | **검출** | **4단계 캐스케이드** |
| **우리 E2** | **σ (산포)** | **검출** | 캐스케이드 순서 |

**Bae 의 confidence 는 트랙 이력 양**(길이·놓친 프레임·평균 affinity)이라
검출기 출력 불확실성과 **겹치지 않는다.** E2 와 혼동될 여지가 없다.

**E2 의 정확한 위치**: LG-Track 과 **같은 쪽·같은 통로**, **다른 소스**
(품질 아닌 산포). 그렇게 써야 정직하다.
