# 논문 읽기

읽을 때마다 **같은 세 가지**를 적는다. 이게 모이면 서베이가 된다.

1. 비용행렬에 **무엇이** 들어가는가
2. 그게 **어디서** 오는가
3. 안 들어가는 건 **왜** 안 들어가는가

---

| # | 논문 | 상태 | 노트 |
|---|---|---|---|
| 1 | SORT (Bewley et al., ICIP 2016) | 정독 완료 | `sort.md` |
| 2 | DeepSORT (Wojke et al., ICIP 2017) | **정독 완료** | `primary_reading.md` 2절 |
| 3 | ByteTrack (Zhang et al., ECCV 2022) | **정독 완료** | `primary_reading.md` 1절 |
| 4 | OC-SORT (Cao et al., CVPR 2023) | | |
| 5 | UCMCTrack (Yi et al., AAAI 2024) | **정독 완료** | `primary_reading.md` 4절 |
| 6 | Bae & Yoon, CVPR 2014 — Tracklet Confidence | **정독 완료** | `primary_reading.md` 5절 |
| 7 | **Bae & Yoon, TPAMI 40(3) 2018 — Confidence-Based Data Association** | 초록 (IEEE 유료). **CVPR'14 원판으로 대체 정독** | `primary_reading.md` 5절 |
| 8 | **UTrack (Solano-Carrillo et al., ECCV24 UnCV)** | 정독 + **재현 완료** | `../experiments/exp02_utrack_replication/` |
| 9 | **UncertaintyTrack (Lee & Waslander, arXiv:2402.12303)** | 절제표 확인 | `uncertainty_related_work.md` |
| 10 | Localization-Guided Track (arXiv:2309.09765) | **정독 완료** | `primary_reading.md` 3절 |
| 10b | Deep LG-Track (2504.01457) | 미독 | 후속. 캐스케이드 변경 여부 확인 필요 |

**7~10 은 `uncertainty_related_work.md` 에서 "불확실성이 어느 통로로 할당에
도달하는가" 라는 네 번째 질문으로 함께 대조한다.** 그 표가 이 연구의 논지다.

---

## 각 논문에서 특별히 볼 것

**DeepSORT**
- 마할라노비스 거리 $d^{(1)}$ 정의, 게이팅 임계값 9.4877의 출처
- $d^{(1)}$ 과 외형 거리 $d^{(2)}$ 를 합치는 식의 $\lambda$
- **카메라 모션이 클 때 $\lambda=0$ 을 쓴다는 대목** ← 마할라노비스를 게이팅에만 쓰고
  비용에서 빼는 것. 이 연구의 질문과 정면으로 만난다
- Matching cascade 를 age 순서로 나누는 이유

**ByteTrack**
- 낮은 점수 검출을 2단계로 다시 연관하는 절차
- 그때 쓰는 비용이 무엇인지 (여전히 IoU인지)
- **코드 대조는 2026-08-11 에 먼저 해뒀다** (ultralytics 8.4.46). 확인된 것:
  - 1단계·2단계 모두 `iou_distance` + `fuse_score`. 2단계 임계값은 0.5 하드코딩
  - `fuse_score` 가 `cost = 1 - IoU*s_j` 로 **신뢰도를 곱해 넣는다**. 기본값 True
  - 마할라노비스 게이팅 없음. `gating_distance` 는 호출되지 않는 죽은 코드
  - 헝가리안 후 `cost <= match_thresh` 로 자른다 → 불변성 논증이 여기서 깨진다
- **논문에서 확인할 것: 위 `fuse_score` 식이 논문에 있나, 구현에만 있나.**
  구현에만 있다면 근거 없이 표준이 된 설계이고, 그 자체가 이 연구의 소재다

**OC-SORT**
- 가림 중 칼만 추정이 어긋나는 걸 어떻게 표현하는지
- observation-centric 보정이 정확히 무엇을 고치는지
- 이 연구의 공분산 역설과 무엇이 같고 무엇이 다른지

**UCMCTrack**
- 지면 평면 마할라노비스, $\ln|\Sigma|$ 항의 역할
- $\Sigma$ 가 트랙 쪽인지 검출 쪽인지 ← **핵심**
- `detector/mapper.py` 의 `getUVError` 확인

**Bae CVPR 2014**
- tracklet confidence 를 detectability 와 continuity 로 정의하는 방식
- confidence 값에 따라 연관을 어떻게 나누는지
- confidence 를 **비용 안에 넣는 것**과 **문제를 쪼개는 것**의 차이

---

## 2026-08-18 — **원문 정독으로 숙제 하나가 풀렸다**

위 "ByteTrack" 항목에 이렇게 적어 뒀었다:

> *"논문에서 확인할 것: 위 `fuse_score` 식이 논문에 있나, 구현에만 있나.
> 구현에만 있다면 근거 없이 표준이 된 설계이고, 그 자체가 이 연구의 소재다."*

**확인했다. 구현에만 있다.** 전문 64,480자에서 `fuse` 는 1회(무관한 문맥),
`multiply`·`weighted by` 는 0회. 논문은 점수를 **높음/낮음 문지기**로만 쓰고
Similarity#1 은 *"IoU or Re-ID feature distances"*, Similarity#2 는 **IoU 단독**이다.

상세는 `primary_reading.md`.
