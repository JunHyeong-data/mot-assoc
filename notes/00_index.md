# 논문 읽기

읽을 때마다 **같은 세 가지**를 적는다. 이게 모이면 서베이가 된다.

1. 비용행렬에 **무엇이** 들어가는가
2. 그게 **어디서** 오는가
3. 안 들어가는 건 **왜** 안 들어가는가

---

| # | 논문 | 상태 | 노트 |
|---|---|---|---|
| 1 | SORT (Bewley et al., ICIP 2016) | 정독 완료 | `sort.md` |
| 2 | DeepSORT (Wojke et al., ICIP 2017) | | |
| 3 | ByteTrack (Zhang et al., ECCV 2022) | | |
| 4 | OC-SORT (Cao et al., CVPR 2023) | | |
| 5 | UCMCTrack (Yi et al., AAAI 2024) | | |
| 6 | Bae & Yoon, CVPR 2014 — Tracklet Confidence | | |

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
- `ultralytics/trackers/byte_tracker.py` 로 코드 대조 가능

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
