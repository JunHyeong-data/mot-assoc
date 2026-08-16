# Colab 실행 절차 — UTrack 재현

로컬(Windows)에서는 못 돌린다. 확인된 이유:

- UTrack 은 Python 3.8 기준. 로컬은 3.12/3.13/3.14 뿐
- `lap==0.4.0`, `nms_var`, `fuzzy_cython_bbox`, `fast_gmc`, `cocoapi` 가 C 컴파일 필요
- **MSVC 컴파일러 없음** (`cl.exe` 부재). `lap==0.4.0` 빌드 실패 확인
- WSL 은 설치돼 있으나 배포판 없음, conda 없음

Colab 은 Linux + gcc + 무료 T4 라 컴파일 문제와 GPU 문제가 같이 풀린다.
검출 추론이 병목인데 T4 면 7개 시퀀스가 분 단위다.

---

## 셀 1 — 저장소와 의존성

```bash
!git clone https://github.com/DLR-MI/UTrack.git
%cd UTrack
!pip install -q cython
!pip install -q -r requirements.txt
```

`requirements.txt` 가 ultralytics 포크(`DLR-MI/ultralytics@nms-var`)를 끌어온다.
**표준 ultralytics 가 아니다.** NMS 에서 상자 분산을 같이 뱉도록 고친 버전이다.
설치 후 `import ultralytics; print(ultralytics.__file__)` 로 포크가 잡혔는지 확인할 것.

## 셀 2 — 데이터

공식 `motchallenge.net` 은 2026-08-16 기준 응답 없음. 검증된 HF 미러를 쓴다
(주석 21/21 · 이미지 120/120 SHA256 일치, `notes/data_sources.md` 참고).

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Morrison1025/MOT17', repo_type='dataset',
                  allow_patterns=['train/**','ablation/**','val/**'],
                  local_dir='/content/data/MOT17', max_workers=8)
```

UTrack 은 `--data_root /data/MOT17` 아래에 표준 MOT 구조를 기대하고,
COCO 형식 주석을 따로 만든다. 미러의 폴더명이 UTrack 기대와 다를 수 있으니
`tools/convert_mot17_to_coco.py` 의 `DATA_PATH` 를 실제 경로에 맞춘 뒤:

```bash
!python tools/convert_mot17_to_coco.py
!python tools/fix_yolo_annotations.py --folder /content/data/MOT17
```

설정이 `val_ann: val_half.json` 을 찾으므로 이 산출물이 나와야 한다. 안 나오면
변환 스크립트의 split 로직을 먼저 맞출 것. **여기가 첫 번째 실패 지점이다.**

## 셀 3 — 가중치

```bash
!mkdir -p yolov8l-mix/ablation_17/weights
!wget -O yolov8l-mix/ablation_17/weights/best.pt \
  "https://zenodo.org/records/13604403/files/ablation_17_best.pt?download=1"
```

논문이 쓴 그 가중치다. **직접 학습하지 말 것** — 재현에서 학습 차이를 배제해야 한다.

## 셀 4 — [1] 기준선 재현

```bash
!python track.py --project yolov8l-mix --exp ablation_17 \
  --data_root /content/data/MOT17 --association botsort --gpu_id 0
```

> **판정**: 저장소가 MOT17-ablation 에 대해 보고하는 BoT-SORT 수치와 비교.
> 벗어나면 이후 단계를 우리 재현값 기준으로 삼고 그 사실을 명시한다.

## 셀 5 — [2] 논문이 표로 안 준 ablation

변형이 이미 구현돼 있다 (`tracker/associations/collections.py`).
접두사: `uk_`=U-Kalman, `us_`=size disambiguator, `up_`=phase disambiguator.

```python
for assoc in ['botsort', 'uk_botsort', 'us_botsort', 'up_botsort', 'ukp_botsort']:
    !python track.py --project yolov8l-mix --exp ablation_17 \
      --data_root /content/data/MOT17 --association {assoc} --gpu_id 0
```

논문은 이걸 산점도(Fig.3)로만 보여준다. **표로 만드는 것이 우리 기여의 일부다.**

> **판정**: HOTA 기준 (uk−base) 와 (us−base), (up−base) 를 분리 보고.
> 논문 주장("U-Kalman 이 가장 두드러진다")이 맞으면 uk 가 최대여야 한다.

### 교란 하나 — NSA 를 통제할 것

`track_ablation_17.yaml` 의 `uncertain.kalman.nsa: True` 가 기본이다.
NSA 는 검출 score 로 R 을 이미 조정한다. 즉 기준선도 순수하지 않다.
**`nsa: False` 로도 같은 격자를 돌려** U-Kalman 의 순효과를 분리한다.

## 셀 6 — [3] 결정적 대조군 E

`tracker/kalman_filter.py` `project()` 의 해당 부분:

```python
std = self._std_weight_pos * whwh          # A: 트랙 자신의 w,h 에서
innovation_cov = np.diag(np.square(std))
...
if var_measurement is not None:
    innovation_cov = np.diag(var_measurement)   # B: 검출의 NMS 분산
```

E 는 `var_measurement` 를 **검출 높이만으로 만든 값**으로 바꾼다:

```python
# E: 측정 없이 크기만. h_det 은 해당 검출의 높이.
var_measurement = k * (h_det ** 2) * np.ones(4)
```

`k` 는 val_half 에서 HOTA 최대가 되도록 격자탐색한다. **E 에 유리하게 준다** —
불리하게 잡아 이기는 것은 의미가 없다.

> **판정 (사전 선언, 바꾸지 않는다)**
> - E 와 B 의 HOTA 차이 **0.3 이내** → 이득은 측정된 불확실성이 아니라
>   **R 의 출처를 트랙에서 검출로 옮긴 것**에서 온다. 논문의 핵심 주장이 무너진다.
> - E 가 B 보다 **0.3 이상 낮음** → 측정에 크기 이상의 정보가 있다.
>   실험 1 의 부정 결과와 모순되므로 **실험 1 을 먼저 재검토**한다.
> - E 가 B 보다 **0.3 이상 높음** → 측정이 오히려 해롭다. 실험 1 과 일치한다.

## 셀 7 — [4] 보정 분석

`experiments/exp01_nms_variance/analyze_covariance.py` 를 그대로 쓴다.
포크된 ultralytics 가 NMS 분산을 직접 주므로 우리 가로채기 코드보다 정확하다.
**우리 구현과 포크의 분산이 일치하는지 먼저 대조할 것** — 어긋나면
실험 1 의 부정 결과가 우리 추출 오류였을 가능성을 배제해야 한다.

## 셀 8 — [5] 혼잡 층화

시퀀스별로 (uk−base) 를 내고, `exp00/census.py` 의 M·N·절단률·게이트밀도와 나란히 둔다.
GT visibility 구간별 z² 도 같이. 논문이 "MOT20 에서 baseline 이 더 낫다"고
**말만 하고 수치를 안 준** 부분이다.

---

## 결과를 가져올 것

`track_results/` 아래 지표가 쌓인다. 로컬로 내려받아 `data/exp02/` 에 두고
표로 정리한다. 원자료(시퀀스별 HOTA/AssA/DetA/IDF1)를 같이 보관할 것.

## 부수 확인 — ultralytics 2단계 버그

이 설정 파일이 exp00 발견을 확증한다:

```yaml
matching:
  fuse_scores: [True, False]   # 1단계만 True, 2단계는 False
```

**참조 구현은 2단계에서 fuse_score 를 끈다.** ultralytics 는 두 단계 모두 켜서
`cost ≥ 1−s > 0.75 > 0.5` 로 2단계가 수학적으로 매칭 불가능해진다.
즉 그것은 ByteTrack 설계의 문제가 아니라 **ultralytics 고유의 버그**다.
보고할 때 이 구분을 반드시 지킬 것.
