# Colab 실행 절차 — UTrack 재현

로컬(Windows)에서는 못 돌린다. MSVC 없음, `lap==0.4.0` 빌드 실패, WSL 배포판 없음.
Colab 은 Linux + gcc + CUDA + 무료 T4 라 컴파일과 GPU 문제가 같이 풀린다.

**아래 셀을 순서대로 그대로 돌린다.** 각 셀에 검증이 붙어 있으니 실패하면 거기서 멈추고
다음으로 넘어가지 말 것.

---

## 사전에 알아둘 것 (실패 원인 세 가지를 미리 제거한다)

| 문제 | 원인 | 처방 |
|---|---|---|
| `pip install -r requirements.txt` 가 죽는다 | `lap==0.4.0` 은 PyPI 에 **휠이 0개**. 소스 빌드가 `numpy.distutils` 를 쓰는데 최신 numpy 에서 제거됨 | `lap>=0.5.12` 로 교체. cp37~cp314 휠 제공, 컴파일 불필요. API 동일 |
| `nms_var` 설치 실패 | pybind11 을 git 서브모듈로 씀. `pip install git+` 는 서브모듈을 안 받음. `setup.py` 가 CUDA/nvcc 요구 | **GPU 런타임** + 클론 후 로컬 설치 |
| `fast_gmc` 설치 실패 | `setup.py:8` 이 `os.environ['CONDA_PREFIX']` 를 직접 읽어 **KeyError 즉사**. 이어서 `conda install libopencv` 시도 | `CONDA_PREFIX=/usr` 로 위장 + 시스템 opencv 헤더 설치 |

Python 3.8 은 원인이 아니다. UTrack 자체 코드에 버전 의존 문법이 없다
(3.10+ 전용 문법은 docstring 1건뿐, 구형 numpy 관용구 없음). Colab 기본 3.12 로 간다.

---

## 셀 0 — 런타임을 T4 GPU 로 (필수)

메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기 **T4 GPU**.
`nms_var` 의 `setup.py` 가 `torch.cuda.is_available()` 와 `nvcc` 를 요구한다.
CPU 런타임에서는 설치 자체가 불가능하고, 포크된 ultralytics 의 `ops.py:17` 이
`import nms_var` 라 임포트부터 실패한다.

```python
import torch
print('cuda:', torch.cuda.is_available())     # True 여야 함
!nvcc --version | tail -2
```

## 셀 1 — 깨끗한 클론 (중첩 방지)

이전 시도가 남아 있으면 `/content/UTrack/UTrack` 처럼 중첩되어 어느 쪽
`requirements.txt` 를 고쳤는지 헷갈린다. 항상 지우고 시작한다.

```bash
%cd /content
!rm -rf /content/UTrack /content/nms_var /content/fast_gmc
!git clone -q https://github.com/DLR-MI/UTrack.git /content/UTrack
%cd /content/UTrack
!pwd
```

`/content/UTrack` 이 찍혀야 한다. 중첩 경로가 보이면 다시 셀 1 부터.

## 셀 2 — requirements 손보기

`lap` 을 올리고, 별도 설치할 두 개는 목록에서 뺀다.

```bash
!sed -i 's/^lap==0.4.0/lap>=0.5.12/' requirements.txt
!sed -i '/nms_var/d;/fast_gmc/d' requirements.txt
!cat requirements.txt
```

`lap>=0.5.12` 로 바뀌고 `nms_var`·`fast_gmc` 줄이 사라졌는지 눈으로 확인.
나머지 핀은 건드리지 않는다 — 재현이 어긋났을 때 원인 후보를 줄이기 위한 최소 개입.

## 셀 3 — 빌드 도구와 OpenCV 헤더

`fast_gmc` 가 CMake 로 빌드되고 `include/opencv4` 를 찾는다.

```bash
!apt-get -qq update > /dev/null 2>&1
!apt-get -qq install -y libopencv-dev cmake > /dev/null 2>&1
!ls /usr/include/opencv4 | head -3
!cmake --version | head -1
```

`opencv2` 가 보이고 cmake 버전이 찍혀야 한다.

## 셀 4 — `nms_var` (서브모듈 포함, 로컬 설치)

```bash
!git clone -q --recursive https://github.com/DLR-MI/nms_var.git /content/nms_var
!pip install -q /content/nms_var 2>&1 | tail -3
!python -c "import nms_var; print('nms_var OK')"
```

`extern/pybind11` 이 비어 있어도 빌드가 성공하는 경우가 있다 (pip 가 pybind11 을
따로 잡아줌). 실패하면:

```bash
!cd /content/nms_var && git submodule update --init --recursive && pip install .
```

## 셀 5 — `fast_gmc` (conda 검사 우회)

`setup.py` 가 `os.environ['CONDA_PREFIX']` 를 직접 읽으므로 값이 **존재해야** 한다.
`/usr` 로 두면 `/usr/include/opencv4` 가 있어 `conda install` 시도도 건너뛴다.

```bash
!git clone -q https://github.com/DLR-MI/fast_gmc.git /content/fast_gmc
!CONDA_PREFIX=/usr pip install /content/fast_gmc 2>&1 | tail -15
!python -c "from fast_gmc import gmc; print('fast_gmc OK')"
```

`pip install git+...` 가 아니라 **클론 후 로컬 설치**여야 한다. CMake 빌드가 소스 트리를
필요로 한다.

`fast_gmc` 는 선택이 아니다 — `tracker/update.py:10` 이 `camera_motion` 을 무조건
임포트하고 그 모듈이 `from fast_gmc import gmc` 를 최상단에서 한다. 게다가
`collections.py` 의 BoT-SORT 계열 전부가 `camera_motion.method='fast-gmc'` 를 설정한다.

## 셀 6 — 나머지 의존성 (하나씩, 진단 가능하게)

`-q` 를 쓰지 않는다. 실패 시 어느 패키지인지 즉시 드러나야 한다.

```python
import sys, subprocess
reqs = [l.strip() for l in open('requirements.txt') if l.strip() and not l.startswith('#')]
fail = []
for r in reqs:
    print(f"\n{'='*60}\n>>> {r}")
    if subprocess.run([sys.executable, '-m', 'pip', 'install', r]).returncode:
        fail.append(r)
print("\n실패:", fail or "없음")
```

## 셀 7 — 관문: 포크가 진짜 잡혔는지

**여기를 통과하지 못하면 이후 실험이 전부 무의미하다.** 표준 ultralytics 가 잡히면
NMS 분산이 나오지 않는데, 오류 없이 조용히 그냥 돌아가기 때문에 알아채기 어렵다.

```python
import ultralytics, lap, nms_var, os
print('ultralytics', ultralytics.__version__)
ops = os.path.join(os.path.dirname(ultralytics.__file__), 'utils', 'ops.py')
hit = [l for l in open(ops) if 'nms_var' in l]
print('ops.py 의 nms_var 임포트:', hit or '없음  <-- 표준 ultralytics 다!')
from ultralytics.utils.ops import non_max_suppression
print('lap', lap.__version__)
```

`import nms_var` 줄이 보여야 포크다. 안 보이면:

```bash
!pip uninstall -y ultralytics
!pip install "git+https://github.com/DLR-MI/ultralytics.git@nms-var"
```

경로만 보고 판단하지 말 것 — 포크도 표준도 같은 `site-packages` 에 깔린다.

---

## 셀 8 — 데이터

공식 `motchallenge.net` 은 2026-08-16 기준 응답 없음. 검증된 HF 미러를 쓴다
(독립 미러 2개 바이트 대조: 주석 21/21 · 이미지 120/120 SHA256 일치.
`notes/data_sources.md` 참고).

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Morrison1025/MOT17', repo_type='dataset',
                  allow_patterns=['train/**','ablation/**','val/**'],
                  local_dir='/content/data/MOT17', max_workers=8)
!ls /content/data/MOT17
```

COCO 형식 주석을 만든다. `DATA_PATH` 를 실제 경로에 맞춰야 한다.

```bash
!sed -n '1,20p' tools/convert_mot17_to_coco.py | grep -n DATA_PATH
```

경로를 고친 뒤:

```bash
!python tools/convert_mot17_to_coco.py
!python tools/fix_yolo_annotations.py --folder /content/data/MOT17
!find /content/data/MOT17 -name "val_half.json"
```

설정이 `val_ann: val_half.json` 을 찾으므로 이 파일이 나와야 한다.
**여기가 남은 실패 지점이다.** 안 나오면 변환 스크립트의 split 로직부터 맞춘다.

## 셀 9 — 가중치

논문이 쓴 그 가중치다. **직접 학습하지 말 것** — 재현에서 학습 차이를 배제해야 한다.

```bash
!mkdir -p yolov8l-mix/ablation_17/weights
!wget -q -O yolov8l-mix/ablation_17/weights/best.pt \
  "https://zenodo.org/records/13604403/files/ablation_17_best.pt?download=1"
!ls -la yolov8l-mix/ablation_17/weights/
```

---

## 셀 10 — [1] 기준선 재현

```bash
!python track.py --project yolov8l-mix --exp ablation_17 \
  --data_root /content/data/MOT17 --association botsort --gpu_id 0
```

> **판정**: 저장소가 MOT17-ablation 에 대해 보고하는 BoT-SORT 수치와 비교.
> 벗어나면 이후 단계를 우리 재현값 기준으로 삼고 그 사실을 명시한다.

## 셀 11 — [2] 논문이 표로 안 준 ablation

변형이 이미 구현돼 있다 (`tracker/associations/collections.py`).
접두사: `uk_`=U-Kalman, `us_`=size disambiguator, `up_`=phase disambiguator.

```python
for assoc in ['botsort', 'uk_botsort', 'us_botsort', 'up_botsort', 'ukp_botsort']:
    print(f"\n===== {assoc} =====")
    !python track.py --project yolov8l-mix --exp ablation_17 \
      --data_root /content/data/MOT17 --association {assoc} --gpu_id 0
```

논문은 이걸 산점도(Fig.3)로만 보여준다. **표로 만드는 것이 우리 기여의 일부다.**

> **판정**: HOTA 기준 (uk−base), (us−base), (up−base) 를 분리 보고.
> 논문 주장("U-Kalman 이 가장 두드러진다")이 맞으면 uk 가 최대여야 한다.

**교란 하나** — `track_ablation_17.yaml` 의 `uncertain.kalman.nsa: True` 가 기본이다.
NSA 는 검출 score 로 R 을 이미 조정하므로 기준선도 순수하지 않다.
`nsa: False` 로도 같은 격자를 돌려 U-Kalman 의 순효과를 분리한다.

## 셀 12 — [3] 결정적 대조군 E

`tracker/kalman_filter.py` `project()`:

```python
std = self._std_weight_pos * whwh          # A: 트랙 자신의 w,h 에서
innovation_cov = np.diag(np.square(std))
...
if var_measurement is not None:
    innovation_cov = np.diag(var_measurement)   # B: 검출의 NMS 분산
```

E 는 `var_measurement` 를 **검출 높이만으로 만든 값**으로 바꾼다:

```python
var_measurement = k * (h_det ** 2) * np.ones(4)   # 측정 없이 크기만
```

`k` 는 val_half 에서 HOTA 최대가 되도록 격자탐색한다. **E 에 유리하게 준다** —
불리하게 잡아 이기는 것은 의미가 없다.

> **판정 (사전 선언, 바꾸지 않는다)**
> - E 와 B 의 HOTA 차이 **0.3 이내** → 이득은 측정된 불확실성이 아니라
>   **R 의 출처를 트랙에서 검출로 옮긴 것**에서 온다. 논문의 핵심 주장이 무너진다.
> - E 가 B 보다 **0.3 이상 낮음** → 측정에 크기 이상의 정보가 있다.
>   실험 1 의 부정 결과와 모순되므로 **실험 1 을 먼저 재검토**한다.
> - E 가 B 보다 **0.3 이상 높음** → 측정이 오히려 해롭다. 실험 1 과 일치한다.

## 셀 13 — [4] 보정 분석

`experiments/exp01_nms_variance/analyze_covariance.py` 를 그대로 쓴다.
포크가 NMS 분산을 직접 주므로 우리 가로채기 코드보다 정확하다.
**우리 구현과 포크의 분산이 일치하는지 먼저 대조할 것** — 어긋나면 실험 1 의
부정 결과가 우리 추출 오류였을 가능성을 배제해야 한다.

## 셀 14 — [5] 혼잡 층화

시퀀스별 (uk−base) 를 `exp00/census.py` 의 M·N·절단률·게이트밀도와 나란히 둔다.
GT visibility 구간별 z² 도 함께. 논문이 "MOT20 에서 baseline 이 더 낫다"고
말만 하고 수치를 안 준 부분이다.

---

## 결과 회수

`track_results/` 아래 지표가 쌓인다. 로컬 `data/exp02/` 로 내려받아 표로 정리한다.
시퀀스별 HOTA/AssA/DetA/IDF1 원자료를 같이 보관할 것.

```python
!zip -qr /content/track_results.zip track_results
from google.colab import files; files.download('/content/track_results.zip')
```

## 부수 확인 — ultralytics 2단계 버그

이 저장소의 설정이 exp00 발견을 확증한다:

```yaml
matching:
  fuse_scores: [True, False]   # 1단계만 True, 2단계는 False
```

**참조 구현은 2단계에서 fuse_score 를 끈다.** ultralytics 는 두 단계 모두 켜서
`cost ≥ 1−s > 0.75 > 0.5` 로 2단계가 수학적으로 매칭 불가능해진다.
즉 그것은 ByteTrack 설계의 문제가 아니라 **ultralytics 고유의 버그**다.
보고할 때 이 구분을 반드시 지킬 것.
