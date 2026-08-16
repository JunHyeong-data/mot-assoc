# 실험 3 — Colab 실행 절차

**전제**: exp02 환경이 살아 있을 것. 없으면
`experiments/exp02_utrack_replication/colab_setup.md` 를 먼저 끝낸다.
필요한 것은 `/content/UTrack` (의존성 설치 완료), `/content/data/MOT17`
(COCO 변환 완료, `annotations/val_half.json` 존재), 저자 가중치.

**런타임은 T4 GPU.** `nms_var` 가 CUDA 를 요구한다.

셀은 **순서대로 그대로** 돌린다. 셀 3 에 판단이 필요한 지점이 하나 있고,
거기 말고는 붙여넣기만 하면 된다.

---

## 셀 0 — 환경 확인 (30초)

건너뛰지 말 것. 여기서 안 걸리면 30분 뒤에 걸린다.

```python
import torch, subprocess, os
print('cuda           :', torch.cuda.is_available())
print('UTrack         :', os.path.isdir('/content/UTrack'))
print('val_half.json  :', os.path.isfile('/content/data/MOT17/annotations/val_half.json'))
print('weights        :', os.path.isfile('/content/UTrack/yolov8l-mix/ablation_17/weights/best.pt'))
import ultralytics
ops = os.path.join(os.path.dirname(ultralytics.__file__), 'utils', 'ops.py')
print('ultralytics 포크:', any('nms_var' in l for l in open(ops)))
```

다섯 줄이 전부 `True` 여야 한다. **`ultralytics 포크`가 False 면 표준
ultralytics 가 잡힌 것이고, 그러면 NMS 분산이 안 나와서 실험 전체가 무의미하다.**
그 경우:

```bash
!pip uninstall -y ultralytics
!pip install -q "git+https://github.com/DLR-MI/ultralytics.git@nms-var"
```

## 셀 1 — 실험 코드 받고 설치 (1분)

```bash
!rm -rf /content/mot-assoc
!git clone -q https://github.com/JunHyeong-data/mot-assoc.git /content/mot-assoc
!python /content/mot-assoc/experiments/exp03_box_relaxation/patch_utrack.py /content/UTrack
```

```python
import sys
sys.path.insert(0, '/content/UTrack')
from tracker.associations.collections import ASSOCIATIONS
print('relax_botsort 등록:', 'relax_botsort' in ASSOCIATIONS)
```

`True` 여야 한다.

## 셀 2 — 관문 A: measure 갈래 (약 10분)

확장량이 0 이므로 **기준선과 수치가 같아야 한다.** 동시에 σ 통계를 시퀀스별로 남긴다.

```bash
!python /content/mot-assoc/experiments/exp03_box_relaxation/run_colab.py measure
```

> **여기가 관문이다.** 출력 끝의 `COMBINED` HOTA 가 exp02 기준선
> `botsort` **64.494** 와 같아야 한다. 다르면 훅이 무언가 바꾸고 있다는 뜻이고,
> **이후 숫자는 전부 무의미하다.** 멈추고 원인을 찾을 것.
>
> 참고: 이 갈래는 `relax_botsort` 를 쓰지만 확장이 0 이라 `botsort` 와
> 계산이 동일해야 한다. 로컬 자체 시험에서 `max|diff| = 0.000e+00` 을 확인했다.

중간에 끊기면 같은 명령을 다시 돌린다. **끝난 시퀀스는 건너뛴다.**

## 셀 3 — 확장량 맞추기 (10초) · **눈으로 볼 것**

```bash
!python /content/mot-assoc/experiments/exp03_box_relaxation/run_colab.py calibrate
```

출력에서 **세 줄을 먼저 확인한다.**

| 확인할 것 | 통과 기준 | 걸리면 |
|---|---|---|
| `boxes with zero variance` | 50% 미만 | `nms_var` 가 안 붙었다. 셀 0 으로 |
| `CV(s_x)` | 0.1 이상 | σ 가 상자마다 안 갈린다. R 과 K1 이 원리적으로 구별 불가 → **중단하고 이 숫자를 보고** |
| `corr(s_x, w)` | (판단용) | 높으면 K2 가 R 을 재현할 것으로 예상. 실험 1 과 일치 |

통과했으면 `alpha = 2` 항목의 `RELAX_DX / DY / CW / CH` 네 숫자를 적어둔다.
다음 셀에 넣는다.

## 셀 4 — 갈래 실행 (약 60~70분)

α 격자 5개 + K1 + K2 = 7갈래 × 7시퀀스. 셀 3 이 준 숫자를 그대로 넣는다.

```bash
!python /content/mot-assoc/experiments/exp03_box_relaxation/run_colab.py arms \
    --dx 4.028242 --dy 3.913168 --cw 0.074845 --ch 0.027837
```

> 위 네 숫자는 **예시다.** 셀 3 출력의 `alpha = 2` 줄 값으로 바꿔야 한다.
>
> K1·K2 는 α 하나에만 맞춰진다. R 격자에서 다른 α 가 이기면, 그 α 의 상수로
> `--dx ... --cw ...` 를 바꿔 **한 번 더** 돌린다 (끝난 갈래는 건너뛴다).

진단용 K3·K4 도 보려면 `--diagnostics --alpha 2` 를 붙인다.
**판정에는 쓰지 않는다** — 짝 일관성 교란에 걸린다 (README 참고).

## 셀 5 — 표 (즉시)

```bash
!python /content/mot-assoc/experiments/exp03_box_relaxation/run_colab.py table
```

## 셀 6 — 백업 (**끊기기 전에 반드시**)

```bash
!cd /content/UTrack && zip -qr /content/exp03_results.zip track_results
!zip -qr /content/exp03_stats.zip /content/relax_stats
!ls -lh /content/exp03_*.zip
```

```python
from google.colab import drive; drive.mount('/content/drive')
!cp /content/exp03_results.zip /content/exp03_stats.zip /content/drive/MyDrive/
```

---

## 사고 났을 때

**`val_half.json` 이 이상해졌다** — 실행기가 시퀀스별로 갈아끼우다 죽으면
한 시퀀스짜리로 남아 있을 수 있다. 원본은 `val_half.json.orig` 에 있다.

```bash
!python /content/mot-assoc/experiments/exp03_box_relaxation/run_colab.py restore
```

**중간에 런타임이 끊겼다** — 같은 명령을 다시 돌린다. 결과 txt 가 이미 있는
시퀀스는 건너뛴다.

**MOT17-02 가 표에 없다** — seqmap 헤더 버그다 (exp02 에서 잡은 것).
실행기가 `videos` 맨 앞에 더미를 넣지만, `val_half.json.orig` 자체가 더미 없이
저장돼 있으면 소용없다. `.orig` 의 첫 video 를 확인할 것.

---

## 결과 적는 법

판정은 **K2 하나로** 한다. 사전 선언 기준은 `README.md` 에 있다.

| 갈래 | 평균 확장(px) | HOTA | AssA | IDF1 | IDSW | 게이트밀도 | ΔHOTA |
|---|---|---|---|---|---|---|---|
| A measure | 0 | | | | | | — |
| R sigma α=? | | | | | | | |
| K1 const | | | | | | | |
| K2 prop | | | | | | | |

**R > K2 가 나오면 짝 일관성부터 의심할 것.** 트랙은 마지막 매칭된 검출의
`_var_xywh` 를 물고 있어서, R 은 정답쌍을 같은 만큼 키운다. 사전 검정력 확인에서
σ 가 순전한 잡음일 때도 이 통로로 +0.18 이 나왔다. 진단은 **트랙 σ 와 매칭된
검출 σ 의 상관**을 함께 보고하는 것.

그리고 **HOTA 만 보고 "정보가 전달됐다" 고 쓰지 않는다.** 확장은 임계값 통로로
작동하므로 게이트가 실제로 열렸는지 같이 봐야 한다.
