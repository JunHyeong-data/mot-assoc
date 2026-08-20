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

## 셀 1 — 실험 코드 설치 (10초)

저장소가 공개라 인증 없이 받아진다 (익명 clone 으로 확인함).

```bash
!rm -rf /content/mot-assoc /content/exp03
!git clone -q https://github.com/JunHyeong-data/mot-assoc.git /content/mot-assoc
!cp -r /content/mot-assoc/experiments/exp03_box_relaxation /content/exp03
!python /content/exp03/patch_utrack.py /content/UTrack
```

```python
import sys; sys.path.insert(0, '/content/UTrack')
from tracker.associations.collections import ASSOCIATIONS
print('relax_botsort 등록:', 'relax_botsort' in ASSOCIATIONS)
```

`True` 여야 한다.

> **로컬에서 고치고 아직 push 안 했으면 clone 은 옛 코드를 받는다.**
> 먼저 push 하든지, 아래 부트스트랩을 쓰든지 할 것.

<details>
<summary>대안 — 부트스트랩 셀 (push 안 한 로컬 수정본을 그대로 올릴 때)</summary>

`bootstrap_cell.txt` 파일 **전체를 복사해 셀에 붙여넣는다.** 실행에 필요한
파일 4개가 tar+gzip+base64 로 들어 있어 인증도 clone 도 필요 없다. 17 KB.

로컬에서 파일을 고쳤으면 먼저 다시 만든다:

```bash
python experiments/exp03_box_relaxation/make_bootstrap.py
```

돌리면 마지막에 이렇게 나온다.

```
설치: ['box_relax.py', 'calibrate.py', 'patch_utrack.py', 'run_colab.py']
copied  /content/UTrack/tracker/box_relax.py
patched /content/UTrack/tracker/associations/collections.py
relax_botsort 등록: True
```
</details>

## 셀 2 — 사전 점검 A: measure 조건 (약 10분)

확장량이 0 이므로 **기준선과 수치가 같아야 한다.** 동시에 σ 통계를 시퀀스별로 남긴다.

```bash
!python /content/exp03/run_colab.py measure
```

> **여기가 사전 점검이다.** 출력 끝의 `COMBINED` HOTA 가 exp02 기준선
> `botsort` **64.494** 와 같아야 한다. 다르면 훅이 무언가 바꾸고 있다는 뜻이고,
> **이후 숫자는 전부 무의미하다.** 멈추고 원인을 찾을 것.
>
> 참고: 이 조건은 `relax_botsort` 를 쓰지만 확장이 0 이라 `botsort` 와
> 계산이 동일해야 한다. 로컬 자체 시험에서 `max|diff| = 0.000e+00` 을 확인했다.

중간에 끊기면 같은 명령을 다시 돌린다. **끝난 시퀀스는 건너뛴다.**

## 셀 3 — 확장량 맞추기 (10초) · **눈으로 볼 것**

```bash
!python /content/exp03/run_colab.py calibrate
```

출력에서 **세 줄을 먼저 확인한다.**

| 확인할 것 | 통과 기준 | 걸리면 |
|---|---|---|
| `boxes with zero variance` | 50% 미만 | `nms_var` 가 안 붙었다. 셀 0 으로 |
| `CV(s_x)` | 0.1 이상 | σ 가 박스마다 안 갈린다. R 과 K1 이 원리적으로 구별 불가 → **중단하고 이 숫자를 보고** |
| `corr(s_x, w)` | (판단용) | 높으면 K2 가 R 을 재현할 것으로 예상. 실험 1 과 일치 |

통과했으면 `alpha = 2` 항목의 `RELAX_DX / DY / CW / CH` 네 숫자를 적어둔다.
다음 셀에 넣는다.

## 셀 4 — 조건 실행 (약 60~70분)

α 격자 5개 + K1 + K2 = 7조건 × 7시퀀스. 셀 3 이 준 숫자를 그대로 넣는다.

```bash
!python /content/exp03/run_colab.py arms \
    --dx 4.028242 --dy 3.913168 --cw 0.074845 --ch 0.027837
```

> 위 네 숫자는 **예시다.** 셀 3 출력의 `alpha = 2` 줄 값으로 바꿔야 한다.
>
> K1·K2 는 α 하나에만 맞춰진다. R 격자에서 다른 α 가 이기면, 그 α 의 상수로
> `--dx ... --cw ...` 를 바꿔 **한 번 더** 돌린다 (끝난 조건은 건너뛴다).

진단용 K3·K4 도 보려면 `--diagnostics --alpha 2` 를 붙인다.
**판정에는 쓰지 않는다** — 짝 일관성 교란에 걸린다 (README 참고).

## 셀 5 — 표 (즉시)

```bash
!python /content/exp03/run_colab.py table
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
!python /content/exp03/run_colab.py restore
```

**중간에 런타임이 끊겼다** — 같은 명령을 다시 돌린다. 결과 txt 가 이미 있는
시퀀스는 건너뛴다.

**MOT17-02 가 표에 없다** — seqmap 헤더 버그다 (exp02 에서 잡은 것).
실행기가 `videos` 맨 앞에 더미를 넣지만, `val_half.json.orig` 자체가 더미 없이
저장돼 있으면 소용없다. `.orig` 의 첫 video 를 확인할 것.

---

## 결과 적는 법

판정은 **K2 하나로** 한다. 사전 등록 기준은 `README.md` 에 있다.

| 조건 | 평균 확장(px) | HOTA | AssA | IDF1 | IDSW | 게이트밀도 | ΔHOTA |
|---|---|---|---|---|---|---|---|
| A measure | 0 | | | | | | — |
| R sigma α=? | | | | | | | |
| K1 const | | | | | | | |
| K2 prop | | | | | | | |

**R > K2 가 나오면 짝 일관성부터 의심할 것.** 트랙은 마지막 매칭된 검출의
`_var_xywh` 를 물고 있어서, R 은 정답쌍을 같은 만큼 키운다. 사전 검정력 확인에서
σ 가 순전한 잡음일 때도 이 경로로 +0.18 이 나왔다. 진단은 **트랙 σ 와 매칭된
검출 σ 의 상관**을 함께 보고하는 것.

그리고 **HOTA 만 보고 "정보가 전달됐다" 고 쓰지 않는다.** 확장은 임계값 경로로
작동하므로 게이트가 실제로 열렸는지 같이 봐야 한다.

---

# 보론 — **확장이 문을 여는가 닫는가** (2026-08-20)

사전 등록은 `PREREG-direction.md`. **읽는 법이 자료보다 먼저 커밋돼 있다.**

**왜 도는가.** 본 실행의 α 는 **10** 인데(위 「[1] R 격자」) 방향을 검증한
`selftest [3b]` 는 **α=2, 합성 장면**이다. **검증한 구간과 사용한 구간이
5 배 떨어져 있다.** exp19 에서 NMS 소스가 큰 α 에서 채택률을 **떨어뜨리는**
것이 나왔으므로(확장이 문을 여는 게 아니라 닫는다) exp03 이 같은 자리인지
확인해야 한다. **여기 걸리면 −4.33 과 88% 가 같이 걸린다.**

위 셀 0·1 이 이미 돌아 있어야 한다. **셀 2~5 는 안 돌려도 된다** —
이 보론은 measure 조건만 쓴다.

## 보론 셀 1 — 코드 갱신 (10초)

기록 훅이 들어간 판이어야 한다. 셀 1 을 이미 돌렸어도 **다시 받는다.**

```bash
!rm -rf /content/mot-assoc /content/exp03
!git clone -q https://github.com/JunHyeong-data/mot-assoc.git /content/mot-assoc
!cp -r /content/mot-assoc/experiments/exp03_box_relaxation /content/exp03
!python /content/exp03/patch_utrack.py /content/UTrack
!grep -c RELAX_DUMP_CALLS /content/UTrack/tracker/box_relax.py
```

마지막 줄이 **`2`** 여야 한다. `0` 이면 옛 판이 복사된 것이고
**그대로 돌리면 덤프가 안 나온다.**

## 보론 셀 2 — 원자료 남기기 (약 10분)

확장 0 으로 한 번 돌며 연관 호출마다 **확장 전** 입력을 남긴다.
끝나면 재생·판정까지 자동으로 이어진다.

```bash
!python /content/exp03/run_colab.py direction
```

> **훅은 기본이 꺼져 있고 비용 경로를 안 건드린다.** 로컬 `selftest.py`
> PASS 로 확인했다 (`[1] measure vs plain IoU  max|diff| = 0.000e+00`).
> 따라서 이 실행의 HOTA 는 여전히 **64.494** 여야 한다.

메모리가 걱정되면 `RELAX_DUMP_MAX` 로 호출 수를 자를 수 있다. 다만
**자르면 시퀀스가 조용히 빠지는 것과 같은 일이 생기므로** 자른 사실을
결과에 함께 적을 것.

```bash
!RELAX_DUMP_MAX=20000 python /content/exp03/run_colab.py direction
```

## 보론 셀 3 — 재생만 다시 (즉시)

덤프가 이미 있으면 추적을 다시 돌 필요가 없다.

```bash
!python /content/exp03/direction.py /content/exp03_dumps
```

## 읽는 법 — **스크립트가 판정까지 찍는다. 고쳐 읽지 말 것**

출력 마지막의 「판정」 블록이 사전 등록한 세 경우 중 하나를 고른다.
문턱은 exp19 `[0b]` 와 같은 **5 쌍**이다 — *"기준선 위인지 아래인지"* 로
읽으면 **한 쌍짜리 차이가 철회를 결정한다.**

| α=10 에서 채택 쌍 변화 | 결론 |
|---|---|
| **+5 건 초과** | **개입 성립.** −4.33 과 88% 그대로 유효 |
| **−5 건 미만** | **개입 불성립.** `PREREG-direction.md` 「파급」표대로 고친다 |
| **±5 건 안** | **중립.** 개수로 판정하지 않고 **대칭차**를 보고 정한다 |

> **중립이 나오면 개수만 보고 "개입이 안 돌았다" 고 읽지 말 것.**
> 박스를 키우면 **어느 쌍이 채택되는지는 완전히 바뀌면서 개수만 같을 수
> 있다.** 그래서 대칭차를 함께 센다.

**부수 결과로 두 가지가 공짜로 나온다.**

- **단조인가** — 보정 2 의 (6) 이 *"이 자료에서 단조"* 라고 적었는데 **안 쟀다.**
  채택 쌍 열로 확인한다
- **쌍별 pad 비대칭** — `[3b]` 는 **평균 비**(0.940)만 쟀다. 평균이 1 이어도
  쌍마다 어긋날 수 있다. 중앙값·90분위를 함께 낸다

## 백업 (**끊기기 전에**)

```bash
!zip -qr /content/drive/MyDrive/exp03_direction.zip /content/exp03_dumps
```

## 결과가 나오면

`notes/progress.md` 에 **판정 블록을 그대로** 붙이고, 「파급」표에서 해당
줄을 실행한다. **어느 쪽이 나와도 사전 등록에 이미 적혀 있으므로 새 판단은
필요 없다.**
