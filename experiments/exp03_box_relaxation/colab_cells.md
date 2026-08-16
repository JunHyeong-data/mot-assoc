# 실험 3 — Colab 실행 절차

exp02 환경이 이미 있다는 전제다. 없으면
`experiments/exp02_utrack_replication/colab_setup.md` 를 먼저 끝낼 것.
**GPU 런타임(T4) 필수** (`nms_var` 가 CUDA 를 요구한다).

시퀀스를 통합 실행하면 MOT17-11 구간에서 17초/프레임으로 붕괴한다.
exp02 에서 만든 **시퀀스별 `val_half_<SEQ>.json`** 을 그대로 쓴다.

---

## 셀 1 — 설치

```python
!git clone https://github.com/JunHyeong-data/mot-assoc.git /content/mot-assoc 2>/dev/null || (cd /content/mot-assoc && git pull)
!python /content/mot-assoc/experiments/exp03_box_relaxation/patch_utrack.py /content/UTrack
```

확인 — `True` 가 나와야 한다.

```python
import sys; sys.path.insert(0, '/content/UTrack')
from tracker.associations.collections import ASSOCIATIONS
print('relax_botsort' in ASSOCIATIONS)
```

## 셀 2 — 관문 A: measure 실행

확장이 0 이므로 **HOTA 가 exp02 기준선 `botsort` 64.494 와 같아야 한다.**
동시에 σ 통계를 시퀀스별로 남긴다.

```python
import os, subprocess
SEQS = ['MOT17-02','MOT17-04','MOT17-05','MOT17-09','MOT17-10','MOT17-11','MOT17-13']
os.makedirs('/content/relax_stats', exist_ok=True)

for seq in SEQS:
    env = dict(os.environ,
               RELAX_MODE='measure',
               RELAX_APPLY='both',
               RELAX_CAP='1.0',
               RELAX_STATS=f'/content/relax_stats/{seq}.json')
    subprocess.run([
        'python','track.py',
        '--project','yolov8l-mix','--exp','ablation_17',
        '--data_root','/content/data/MOT17',
        '--association','relax_botsort','--gpu_id','0',
        '--config',f'tracker/config/track_ablation_17_{seq}.yaml',
    ], cwd='/content/UTrack', env=env, check=True)
```

> `--config` 는 exp02 에서 시퀀스별로 쪼갠 yaml 이다. 이름이 다르면 맞춰 고칠 것.
> `val_half.json` 의 `videos` 맨 앞 더미 항목(seqmap 헤더 버그 우회)이 그대로
> 있어야 MOT17-02 가 평가에 들어간다.

## 셀 3 — 확장량 맞추기

```python
!python /content/mot-assoc/experiments/exp03_box_relaxation/calibrate.py \
    /content/relax_stats/*.json --alphas 0.5 1 2 3 5 --cap 1.0
```

**여기서 멈추고 세 줄을 확인한다.**

- `CV(s_x)` 가 **0.1 미만이면 중단**. R 과 K1 이 구별 안 된다
- `boxes with zero variance` 가 **50% 넘으면 중단**. `nms_var` 가 안 붙었다
- `corr(s_x, w)` — 높으면 K2 가 R 을 재현할 것으로 예상 (실험 1 과 일치)

출력의 `RELAX_*` 줄을 그대로 다음 셀에 옮긴다.

## 셀 4 — R 갈래 α 격자

α 는 **R 에 유리하게** 고른다.

```python
for alpha in [0.5, 1, 2, 3, 5]:
    for seq in SEQS:
        env = dict(os.environ,
                   RELAX_MODE='sigma', RELAX_ALPHA=str(alpha),
                   RELAX_APPLY='both', RELAX_CAP='1.0')
        subprocess.run([...같음...], cwd='/content/UTrack', env=env, check=True)
```

결과 폴더가 갈래마다 갈리므로 α 마다 HOTA 를 적어두고 최고인 α 를 고른다.

## 셀 5 — K1, K2 (셀 3 이 준 상수로)

```python
BEST_ALPHA = 2          # 셀 4 에서 고른 값
ARMS = {
  'K1': dict(RELAX_MODE='const', RELAX_DX='<셀3 값>', RELAX_DY='<셀3 값>'),
  'K2': dict(RELAX_MODE='prop',  RELAX_CW='<셀3 값>', RELAX_CH='<셀3 값>'),
}
for name, cfg in ARMS.items():
    for seq in SEQS:
        env = dict(os.environ, RELAX_APPLY='both', RELAX_CAP='1.0', **cfg)
        subprocess.run([...같음...], cwd='/content/UTrack', env=env, check=True)
```

## 셀 6 — 백업 (끊기기 전에)

```python
!cd /content/UTrack && zip -qr /content/exp03_results.zip track_results
!cp /content/exp03_results.zip /content/relax_stats -r /content/drive/MyDrive/
```

---

## 결과 적는 법

`experiments/exp03_box_relaxation/README.md` 의 사전 선언 기준에 대고 판정한다.
표는 exp02 와 같은 형식으로, **HOTA 와 함께 게이트밀도를 반드시 같이 적는다.**

| 갈래 | 평균 확장(px) | HOTA | AssA | IDF1 | IDSW | 게이트밀도 | ΔHOTA |
|---|---|---|---|---|---|---|---|
| A measure | 0 | | | | | | — |
| R sigma α=? | | | | | | | |
| K1 const | | | | | | | |
| K2 prop | | | | | | | |

세 갈래의 **평균 확장(px) 이 같은지 먼저 확인할 것.** 안 맞으면 비교가 무의미하다.
