# -*- coding: utf-8 -*-
"""실험 5 [2단계] -- 와서스타인 비용함수. 갈래별 Sigma 만 다르다.

**사전 선언은 README. 자료보다 먼저 커밋했다.**

## 대각 공분산에서의 단순화 (검산할 것)

일반형:
    W^2 = ||eps||^2 + tr(St + Sd - 2 (St^.5 Sd St^.5)^.5)

우리 경우 St, Sd 가 **전부 대각**이다 (DFL 은 변마다 독립 softmax, 상자크기 모형은
정의상 대각, 칼만 P 의 위치 성분도 대각). 대각이면 행렬제곱근이 원소별이 되어

    tr(...) = sum_i ( sqrt(st_i) - sqrt(sd_i) )^2

**표준편차 차이의 제곱합**으로 떨어진다. 싸고 정확하다.
`selftest.py [1]` 이 일반형과 수치로 대조한다.

## 트래커에 넣는 형태 -- NWD 를 따른다

W^2 은 px^2 단위이고 무한대까지 간다. ByteTrack 의 비용은 [0,1] 이고
`match_thresh=0.8` 로 자른다. 그래서 NWD(Normalized Wasserstein Distance)가 쓰는
지수 사상을 그대로 쓴다:

    NWD = exp( - sqrt(W^2) / C )        cost = 1 - NWD

`C` 는 데이터셋 상수다. **채택률을 기준선과 맞추도록 보정한다** (사전 선언 함정 3).
보정 안 하면 통로 효과와 임계값 효과가 섞인다.

## 갈래

| 갈래 | Sigma_d | Sigma_t |
|---|---|---|
| `iou` | -- | -- (맨 IoU, 기준선) |
| `w_dfl` | DFL 분포 분산 | 칼만 P |
| `w_size` | `diag(w^2, h^2)/4` | 칼만 P |
| `w_nms` | NMS 후보 분산 | 칼만 P |
| `nwd_exact` | `diag(w^2,h^2)/4` | `diag(w^2,h^2)/4` | <- 문헌 그대로 (참조점) |

**Sigma_t 를 갈래와 무관하게 칼만 P 로 고정하는 것이 핵심 통제다** (함정 2).
갈래마다 다른 궤적을 타면 통제가 깨지고, 동시에 exp03 의 짝 일관성 교란도
원천 차단된다 -- 트랙이 검출의 Sigma 를 물려받지 않기 때문이다.
`nwd_exact` 만 그 통제를 깨는데, 그건 **문헌 재현용 참조점**이지 판정 갈래가 아니다.
"""
import numpy as np

EPS = 1e-12


def bures_diag(st, sd):
    """대각 공분산의 Bures 항: sum_i (sqrt(st_i) - sqrt(sd_i))^2.

    st, sd: (..., 2) 각 축의 분산. 음수는 0 으로 막는다.
    """
    a = np.sqrt(np.maximum(st, 0.0))
    b = np.sqrt(np.maximum(sd, 0.0))
    return ((a - b) ** 2).sum(-1)


def bures_full(St, Sd):
    """일반형 (2x2). 검산용이라 느려도 된다.

    배치 eigh 는 V (...,2,2) 와 w (...,2) 를 준다. V @ diag(sqrt(w)) @ V^T 를
    쓰려면 고유값을 **열 축**에 브로드캐스트해야 한다 -> [..., None, :].
    """
    w, V = np.linalg.eigh(St)
    rt = (V * np.sqrt(np.maximum(w, 0))[..., None, :]) @ np.swapaxes(V, -1, -2)
    M = rt @ Sd @ rt
    w2, V2 = np.linalg.eigh(M)
    cross = (V2 * np.sqrt(np.maximum(w2, 0))[..., None, :]) @ np.swapaxes(V2, -1, -2)
    return np.trace(St + Sd - 2.0 * cross, axis1=-2, axis2=-1)


def w2_matrix(t_xyxy, d_xyxy, t_var, d_var):
    """트랙 x 검출 W^2 행렬.

    t_xyxy (M,4), d_xyxy (N,4) : xyxy
    t_var  (M,2), d_var  (N,2) : 중심의 축별 분산 (px^2)
    """
    tc = np.stack([(t_xyxy[:, 0] + t_xyxy[:, 2]) / 2,
                   (t_xyxy[:, 1] + t_xyxy[:, 3]) / 2], 1)
    dc = np.stack([(d_xyxy[:, 0] + d_xyxy[:, 2]) / 2,
                   (d_xyxy[:, 1] + d_xyxy[:, 3]) / 2], 1)
    mean = ((tc[:, None, :] - dc[None, :, :]) ** 2).sum(-1)      # (M,N)
    cov = bures_diag(t_var[:, None, :], d_var[None, :, :])       # (M,N)
    return mean + cov


def w2_matrix_norm(t_xyxy, d_xyxy, t_var, d_var):
    """**크기 정규화** W^2 (실험 5b. 사전 선언은 PREREG-norm.md).

    쌍마다 상자 크기로 좌표를 나눈 뒤 같은 W^2 을 쓴다. 아핀 사상
    diag(1/s_x, 1/s_y) 를 두 가우시안에 적용한 것이라 **여전히 2-와서스타인**이다.

        s_x = sqrt(w_t * w_d),  s_y = sqrt(h_t * h_d)      <- 쌍마다. 대칭
        W^2 = (eps_x/s_x)^2 + (eps_y/s_y)^2
              + (sqrt(st_x)-sqrt(sd_x))^2 / s_x^2
              + (sqrt(st_y)-sqrt(sd_y))^2 / s_y^2

    분산은 s^2 으로 나뉘므로 sqrt 차의 제곱은 s^2 으로 나뉜다 (축마다).

    **한쪽 상자로 나누지 않고 기하평균을 쓴다.** 한쪽이면 비용이 비대칭이 되고
    행/열 구조가 인위적으로 갈린다. IoU 가 대칭인 것과 맞춘다.

    부수효과 (사전에 적어둔 것): tr(St)/s^2 이 더 이상 **행상수가 아니다** ->
    할당에 도달하는 Sigma 정보가 실험 5 보다 **늘어난다**.
    """
    tw = np.maximum(t_xyxy[:, 2] - t_xyxy[:, 0], EPS)
    th = np.maximum(t_xyxy[:, 3] - t_xyxy[:, 1], EPS)
    dw = np.maximum(d_xyxy[:, 2] - d_xyxy[:, 0], EPS)
    dh = np.maximum(d_xyxy[:, 3] - d_xyxy[:, 1], EPS)
    sx2 = tw[:, None] * dw[None, :]          # s_x^2 = w_t * w_d
    sy2 = th[:, None] * dh[None, :]

    tcx = (t_xyxy[:, 0] + t_xyxy[:, 2]) / 2
    tcy = (t_xyxy[:, 1] + t_xyxy[:, 3]) / 2
    dcx = (d_xyxy[:, 0] + d_xyxy[:, 2]) / 2
    dcy = (d_xyxy[:, 1] + d_xyxy[:, 3]) / 2
    mean = ((tcx[:, None] - dcx[None, :]) ** 2 / sx2
            + (tcy[:, None] - dcy[None, :]) ** 2 / sy2)

    bx = (np.sqrt(np.maximum(t_var[:, None, 0], 0.0))
          - np.sqrt(np.maximum(d_var[None, :, 0], 0.0))) ** 2 / sx2
    by = (np.sqrt(np.maximum(t_var[:, None, 1], 0.0))
          - np.sqrt(np.maximum(d_var[None, :, 1], 0.0))) ** 2 / sy2
    return mean + bx + by


def size_var(xyxy):
    """NWD 의 상자-가우시안: Sigma = diag(w^2, h^2)/4."""
    w = np.maximum(xyxy[:, 2] - xyxy[:, 0], EPS)
    h = np.maximum(xyxy[:, 3] - xyxy[:, 1], EPS)
    return np.stack([w ** 2 / 4.0, h ** 2 / 4.0], 1)


def match_scale(var, target_tr):
    """전역 배율로 E[tr Sigma] 를 target 에 맞춘다 (사전 선언 함정 1).

    DFL 은 0.26배 과대, NMS 는 80배 과소다. 날것으로 넣으면 갈래 간 차이가
    **정보가 아니라 규모**가 된다. exp03 이 확장량을 맞춘 것과 같은 이유.
    """
    cur = float(np.mean(var.sum(-1)))
    if cur <= EPS:
        return var
    return var * (target_tr / cur)


def nwd_cost(w2, C):
    """NWD 사상: cost = 1 - exp(-sqrt(W^2)/C). [0,1] 이라 match_thresh 와 맞는다."""
    return 1.0 - np.exp(-np.sqrt(np.maximum(w2, 0.0)) / max(C, EPS))


def solve_C(w2_samples, target_cost_median):
    """채택률을 맞추기 위한 C (사전 선언 함정 3).

    기준선(맨 IoU)의 비용 중앙값과 같아지도록 C 를 푼다. 단조라 닫힌해가 있다:
        1 - exp(-r/C) = m   =>   C = r / -ln(1-m)
    r 은 sqrt(W^2) 의 중앙값.
    """
    r = float(np.median(np.sqrt(np.maximum(w2_samples, 0.0))))
    m = float(np.clip(target_cost_median, 1e-6, 1 - 1e-6))
    return r / (-np.log(1.0 - m))
