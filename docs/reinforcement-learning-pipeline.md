# 강화학습 기반 트레이딩 파이프라인

## 1. 목적 및 시스템 개요

본 시스템은 삼성전자(`005930`)의 분봉 데이터를 기반으로, 시장 상태와 현재
포트폴리오 상태에 따라 목표 주식 비중을 결정하는 강화학습 정책을 구축한다.
가격의 단기 방향을 직접 예측하는 대신, 거래비용이 반영된 포트폴리오의 장기
성장을 최대화하는 순차적 의사결정 문제로 트레이딩을 정의한다.

전체 파이프라인은 다음과 같다.

```text
KIS 1분봉 수집
    ↓
5분·30분 시장 feature 생성
    ↓
시간순 Train / Validation / Test 분할
    ↓
Train 기준 feature 정규화
    ↓
Gymnasium 거래 환경 구성
    ↓
MaskablePPO 학습
    ↓
Validation 기반 checkpoint 선택
    ↓
독립 Test 및 baseline 비교
    ↓
Multi-seed·walk-forward 강건성 평가
```

파이프라인 설계에서 가장 중요한 원칙은 시간적 인과성이다. 시점 \(t\)의 의사결정은
해당 시점까지 확정된 정보만 사용해야 하며, 체결가격과 평가가격 역시 실제 주문
가능 순서를 반영해야 한다. 이를 위반하면 미래정보 누출로 인해 백테스트 성과가
과대평가될 수 있다.

## 2. 강화학습 문제 정의

트레이딩 환경은 부분적으로 관측 가능한 순차적 의사결정 문제이지만, 구현에서는
관찰 가능한 feature와 포트폴리오 상태를 하나의 상태 벡터로 구성한 Markov Decision
Process(MDP) 근사로 다룬다.

| 구성 요소 | 프로젝트 내 정의 |
| --- | --- |
| Agent | 목표 보유 비중을 결정하는 MaskablePPO 정책 |
| Environment | 시장 전이, 주문 체결, 비용 및 포트폴리오 회계를 처리하는 환경 |
| Observation | 시장 feature 9개와 포트폴리오 상태 4개 |
| Action | 목표 비중 `0/20/40/60/80/100%` |
| Reward | 거래비용 반영 포트폴리오 로그수익률 |
| Episode | 연속 20거래일 |

시점 \(t\)에서 정책 \(\pi_\theta(a_t|s_t)\)는 관찰 \(s_t\)를 바탕으로 행동
\(a_t\)를 선택한다. 환경은 주문 체결과 시장가격 변화를 반영하여 다음 관찰
\(s_{t+1}\)과 보상 \(r_t\)를 반환한다. 학습의 목적은 할인 누적보상의 기대값을
최대화하는 정책 파라미터 \(\theta\)를 찾는 것이다.

```text
max Eπ [Σ γ^t r_t]
```

## 3. 데이터 처리와 시간적 계약

원본 데이터는 KIS Open API에서 수집한 1분봉 OHLCV와 거래대금이다. 의사결정
주기는 5분이며, 5분 미시 feature와 30분 시장 맥락 feature를 결합한다.

5분봉의 라벨 시각은 해당 구간의 정보가 모두 확정된 시점을 의미한다. 예를 들어
`09:05` 관찰은 `09:00:00~09:04:59` 데이터를 집계한 결과다. 에이전트는 이
관찰을 받은 이후에만 행동할 수 있다.

주문 체결에는 별도 컬럼인 `ExecPrice`를 사용한다.

- 장중 의사결정: 라벨 시각 이후 첫 정규 1분봉의 시가
- 장 마감 의사결정: 15:30 종가 동시호가 단일가
- `ExecPrice`는 체결 계산에만 사용하며 observation에는 포함하지 않음

이 계약은 관찰한 봉의 종가에 즉시 체결되는 비현실적인 zero-latency 가정을
제거한다. 포트폴리오 평가는 다음 5분 관찰 시점의 시장가격으로 수행한다.

데이터는 거래일 단위로 Train, Validation, Test에 시간순 분할한다. feature
정규화에 필요한 평균과 표준편차는 Train에서만 추정하고 Validation과 Test에는
동일한 통계를 적용한다. 평가 구간의 분포 정보를 정규화에 사용하는 것도 데이터
누출에 해당하므로 허용하지 않는다.

## 4. Observation 설계

Observation은 총 13차원이며 시장 상태 9개와 포트폴리오 상태 4개로 구성된다.

### 4.1 시장 feature

| Feature | 의미 |
| --- | --- |
| `log_ret_1` | 직전 5분 로그수익률 |
| `log_ret_3` | 최근 15분 누적 로그수익률 |
| `log_ret_12` | 최근 1시간 누적 로그수익률 |
| `realized_vol_12` | 최근 1시간 실현변동성 |
| `relative_volume` | 최근 거래량의 이동평균 대비 비율 |
| `vwap_dev` | 당일 VWAP 대비 현재가격 이격도 |
| `trend_strength_30m` | 30분 주기의 추세 강도 |
| `macd_hist_30m` | 30분 MACD histogram |
| `vol_regime_30m` | 30분 변동성 국면 |

시장 feature는 Train 통계로 표준화한 뒤 `[-5, 5]` 범위로 clipping한다. 극단값이
정책 네트워크의 업데이트를 지배하는 현상을 완화하면서도, Validation과 Test에
동일한 변환을 적용할 수 있도록 정규화 통계를 artifact에 저장한다.

### 4.2 포트폴리오 상태

| 상태 변수 | 의미 |
| --- | --- |
| `units_held_frac` | 최대 허용 비중 대비 현재 목표 비중 |
| `unrealized_pnl_norm` | 초기자산으로 정규화한 미실현 손익 |
| `holding_duration_norm` | 기준 보유기간 대비 현재 보유기간 |
| `tod_frac` | 정규장 내 현재 시점의 상대적 위치 |

동일한 시장 feature에서도 현재 노출도와 손익 상태에 따라 적절한 행동이 달라질 수
있으므로 포트폴리오 상태를 정책 입력에 포함한다.

## 5. Action 및 주문 실행

행동 공간은 여섯 개의 이산적인 목표 비중으로 정의한다.

| Action | 목표 주식 비중 |
| ---: | ---: |
| 0 | 0% |
| 1 | 20% |
| 2 | 40% |
| 3 | 60% |
| 4 | 80% |
| 5 | 100% |

행동이 선택되면 환경은 현재 포트폴리오 가치와 거래비용을 고려하여 목표 비중에
도달하는 주문금액을 계산한다. 현재 비중과 목표 비중이 동일하면 주문이 발생하지
않는 no-op으로 처리한다.

거래비용은 다음 항목을 포함한다.

- 매수·매도 수수료
- KRX 호가단위에 따른 동적 half-spread
- 설정된 slippage 및 execution uncertainty
- 체결일 기준 매도 거래세

비용은 주문 시점의 현금흐름에 직접 반영된다. 따라서 reward에서 동일 비용을 다시
차감하지 않는다.

현재 목표 비중 방식에서는 여섯 행동이 모두 실행 가능하다. `MaskablePPO`
인터페이스는 유지되지만 action mask는 모두 `True`이며, 과거 3행동
`Hold/Add/Clear` 환경에서 사용하던 invalid-action 제거 기능은 실질적으로
활성화되지 않는다.

## 6. 환경 전이와 Reward

한 환경 step의 처리 순서는 다음과 같다.

1. 시점 \(t\)의 observation을 정책에 제공한다.
2. 정책이 목표 비중 \(a_t\)를 선택한다.
3. `ExecPrice_t`를 기준으로 리밸런싱 주문을 체결한다.
4. 거래비용을 현금에서 차감한다.
5. 환경을 다음 5분 관찰 시점 \(t+1\)로 전이한다.
6. 포트폴리오 가치를 mark-to-market 방식으로 계산한다.
7. reward와 다음 observation을 반환한다.

기본 reward는 다음과 같다.

```text
r_t = 100 × log(V_t / V_(t-1))
```

\(V_t\)는 거래비용이 반영된 시점 \(t\)의 포트폴리오 가치다. 로그수익률은 시간에
대해 가법적이므로 에피소드 누적보상이 복리 자산 성장과 직접 연결된다. 배율
`100`은 분봉 수익률의 작은 수치 범위를 신경망 최적화에 적합한 규모로 조정한다.

환경에는 다음 reward shaping 항목도 구현되어 있다.

- inventory penalty
- turnover penalty
- incremental drawdown penalty
- downside return penalty
- benchmark-relative reward

현재 기본 설정에서는 이 계수들이 모두 0이므로 실제 목적함수는 거래비용 반영
포트폴리오 성장이다. 각 shaping 항목은 독립적인 ablation을 통해 효과를 검증한
후 적용해야 한다.

## 7. PPO 학습

에이전트는 Stable-Baselines3 Contrib의 `MaskablePPO`와 다층 퍼셉트론 정책을
사용한다. Actor는 행동별 선택 확률을 출력하고 Critic은 현재 상태의 기대
누적보상 \(V_\phi(s_t)\)를 추정한다.

Rollout에서 수집한 전이에 대해 Generalized Advantage Estimation(GAE)을 사용하여
advantage \(\hat{A}_t\)를 계산한다. 정책은 advantage가 양수인 행동의 확률을
증가시키고 음수인 행동의 확률을 감소시키는 방향으로 업데이트된다.

PPO는 다음 clipped surrogate objective를 사용하여 정책이 한 번의 업데이트에서
과도하게 변하는 것을 제한한다.

```text
L_clip(θ) =
E[min(r_t(θ) A_t,
      clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

여기서 \(r_t(\theta)\)는 이전 정책 대비 현재 정책의 확률비이고,
\(\epsilon=0.2\)는 clipping 범위다. 전체 손실은 policy loss, value loss,
entropy regularization의 조합으로 구성된다.

| Hyperparameter | 값 |
| --- | ---: |
| Total timesteps | 300,000 |
| Learning rate | 0.0003 |
| `n_steps` | 2,048 |
| Batch size | 256 |
| Epochs per update | 10 |
| Discount factor `gamma` | 0.999 |
| GAE lambda | 0.98 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Target KL | 0.03 |
| Network | MLP 128 × 128 |

`n_steps=2,048`은 한 번의 정책 업데이트 전에 수집하는 환경 전이 수를 의미한다.
수집된 rollout은 256개 단위 mini-batch로 나누어 10 epoch 동안 반복 사용한다.
`gamma=0.999`는 단기 체결 결과뿐 아니라 비교적 긴 보유기간의 성과를 반영하기
위한 설정이다.

## 8. Episode와 Truncation

학습 episode는 Train 구간에서 임의로 선택한 시작일부터 연속 20거래일로 구성된다.
하루 64개 관찰을 기준으로 약 1,280개 bar가 포함되며, 포지션과 현금은 거래일
경계를 넘어 유지된다. 이로써 overnight gap의 손익도 환경 전이에 반영된다.

20거래일 도달 시 환경은 `terminated=False`, `truncated=True`를 반환한다. 이는
경제적 의미의 종료상태가 아니라 학습을 위한 시간 제한이므로, PPO는 마지막
상태의 가치추정치를 이용하여 bootstrap할 수 있다. 에피소드 종료 후에는 Train
구간에서 새로운 시작일을 선택하여 경험 수집을 계속한다.

Episode 분할은 동일한 Train 데이터 내 경험 순서를 다양화하지만, 원본 데이터에
존재하지 않는 시장 국면을 생성하지는 않는다. 따라서 Train이 특정 상승장에
편향되어 있다면 episode randomization만으로 일반화 문제를 해결할 수 없다.

## 9. Validation과 Checkpoint 선택

학습 시작 시점, 매 25,000스텝 및 학습 종료 시점에 Validation 전체 구간을
결정론적으로 평가한다. Validation 과정에서는 정책 파라미터를 업데이트하지
않으며 checkpoint 선택에만 결과를 사용한다.

단일 Validation 누적수익률의 최댓값은 특정 구간의 우연한 수익이나 과잉매매에
민감할 수 있다. 현재는 Validation을 시간순 세 구간으로 나누고 다음 안정성 점수를
checkpoint 선택 기준으로 사용한다.

```text
selection_score
  = validation 구간별 수익률 중앙값
  - 0.25 × |validation 전체 MDD|
  - 0.001 × validation 전체 turnover
```

이 기준은 특정 하위 기간의 극단적인 수익이 전체 선택을 지배하는 현상을 줄이고,
낙폭과 거래빈도를 함께 통제하려는 목적을 가진다. 학습 종료 후 마지막 정책이
아니라 selection score가 가장 높았던 파라미터로 복원하여 artifact를 생성한다.

Artifact에는 다음 항목을 함께 저장한다.

- PPO 모델 파라미터
- feature schema와 observation 계약
- action space 계약
- Train 기준 정규화 통계
- 환경 및 거래비용 설정
- 학습 hyperparameter와 seed
- Validation checkpoint 기록

로드 시 현재 환경과 artifact의 schema 및 행동 계약을 비교하여 비호환 모델의
실행을 차단한다.

## 10. Baseline 정책

강화학습 정책의 절대수익률만으로는 유효성을 판단할 수 없다. 상승장에서는 단순
보유 정책도 높은 수익을 기록하므로, RL의 추가적인 의사결정 가치가 존재하는지
확인하려면 복잡도가 낮은 정책들과 동일 조건에서 비교해야 한다.

| 정책 | 정의 | 평가 목적 |
| --- | --- | --- |
| Cash | 주식 비중 0% 유지 | 무위험 노출 기준 |
| Static 20~80% | 일정한 부분 투자 비중 유지 | 시장 노출 수준별 통제군 |
| Buy & Hold | 초기 100% 매수 후 보유 | 완전 시장 노출 기준 |
| Volatility-scaled | 최근 변동성에 반비례하여 비중 조정 | 단순 위험관리 기준 |
| MA crossover | 이동평균 교차에 따른 0/100% 전환 | 전통적 추세추종 기준 |
| Random | 목표 비중을 균등 무작위 선택 | 비학습 정책의 하한선 |
| MaskablePPO | 상태에 따라 목표 비중 결정 | 평가 대상 정책 |

특히 Static 80%는 학습된 RL 정책이 80% 행동에 집중될 때 필요한 통제군이다. RL이
Static 80%와 유사한 노출을 유지하면서 더 많은 거래를 발생시킨다면, 초과성과가
없는 한 정책 복잡도와 비용을 정당화하기 어렵다.

비교의 공정성을 위해 모든 정책에 동일한 Test 데이터, 초기자산, 체결가격,
거래비용 및 최종 미청산 포지션의 가상 청산비용을 적용한다.

## 11. 평가 지표

정책 평가는 다음 지표를 종합하여 수행한다.

| 지표 | 의미 |
| --- | --- |
| Total return | 평가기간 전체 포트폴리오 수익률 |
| Sharpe ratio | 일별 수익률의 위험조정 성과 |
| Maximum drawdown | 이전 고점 대비 최대 누적손실 |
| Turnover | 누적 거래대금 / 초기자산 |
| Trade count | 실제 거래가 발생한 step 수 |
| Action rate | 각 목표 비중의 선택 비율 |
| Overnight hold rate | 거래일 경계에서 포지션을 보유한 비율 |

높은 수익률이 과도한 turnover와 큰 drawdown을 수반한다면 실거래 적용 가능성은
낮다. 또한 특정 action rate가 지나치게 높으면 정책이 단일 행동으로 수렴하거나
시장 국면을 충분히 구분하지 못했을 가능성을 검토해야 한다.

## 12. Multi-seed 평가

PPO 학습 결과는 신경망 초기화, episode 시작점, 행동 표본추출 및 mini-batch 구성
등의 확률적 요소에 영향을 받는다. Seed는 이 무작위 과정을 재현하기 위한 난수
초기값이다.

Multi-seed 평가는 동일한 데이터와 hyperparameter에서 seed만 변경하여 정책을
처음부터 독립적으로 재학습한다.

```text
seed 11 → 독립 학습 → artifact A → Test
seed 22 → 독립 학습 → artifact B → Test
seed 33 → 독립 학습 → artifact C → Test
```

저장된 artifact 하나를 평가 seed만 바꾸어 반복 실행하는 것은 학습 안정성을
검증하지 못한다. 결정론적 policy inference에서는 동일 모델이 거의 동일한 행동을
출력하기 때문이다.

Multi-seed 결과는 최고 성능 하나보다 다음 통계를 중심으로 해석한다.

- seed별 개별 Test 성과
- 수익률 평균과 중앙값
- seed 간 표준편차
- 최악 seed 성과
- 각 baseline 대비 승률

Seed 간 분산이 크다면 보고된 단일 성과가 학습 알고리즘의 일관된 결과가 아니라
초기조건에 따른 우연일 가능성이 높다.

## 13. Expanding Walk-forward 평가

고정된 Train/Validation/Test 분할 하나만으로는 특정 시장 국면에 대한 적합 여부와
시간적 일반화 능력을 분리하기 어렵다. 이를 보완하기 위해 Train 기간을 확장하면서
서로 겹치지 않는 미래 Test 구간을 순차적으로 평가한다.

```text
Fold 1: Train A     → Validation B → Test C
Fold 2: Train A+B   → Validation C → Test D
Fold 3: Train A+B+C → Validation D → Test E
```

각 fold에서 Test는 해당 시점의 Train과 Validation보다 뒤에 위치한다. 이전
fold의 Validation은 다음 fold에서 Train에 편입되고, 이전 Test는 다음 fold의
Validation으로 이동한다. 한 기간이 Test에서 곧바로 다음 Train으로 이동하지
않으므로 각 평가 시점에서 Train과 Validation의 시간적 선후관계가 유지된다.
각 fold의 최종 Test 구간은 서로 중복되지 않는다.

현재 240거래일 데이터에 적용되는 실제 구간은 다음과 같다.

| Fold | Train | Validation | Test |
| ---: | --- | --- | --- |
| 1 | 2025-07-03~2026-02-04, 144일 | 2026-02-05~2026-03-19, 24일 | 2026-03-20~2026-04-22, 24일 |
| 2 | 2025-07-03~2026-03-19, 168일 | 2026-03-20~2026-04-22, 24일 | 2026-04-23~2026-05-29, 24일 |
| 3 | 2025-07-03~2026-04-22, 192일 | 2026-04-23~2026-05-29, 24일 | 2026-06-02~2026-07-10, 24일 |

Fold는 학습과 평가에 사용하는 날짜 구간을 변경하고, seed는 동일한 fold 내에서
학습의 확률적 초기조건을 변경한다. 따라서 `3 folds × 5 seeds` 실험은 세 개의
미래 시장 구간에서 각각 다섯 개 정책을 독립 학습하는 총 15개 모델 평가를
의미한다.

| 단위 | 정의 | 검증 목적 |
| --- | --- | --- |
| Step | 5분 단위 환경 전이 | 개별 행동과 보상 |
| Episode | Train 내 연속 20거래일 | 경험 수집 단위 |
| Checkpoint | 한 학습 실행의 특정 timestep | 조기 종료 시점 선택 |
| Seed | 확률적 학습 초기조건 | 학습 안정성 |
| Fold | 시간순 데이터 구간 | 시장 국면 간 일반화 |

## 14. 통합 평가 프로토콜

최종 강건성 평가는 다음 순서로 수행한다.

```text
for each walk-forward fold:
    1. Train 통계로 feature normalizer 추정
    2. 각 seed에서 PPO를 독립적으로 초기화하고 학습
    3. 25,000스텝마다 Validation stability score 계산
    4. seed별 최고 checkpoint를 artifact로 저장
    5. 해당 fold의 독립 Test 구간에서 결정론적 평가
    6. 동일 Test에서 baseline 정책 평가

aggregate:
    seed·fold별 수익률, MDD, turnover 집계
    중앙값, 분산, 최악 성과, baseline 대비 승률 보고
```

정책의 유효성은 단일 최고수익률이 아니라 여러 seed와 여러 fold에서의 반복성으로
판단한다. 최소한 단순 고정 비중 및 Buy & Hold 대비 성과가 반복적으로 우수하고,
거래비용 변화에도 결과가 유지되며, 위험과 turnover가 허용 가능한 범위여야 한다.

## 15. 해석 및 한계

본 파이프라인이 학습하는 정책은 특정 시점의 독립적인 매수·매도 신호가 아니라,
시장 feature와 현재 포트폴리오 상태를 조건으로 목표 비중을 연속적으로 조절하는
상태의존적 자산배분 규칙이다.

다만 정책의 일반화 성능은 데이터가 포함하는 시장 국면에 의해 제한된다. Train이
강한 상승장에 편향되면 높은 주식 노출을 유지하는 행동이 보상 측면에서 우세해지고,
하락장이나 고변동 Test에서 성능이 급격히 저하될 수 있다. 이 문제는 학습 스텝을
늘리는 것만으로 해결되지 않으며, 장기간의 다양한 시장 데이터, 목적함수 ablation,
거래빈도 통제 및 반복적인 walk-forward 검증이 필요하다.

현재 multi-seed 및 walk-forward 실험의 수치 결과는
`docs/rl-robustness-results.md`에 별도로 기록한다.
