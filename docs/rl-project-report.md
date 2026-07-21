# 실제 시장 데이터 기반 강화학습 트레이딩 프로젝트 보고서

**프로젝트명:** `real_market_rl_trader`
**작성 기준일:** 2026-07-20
**담당 영역:** 강화학습 환경 연동, 정책 학습, 모델 artifact, validation 및 backtest 평가
**현재 단계:** 취업 포트폴리오용 연구 프로토타입

---

## 1. 요약

본 프로젝트는 한국 주식시장의 실제 분봉 데이터를 이용해 단일 종목의 포지션을
순차적으로 제어하는 강화학습 트레이딩 시스템을 구현하는 것을 목표로 한다. 가격
방향을 직접 예측하는 대신, 5분 및 30분 시장 feature와 현재 포지션 상태를 관찰하여
포지션을 유지하거나 확대하고 전량 청산하는 정책을 학습한다.

현재 시스템은 KIS 데이터 수집, 멀티 타임프레임 feature 생성, Gymnasium 환경,
거래비용 모델, MaskablePPO 학습, feature normalization, TensorBoard 진단,
versioned artifact 저장, 시간순 Train/Validation/Test 평가와 baseline 비교까지
연결되어 있다. 전체 테스트 231개가 통과하며, 단순 노트북을 넘어 재현 가능한 연구
프로토타입 형태를 갖추었다.

최종 300,000스텝 MaskablePPO 실험에서는 validation 최고 성능이 75,000스텝에서
관찰되어 해당 파라미터가 artifact에 저장되었다. 선택된 모델은 validation에서
`+10.29%`를 기록했지만, 완전히 뒤의 test 구간에서는 `-11.93%`를 기록해 일반화에
실패했다. 따라서 현재 성과는 수익 전략의 발견이 아니라, 실제 데이터 기반 RL
파이프라인 구축과 일반화 한계의 실험적 확인으로 해석해야 한다.

또한 코드 검토 과정에서 현재 5분봉 feature를 확인한 후 같은 봉의 마지막 가격으로
체결하는 실행 시점 편향이 발견되었다. 이 문제를 수정하기 전의 수익률은 최종 연구
결과로 사용할 수 없으며, 수정 후 모든 모델을 재학습해야 한다.

---

## 2. 프로젝트 목표

### 2.1 기술 목표

- 실제 KIS 분봉 데이터를 재현 가능한 형태로 수집하고 저장한다.
- 미래 정보 누수를 방지하는 멀티 타임프레임 feature를 생성한다.
- 거래비용과 포지션 상태를 포함한 Gymnasium 환경을 구현한다.
- PPO 계열 정책을 학습하고 baseline 정책과 동일한 환경에서 비교한다.
- 학습 데이터와 평가 데이터를 시간순으로 분리한다.
- 모델, normalization 통계와 환경 계약을 하나의 artifact로 저장한다.
- 학습 및 평가 결과를 테스트, 로그, metric으로 재현할 수 있게 한다.

### 2.2 연구 질문

1. PPO 계열 정책이 Buy & Hold, 이동평균, Random 정책보다 나은 포지션 제어를
   학습할 수 있는가?
2. 무효 행동 마스킹이 중복 no-op 행동을 제거하고 정책 학습을 개선하는가?
3. Validation에서 선택한 정책이 미래 Test 구간에서도 성능을 유지하는가?
4. 거래비용과 시장 regime 변화가 정책 행동과 일반화 성능에 어떤 영향을 주는가?

### 2.3 비목표

현재 단계에서는 다음을 목표로 하지 않는다.

- 실전 수익 보장
- 실제 계좌 자동 주문
- 다종목 포트폴리오 최적화
- Transformer, 계층적 RL, Multi-agent RL
- 대규모 하이퍼파라미터 탐색

---

## 3. 시스템 구조

```text
[KIS Open API]
      |
      v
[1분봉 수집 및 Parquet 저장]
      |
      v
[결손일 처리 및 5분/30분 Feature 생성]
      |
      v
[TradingEnvironment + FrictionModel]
      |
      v
[Feature Normalization + MaskablePPO]
      |
      v
[Full Validation Checkpoint Selection]
      |
      v
[Versioned Model Artifact]
      |
      v
[Backtest + Baseline Comparison + Metrics]
```

### 3.1 모듈 책임

| 경로 | 책임 |
|---|---|
| `env/src/pipeline/` | KIS 인증 및 시장 데이터 수집 |
| `env/src/data/` | resample, feature 생성, 데이터 품질 처리 |
| `env/src/env/` | 상태 전이, 포지션, reward, episode 관리 |
| `env/src/friction/` | 수수료, spread, slippage, 매도세 |
| `agent/src/models/` | PPO wrapper, normalization, 학습, artifact, validation |
| `agent/src/policies/` | baseline 정책 및 평가 orchestration |
| `experiments/` | 학습과 backtest 실행 진입점 |
| `evaluation/` | 수익률, Sharpe, drawdown, turnover metric |

현재 구조는 시장 환경, agent, application을 분리하는 FinRL의 3계층 구조와 유사한
방향을 갖지만, 단일 종목 연구 범위에 맞게 더 작고 명시적으로 구성되어 있다.

---

## 4. 데이터

### 4.1 데이터 현황

| 항목 | 값 |
|---|---:|
| 대상 종목 | 삼성전자 `005930` |
| 원본 주기 | 1분봉 |
| 원본 행 수 | 96,481 |
| 원본 월별 파일 | 13개 |
| 원본 기간 | 2025-07-02 ~ 2026-07-10 |
| Feature 행 수 | 15,318 |
| Feature 거래일 | 240일 |
| Feature 기간 | 2025-07-03 ~ 2026-07-10 |
| 일반적인 일별 5분봉 수 | 64개 |

시간순 기본 분할은 다음과 같다.

| Split | 거래일 | 비율 |
|---|---:|---:|
| Train | 168일 | 70% |
| Validation | 36일 | 15% |
| Test | 36일 | 15% |

### 4.2 데이터 품질 관찰

Feature 데이터의 첫 가격은 `62,750`, 마지막 가격은 `285,000`으로 전체 기간 동안
약 `+354%` 변화했다. 일부 overnight 변동은 약 `+12.7%`에서 `-9.6%` 범위로
관찰되었다. 이 값이 실제 시장 변화인지, 수정주가·단위·월 파일 결합 문제인지 원본
KIS 응답 및 외부 기준 데이터와 대조해야 한다.

현재 feature는 일별 warm-up 때문에 대부분의 거래일에서 약 10시 5분부터 시작한다.
따라서 장 시작부터 feature 준비 시점까지의 거래는 정책이 관찰하지 못하며, 전일
마지막 관측과 당일 첫 관측 사이의 수익은 overnight transition으로 반영된다.

---

## 5. Observation과 Action

### 5.1 Feature 계약

`FEATURE_SCHEMA_VERSION = 2`이며 시장 feature는 다음 9개다.

```text
log_ret_1
log_ret_3
log_ret_12
realized_vol_12
relative_volume
vwap_dev
trend_strength_30m
macd_hist_30m
vol_regime_30m
```

포트폴리오 상태 4개가 추가되어 최종 observation은 13차원이다.

```text
units_held_frac
unrealized_pnl_norm
holding_duration_norm
tod_frac
```

시장 feature 9개는 Train split 통계로 표준화하고 `[-5, 5]`로 clipping한다.
포트폴리오 상태는 원래 의미를 유지하기 위해 정규화하지 않는다. 학습 통계는
artifact의 `feature_normalization.json`에 저장되어 backtest와 inference에서
동일하게 복원된다.

### 5.2 Action 계약

| Action | 의미 |
|---:|---|
| 0 | Hold |
| 1 | Add 1 Unit |
| 2 | Clear Position |

1 Unit은 초기 자본의 20%이고 최대 5 Unit까지 보유할 수 있다. 현재 환경은
long-only이며 부분 축소 없이 전량 Clear만 지원한다.

MaskablePPO에서는 다음 행동을 마스킹한다.

- Flat 상태의 Clear
- 5 Unit 보유 상태의 Add

이를 통해 Hold와 동일한 결과를 만드는 중복 행동이 PPO의 학습 분포에 들어가는 것을
방지한다.

---

## 6. 환경과 Reward

### 6.1 Episode

- 기본 episode 길이: 연속 20거래일
- 포지션과 현금은 거래일 경계를 넘어 유지
- 하루 종료 강제청산 없음
- episode 경계는 `truncated=True`인 continuing task로 처리
- 전체 split backtest 종료 시 미청산 포지션의 가상 청산 비용 반영

### 6.2 거래비용

현재 기본 비용은 다음과 같다.

| 비용 | 비율 |
|---|---:|
| Fee | 0.10% |
| Spread | 0.05% |
| Slippage | 0.05% |
| Sell tax | 매도 시 0.20% |

고정 비용 기준 왕복 거래비용은 대략 0.60%다. 비용은 별도 reward penalty로 다시
차감하지 않고 실제 cash flow에서 한 번 차감한다. 다만 현재 비율은 실제 KIS 계좌
조건 및 체결 자료와 다시 보정해야 한다.

### 6.3 Reward

현재 기본 reward는 비용 차감 포트폴리오 로그수익률이다.

```text
reward_t = 100 * log(portfolio_value_t / portfolio_value_t-1)
```

다음 항은 코드에서 ablation 가능하지만 현재 기본값은 0이다.

- Inventory penalty
- Turnover penalty
- Incremental drawdown penalty
- Downside return penalty
- Benchmark-relative reward

단순 reward는 목적을 명확하게 하지만, 거래하지 않는 Cash 정책이 안전한 지역해가
될 수 있다. Benchmark-relative reward를 추가하면 시장 대비 초과성과를 목표로
바꾸게 되므로, 적용 전 프로젝트의 최적화 목표를 명시해야 한다.

---

## 7. 강화학습 설정

현재 기본 agent는 SB3 Contrib의 MaskablePPO다.

| Parameter | 값 |
|---|---:|
| Total timesteps | 300,000 |
| Learning rate | 0.0003 |
| `n_steps` | 2,048 |
| Batch size | 256 |
| Epochs/update | 10 |
| Gamma | 0.999 |
| GAE lambda | 0.98 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Target KL | 0.03 |
| Network | MLP 128 x 128 |

### 7.1 Validation 최고 모델 선택

학습 시작, 매 25,000스텝, 학습 종료 시점에 validation 전체 36일을 결정론적으로
평가한다. Validation terminal return이 가장 높은 시점의 모델 파라미터를 메모리에
보존하고, 학습 종료 후 해당 파라미터를 복원하여 artifact에 저장한다.

최종 실험에서는 총 14회 validation 평가가 수행되었고, 75,000스텝이 최고
checkpoint로 선택되었다.

```text
75,000 steps validation return:  +10.29%
300,000 steps validation return:  +2.10%
```

이는 최고 모델 선택 기능이 정상 작동했음을 보여주는 동시에, 추가 학습이 반드시
일반화 성능 개선으로 이어지지 않음을 보여준다.

---

## 8. 실험 결과

### 8.1 초기 PPO

초기 300,000스텝 PPO artifact는 대부분의 시간 동안 현금 상태를 유지하는 정책에
가까웠다.

| Split | PPO | Buy & Hold | PPO 거래 수 |
|---|---:|---:|---:|
| Train | +11.48% | 시장 +219.52% | 26 |
| Validation | -1.92% | +38.29% | 22 |
| Test | -1.31% | +3.59% | 2 |

거래비용이 존재하고 reward가 불확실한 환경에서 PPO가 거래를 회피하는 지역해에
수렴한 것으로 해석했다.

### 8.2 10,000스텝 MaskablePPO 구조 검증

행동 마스킹과 validation callback을 연결한 후 실제 데이터에서 10,000스텝 smoke
학습을 수행했다.

| Split | MaskablePPO | Buy & Hold |
|---|---:|---:|
| Validation | -2.57% | +38.29% |
| Test | +22.29% | +3.59% |

이 결과는 구조의 정상 작동을 확인하기 위한 smoke result이며, 짧은 학습 한 번의
우연한 Test 성과이므로 모델 우위의 증거로 사용하지 않는다.

### 8.3 300,000스텝 MaskablePPO

최종 artifact:

```text
artifacts/maskableppo-fs2-20260720-130052
```

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| PPO total return | +43.60% | +10.29% | -11.93% |
| Market return | +219.52% | +38.81% | +5.17% |
| Buy & Hold return | - | +38.29% | +3.59% |
| Final portfolio value | 14,360.28 | 11,029.39 | 8,807.47 |
| Sharpe ratio | 2.44 | 2.17 | -2.31 |
| Max drawdown | -7.71% | -6.76% | -14.37% |
| Trade count | 343 | 106 | 165 |
| Turnover | 95.12 | 25.78 | 37.58 |
| Overnight hold rate | 90.42% | 85.71% | 74.29% |

Test baseline 비교:

| Policy | Test return | Max drawdown | Trade count |
|---|---:|---:|---:|
| Buy & Hold | +3.59% | -23.48% | 5 |
| Random | -89.45% | -89.45% | 1,089 |
| MA Crossover | -9.88% | -16.32% | 383 |
| MaskablePPO | -11.93% | -14.37% | 165 |

MaskablePPO는 Random보다 합리적인 정책을 학습했고 MA와 비슷한 손실 범위를 보였지만,
Test에서 Buy & Hold보다 낮은 성능을 기록했다. Validation에서 선택된 모델이 Test에서
손실을 기록했으므로 미래 시장 regime으로 일반화되지 않았다.

### 8.4 학습 로그 진단

최종 학습 로그의 주요 값은 다음과 같다.

| 로그 | 최종 값 | 해석 |
|---|---:|---|
| `approx_kl` | 0.000995 | 정책 업데이트 폭주 없음 |
| `clip_fraction` | 0.0081 | 후반 정책 업데이트가 작음 |
| `entropy_loss` | -0.0301 | 정책이 거의 결정론적으로 수렴 |
| `value_loss` | 0.0990 | 절대값만으로 오류 판단 불가 |
| `explained_variance` | 0.0292 | Critic의 누적보상 예측력이 매우 낮음 |
| `rollout/ep_rew_mean` | 0.2284 | 학습 episode 평균 reward 개선 |

KL과 clip 지표에서 수치적 발산은 확인되지 않았다. 그러나 explained variance가 거의
0이므로 value function이 미래 return을 충분히 설명하지 못했고, validation return도
checkpoint에 따라 `+10.29%`에서 `-14.61%`까지 크게 변동했다.

---

## 9. 발견된 문제

### 9.1 체결 시점 편향

5분봉은 `label=right`, `closed=left`로 생성된다. 예를 들어 09:00~09:04 데이터가
09:05 feature로 완성된다. 현재 환경은 이 feature를 관찰한 후 해당 row의 `Close`,
즉 이미 지나간 09:04 가격으로 체결한다.

```text
09:00~09:04 정보 관찰
        -> 09:05 feature 완성
        -> 09:04 마지막 가격으로 체결
```

이는 실제 주문에서 불가능한 낙관적 체결이다. 다음 봉 Open 또는 명시적으로 지연된
실행 가격을 사용하도록 환경 계약을 수정해야 한다. 수정 시 기존 artifact와 실험
결과는 폐기하고 재학습해야 한다.

### 9.2 제한된 데이터와 Regime 변화

Train 시장수익은 `+219.52%`, Validation은 `+38.81%`, Test는 `+5.17%`로 시장
분포가 크게 다르다. 한 종목 240거래일만으로 다양한 상승·하락·횡보 regime을
학습하기 어렵다. 15,318개의 5분봉이 존재해도 같은 날과 같은 시장 추세의 관측은
독립 표본이 아니다.

### 9.3 단일 Seed

현재 최종 결과는 seed 42 한 번의 학습 결과다. PPO는 초기화 및 rollout sampling에
따라 결과 분산이 크므로 최소 3개, 권장 5개 seed의 중앙값과 분산을 보고해야 한다.

### 9.4 Validation 및 Test 재사용

개발 과정에서 같은 Validation과 Test 결과를 여러 번 확인했다. 이후 설정을 이 결과에
맞춰 변경하면 Test도 사실상 모델 선택에 사용된 데이터가 된다. 다음 최종 보고에서는
새로운 미래 holdout을 확보하거나 반복 walk-forward 구간을 사전에 고정해야 한다.

### 9.5 높은 Turnover

선택된 정책의 turnover는 Validation 25.78, Test 37.58이다. `ent_coef=0.01`은
MaskablePPO 기본값 0보다 높은 탐험 설정이며, 거래 환경에서는 불필요한 매매를
증가시킬 수 있다. 환경 수정 후 `0`, `0.001`, `0.005`를 제한적으로 비교해야 한다.

### 9.6 단순한 체결 모델

현재 spread와 slippage는 고정 비율이다. 실제 호가, 주문 크기 대비 거래량, latency,
부분 체결, 주문 거부와 market impact는 반영되지 않는다. 현재 결과는 실제 체결
성과가 아닌 friction-adjusted simulation 결과다.

### 9.7 Action 비대칭

정책은 1 Unit씩 포지션을 늘릴 수 있지만 감소는 전량 Clear만 가능하다. 점진적인
위험 축소가 불가능하므로 잦은 전량 청산과 재진입이 발생할 수 있다. 향후 target
position action 또는 Remove 1 Unit을 별도 ablation으로 비교할 수 있다.

### 9.8 보안

개발 과정에서 KIS credential이 외부 대화에 노출되었다. Git에 포함되지 않았더라도
기존 key는 폐기하고 재발급해야 한다. 저장소 공개 전 Git history와 현재 파일에 대한
secret scan도 수행해야 한다.

---

## 10. 관련 연구 및 프로젝트 비교

### 10.1 Moody & Saffell

초기 직접 강화학습 트레이딩 연구는 가격 예측 오차보다 wealth, Sharpe와 거래비용을
직접 최적화했다. 이전 포지션과 거래비용에 따른 시간 의존성을 중요한 상태로 다뤘다.

- [Reinforcement Learning for Trading, NeurIPS 1998](https://papers.neurips.cc/paper_files/paper/1998/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html)
- [Learning to Trade via Direct Reinforcement, 2001](https://pubmed.ncbi.nlm.nih.gov/18249919/)

본 프로젝트는 포트폴리오 상태와 비용을 환경에 포함한다는 점에서 같은 문제의식을
갖지만, differential Sharpe 등 목적함수 비교와 장기간 민감도 분석은 부족하다.

### 10.2 Deep Reinforcement Learning for Trading

Zhang, Zohren, Roberts는 2011~2019년의 50개 유동성 높은 선물 계약을 이용해
discrete/continuous action, volatility scaling, 거래비용과 여러 자산군을 비교했다.

- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107)

본 프로젝트와 가장 큰 차이는 모델보다 데이터 규모와 다양성이다. 현재 프로젝트는
삼성전자 한 종목과 약 1년 데이터만 사용한다.

### 10.3 FinRL-Meta

FinRL-Meta는 DataOps 기반 데이터 처리, 다양한 Gym 시장 환경, agent와 application
계층, 여러 알고리즘 benchmark를 제공한다. 낮은 신호대잡음비, survivorship bias와
backtest overfitting을 금융 RL의 핵심 문제로 제시한다.

- [FinRL Three-layer Architecture](https://finrl.readthedocs.io/en/latest/start/three_layer.html)
- [FinRL-Meta](https://arxiv.org/abs/2211.03107)

본 프로젝트는 작은 범위에서 유사한 모듈 분리를 달성했지만, 데이터 소스, 환경 수,
알고리즘 비교와 benchmark 규모는 FinRL보다 제한적이다.

### 10.4 TradeMaster

TradeMaster는 멀티모달 시장 데이터, 여러 simulator, 13개 이상의 RL 알고리즘과
다축 평가 toolkit을 제공하는 연구 플랫폼이다.

- [TradeMaster GitHub](https://github.com/TradeMaster-NTU/TradeMaster)

본 프로젝트는 범용 연구 플랫폼보다 특정 KIS 단일 종목 lifecycle 제어 문제를 깊게
검증하는 방향이 적합하다.

### 10.5 재현성과 Backtest 과적합

RL 연구는 seed와 구현 세부사항에 따른 결과 분산이 크고, 금융에서는 많은 전략을
시험한 후 최고 결과만 선택할 때 backtest 과적합 위험이 더 커진다.

- [Deep Reinforcement Learning That Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
- [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)

따라서 단일 최고 수익률보다 여러 seed, 여러 시간 구간, 비용 stress 결과를 함께
보고해야 한다.

---

## 11. 현재 프로젝트 수준

| 영역 | 평가 | 수준 |
|---|---|---:|
| 소프트웨어 구조 | 환경·agent·experiment·평가 분리 | 6.5/10 |
| RL 파이프라인 | MaskablePPO, 정규화, validation 선택 | 5/10 |
| 연구 신뢰성 | 단일 종목·짧은 기간·단일 seed | 3/10 |
| 시장 시뮬레이션 | 비용 반영, 체결 모델은 단순 | 3/10 |
| 실전 준비 | Paper trading과 주문 실행 미완성 | 1/10 |

종합하면 단순 교육용 예제를 넘어선 고급 학부 또는 초기 대학원 수준의 연구
프로토타입이다. 취업 포트폴리오로 경쟁력이 있지만, 공개 전 환경 정확성, 반복 실험,
문서화와 inference demo를 보강해야 한다.

---

## 12. 발전 계획

### Phase 0. 실험 유효성 복구

- 다음 거래 가능 가격으로 체결 시점 변경
- 실행 가격과 feature timestamp 계약 문서화
- 같은 봉 체결을 막는 회귀 테스트 추가
- 원본 가격과 수정주가 검증
- KIS credential 재발급 및 secret scan
- 기존 artifact를 legacy 결과로 표시

### Phase 1. 평가 자동화

- 3~5개 학습 seed 자동 실행
- 여러 walk-forward 구간 생성
- 평균, 중앙값, 표준편차, 최악 성능 보고
- 거래비용 1배, 1.5배, 2배 stress test
- 실험 설정과 결과를 JSON/CSV로 저장
- Test를 모델 선택에 사용하지 않는 protocol 고정

### Phase 2. 제한적 PPO Ablation

환경 수정 후 다음 축만 작은 grid로 비교한다.

```text
ent_coef:   0 / 0.001 / 0.005
gamma:      0.995 / 0.997 / 0.999
gae_lambda: 0.95 / 0.98
n_steps:    1024 / 2048 / 4096
```

Reward는 다음 순서로 비교한다.

1. 비용 차감 log return
2. log return + 작은 downside penalty
3. benchmark-relative return
4. 필요할 때만 drawdown 또는 turnover regularization

### Phase 3. 데이터 및 Baseline 확장

- SK하이닉스, 현대차, NAVER 등 대형주 추가
- Cash, Buy & Hold, MA 외 momentum과 volatility-scaled baseline 추가
- 상승, 하락, 횡보 regime별 성능 분해
- 여러 종목을 공유하는 단일 정책 검토
- MLP가 안정적으로 일반화한 후에만 recurrent policy 검토

### Phase 4. 취업 포트폴리오 완성

- `POST /predict`, `GET /model/metadata`, `GET /health` FastAPI 구현
- Docker 기반 재현 실행
- GitHub Actions에서 테스트와 training smoke 실행
- Equity curve, drawdown, action, seed 분포 그래프 생성
- 실제 체결 없이 historical event replay 기반 paper demo 구현
- 3분 내 이해 가능한 README와 짧은 데모 영상 제작

---

## 13. 재현 방법

환경 설치:

```powershell
cd C:\Users\AN\Desktop\rl_trader\real_market_rl_trader
conda activate rl-trader-py310
pip install -r requirements.txt
```

학습:

```powershell
python experiments/train.py --symbol 005930 --split train --total-timesteps 300000 --seed 42
```

TensorBoard:

```powershell
tensorboard --logdir runs\tensorboard
```

Backtest:

```powershell
python experiments/backtest.py `
  --symbol 005930 `
  --split test `
  --compare-baselines `
  --artifact artifacts\maskableppo-fs2-20260720-130052
```

테스트:

```powershell
python -m pytest -q
```

---

## 14. 취업 포트폴리오에서 강조할 내용

### 핵심 기여

- 실제 KIS 분봉 데이터를 사용한 end-to-end 강화학습 파이프라인 구현
- 5분/30분 causal feature와 13차원 observation 계약 설계
- 거래비용을 포함한 다거래일 Gymnasium 환경 구현
- MaskablePPO 기반 invalid-action masking 적용
- Train 기준 normalization 및 versioned artifact 복원 구현
- Full-validation 최고 checkpoint 선택과 baseline 비교 자동화
- 환경, leakage, artifact, 학습 및 backtest를 포함한 231개 테스트 구축
- Validation/Test 성능 차이와 체결 시점 편향을 분석해 일반화 실패 원인 식별

### 이력서 문장 예시

> 실제 KIS 분봉 데이터를 활용한 강화학습 트레이딩 파이프라인을 설계하고,
> Gymnasium 기반 다거래일 환경, 거래비용 모델, MaskablePPO, 시간순 validation,
> versioned model artifact를 구현했다.

> 데이터 누수, 환경 전이, 학습 및 backtest를 검증하는 231개 회귀 테스트와
> TensorBoard 진단을 구축하고, validation 최고 모델의 test 성능 저하를 분석해
> 체결 시점 편향과 시장 regime 변화에 따른 일반화 문제를 식별했다.

---

## 15. 결론

본 프로젝트는 실제 시장 데이터 기반 강화학습 시스템의 전체 흐름을 구현했다는 점에서
취업 포트폴리오로 의미가 있다. 특히 데이터, 환경, agent, artifact와 평가 계층을
분리하고 테스트 가능한 형태로 만든 점은 단순 모델 학습보다 강한 엔지니어링 성과다.

반면 현재 MaskablePPO는 Test에서 `-11.93%`를 기록해 수익 전략으로서 유효하지 않다.
동일 봉 종가 체결 문제, 제한된 데이터, 단일 seed, 높은 turnover와 낮은 critic 설명력
때문에 현재 수치를 최종 성과로 주장할 수 없다.

다음 단계의 핵심은 더 복잡한 신경망을 추가하는 것이 아니다. 체결 시점 계약을 먼저
수정하고, 여러 seed와 walk-forward 구간에서 반복 가능한 평가 체계를 만드는 것이다.
그 과정과 실패 원인을 투명하게 기록할 때 본 프로젝트는 "주가를 맞히는 데모"가 아니라
"실제 ML 시스템을 설계하고 검증할 수 있는 엔지니어링 프로젝트"로 완성될 수 있다.
