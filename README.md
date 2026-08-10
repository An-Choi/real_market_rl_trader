# real-market-rl-trader

실제 시장 데이터로 강화학습 트레이딩을 실험하는 플랫폼입니다. 목표는 "잘 맞히는 모델"이 아니라 **재현 가능한 RL baseline** — 시장을 예측하는 대신, 시장 상태를 보고 단일 종목 포지션을 언제 키우고 정리할지를 학습합니다.

## 개요

실제 OHLCV를 환경의 backbone으로 두고, agent는 5분/30분 멀티타임프레임(MTF) feature와 포지션 상태를 관찰해 **포지션을 점진적으로 제어**합니다.

- **Action (3개, discrete):** `0` Hold · `1` Add 1 Unit · `2` Clear
- **Unit scaling-in:** 1 Unit = 자본의 20%, 최대 5 Unit(=100%). Turtle Trading식 분할 진입. long-only.
- **에피소드:** 1 종목 × 연속 N거래일(기본 20, config `episode_days`). Overnight 보유 허용, 강제청산 없음 — episode 끝은 truncation(continuing task)이며, 미청산 종료분은 평가 단계에서 가상 청산비용으로 정산.
- **거래비용:** 수수료·스프레드·슬리피지 + 한국장 매도 증권거래세 0.20%. reward는 비용 차감 로그수익률과 분리 가능한 risk/benchmark 항으로 구성.

## 구조

```text
real-market-rl-trader/
│
├── env/
│   ├── src/
│   │   ├── env/
│   │   ├── data/
│   │   ├── friction/
│   │   ├── pipeline/
│   │   └── utils/
│   ├── configs/
│   └── tests/
│
├── agent/
│   └── src/
│       ├── models/
│       └── policies/
│
├── experiments/
├── evaluation/
├── scripts/
│
├── data/
│   ├── raw/
│   └── processed/
│
└── docs/
```

## 시작하기

Python 3.10+ 권장. 인터프리터는 `python3`을 씁니다.

```bash
pip install -r requirements.txt

python3 experiments/train.py                    # PPO 학습
python3 experiments/backtest.py                 # baseline/agent 백테스트
python3 experiments/walk_forward_train.py --plan-only  # 다중 시계열 fold 확인
python3 scripts/backfill.py --symbols 005930    # KIS 분봉 백필
python3 -m pytest env/tests/ -v                 # 테스트
```

KIS API 키는 `.env`에 둡니다(`.env.example` 참고). `train.py`는 real 분봉 데이터를 요구합니다 — 먼저 `scripts/backfill.py --symbols 005930`으로 `data/raw/`에 분봉을 백필해야 합니다(`data/raw/`·`data/processed/`는 git에서 제외되므로 fresh clone에는 없음). 이후 `build_features`가 `data/processed/`에 feature parquet 캐시를 만듭니다.

## 현재 구현 범위

- real-OHLCV 단일채널 `TradingEnvironment` (Gymnasium, Unit scaling-in, 연속 N거래일 에피소드)
- leakage-safe MTF feature 생성 (5분 micro + 30분 context, 총 9개; 1분봉→5분 그리드 causal resample)
- 매도 거래세 포함 friction model
- SB3 Contrib `MaskablePPO` 연결용 `RLAgent` wrapper와 invalid-action masking
- train split 기준 feature standardization과 artifact stats 복원
- 학습 중 validation 전체 구간 평가와 최고 checkpoint artifact 저장
- baseline agent (buy-and-hold, MA crossover, random 등)
- split 전체 날짜 agent ↔ env 백테스트, 평가 metric (return / Sharpe / max drawdown / turnover 등)
- expanding walk-forward 다중 학습/검증/test와 bull·bear·sideways 사후 구간 표시
- 중앙 validation 수익률 + 최악 window + 평균 낙폭 기반 robust checkpoint 선택
- feature 품질·미래수익 상관·정책 permutation sensitivity 리포트
- 거래소 API 없이 동작 구조만 잡은 paper trading engine

## 다중 시장 구간 평가

최근 test 한 구간에 의존하지 않고, 시간 순서를 지킨 독립 fold들을 먼저 확인합니다.

```bash
python experiments/walk_forward_train.py --plan-only \
  --folds 3 --validation-days 20 --test-days 20

# fold마다 별도 학습 → validation checkpoint 선택 → 다음 test window 평가
python experiments/walk_forward_train.py \
  --folds 3 --validation-days 20 --test-days 20 --total-timesteps 300000
```

각 test window의 시장 수익률과 bull/bear/sideways 라벨은 결과 설명용이며, fold를
선택하거나 모델을 학습하는 입력으로 사용하지 않습니다.

## Feature 진단

```bash
python experiments/feature_report.py \
  --artifact artifacts/<artifact-id> --split validation
```

리포트는 결측·비정상값·정규화 clip 비율, 미래 1/3/12 bar 수익률과의 순위상관,
feature 간 중복도, feature permutation에 따른 정책 행동확률 변화와 action flip 비율을
한 파일에 기록합니다.

TensorBoard 사용자 정의 로그는 다음 핵심 그룹만 기록합니다.

- `performance/`: 전략·시장·초과수익
- `validation/`: 선택점수·중앙·최악수익·평균 낙폭
- `risk/`: 평균 노출
- `policy/`: Hold/Add/Clear 비율
- `trading/`: 비용과 강제청산
- `reward/`: 실제 활성화된 보상 항
