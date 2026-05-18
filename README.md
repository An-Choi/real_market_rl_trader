# real-market-rl-trader

Python 기반 강화학습 트레이딩 시스템의 초기 코드 구조입니다. 이 프로젝트는 완성된 모델이나 수익성 있는 전략을 제공하는 것이 아니라, 실제 시장 데이터를 backbone으로 사용하는 RL 트레이딩 실험을 이어가기 위한 skeleton code를 제공합니다.

## 프로젝트 개요

`real-market-rl-trader`는 가격을 직접 예측하는 모델이 아닙니다. 시장 가격 데이터는 환경의 backbone으로 사용하고, agent는 미래 가격 자체를 예측하지 않습니다. 대신 market regime, liquidity, pressure, friction 정보를 관찰한 뒤 거래 여부와 포지션 방향을 결정합니다.

초기 action space는 discrete action입니다.

- `0`: Hold
- `1`: Buy
- `2`: Sell

구조는 이후 position size, leverage, continuous action, portfolio-level allocation으로 확장할 수 있도록 열어두었습니다.

## 핵심 아이디어

- 실제 OHLCV 시장 데이터를 기반으로 환경을 구성합니다.
- feature는 regime, liquidity, pressure 계열로 분리합니다.
- friction은 fee, spread, slippage, execution uncertainty를 포함합니다.
- reward는 `portfolio return - friction cost - risk penalty` 형태로 확장 가능한 틀만 둡니다.
- RL agent는 Stable-Baselines3와 연결할 수 있는 wrapper 형태로 둡니다.
- baseline agent와 backtest engine을 먼저 두어 RL 이전에도 실험 루프를 검증할 수 있게 합니다.

## 폴더 구조

```text
real-market-rl-trader/
|
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
|
├── data/
│   ├── raw/
│   └── processed/
|
├── src/
│   ├── main.py
│   │
│   ├── data/
│   │   ├── data_collector.py
│   │   └── data_loader.py
│   │
│   ├── features/
│   │   ├── feature_engineer.py
│   │   ├── regime_features.py
│   │   ├── liquidity_features.py
│   │   └── pressure_features.py
│   │
│   ├── env/
│   │   ├── trading_env.py
│   │   └── friction_model.py
│   │
│   ├── agent/
│   │   ├── rl_agent.py
│   │   └── baseline_agents.py
│   │
│   ├── backtest/
│   │   ├── backtest_engine.py
│   │   └── metrics.py
│   │
│   ├── paper_trading/
│   │   └── paper_trading_engine.py
│   │
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
|
└── notebooks/
    └── quick_start.ipynb
```

## 실행 방법

Python 3.10+ 환경을 권장합니다.

```bash
pip install -r requirements.txt
python src/main.py
```

`data/raw/sample_ohlcv.csv`가 있으면 해당 파일을 사용하고, 없으면 `main.py`에서 간단한 synthetic OHLCV 데이터를 만들어 전체 파이프라인을 smoke test합니다.

## 현재 구현 범위

- OHLCV 데이터 수집 및 로딩을 위한 skeleton
- regime, liquidity, pressure feature 생성 함수 틀
- fee, spread, slippage, execution uncertainty를 포함하는 friction model
- Gymnasium 스타일의 discrete-action `TradingEnvironment`
- Stable-Baselines3 연결을 위한 `RLAgent` wrapper
- Buy-and-hold, moving average crossover, random, rule-based baseline agent 틀
- agent와 environment를 연결하는 backtest loop
- total return, Sharpe ratio, max drawdown, win rate, trade count, turnover metric 함수 틀
- 실제 거래소 API 없이 동작 구조만 잡은 paper trading engine

## 향후 확장 계획

- 실제 yfinance 또는 broker API 기반 데이터 수집 구현
- feature normalization, train/test split, walk-forward validation 추가
- order book, tick, quote 데이터 기반 liquidity/pressure feature 확장
- position size와 continuous action space 지원
- slippage와 partial fill을 반영한 execution simulator 확장
- Stable-Baselines3 PPO, DQN, A2C 학습 파이프라인 연결
- 실험 tracking, checkpointing, model registry 추가
- paper trading sandbox와 실거래 API adapter 분리
