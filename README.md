# real-market-rl-trader

실제 시장 데이터로 강화학습 트레이딩을 실험하는 플랫폼입니다. 목표는 "잘 맞히는 모델"이 아니라 **재현 가능한 RL baseline** — 시장을 예측하는 대신, 시장 상태를 보고 단일 종목 포지션을 언제 키우고 정리할지를 학습합니다.

## 개요

실제 OHLCV를 환경의 backbone으로 두고, agent는 regime·liquidity·pressure·friction 신호를 관찰해 **포지션을 점진적으로 제어**합니다.

- **Action (3개, discrete):** `0` Hold · `1` Add 1 Unit · `2` Clear
- **Unit scaling-in:** 1 Unit = 자본의 20%, 최대 5 Unit(=100%). Turtle Trading식 분할 진입. long-only.
- **에피소드:** 1 종목 × 1 거래일. 장 마감에 강제 청산(overnight 없음).
- **거래비용:** 수수료·스프레드·슬리피지 + 한국장 매도 증권거래세 0.20%. reward는 비용 차감 후 realized return.

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
python3 scripts/backfill.py --symbols 005930    # KIS 분봉 백필
python3 -m pytest env/tests/ -v                 # 테스트
```

KIS API 키는 `.env`에 둡니다(`.env.example` 참고). 샘플 데이터가 없으면 `train.py`가 synthetic OHLCV로 전체 파이프라인을 smoke test합니다.

## 현재 구현 범위

- real-OHLCV 단일채널 `TradingEnvironment` (Gymnasium, Unit scaling-in, 1거래일 에피소드)
- regime·liquidity·pressure feature 생성
- 매도 거래세 포함 friction model
- Stable-Baselines3 연결용 `RLAgent` wrapper
- baseline agent (buy-and-hold, MA crossover, random 등)
- agent ↔ env 백테스트 루프, 평가 metric (return / Sharpe / max drawdown / turnover 등)
- 거래소 API 없이 동작 구조만 잡은 paper trading engine
