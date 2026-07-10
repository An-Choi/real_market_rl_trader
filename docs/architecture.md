# Project Architecture — real_market_rl_trader

**최종 업데이트:** 2026-07-05
**목적:** 이 문서는 팀 공유 컨텍스트의 source of truth다. 설계 결정과 구조를 여기서
확인하고, 코드와 이 문서가 어긋나면 이 문서를 먼저 갱신한다.

---

## 1. 프로젝트 한 줄 요약

KOSPI 대형주 대상, **시장 상태를 보고 단일 종목 포지션의 lifecycle(진입/추가/청산)을
sequential하게 제어하는 PPO 정책**을 학습하고, 이를 실시간 trading backend와 연결하는
end-to-end 플랫폼. 가격 예측 프로젝트가 아니다. 1차 목표는 수익률이 아니라
**재현 가능한 baseline + leakage 없는 파이프라인**.

---

## 2. 핵심 설계 결정 (고정)

| 항목 | 결정 |
|------|------|
| 거래 단위 | 5분봉 execution, 1종목 × 1거래일 에피소드, 마감(15:30) 강제청산 |
| Action | Discrete 3-action: `0=Hold / 1=Add 1 Unit / 2=Clear` |
| Unit | 1 Unit = 자본 20%, 최대 5 Units (Turtle-style scaling-in). continuous sizing 금지 |
| Observation | 5분 micro + 30분 context feature + portfolio state (일봉 regime은 ablation 후보) |
| Reward | scaled log return + ablatable downside/drawdown/benchmark-relative terms |
| 거래비용 | env transition에서 직접 차감: 수수료·스프레드·슬리피지 + 매도 증권거래세 0.20% |
| 종목 | 005930 1종목으로 baseline 완주. 코드는 처음부터 multi-symbol-ready |
| 정책 구조 | Single shared PPO policy (SB3). HRL 사용 안 함 |
| 검증 | strict time-order split 전체 날짜 평가 + multi-seed + baseline 비교 (B&H / MA cross / Random) |
| 배포 | FastAPI inference server + Spring trading service + KIS 모의투자 |

**baseline 완성 전 금지:** HRL/듀얼 에이전트, 별도 sizing/execution agent, continuous
sizing, portfolio allocation RL, transformer, 대규모 HP search. (폐기가 아니라 baseline
완성 후 ablation으로 미룬 것)

**데이터 한계 (설계 제약):** KIS 분봉은 rolling ~1년이 하드 한계 (가장 이른 가용일
2025-05-22). 이 때문에 (a) 일일 자동 수집으로 데이터를 계속 누적하고, (b) 다년치
일봉으로 robustness sanity-check를 보완한다.

---

## 3. 시스템 아키텍처

```text
[KIS API]
   ↓
[Data Pipeline]            scripts/backfill.py, env/src/pipeline/
   ↓
[Feature Engineering]      env/src/data/  (FeatureEngineer — §5 contract)
   ↓
[Trading Environment]      env/src/env/trading_env.py (Gymnasium)
   ↓
[PPO Training]             experiments/train.py (SB3)
   ↓
[Evaluation / Walk-forward] experiments/backtest.py, evaluation/
   ↓
[Versioned Model Artifact]
   ↓
[FastAPI Inference Server]
   ↓
[Spring Trading Service]
   ├─ 5분 스케줄러 / OrderManager / RiskGate / PortfolioManager
   ├─ Audit/Event Log, Prometheus/Grafana
   └─ KIS 모의투자 API
```

**서빙 경계 원칙:** feature 계산과 inference는 Python(FastAPI) 쪽이 담당하고
(학습 코드와의 파리티, §5), Spring은 스케줄링·주문 실행·리스크·포트폴리오 상태
관리·감사 로그만 담당한다. Spring에서 feature를 재구현하지 않는다.

---

## 4. 디렉터리 구조

```text
real_market_rl_trader/
|-- env/                  # 핵심: 환경·데이터·feature·friction
|   |-- src/
|   |   |-- data/         # FeatureEngineer, resample, micro/context features, defect_days
|   |   |-- env/          # trading_env.py — Unit 기반 Gymnasium env
|   |   |-- friction/     # friction_model.py — 수수료/스프레드/슬리피지/매도 거래세
|   |   |-- pipeline/     # KIS auth, data_collector, kis_historical
|   |   |-- execution/    # order_executor
|   |   `-- utils/        # config_loader, logger
|   |-- configs/config.yaml
|   `-- tests/            # 결정론·leakage 불변식 테스트 포함
|-- agent/                # src/models/rl_agent.py (SB3 PPO), src/policies/baseline_agents.py
|-- experiments/          # train.py, backtest(_engine).py, paper_trading(_engine).py
|-- evaluation/           # metrics.py, sim_to_live.py
|-- scripts/              # backfill.py — KIS 분봉 다종목 백필 (재개 가능)
|-- data/                 # raw/, processed/ (parquet, 월별 파티션) — git 미포함
`-- docs/                 # 이 문서
```

---

## 5. Observation / Feature Contract

`FeatureEngineer` (env/src/data/feature_engineer.py)가 관측 feature의 단일 진실이다.

- **`FEATURE_SCHEMA_VERSION = 2`** — feature 정의가 바뀌면 반드시 올린다.
- **`FEATURE_COLUMNS`** (순서 고정):
  `log_ret_1, log_ret_3, log_ret_12, realized_vol_12, relative_volume, vwap_dev,
  trend_strength_30m, macd_hist_30m, vol_regime_30m`
- 파이프라인 순서(불변): 결손일 drop → 거래일별 MinuteTradingValue diff → 종가
  동시호가(15:30) fold-in → 5분 resample → 5분 micro → 30분 context → warm-up dropna
- `transform()`은 **순수 함수** (self·입력 미변경). 같은 입력 → 같은 출력.
- env는 여기에 portfolio state(current position, `units_held_frac`, unrealized pnl,
  holding duration)를 붙여 observation을 완성한다.

**파리티 원칙 (중요):** 실시간 서빙에서도 이 배치 `transform()`을 그대로 재사용한다
(5분마다 최근 N일치 1분봉 재변환). 실시간용 incremental feature 코드를 별도로 만들지
않는다 — train/serve skew를 구조적으로 차단하기 위함이다.

**Leakage 규칙:** 상위 timeframe feature는 "완료된 bar만" 사용. 정규화는 causal
rolling(`center=False`). env/tests에 no-future-leakage 불변식 테스트가 있으며, env나
feature 변경 시 반드시 통과해야 한다.

