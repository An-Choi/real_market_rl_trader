# Observation Contract — RL 학습 ↔ 서빙

**기준:** `FEATURE_SCHEMA_VERSION = 2` / artifact format v2
**진실의 원천:** `env/src/data/feature_engineer.py`의 `FeatureEngineer.FEATURE_COLUMNS`·
`FEATURE_SCHEMA_VERSION`, `env/src/env/trading_env.py`의 `_get_observation()`.
이 문서는 코드를 미러링하며, 불일치 시 코드가 우선한다 (문서를 갱신할 것).

RL 학습과 서빙(FastAPI/Spring)이 지켜야 하는 계약. 이 계약이 깨지면
모델은 에러 없이 조용히 잘못된 action을 내므로, 변경은 반드시 versioning 규칙을 따른다.

## 1. Observation 벡터 (13차원, float32)

순서 고정: feature 9개 → portfolio state 4개.

| idx | 이름 | 구간 |
|---|---|---|
| 0 | `log_ret_1` | 5분 micro |
| 1 | `log_ret_3` | 5분 micro |
| 2 | `log_ret_12` | 5분 micro |
| 3 | `realized_vol_12` | 5분 micro |
| 4 | `relative_volume` | 5분 micro |
| 5 | `vwap_dev` | 5분 micro |
| 6 | `trend_strength_30m` | 30분 context |
| 7 | `macd_hist_30m` | 30분 context |
| 8 | `vol_regime_30m` | 30분 context |
| 9 | `units_held_frac` | portfolio state |
| 10 | `unrealized_pnl_norm` | portfolio state |
| 11 | `holding_duration_norm` | portfolio state |
| 12 | `tod_frac` | portfolio state |

feature(idx 0–8)는 `FeatureEngineer.transform()`이 계산한다. 서빙 경로도 **같은 코드를
그대로 재사용**한다(별도 재구현 금지 — train/serve parity 원칙).

## 2. Portfolio state 재현 공식 (idx 9–12)

서빙 쪽이 정확히 같은 값을 만들어야 하는 parity의 핵심.
`trading_env.py`의 `_get_observation()` 기준:

- 시간 grid: `n` = 당일 5분 bar 개수(종가 동시호가 fold-in 후 grid),
  `t` = 당일 내 0-based step
- price source: 해당 5분 bar의 `Close`
- `units_held_frac = units_held / max(max_units, 1)`
- `unrealized_pnl_norm = (held_market_value − cost_basis) / max(initial_cash, 1e-9)`
  - `held_market_value` = **실제 보유 주식 수 × 현재 bar Close** (`shares_held * price`).
    명목값(`units × initial_cash × unit_fraction`)으로 재구성하면 안 됨.
  - `cost_basis`는 반대로 **실제 체결가가 아닌 명목가 convention**:
    `units_held × initial_cash × unit_fraction`
- `holding_duration_norm = min(진입 후 경과 bar 수 / duration_horizon_bars, 1.0)` (미보유 시 0)
  - `duration_horizon_bars`는 artifact `env_params`의 고정 상수
    (학습 config의 `episode_days × nominal_bars_per_day`). runtime episode
    길이와 무관 — 서빙은 실거래 진입 후 경과 bar 수를 세서 clip만 하면 된다.
- `tod_frac = 당일 내 step / max(당일 bar 수 − 1, 1)` (당일 기준, episode가 여러 날이어도 매일 리셋)
- 분모의 `max(·, 1)`은 1-bar day(결손/부분장/fixture)에서도 코드와 계약이
  갈리지 않게 하기 위한 코드 그대로의 표기

계산에 필요한 파라미터(`unit_fraction`, `max_units`, `initial_cash`, `episode_days`,
`duration_horizon_bars`)는 artifact의 `env_params`에 기록된다. 서빙 쪽은 **artifact의
값**을 사용한다 (config 재정의 금지).

## 3. Action space

`Discrete(3)`:

| action | label | 의미 |
|---|---|---|
| 0 | `hold` | 아무것도 하지 않음 |
| 1 | `add_unit` | 1 Unit 추가 진입 (1 Unit = 자본 20%, 최대 5) |
| 2 | `clear` | 전체 포지션 청산 |

**Step 의미론 (artifact v2):** bar t에서 실행 → t+1로 전진해 평가·관측. B개 bar
episode는 B−1 step이고 마지막 bar는 valuation 전용이다. 강제청산은 없다 —
episode 끝은 `truncated=True`(continuing task)이며, 미청산 종료의 가상
청산비용은 평가(metrics) 전용 convention으로 서빙·reward와 무관하다.
날짜 경계 step의 수익(boundary portfolio return)은
`C_{d,first5m}/C_{d-1,last5m}` 가격 성분 + 경계 거래 friction의 합성이다.

## 4. Versioning·거부 규칙

- feature의 **이름/순서/계산**이 바뀌면 `FEATURE_SCHEMA_VERSION` bump 필수.
- 서버는 artifact의 `feature_schema_version`이 자신의 `FeatureEngineer` 버전과 다르면
  **기동 거부**.
- 서버는 artifact의 `action_space`가 기대 정의(discrete, n=3, labels 순서)와 다르면
  **기동 거부**.
- artifact 모듈(`models/artifact.py`)은 내부 일관성만 검증한다 — 기대치 비교는 서버 책임.
- artifact format v1(당일 기준 holding_duration_norm)은 v2 환경에서 실행 거부.
  `load_metadata`는 v1·v2 모두 파싱하지만 `load_artifact`는 v1을
  `allow_legacy_semantics=True` 없이는 거부한다.

## 5. Normalization 규약

artifact `metadata.json`의 `normalization` 필드:

- `null` — legacy artifact with no observation normalization.
- `{"type": "feature_standardization", "file": "feature_normalization.json"}` — current
  contract. Mean/std are fit on the training split only, frozen in the artifact,
  and applied to feature indices 0-8. Portfolio state indices 9-12 are unchanged.
- `{"type": "sb3_vecnormalize", "file": "vecnormalize.pkl"}` — 학습이 VecNormalize를
  사용한 경우. 서버는 해당 stats로 observation을 정규화한 뒤 모델에 넣는다
  (stats는 학습 시점에 freeze된 값 — 서빙 중 업데이트 금지).
  `file`은 artifact 디렉토리 안의 **단순 파일명**만 허용된다 (경로 구분자/`..`/절대경로 거부).
  참고: legacy `sb3_vecnormalize` artifacts are rejected by direct prediction.
  `feature_standardization` artifacts are restored automatically by
  `models.artifact.load_artifact()`.

## 6. metadata.json 필드 명세

```json
{
  "artifact_format_version": 2,
  "artifact_id": "ppo-fs2-20260709-153000",
  "created_at": "2026-07-09T06:30:00Z",
  "algo": "PPO",
  "policy": "MlpPolicy",
  "feature_schema_version": 2,
  "feature_columns": ["log_ret_1", "log_ret_3", "log_ret_12",
    "realized_vol_12", "relative_volume", "vwap_dev",
    "trend_strength_30m", "macd_hist_30m", "vol_regime_30m"],
  "portfolio_state_fields": ["units_held_frac", "unrealized_pnl_norm",
    "holding_duration_norm", "tod_frac"],
  "observation_dim": 13,
  "action_space": {"type": "discrete", "n": 3, "labels": ["hold", "add_unit", "clear"]},
  "normalization": {"type": "feature_standardization", "file": "feature_normalization.json"},
  "train_git_sha": "ef4b63a",
  "train_data": {"symbols": ["005930"], "start": "2025-05-22", "end": "2026-06-30"},
  "env_params": {"unit_fraction": 0.2, "max_units": 5, "initial_cash": 100000000,
    "episode_days": 20, "duration_horizon_bars": 1280},
  "training_params": {"total_timesteps": 300000, "seed": 42, "ppo": {}}
}
```

(예시의 수치 값들은 illustrative — 실제 값은 학습 시점의 config/데이터 구간을 따른다.)

| 필드 | 의미 |
|---|---|
| `artifact_format_version` | 이 포맷 자체의 버전. 로더는 아는 버전만 수용 |
| `feature_schema_version` | 서버가 자기 FeatureEngineer 버전과 비교, 불일치 시 기동 거부 |
| `feature_columns` / `portfolio_state_fields` | observation 앞/뒤 구간, 순서 그대로 |
| `observation_dim` | 위 둘 길이의 합 (저장/로드 시 검증) |
| `normalization` | §5 참조 |
| `train_git_sha` | 학습 코드 커밋 (uncommitted 변경 시 `-dirty`, 조회 실패 시 `unknown`) |
| `train_data` | 학습 데이터 구간 추적 |
| `env_params` | §2의 portfolio state 계산 파라미터 |

## 7. 학습 쪽 사용 예시

학습 종료 후:

```python
from datetime import datetime, timezone

from data.feature_engineer import FeatureEngineer
from models.artifact import (
    ArtifactMetadata, current_git_sha, make_artifact_id, save_artifact,
)

meta = ArtifactMetadata(
    artifact_format_version=2,
    artifact_id=make_artifact_id("PPO", FeatureEngineer.FEATURE_SCHEMA_VERSION),
    created_at=datetime.now(timezone.utc).isoformat(),
    algo="PPO",
    policy="MlpPolicy",
    feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
    feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
    portfolio_state_fields=["units_held_frac", "unrealized_pnl_norm",
                            "holding_duration_norm", "tod_frac"],
    observation_dim=len(FeatureEngineer.FEATURE_COLUMNS) + 4,
    action_space={"type": "discrete", "n": 3, "labels": ["hold", "add_unit", "clear"]},
    normalization=None,  # VecNormalize 사용 시 §5의 블록 + vecnormalize_path 전달
    train_git_sha=current_git_sha(),
    train_data={"symbols": ["005930"], "start": "2025-05-22", "end": "2026-06-30"},
    env_params={"unit_fraction": 0.2, "max_units": 5, "initial_cash": 100_000_000,
                "episode_days": 20, "duration_horizon_bars": 1280},
)
save_artifact(agent, meta, "artifacts/")
```
