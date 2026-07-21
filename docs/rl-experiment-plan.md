# RL Experiment Plan

## Scope

This document defines the RL-owner workflow for the current Phase 1 baseline.
The goal is not to claim profitable trading, but to produce a reproducible PPO
policy-learning and evaluation loop for adaptive single-symbol position control.

## Problem Definition

- Task: sequential position lifecycle control for one symbol.
- Policy: single shared MaskablePPO policy.
- Observation: multi-timeframe features plus portfolio state.
- Action space:
  - `0..5`: target stock allocation `0/20/40/60/80/100%`
- Position sizing: friction-aware target allocation based on current portfolio value.
  Repeating the selected target is a no-op hold.
- Episode: configurable contiguous trading-day window (`environment.episode_days`).
  The current default uses 20 trading days, carries positions overnight, and
  treats the artificial window boundary as truncation. Evaluation subtracts a
  virtual liquidation cost at the end of the selected split.
- Invalid actions: none; all six target allocations are always valid.

## Current Workflow

Prepare data:

```bash
conda activate rl-trader-py310
python scripts/backfill.py --symbols 005930 --skip-daily
```

Train MaskablePPO on the train split and select the saved parameters on the
full validation split:

```bash
python experiments/train.py --symbol 005930 --split train --total-timesteps 300000
```

Choose the episode length per run without editing YAML:

```bash
python experiments/train.py --symbol 005930 --split train --episode-days 1
python experiments/train.py --symbol 005930 --split train --episode-days 20
```

Use the same `--episode-days` value when backtesting the saved artifact.

Watch PPO training in TensorBoard:

```bash
tensorboard --logdir runs/tensorboard
```

TensorBoard logging is enabled by default in `env/configs/config.yaml`. Override it
per run with `--tensorboard-log-dir <dir>`, `--tensorboard-log-name <name>`, or
disable it with `--no-tensorboard`.

Validation selection is enabled by default and runs every 25,000 steps plus at
step zero and training end. Disable only for diagnostics with `--no-validation`.

Custom training charts are grouped under:

- `returns/*`: current/completed episode return, compounded strategy return,
  compounded benchmark return, and benchmark-relative excess return.
- `portfolio/*`: mean/latest portfolio value and mean exposure.
- `actions/*`: Hold/Add/Clear rates for the rollout and the full run.
- `cost/*`: mean and summed transaction friction.
- `trading/*`: rollout and cumulative forced-clear counts.
- `reward/*`: shaped reward and each base, benchmark-relative, inventory,
  turnover, drawdown, and downside component.
- `daily/*`: one point per completed trading date for actual/benchmark/excess
  return, closing portfolio value, friction, exposure, action rates, forced
  clears, and reward components. `daily/date` stores the date as `YYYYMMDD`.

Backtest baselines on the test split:

```bash
python experiments/backtest.py --compare-baselines --split test
```

Compare a MaskablePPO artifact against baselines:

```bash
python experiments/backtest.py --compare-baselines --artifact artifacts/maskableppo-fs2-... --split test
```

Run multi-seed robustness:

```bash
python experiments/backtest.py --compare-baselines --artifact artifacts/maskableppo-fs2-... --split test --seeds 1,2,3
```

## Splits

Walk-forward splits are date-based, not row-based, so a trading day is never
split across train/validation/test.

Default split ratio:

- Train: 70% of trading days
- Validation: 15% of trading days
- Test: remaining 15% of trading days

Default CLI behavior:

- `train.py`: `--split train`
- `backtest.py`: `--split test`

Validation chooses the saved training checkpoint and experiment settings. Test
must remain untouched until the experiment is frozen and is used only for final
out-of-sample comparison.

Backtests run the entire selected split as one continuous episode. Cash and
positions carry across all date boundaries, and final metrics include virtual
liquidation friction for any open position.

## Baselines

Every PPO artifact must be compared against:

- `buy_and_hold`
- `random`
- `ma_crossover`

The comparison command is:

```bash
python experiments/backtest.py --compare-baselines --artifact <artifact_dir>
```

## Metrics

Backtest summaries currently include:

- `total_return`
- `sharpe_ratio`
- `max_drawdown`
- `final_portfolio_value`
- `win_rate`
- `trade_count`
- `number_of_trades` compatibility alias
- `turnover`
- `evaluated_days`
- `overnight_hold_rate`, `open_at_end`, `terminal_liquidation_cost`
- `target_0pct_action_rate`부터 `target_100pct_action_rate`까지 6개 목표 비중 행동률

For multi-seed runs, output includes:

- per-seed `runs`
- `mean_metrics`
- `std_metrics`

## Reward Ablation

Default reward:

```text
100 * log(portfolio_value_t / portfolio_value_t-1)
```

All terms are logged separately for ablation. Inventory and turnover penalties
remain available but default to zero because trading friction is already deducted
from cash.

```text
reward = reward_scale * (
  base_return
  + benchmark_relative_reward
  - inventory_penalty
  - turnover_penalty
  - drawdown_penalty
  - downside_penalty
)
```

Config fields:

```yaml
environment:
  risk_penalty_rate: 0.0
  turnover_penalty_rate: 0.0
  drawdown_penalty_rate: 0.0
  downside_penalty_rate: 0.0
  benchmark_relative_rate: 0.0
  reward_scale: 100.0
  reward_return_mode: log_return
```

Initial ablation order:

1. Scaled log return only.
2. Add downside penalty.
3. Add incremental drawdown penalty.
4. Add benchmark-relative term.
5. Add inventory or turnover penalties only if validation behavior requires them.

Only keep a reward variant if it improves validation robustness without making
test turnover or max drawdown worse.

## PPO Baseline Discipline

Start simple:

- `MaskablePPO` with `MlpPolicy`
- invalid-action masking while flat or at maximum allocation
- training-split feature standardization with clipping; stats saved in each artifact
- deterministic full-validation checkpoint selection by terminal return
- no HRL
- no portfolio allocation RL
- no continuous action sizing
- no transformer policy

The first acceptable MaskablePPO result is not required to beat all baselines. It must:

- train without crashing
- save a loadable artifact
- run on validation/test splits
- produce actions in the expected action space
- report metrics against all baselines
- be reproducible across seeds

## Success Criteria

Minimum success:

- MaskablePPO artifact trains and loads.
- Backtest runs on validation and test.
- Baseline comparison prints JSON metrics.
- Multi-seed evaluation prints mean and standard deviation.

Research success:

- MaskablePPO improves at least one risk-adjusted metric on validation.
- MaskablePPO does not increase turnover excessively.
- MaskablePPO does not worsen max drawdown materially versus baselines.
- Behavior is explainable from action sequences and reward terms.

## Current Commands To Keep Green

```bash
conda activate rl-trader-py310
python -m pytest env/tests agent/tests -q
```

Current smoke coverage includes:

- synthetic PPO and MaskablePPO training with artifact roundtrip
- full-validation best-checkpoint selection
- backtest CLI on synthetic data
- baseline comparison
- PPO artifact comparison
- walk-forward split
- reward terms
- multi-seed aggregation
