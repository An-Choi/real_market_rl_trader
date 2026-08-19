# Schema-v4 feature audit — 2026-08-18

## Run

- Five KRX symbols, feature data from 2025-06-23 through 2026-08-18
- Three expanding chronological folds
- 60-day validation split into three 20-day windows; every fold contains bull
  and bear validation windows
- 20-day test windows: bull, bear, then sideways
- 300,000 MaskablePPO steps per fold, one seed per fold
- CPU training (`rl-trader-py310` has a CPU-only PyTorch build)

Output: `runs/walk_forward/v4-feature-audit-20260818`

## Result

| Fold | Test regime | Mean return | Median | Worst symbol | Mean MDD | Revised validation |
|---|---|---:|---:|---:|---:|---|
| 1 | bull | -2.47% | -1.42% | -5.76% | -2.71% | reject: negative robust score |
| 2 | bear | -0.91% | +2.10% | -11.84% | -15.59% | reject: validation worst -5.81% |
| 3 | sideways | 0.00% | 0.00% | 0.00% | 0.00% | reject: 100% hold |

Across-fold mean return is -1.13%; no fold is positive. The model is not a
deployment candidate. A cash-only checkpoint now fails validation through
`maximum_hold_action_rate: 0.995`.

## Feature use

Mean permutation probability shift over the three validation folds:

| Feature | Mean shift | Finding |
|---|---:|---|
| `macd_hist_30m` | 0.03919 | Highest reliance, dominated the risky fold 2, redundant with trend/gap |
| `realized_vol_12` | 0.02868 | Strong only in fold 2; unstable reliance |
| `gap_open` | 0.02788 | New feature is used, but its return direction changes by period |
| `trend_strength_30m` | 0.02450 | Useful context, correlated 0.66-0.69 with MACD |
| `vol_regime_30m` | 0.02192 | Context feature, unstable magnitude |
| `log_ret_12` | 0.01689 | Moderate use; correlated with VWAP deviation |
| `log_ret_1` | 0.01543 | Moderate/low use |
| `log_ret_3` | 0.01275 | Low middle-horizon input, redundant with 1-bar return |
| `relative_volume` | 0.00829 | Weak and overlaps with the time-of-day-adjusted replacement |
| `relative_volume_tod` | 0.00588 | New feature is consistently weak but better specified than raw volume |
| `vwap_dev` | 0.00448 | Policy barely uses it despite a stable negative 12-bar association later |

`liquidity_pressure` has only about 0.0042 average probability shift. It is
always populated, but the environment currently has zero slippage and does not
derive friction from `Adv20`; therefore the policy has little reward incentive
to use it.

Portfolio state is also unstable. Folds 1 and 3 are effectively cash-only.
Fold 2 holds positions on roughly 42% of validation observations, yet
`unrealized_pnl_norm` and `holding_duration_norm` still have almost zero policy
sensitivity.

## Recommended changes

Do not expand the indicator set yet. Run controlled ablations in this order:

1. Remove legacy `relative_volume`; retain `relative_volume_tod` as the single
   volume anomaly input.
2. Test removal of `macd_hist_30m`. It is highly relied on but redundant and its
   largest reliance occurs in the fold with the worst drawdown. Keep
   `trend_strength_30m` as the simpler trend input during this test.
3. Test removal of `log_ret_3`; retain the short (`log_ret_1`) and hourly
   (`log_ret_12`) horizons.
4. Replace raw `liquidity_pressure` with an observable expected round-trip cost,
   or make `Adv20` affect the environment slippage model. A feature disconnected
   from reward friction is unlikely to become useful.
5. Keep `gap_open` for the next ablation because the policy demonstrably uses
   it, but require multi-seed stability before promotion.
6. Add benchmark-relative return and market volatility only after the reduced
   feature baseline is measured.

Feature changes alone will not fix the cash-policy collapse. Reward/opportunity
cost and checkpoint selection need separate ablations; forcing trades is not an
acceptable substitute for predictive signal.

Machine-readable reports:

- `runs/feature_reports/v4-fold-01-validation.json`
- `runs/feature_reports/v4-fold-02-validation.json`
- `runs/feature_reports/v4-fold-03-validation.json`
