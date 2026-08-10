# Walk-forward evaluation report (2026-08-10)

## Scope

- Model: MaskablePPO, five symbols, 1-minute bars
- Device: CUDA
- Training: 300,000 configured timesteps per fold (rollout-aligned execution can reach 307,200)
- Validation: every 25,000 timesteps
- Checkpoint score: `median_return + 0.5 * min(worst_return, 0) + 0.5 * mean_max_drawdown`
- Data separation: expanding training window, 20 trading-day validation and test windows, with a five trading-day purge between them
- Seed: 42 for Fold 1 and 43 for Fold 2
- Fold 3 was intentionally stopped before completing its first training rollout at the user's request.

## Implemented changes

- Prefer CUDA automatically when a GPU is available.
- Add expanding walk-forward training and future-only fold evaluation.
- Select checkpoints using median return, worst return, and maximum drawdown instead of mean return alone.
- Add cash, buy-and-hold, random, and moving-average crossover baselines.
- Add feature quality, forward correlation, redundancy, and policy permutation-sensitivity reports.
- Reduce TensorBoard custom metrics to performance, policy, reward, risk, trading friction, and speed signals.
- Keep generated datasets, model artifacts, logs, TensorBoard events, and credentials out of Git.

## Results

| Fold | Test regime | Test dates | Model mean | Market mean | Excess vs market | Mean max drawdown | Positive symbols | Worst symbol |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Bull | 2026-04-30 to 2026-06-01 | +6.52% | +69.76% | -63.24%p | -2.14% | 80% | -4.39% |
| 2 | Bear | 2026-06-02 to 2026-07-03 | -7.41% | -14.98% | +7.57%p | -19.42% | 20% | -30.28% |

Fold 1 selected the 307,200-timestep checkpoint. Its validation mean return was +3.45%, median return +2.92%, worst return +0.45%, and mean maximum drawdown -1.18%.

Fold 2 selected the 300,000-timestep checkpoint. Its validation mean return was +30.09%, median return +29.97%, worst return +12.73%, and mean maximum drawdown -9.39%. The immediately following bear test then lost 7.41% on average, showing that strong validation performance in one market regime did not transfer reliably to the next regime.

## Feature diagnostics

The current feature schema was not changed. The strongest policy-sensitivity signals in the existing validation report were `log_ret_1`, `trend_strength_30m`, `log_ret_12`, `vol_regime_30m`, `relative_volume`, and `realized_vol_12`.

Potentially redundant pairs included:

- `trend_strength_30m` and `macd_hist_30m`: Spearman 0.689
- `log_ret_12` and `vwap_dev`: Spearman 0.548
- `log_ret_1` and `log_ret_3`: Spearman 0.515

These are ablation candidates, not features to remove without another controlled run.

## Conclusion

The model is not ready for production. It participated too little in the bull market and still produced a material loss and drawdown in the bear market. The next experiment should make checkpoint validation span multiple regimes and compare candidates against both cash and market baselines. Feature ablations should then be run one group at a time before training a final deployment model.

## Verification

- Agent and environment tests: 343 passed
- Walk-forward plan generation: passed
- Real-data one-fold smoke test: passed
- Feature report smoke test: passed
- Compact TensorBoard logging smoke test: passed
