# Local research: action representation and temporal observations

This is an offline, retrospective study. The existing trading/serving v5 contract
is preserved. New models are saved as **research-only** bundles, not approved
trading models. No broker/order API is called by the ablation runner.

## Research question

Under identical price paths, costs and transition budgets, how do target-weight
actions and short market histories change PPO learning, exposure and turnover?

| Condition | Action | Market history |
|---|---|---|
| A | Hold / Add / Reduce / Clear | current feature vector |
| B | target 0/20/40/60/80/100% | current feature vector |
| C | Hold / Add / Reduce / Clear | last 12 feature vectors |
| D | target 0/20/40/60/80/100% | last 12 feature vectors |

A is a new research baseline, **not** an unchanged historical model (A0). The
same common account fields, quantity-planning and settlement rules are used in
all four conditions. Original source/legacy artifact handling remains separate.

## Explicit assumptions

- Quantities are planned with the observable Close. An unseen future discount
  cannot increase the requested number of shares. At the next execution price,
  a buy may be partially filled down to available cash including friction. This
  is an idealized cash-limited execution rule, not a full broker simulator.
- Missing ExecPrice means no fill and no cost. Action masks do not inspect the
  future execution price. Missing prints and raw-quality exclusions are recorded.
- Incremental actions retain the LIFO lot convention. Target weights use average
  cost on partial reductions; they never fabricate unit counts. The common
  `unit_occupancy` field is zero/unused for target-weight environments.
- Target weight uses current marked equity. Long-only, no leverage. 100% means
  the largest cash-affordable fill of the planned quantity, including costs.
- A 1 percentage-point rebalance tolerance avoids tiny repetitive adjustments;
  a zero target always clears. This tolerance is fixed before evaluation.
- All six bounded target intents are valid, so their masks are all true. Cash
  limits are enforced by execution. MaskablePPO is kept for a common learner;
  target actions are not given future-return-based masks.
- Market history resets at every trading day or intraday gap >5 minutes. Missing
  history is zero-padded **after normalization**, with validity bits. Episode
  starts may read earlier causal feature context within the same supplied split
  and day. No cross-day or cross-split feature context is preloaded in v1.
- Normalization is fitted on the train segment only. Features, including the
  preexisting rolling indicators, are constructed causally before splitting.
- Final evaluation settles at the last observed Close plus sell friction,
  without using an execution print beyond the evaluation boundary.
- The default zero penalty settings retain cost-deducted log-return reward.
  Any enabled inventory penalty uses actual exposure identically across actions.
- MLP widths are chosen deterministically to approximately match a 40,000
  actor+critic parameter budget, accounting for different observation/action sizes.

## Data and evaluation

The existing default 3-fold/60-day-validation plan is not rewritten. Its coverage
failures remain documented. This **separate screening protocol** uses one fold,
20 validation days, 20 reserved test days, and five-day purges. Its dates are
defined from the daily-data calendar proxy, not returns. This proxy is not an
independent official exchange calendar. Screening is retrospective: these data
have already been explored, so they are not a pristine final holdout.

Raw days are audited with `require_complete_session=True` after deterministic
duplicate sanitation. Per-segment/per-symbol day coverage must be at least 95%.
The runner refuses to train if a segment fails. Original raw files are not
changed. Collect missing data separately rather than relaxing this threshold.

Checkpoint selection uses validation only, including a step-zero control:
`median return + .5*min(worst return,0) + .5*mean bar drawdown`.
This is a research score, not the existing production approval gate. A selected
step-zero policy is possible and must be reported. Test evaluation is opt-in via
`--evaluate-test`, after freezing the study settings. Do not tune on that result.

Cash, near-full buy-and-hold (one .995-initial-capital buy) and a fixed mean-
reversion rule are evaluated under the same costs and calendar. The rule enters
when standardized log_ret_1 < -1, clears when >0, and otherwise holds. These
baselines are controls, not tuned investment recommendations.

## Commands (from the Python repository)

```bash
source ../env.sh
python scripts/research_data_audit.py
python experiments/run_ablation.py --output runs/ablation/plan --plan-only
python experiments/run_ablation.py --synthetic --output runs/ablation/synthetic \
  --seeds 42 --timesteps 512 --n-steps 128 --eval-freq 256
python experiments/run_ablation.py --output runs/ablation/screening \
  --seeds 42,43,44 --timesteps 50000
python experiments/summarize_ablation.py runs/ablation/screening
```

Existing output directories are rejected to preserve earlier results. A full
3-fold run can be requested with explicit CLI arguments, but does not bypass
quality checks. A functioning CUDA-enabled PyTorch and GPU driver are required
for `--device cuda`; the current workspace uses CPU PyTorch.

Outputs: study/data/code manifest, validation checkpoints, PPO loss/KL/entropy
CSV, per-bar traces, action probabilities/masks, flat fraction, exposure, costs,
turnover, drawdown, seed tables, plots, model/normalization and round-trip checks.
File checksums detect accidental bundle changes; they do not authenticate an
untrusted model. Load only local/team-created bundles.

## Minimal local security

Keep `.env`, KIS credentials and token files out of Git and shared logs. Token
cache files are private (0600) and atomically replaced. The Python config repr
does not print keys. Existing serving binds to loopback; Java and Compose DB
defaults also bind to loopback. Python `KIS_ENV=demo` is the example default;
Java's default profile is dry-run and `live` enables broker interaction (its
configured default endpoint is the demo broker). Offline study commands above
need no keys and place no orders. External deployment security is out of scope.
