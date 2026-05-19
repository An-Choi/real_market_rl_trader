# Project Architecture

```text
real_market_rl_trader/
|-- env/              # Market data, friction, execution, and Gym envs
|-- agent/            # RL models and policy/baseline implementations
|-- experiments/      # Train, backtest, and paper-trading entrypoints
|-- evaluation/       # Metrics and sim-to-live analysis
|-- data/             # Raw and processed market data
|-- notebooks/        # Exploration notebooks
`-- docs/             # Project notes
```

The next implementation target is `env/src/env/hybrid_market_env.py`, which
will combine real market returns with virtual agent impact.
