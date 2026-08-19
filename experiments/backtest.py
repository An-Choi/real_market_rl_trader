"""Backtest a saved PPO artifact or a simple baseline agent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader, resolve_project_path, resolve_symbols
from models.artifact import check_env_compatibility, load_artifact, load_metadata
from models.walk_forward import SplitBoundaries, compute_split_boundaries, split_by_trading_day
from policies import SUPPORTED_BASELINES
from policies.evaluation import (
    build_backtest_environment,
    compare_baselines,
    compare_baselines_multi_seed,
    evaluate_baseline,
    evaluate_baseline_multi_seed,
    run_agent_backtest,
    summarize_seed_runs,
)
from utils.config_loader import load_config
from utils.logger import setup_logger


def resolve_backtest_symbols(*, config, meta, cli_symbol, cli_symbols) -> list[str]:
    """우선순위: CLI > artifact 학습 종목 > config."""
    if cli_symbol is not None or cli_symbols is not None:
        return resolve_symbols(config=config, cli_symbol=cli_symbol, cli_symbols=cli_symbols)
    if meta is not None:
        symbols = (meta.train_data or {}).get("symbols")
        if symbols:
            return list(symbols)
        print(
            "WARNING: artifact metadata has no train_data.symbols; "
            "falling back to config symbols",
            file=sys.stderr,
        )
    return resolve_symbols(config=config)


def resolve_boundaries(*, meta, data_by_symbol, config) -> SplitBoundaries | None:
    """경계 출처 규칙: v4 metadata 재사용 > 다종목 공유 계산 > 단일 종목 비율 fallback."""
    if meta is not None:
        if meta.artifact_format_version >= 4:
            return SplitBoundaries.from_metadata(meta.train_data["split_boundaries"])
        print(
            f"WARNING: format v{meta.artifact_format_version} artifact has no "
            "split boundaries; using legacy ratio split",
            file=sys.stderr,
        )
        return None
    if len(data_by_symbol) > 1:
        purge_days = int(config.get("data", {}).get("split", {}).get("purge_days", 0))
        return compute_split_boundaries(data_by_symbol, purge_days=purge_days)
    return None


def ensure_oos_artifact(meta) -> None:
    """v4 artifact의 trained_split이 train이 아니면 OOS 결과가 아니므로 거부."""
    if meta is None or meta.artifact_format_version < 4:
        return
    trained_split = meta.train_data.get("trained_split")
    if trained_split != "train":
        raise SystemExit(
            f"artifact was trained on split {trained_split!r}; its backtest would not be "
            "out-of-sample. Re-train with --split train (or use the evaluation API directly)."
        )


def run_artifact_backtests(
    *,
    artifact_path: Path,
    meta,
    data_by_symbol: dict,
    config: dict,
    boundaries,
    split: str,
    max_steps,
    seed: int,
    seeds: "list[int] | None" = None,
) -> dict[str, dict]:
    """모델을 첫 종목 env로 1회만 로드하고 전 종목을 순회 평가한다."""
    agent = None
    payloads: dict[str, dict] = {}
    for symbol, all_data in data_by_symbol.items():
        featured = split_by_trading_day(all_data, split=split, boundaries=boundaries)
        environment = build_backtest_environment(
            featured, config, feature_columns=meta.feature_columns
        )
        if agent is None:
            agent, _ = load_artifact(artifact_path, env=environment)  # 로드 + env 계약 검증
        else:
            check_env_compatibility(meta, environment)                # 이후 env는 검증만
        if seeds is None:
            payloads[symbol] = run_agent_backtest(
                agent=agent,
                agent_name=meta.artifact_id,
                environment=environment,
                max_steps=max_steps,
                seed=seed,
            )
        else:
            runs = [
                run_agent_backtest(
                    agent=agent,
                    agent_name=meta.artifact_id,
                    environment=environment,
                    max_steps=max_steps,
                    seed=s,
                )
                for s in seeds
            ]
            payloads[symbol] = {"agent": meta.artifact_id, **summarize_seed_runs(runs)}
    return payloads


def run_compare_backtests(
    *,
    artifact_path,
    meta,
    data_by_symbol: dict,
    config: dict,
    boundaries,
    split: str,
    max_steps,
    seed: int,
    seeds=None,
) -> dict[str, list]:
    """종목별 baseline suite + (artifact 있으면) 1회 로드된 artifact 요약을 append."""
    artifact_payloads = None
    if artifact_path is not None:
        artifact_payloads = run_artifact_backtests(
            artifact_path=artifact_path, meta=meta, data_by_symbol=data_by_symbol,
            config=config, boundaries=boundaries, split=split,
            max_steps=max_steps, seed=seed, seeds=seeds,
        )
    out: dict[str, list] = {}
    for symbol, all_data in data_by_symbol.items():
        featured = split_by_trading_day(all_data, split=split, boundaries=boundaries)
        if seeds is None:
            summaries = compare_baselines(
                featured_data=featured, config=config, max_steps=max_steps,
                seed=seed, artifact_path=None,
            )
        else:
            summaries = compare_baselines_multi_seed(
                featured_data=featured, config=config, max_steps=max_steps,
                seeds=seeds, artifact_path=None,
            )
        if artifact_payloads is not None:
            summaries.append(artifact_payloads[symbol])
        out[symbol] = summaries
    return out


def resolve_output_dir(explicit: "Path | None", *, base: Path, run_id: str) -> Path:
    """--output-dir 명시 시 생성·검증, 미지정 시 자동 디렉터리 (충돌 시 suffix)."""
    if explicit is not None:
        if explicit.exists() and not explicit.is_dir():
            raise SystemExit(f"--output-dir points to a file: {explicit}")
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = base / f"{run_id}-{ts}"
    counter = 2
    while candidate.exists():
        candidate = base / f"{run_id}-{ts}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def _fmt_metric(metrics, key, width=10):
    value = metrics.get(key) if isinstance(metrics, dict) else None
    return f"{value:>{width}.4f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"


def _print_summary_table(per_symbol_payload: dict[str, dict], days_by_symbol: "dict[str, int] | None" = None) -> None:
    days_by_symbol = days_by_symbol or {}
    lines = [f"{'symbol':<8} {'days':>5} {'agent':<20} {'return':>10} {'market':>10} {'mdd':>8}"]
    for symbol, payload in per_symbol_payload.items():
        days = days_by_symbol.get(symbol, "-")
        # compare 모드 payload는 {"results": [summary, ...]} — 한 종목당 agent별 1행
        entries = payload["results"] if "results" in payload else [payload]
        for entry in entries:
            metrics = entry.get("metrics") or entry.get("mean_metrics") or {}
            lines.append(
                f"{symbol:<8} {days!s:>5} {str(entry.get('agent', '-')):<20} "
                f"{_fmt_metric(metrics, 'total_return')} {_fmt_metric(metrics, 'market_return')} "
                f"{_fmt_metric(metrics, 'max_drawdown', 8)}"
            )
    print("\n".join(lines), file=sys.stderr)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest PPO artifact or baseline")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols, e.g. 005930,000660")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--baseline", choices=SUPPORTED_BASELINES, default=None)
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for robustness evaluation, e.g. 1,2,3")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="test")
    parser.add_argument(
        "--episode-days",
        type=_positive_int,
        default=None,
        help="Trading days per episode; overrides environment.episode_days.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    seeds = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not seeds:
        raise SystemExit("--seeds requires at least one integer")
    return seeds


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    if args.episode_days is not None:
        config["environment"]["episode_days"] = args.episode_days

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seeds = _parse_seeds(args.seeds)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = config["backtest"].get("max_steps")

    artifact_path = resolve_project_path(PROJECT_ROOT, args.artifact) if args.artifact else None
    meta = None
    if artifact_path is not None:
        meta = load_metadata(artifact_path)
        ensure_oos_artifact(meta)

    symbols = resolve_backtest_symbols(
        config=config, meta=meta, cli_symbol=args.symbol, cli_symbols=args.symbols,
    )

    data_loader = make_data_loader(
        project_root=PROJECT_ROOT,
        config=config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    data_by_symbol = {
        symbol: load_feature_data(
            symbol=symbol,
            data_loader=data_loader,
            force_rebuild=args.force_rebuild,
        )
        for symbol in symbols
    }
    boundaries = resolve_boundaries(meta=meta, data_by_symbol=data_by_symbol, config=config)
    logger.info(
        "Evaluation episode length: %d trading day(s)",
        int(config["environment"].get("episode_days", 1)),
    )

    # spec §12: 리포트에 종목별 evaluated split의 실제 거래일수를 표기 — --max-steps
    # 캡과 무관하게 split 크기 자체를 보여준다 (stdout JSON 페이로드는 변경하지 않음).
    days_by_symbol: dict[str, int] = {}
    for symbol in symbols:
        split_df = split_by_trading_day(data_by_symbol[symbol], split=args.split, boundaries=boundaries)
        days_by_symbol[symbol] = int(pd.to_datetime(split_df["Timestamp"]).dt.date.nunique())

    per_symbol_payload: dict[str, dict] = {}

    if args.compare_baselines:
        logger.info("Backtesting baseline suite on %s (%s split)", ", ".join(symbols), args.split)
        compare_out = run_compare_backtests(
            artifact_path=artifact_path,
            meta=meta,
            data_by_symbol=data_by_symbol,
            config=config,
            boundaries=boundaries,
            split=args.split,
            max_steps=max_steps,
            seed=seed,
            seeds=seeds,
        )
        per_symbol_payload = {
            symbol: {"symbol": symbol, "split": args.split, "seeds": seeds, "results": summaries}
            for symbol, summaries in compare_out.items()
        }
        run_id = meta.artifact_id if meta is not None else "baseline-suite"
    elif artifact_path is not None:
        logger.info("Backtesting artifact on %s (%s split)", ", ".join(symbols), args.split)
        artifact_out = run_artifact_backtests(
            artifact_path=artifact_path,
            meta=meta,
            data_by_symbol=data_by_symbol,
            config=config,
            boundaries=boundaries,
            split=args.split,
            max_steps=max_steps,
            seed=seed,
            seeds=seeds,
        )
        per_symbol_payload = {
            symbol: {**run, "symbol": symbol, "split": args.split, "seeds": seeds}
            for symbol, run in artifact_out.items()
        }
        run_id = meta.artifact_id
    else:
        baseline_name = args.baseline or config["agent"].get("baseline_name", "buy_and_hold")
        logger.info("Backtesting %s on %s (%s split)", baseline_name, ", ".join(symbols), args.split)
        for symbol in symbols:
            featured = split_by_trading_day(data_by_symbol[symbol], split=args.split, boundaries=boundaries)
            if seeds is None:
                summary = evaluate_baseline(
                    baseline_name=baseline_name,
                    featured_data=featured,
                    config=config,
                    max_steps=max_steps,
                    seed=seed,
                )
            else:
                summary = evaluate_baseline_multi_seed(
                    baseline_name=baseline_name,
                    featured_data=featured,
                    config=config,
                    max_steps=max_steps,
                    seeds=seeds,
                )
            payload = {"agent": summary["agent"], "symbol": symbol, "split": args.split, "seeds": seeds}
            if seeds is None:
                payload["metrics"] = summary["metrics"]
            else:
                payload.update({
                    "runs": summary["runs"],
                    "mean_metrics": summary["mean_metrics"],
                    "std_metrics": summary["std_metrics"],
                })
            per_symbol_payload[symbol] = payload
        run_id = f"baseline-{baseline_name}"

    reused = args.output_dir is not None and args.output_dir.is_dir()
    output_dir = resolve_output_dir(args.output_dir, base=PROJECT_ROOT / "runs" / "backtest", run_id=run_id)
    if reused:
        logger.info("reusing existing output dir (files may be overwritten): %s", output_dir)

    for symbol, payload in per_symbol_payload.items():
        (output_dir / f"{symbol}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    _print_summary_table(per_symbol_payload, days_by_symbol)

    if len(symbols) == 1:
        print(json.dumps(per_symbol_payload[symbols[0]], indent=2, sort_keys=True))
    else:
        print(json.dumps(
            {
                "symbols": symbols,
                "split": args.split,
                "output_dir": str(output_dir),
                "per_symbol": per_symbol_payload,
            },
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
