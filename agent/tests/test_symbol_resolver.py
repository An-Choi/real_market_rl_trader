"""experiments.common.resolve_symbols 계약 테스트."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import resolve_symbols


def test_cli_symbols_take_priority_over_config():
    config = {"data": {"symbols": ["005930", "000660"]}}
    assert resolve_symbols(config=config, cli_symbols="034220, 066570") == ["034220", "066570"]


def test_cli_symbol_single_alias():
    config = {"data": {"symbols": ["005930", "000660"]}}
    assert resolve_symbols(config=config, cli_symbol="005930") == ["005930"]


def test_config_symbols_default():
    config = {"data": {"symbols": ["005930", "000660"]}}
    assert resolve_symbols(config=config) == ["005930", "000660"]


def test_legacy_config_symbol_promoted():
    config = {"data": {"symbol": "005930"}}
    assert resolve_symbols(config=config) == ["005930"]


def test_both_cli_args_rejected():
    config = {"data": {"symbols": ["005930"]}}
    with pytest.raises(SystemExit):
        resolve_symbols(config=config, cli_symbol="005930", cli_symbols="005930,000660")


def test_duplicates_rejected():
    config = {"data": {"symbols": ["005930"]}}
    with pytest.raises(SystemExit):
        resolve_symbols(config=config, cli_symbols="005930,005930")


def test_config_duplicates_rejected():
    config = {"data": {"symbols": ["005930", "005930"]}}
    with pytest.raises(SystemExit):
        resolve_symbols(config=config)


def test_empty_and_whitespace_rejected():
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {"symbols": []}})
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {"symbols": ["  "]}})
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {}})


def test_non_string_values_rejected():
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {"symbols": [5930]}})
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {"symbols": [None]}})
    with pytest.raises(SystemExit):
        resolve_symbols(config={"data": {"symbols": "005930"}})  # list가 아닌 str


def test_values_are_trimmed():
    config = {"data": {"symbols": [" 005930 "]}}
    assert resolve_symbols(config=config) == ["005930"]
