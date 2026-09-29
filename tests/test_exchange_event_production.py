"""E32実行条件と本番CLIの一致、および既存構成の不変性を検査する。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys
from unittest.mock import Mock

import pytest

from src import production_config as config
from src.exchange_event_cli import parse_exchange_event_args

ROOT = Path(__file__).resolve().parents[1]
RENDER = "scripts.visualize_advantage_overlay"
REPLAY = "scripts.replay_exchange_event_20260926"


def fixed_flags() -> list[str]:
    """実際の検収結果に保存された引数をCLI表記へ変換する。"""
    done = json.loads((ROOT / "logs/e32/on/review/DONE.json").read_text())
    flags = ["--exchange-event-update", "--exchange-event-model-dir", done["model_dir"]]
    if done["live_count"]:
        flags.append("--exchange-event-live-count")
    for name, enabled in done["options"].items():
        assert enabled is True
        prefix = "--exchange-event-" if name == "death_guard" else "--"
        flags.append(prefix + name.replace("_", "-"))
    return flags


def test_fixed_execution_conditions() -> None:
    """保存引数だけでなくランチャーが辿るE32実行設定とも照合する。"""
    from scripts import run_e32

    launcher = (ROOT / "scripts/_launch_e32.sh").read_text()
    assert "-m scripts.run_e32" in launcher
    paths = [ROOT / "logs/e32/on" / name / "DONE.json" for name in ("review", "zenchi")]
    paths.extend((ROOT / "logs/e32/on/renders").glob("*/on/DONE.json"))
    for path in paths:
        done = json.loads(path.read_text())
        assert done["options"] == run_e32.OPTIONS
        assert done["model_dir"] == "models/exchange_event_v3"
        assert done["live_count"] is True
    assert shlex.split(config.exchange_event_flags()) == fixed_flags()
    assert len(set(fixed_flags())) == len(fixed_flags())


@pytest.mark.parametrize("entry", config.EXCHANGE_EVENT_ADOPTED)
def test_provenance(entry: config.AdoptedFlag) -> None:
    """全採用項目に日付と検収根拠が残る。"""
    assert entry.adopted == "2026-09-29"
    for evidence in (".507977", "91.9717%", "7,664/8,333", "1/35", "E32b",
                     "リークなし", "E34c", "114,146/114,146", "user承認"):
        assert evidence in entry.reason
    assert entry.flag in config.describe()


class Parsed(Exception):
    """重い描画・動画読み取り前に実効引数を取り出す。"""


def cli_options(module: str, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> dict:
    """実際のmainを引数解析完了まで走らせる。"""
    import importlib
    target = importlib.import_module(module)
    captured = {}

    def capture(parser: argparse.ArgumentParser) -> argparse.Namespace:
        captured.update(vars(parse_exchange_event_args(parser)))
        raise Parsed

    monkeypatch.setattr(sys, "argv", [module, *argv])
    monkeypatch.setattr("src.exchange_event_cli.parse_exchange_event_args", capture)
    if module == REPLAY:
        monkeypatch.setattr(target, "parse_exchange_event_args", capture)
    with pytest.raises(Parsed):
        target.main()
    return captured


@pytest.mark.parametrize("module", [RENDER, REPLAY])
def test_cli_effective_arguments(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """両CLIで一括指定と固定検収の全実効引数が一致し、未採用はOFF。"""
    required = ["input.jsonl.gz", "--out", "unused"] if module == REPLAY else []
    explicit = cli_options(module, required + fixed_flags(), monkeypatch)
    production = cli_options(module, required + ["--production-exchange-event"], monkeypatch)
    assert production.pop("production_exchange_event") is True
    assert explicit.pop("production_exchange_event") is False
    assert production == explicit
    for name in ("prefire_stage_timeout", "prefire_stage_timeout_only", "prefire_origin_guard"):
        assert production[name] is False


@pytest.mark.parametrize("module", [RENDER, REPLAY])
def test_cli_reads_registry_at_runtime(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """本番関数を実行時に読むことと、指定なしでは読まないことを確認する。"""
    required = ["input.jsonl.gz", "--out", "unused"] if module == REPLAY else []
    getter = Mock(return_value="--exchange-event-model-dir models/runtime-test")
    monkeypatch.setattr(config, "exchange_event_flags", getter)
    cli_options(module, required, monkeypatch)
    getter.assert_not_called()
    parsed = cli_options(module, required + ["--production-exchange-event"], monkeypatch)
    getter.assert_called_once_with()
    key = "model_dir" if module == REPLAY else "exchange_event_model_dir"
    assert parsed[key] == Path("models/runtime-test")


def test_existing_configuration_unchanged() -> None:
    """採用前コミットの既存バケット・取得関数と構文が完全一致する。"""
    previous = json.loads((ROOT / "tests/fixtures/production_config_before_p1.json").read_text())
    current = ast.parse((ROOT / "src/production_config.py").read_text(encoding="utf-8"))
    definitions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
    definitions.update({node.target.id: node for node in current.body if isinstance(node, ast.AnnAssign)})
    for name, expected in previous["sha256_ast"].items():
        assert hashlib.sha256(ast.dump(definitions[name]).encode()).hexdigest() == expected, name
