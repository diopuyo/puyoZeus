"""2026-10-01レビューの入力契約とキャッシュ退行を固定する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.exchange_hidden_row_probability import capped
from src.exchange_event_overlay import ExchangeEventOverlay
from tests.test_exchange_event_production import RENDER, REPLAY, cli_options

ROOT = Path(__file__).resolve().parents[1]
INVALID_CAPS = (0, -1, 1.5, "2", True, False)
GARBAGE_HEIGHT, INCOMING, HANDS = 10, 30, 1
QUEUE = (1, 2, 3, 4)
LOW_LIMIT, HIGH_LIMIT = 1, 1000


@pytest.mark.parametrize("cap", INVALID_CAPS)
def test_invalid_cap_is_rejected_before_evaluation(cap: object) -> None:
    """空候補・非空候補とoverlay公開APIを同じ契約で拒否する。"""
    for rows in ([], [((700,), 1.0)]):
        with pytest.raises(ValueError, match="hidden_scenario_cap"):
            capped(rows, cap)
    with pytest.raises(ValueError, match="hidden_scenario_cap"):
        ExchangeEventOverlay(None, None, None, None, hidden_scenario_cap=cap)


@pytest.mark.parametrize("cap", (None, 1, 256))
def test_valid_cap_preserves_single_candidate(cap: int | None) -> None:
    rows = [((700,), 1.0)]
    assert capped(rows, cap) == rows


@pytest.mark.parametrize("module", (RENDER, REPLAY))
@pytest.mark.parametrize("cap", ("0", "-1", "1.5"))
def test_cli_rejects_invalid_cap(module: str, cap: str, monkeypatch: pytest.MonkeyPatch) -> None:
    required = ["input.gz", "--out", "unused"] if module == REPLAY else []
    with pytest.raises(SystemExit) as exc:
        cli_options(module, required + ["--hidden-scenario-cap", cap], monkeypatch)
    assert exc.value.code == 2


def test_generate_positional_names_match_3802ab7() -> None:
    """基点コミットから保存した全位置引数の順序を固定する。"""
    expected = json.loads((ROOT / "tests/fixtures/generate_positional_3802ab7.json").read_text())
    tree = ast.parse((ROOT / "scripts/visualize_advantage_overlay.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    names = [arg.arg for arg in function.args.posonlyargs + function.args.args]
    assert names[:len(expected)] == expected
    assert names[-1] == "exchange_event_switch_smoothing"


def test_changed_node_limit_recomputes_real_projection() -> None:
    """下10段おじゃまの再現例で、上限変更後の結果を新規計算と照合する。"""
    from src.board import Board, COLOR_OJAMA
    from src.exchange_event_landing import ExchangeLandingProjection
    from src.exchange_event_multilanding import cached_proof
    board = Board()
    board._grid[-GARBAGE_HEIGHT:, :] = COLOR_OJAMA
    projection = ExchangeLandingProjection()
    projection.multilanding_node_limit = LOW_LIMIT
    first = cached_proof(projection, board, QUEUE, INCOMING, HANDS, 0., 0)
    assert (first['dead'], first['reason'], first['nodes']) == (False, 'node_limit', 2)
    projection.multilanding_node_limit = HIGH_LIMIT
    second = cached_proof(projection, board, QUEUE, INCOMING, HANDS, 0., 0)
    assert (second['dead'], second['reason'], second['nodes']) == (True, 'all_responses_dead', 42)
    assert cached_proof(projection, board, QUEUE, INCOMING, HANDS, 0., 0) is second
    projection.multi_landing_cache.clear()
    assert cached_proof(projection, board, QUEUE, INCOMING, HANDS, 0., 0) == second
