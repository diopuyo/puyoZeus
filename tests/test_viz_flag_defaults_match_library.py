"""viz の認識フラグ既定が本番 (load_default) と一致することを恒久検証する。

## なぜこのテストが必要か

`scripts/visualize_advantage_overlay.py` は認識系フラグに**明示的な False** を
渡しており、`RecognitionPipeline.load_default` の既定が True なのに
**viz では認識改善が一切効いていなかった** (2026-07-30 に inspect で実測確認)。

食い違っていたのは 7 つ:
    enable_landing_observed_color / enable_match_start_full_clear /
    enable_recovery_counter_carryover / enable_cnn_flicker_hsv_fallback /
    enable_initial_confirm_vote /
    enable_drift_resync_match_start_guard / enable_drift_resync_hsv_gate

**人のレビューは全て viz を通る**ため、本番と違う挙動を映すレビューは
無いより悪い (「viz で承認したのに収集データは別挙動」が起こりうる)。

2026-07-31 に「未指定 (None) は inspect でライブラリ既定に解決する」方式へ修正した。
本テストは **将来ライブラリ既定が変わったときに viz が置き去りにされないこと**を
保証する (今回の事故の再発防止そのもの)。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

from src.recognition_pipeline import RecognitionPipeline

# viz の generate() 引数名 -> load_default の引数名
# (enable_drift_guards は 1 つで 2 引数に展開されるので別扱い)
DIRECT_FLAGS: tuple[str, ...] = (
    "enable_landing_observed_color",
    "enable_match_start_full_clear",
    "enable_recovery_counter_carryover",
    "enable_cnn_flicker_hsv_fallback",
    "enable_initial_confirm_vote",
    "enable_puyo_to_empty_hsv_guard",
)
# enable_drift_guards が展開される load_default 引数
DRIFT_FLAGS: tuple[str, ...] = (
    "enable_drift_resync_match_start_guard",
    "enable_drift_resync_hsv_gate",
)


@pytest.fixture(scope="module")
def viz() -> Any:
    """visualize_advantage_overlay をモジュールとして読み込む。"""
    path = Path(__file__).resolve().parent.parent / "scripts" / (
        "visualize_advantage_overlay.py"
    )
    spec = importlib.util.spec_from_file_location("_viz_for_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_viz_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _lib_default(name: str) -> bool:
    """load_default の当該引数の既定値。"""
    params = inspect.signature(RecognitionPipeline.load_default).parameters
    assert name in params, f"load_default に {name} が無い"
    return bool(params[name].default)


@pytest.mark.parametrize("name", DIRECT_FLAGS)
def test_generate_default_is_none(viz: Any, name: str) -> None:
    """generate() の既定が None (= ライブラリ既定に委ねる) であること。

    ここが False に戻されると「viz だけ旧挙動」の事故が再発する。
    """
    params = inspect.signature(viz.generate).parameters
    assert name in params, f"generate() に {name} が無い"
    assert params[name].default is None, (
        f"{name} の既定が {params[name].default!r}。"
        "None 以外だとライブラリ既定を追従できない"
    )


def test_drift_guards_default_is_none(viz: Any) -> None:
    """enable_drift_guards の既定も None であること。"""
    params = inspect.signature(viz.generate).parameters
    assert params["enable_drift_guards"].default is None


@pytest.mark.parametrize("name", DIRECT_FLAGS + DRIFT_FLAGS)
def test_unspecified_resolves_to_library_default(viz: Any, name: str) -> None:
    """未指定 (None) がライブラリ既定に解決されること。

    **これが本テストの核心**: 将来ライブラリ既定が変わっても viz が追従する。
    """
    assert viz._resolve_flag(name, None) == _lib_default(name)


@pytest.mark.parametrize("name", DIRECT_FLAGS + DRIFT_FLAGS)
@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_value_is_respected(viz: Any, name: str, explicit: bool) -> None:
    """明示指定 (--flag / --no-flag) は尊重されること (A/B 検証に必要)。"""
    assert viz._resolve_flag(name, explicit) is explicit


def test_unknown_flag_resolves_to_false(viz: Any) -> None:
    """load_default に存在しない名前は False に倒れる (例外を投げない)。

    フラグ名の typo でレンダが落ちるより、無害な既定で進む方が安全。
    """
    assert viz._resolve_flag("enable_definitely_not_a_real_flag", None) is False


def test_argparse_flags_allow_explicit_false(viz: Any) -> None:
    """CLI で --no-<flag> による明示 False が可能であること。

    None 既定にした結果「False を強制する手段がない」と A/B 検証ができない。
    BooleanOptionalAction により --no- 形が生えていることを確認する。
    """
    parser = viz.build_parser() if hasattr(viz, "build_parser") else None
    if parser is None:
        pytest.skip("build_parser が無い構成 (main 内で組み立てている)")
    opts = {s for a in parser._actions for s in a.option_strings}
    for cli in (
        "--no-landing-observed-color",
        "--no-drift-guards",
        "--no-initial-confirm-vote",
    ):
        assert cli in opts, f"{cli} が無い (明示 False ができない)"
