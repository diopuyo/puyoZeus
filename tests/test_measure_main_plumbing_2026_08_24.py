"""物差しの `main()` → `_collect_results` 配線漏れ検出 (2026-08-24、W36 恒久対策)。

## なぜ要るか

`scripts/measure_stable_cell_acc.py` で **CLI フラグが完全に死んでいた**
(`docs/KNOWN_WEAKNESSES.md` W36)。

    CLI --gravity-settle-state              既定 True
    main() のローカル変数                    getattr(args, ..., True) で取得している
    main() → _collect_results(...)          ← **渡していない**
    _collect_results のシグネチャ既定         False
    _collect_results → 下流                  その False を明示的に転送
    RecognitionPipeline.load_default の既定   True (2026-06-06 採用)

結果として物差しは GRAVITY_SETTLE を切った状態、つまり**本番とは違う
フレーム母集団**の認識精度を測り続けていた。Phase I の合格判定 99.54% を含む
過去の全数値がこの構成で出ている。

**これは `--help` を見ても、シグネチャを見ても、`main()` が変数を持っている
ことを見ても検出できない。** 3 つとも揃っていたのに値が届いていなかった
(`feedback_wiring_gap_vs_wiring_error_2026-08-22` の「漏れ」)。

そこで **`ast` で呼び出しの実引数を直接読み、シグネチャとの差を突きつける**。

## 何を検査するか

`_collect_results` のシグネチャにある `enable_*` / `disable_*` 引数のうち、
`main()` の呼び出しで渡されていないものを列挙し、
**許可リストに無ければ失敗させる**。

許可リスト方式にしているのは、「意図的に既定へ委ねる引数」が将来出うるため。
ただし**追加するときは理由をここに書かせる**ことで、無自覚な漏れと区別する。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEASURE_PATH = PROJECT_ROOT / "scripts" / "measure_stable_cell_acc.py"

# `main()` から `_collect_results` へ意図的に渡さない引数。
# **空にしておくこと。追加するなら必ず理由を併記する。**
INTENTIONALLY_UNPASSED: dict[str, str] = {}


def _module_ast() -> ast.Module:
    return ast.parse(MEASURE_PATH.read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} が見つからない")


def _find_call_kwargs(func: ast.FunctionDef, callee: str) -> set[str]:
    """func の中の `callee(...)` 呼び出しで渡しているキーワード名を集める。"""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == callee:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"{func.name} の中に {callee}(...) の呼び出しが無い")


def _flag_params(func: ast.FunctionDef) -> list[str]:
    """シグネチャのうちフラグ系 (enable_/disable_) の引数名。"""
    args = func.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return [n for n in names if n.startswith(("enable_", "disable_"))]


def test_all_flags_are_passed_from_main_to_collect_results() -> None:
    """`_collect_results` のフラグ引数が `main()` から全て渡されていること。

    渡し忘れると**シグネチャ既定値が黙って使われ、CLI が無効になる**。
    W36 はこれで CLI が完全に死んでいた。
    """
    tree = _module_ast()
    collect = _find_func(tree, "_collect_results")
    main_fn = _find_func(tree, "main")
    passed = _find_call_kwargs(main_fn, "_collect_results")

    missing = [
        name for name in _flag_params(collect)
        if name not in passed and name not in INTENTIONALLY_UNPASSED
    ]
    assert not missing, (
        "main() から _collect_results へ渡されていないフラグがある。\n"
        "シグネチャ既定値が黙って使われ、対応する CLI が無効になる "
        "(docs/KNOWN_WEAKNESSES.md W36 と同じ事故)。\n"
        f"未配線: {missing}\n"
        "意図的に既定へ委ねる場合は INTENTIONALLY_UNPASSED に理由つきで登録すること。"
    )


def test_gravity_settle_state_is_wired() -> None:
    """W36 の直接の回帰テスト。

    `--gravity-settle-state` が効かず、物差しが本番と違うフレーム母集団を
    測っていた件。名指しで固定しておく。
    """
    tree = _module_ast()
    passed = _find_call_kwargs(_find_func(tree, "main"), "_collect_results")
    assert "enable_gravity_settle_state" in passed, (
        "enable_gravity_settle_state が main() から渡されていない。"
        "CLI --gravity-settle-state が無効になり、物差しが GRAVITY_SETTLE を"
        "切った状態 (本番と異なる構成) で走る。"
    )


def test_detector_catches_a_deliberately_dropped_flag() -> None:
    """**測定器自身の健全性**: 引数を落としたら本当に検知するか。

    「順序ズレ厳禁」のようなコメントに頼らず機械検査に置き換えるのが狙いなので、
    その機械検査が実際に穴を捕まえることを確かめておく
    (`feedback_measure_under_production_conditions_2026-08-22` の系:
     測定器は壊れた条件で空振りしうる)。
    """
    src = (
        "def _collect_results(a, *, enable_foo=False, enable_bar=False):\n"
        "    return a\n"
        "def main():\n"
        "    return _collect_results(1, enable_foo=True)\n"
    )
    tree = ast.parse(src)
    collect = _find_func(tree, "_collect_results")
    passed = _find_call_kwargs(_find_func(tree, "main"), "_collect_results")
    missing = [n for n in _flag_params(collect) if n not in passed]
    assert missing == ["enable_bar"], f"検知できていない: {missing}"


def test_no_false_positive_when_all_flags_passed() -> None:
    """全部渡していれば検知しない (偽陽性が出ないこと)。"""
    src = (
        "def _collect_results(a, *, enable_foo=False, enable_bar=False):\n"
        "    return a\n"
        "def main():\n"
        "    return _collect_results(1, enable_foo=True, enable_bar=False)\n"
    )
    tree = ast.parse(src)
    collect = _find_func(tree, "_collect_results")
    passed = _find_call_kwargs(_find_func(tree, "main"), "_collect_results")
    assert [n for n in _flag_params(collect) if n not in passed] == []


@pytest.mark.parametrize("name", ["_collect_results", "main"])
def test_target_functions_exist(name: str) -> None:
    """検査対象の関数名が変わっていないこと (検査が空振りするのを防ぐ)。"""
    _find_func(_module_ast(), name)
