"""STABLE 凍結デッドロック根治 3 フラグの収集/表示スクリプトへの配線検証
(2026-08-24)。

## 背景

`src/production_config.py` の `RECOGNITION_ADOPTED` に本日 (2026-08-24)
`--enable-chain-formula-read-verify` / `--enable-formula-chain-count-update` /
`--enable-formula-step-interlude` の 3 フラグを追加登録した。これを
`scripts/collect_boards_lean.py` と `scripts/visualize_recognition.py` の
CLI へ配線した際、`tests/test_measure_main_plumbing_2026_08_24.py`
(W36 恒久対策) と同じ発想の機械検査をこの 2 スクリプトにも適用する。

`--help` が通ること・関数シグネチャに引数があることは「値が実際に
`RecognitionPipeline` まで届く」ことを意味しない
(`feedback_wiring_gap_vs_wiring_error_2026-08-22` の「漏れ」型事故)。
そこで `ast` で `main()` の呼び出し実引数を直接読み、シグネチャとの差を
突きつける。

## スコープについて (重要)

`scripts/visualize_recognition.py` の `main()` →
`RecognitionPipeline.load_default(...)` 呼び出しを全フラグ (91 個) で
汎用監査したところ、**本タスクと無関係な既存フラグ 33 個が既に未配線**
だった (2026-08-24 発見、本ファイル作成時点の実測)。これらは今回の
3 フラグ追加とは無関係な既存の状態であり、正体・要否を未検証のまま
許可リストに機械的に登録するのは「根拠を確認せず問題を握りつぶす」行為に
あたるため、本ファイルでは行わない。**33 件の詳細は本タスクの報告で
別途 user に提示し、対応要否は別タスクとする。**
本ファイルは今回追加した 3 フラグに限定した狭い回帰テストに留める。
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECT_PATH = PROJECT_ROOT / "scripts" / "collect_boards_lean.py"
VISUALIZE_PATH = PROJECT_ROOT / "scripts" / "visualize_recognition.py"
RECOGNITION_PIPELINE_PATH = PROJECT_ROOT / "src" / "recognition_pipeline.py"

# 2026-08-24 に RECOGNITION_ADOPTED へ追加した 3 フラグ (dest 名、
# src.production_config が単一情報源)。
NEW_FLAG_NAMES: tuple[str, ...] = (
    "enable_chain_formula_read_verify",
    "enable_formula_chain_count_update",
    "enable_formula_step_interlude",
)

# collect_boards_lean.py: main() → collect_lean() の全フラグ配線を汎用監査
# する。作成時点 (2026-08-24) で欠落 0 件を実測済みのため空リストにできる。
# **将来ここに追加するときは、なぜ意図的に渡さないのか理由を書くこと。**
COLLECT_INTENTIONALLY_UNPASSED: dict[str, str] = {}


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} が見つからない")


def _flag_params(func: ast.FunctionDef) -> list[str]:
    """シグネチャのうちフラグ系 (enable_/disable_) の引数名。"""
    args = func.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return [n for n in names if n.startswith(("enable_", "disable_"))]


def _find_call_kwargs_by_name(func: ast.FunctionDef, callee: str) -> set[str]:
    """func 内の `callee(...)` (単純名呼び出し) のキーワード名を集める。"""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == callee:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"{func.name} の中に {callee}(...) の呼び出しが無い")


def _find_call_kwargs_by_attr(func: ast.FunctionDef, attr: str) -> set[str]:
    """func 内の `xxx.attr(...)` (属性呼び出し) のキーワード名を集める。"""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == attr:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"{func.name} の中に .{attr}(...) の呼び出しが無い")


def _find_for_loop_str_tuple(func: ast.FunctionDef, target_name: str) -> set[str]:
    """`for <target_name> in (...):` のタプル文字列リテラル一式を集める。

    resolve_production_config_overrides() の
    `for name in ("enable_effect_gate", ...):` を読むために使う。
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.For):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == target_name):
            continue
        if isinstance(node.iter, ast.Tuple):
            return {
                elt.value for elt in node.iter.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    raise AssertionError(f"{func.name} に for {target_name} in (...) が無い")


class TestCollectBoardsLeanFullPlumbing:
    """collect_boards_lean.py: main() → collect_lean() の全フラグ配線監査。

    W35/W36 と同型の配線漏れ (シグネチャは揃っているのに main() が渡して
    いない) を機械検出する。作成時点で欠落 0 件 (46/46 配線済み)。
    """

    def test_all_flags_passed_from_main_to_collect_lean(self) -> None:
        tree = _module_ast(COLLECT_PATH)
        collect_lean = _find_func(tree, "collect_lean")
        main_fn = _find_func(tree, "main")
        passed = _find_call_kwargs_by_name(main_fn, "collect_lean")
        missing = [
            name for name in _flag_params(collect_lean)
            if name not in passed and name not in COLLECT_INTENTIONALLY_UNPASSED
        ]
        assert not missing, (
            "main() から collect_lean() へ渡されていないフラグがある。\n"
            "シグネチャ既定値が黙って使われ、対応する CLI が無効になる "
            "(W35/W36 と同型の配線漏れ)。\n"
            f"未配線: {missing}\n"
            "意図的に既定へ委ねる場合は COLLECT_INTENTIONALLY_UNPASSED に"
            "理由つきで登録すること。"
        )

    def test_new_3flags_wired_in_collect_lean_signature_and_call(self) -> None:
        """今回追加した 3 フラグが collect_lean のシグネチャと main() の呼び
        出しの両方に存在すること (名指しの回帰テスト)。"""
        tree = _module_ast(COLLECT_PATH)
        collect_lean = _find_func(tree, "collect_lean")
        main_fn = _find_func(tree, "main")
        sig_flags = set(_flag_params(collect_lean))
        passed = _find_call_kwargs_by_name(main_fn, "collect_lean")
        for name in NEW_FLAG_NAMES:
            assert name in sig_flags, f"{name} が collect_lean のシグネチャに無い"
            assert name in passed, f"{name} が main() から collect_lean へ渡されていない"


class TestVisualizeRecognitionNew3FlagsPlumbing:
    """visualize_recognition.py: 今回追加した 3 フラグに限定した回帰テスト。

    全 91 フラグの汎用監査は本タスクと無関係な既存の未配線 33 件を含むため
    行わない (モジュール docstring 参照)。今回追加分のみ名指しで固定する。
    """

    def test_new_3flags_wired_in_load_default_call(self) -> None:
        """main() → RecognitionPipeline.load_default(...) へ渡されていること。"""
        vr_tree = _module_ast(VISUALIZE_PATH)
        main_fn = _find_func(vr_tree, "main")
        passed = _find_call_kwargs_by_attr(main_fn, "load_default")
        for name in NEW_FLAG_NAMES:
            assert name in passed, (
                f"{name} が main() から RecognitionPipeline.load_default() へ"
                "渡されていない (CLI が無効になる、W35/W36と同型の配線漏れ)"
            )

    def test_new_3flags_in_resolve_production_config_overrides(self) -> None:
        """production 自動適用 (RECOGNITION_ADOPTED OR 合成) の対象名一覧に
        含まれていること。ここに無いと --no-production-recognition 未指定
        でも本番採用フラグが自動 ON されない。"""
        vr_tree = _module_ast(VISUALIZE_PATH)
        resolver = _find_func(vr_tree, "resolve_production_config_overrides")
        names = _find_for_loop_str_tuple(resolver, "name")
        for name in NEW_FLAG_NAMES:
            assert name in names, (
                f"{name} が resolve_production_config_overrides() の対象名"
                "一覧に無い (RECOGNITION_ADOPTED 自動適用が効かない)"
            )

    def test_new_3flags_have_cli_arguments(self) -> None:
        """argparse に --enable-xxx が定義されていること (dest 名で確認)。"""
        vr_tree = _module_ast(VISUALIZE_PATH)
        main_fn = _find_func(vr_tree, "main")
        dests: set[str] = set()
        for node in ast.walk(main_fn):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr == "add_argument":
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        dests.add(kw.value.value)
        for name in NEW_FLAG_NAMES:
            assert name in dests, f"--enable-{name[len('enable_'):].replace('_', '-')} の argparse 定義が無い"


class TestDetectorSelfCheck:
    """測定器自身の健全性: 引数を落としたら本当に検知できるか
    (`feedback_measure_under_production_conditions_2026-08-22` の系)。"""

    def test_name_call_detector_catches_dropped_kwarg(self) -> None:
        src = (
            "def collect_lean(a, *, enable_foo=False, enable_bar=False):\n"
            "    return a\n"
            "def main():\n"
            "    return collect_lean(1, enable_foo=True)\n"
        )
        tree = ast.parse(src)
        collect_lean = _find_func(tree, "collect_lean")
        main_fn = _find_func(tree, "main")
        passed = _find_call_kwargs_by_name(main_fn, "collect_lean")
        missing = [n for n in _flag_params(collect_lean) if n not in passed]
        assert missing == ["enable_bar"], f"検知できていない: {missing}"

    def test_attr_call_detector_catches_dropped_kwarg(self) -> None:
        src = (
            "class RecognitionPipeline:\n"
            "    @staticmethod\n"
            "    def load_default(*, enable_foo=False, enable_bar=False):\n"
            "        pass\n"
            "def main():\n"
            "    RecognitionPipeline.load_default(enable_foo=True)\n"
        )
        tree = ast.parse(src)
        main_fn = _find_func(tree, "main")
        passed = _find_call_kwargs_by_attr(main_fn, "load_default")
        assert "enable_bar" not in passed
        assert "enable_foo" in passed

    def test_for_loop_tuple_detector_catches_dropped_name(self) -> None:
        src = (
            "def resolve():\n"
            "    for name in ('enable_a', 'enable_b'):\n"
            "        pass\n"
        )
        tree = ast.parse(src)
        resolver = _find_func(tree, "resolve")
        names = _find_for_loop_str_tuple(resolver, "name")
        assert names == {"enable_a", "enable_b"}
        assert "enable_c" not in names


class TestTargetFunctionsExist:
    """検査対象の関数/クラスが変わっていないこと (検査が空振りするのを防ぐ)。"""

    def test_collect_lean_and_main_exist(self) -> None:
        tree = _module_ast(COLLECT_PATH)
        _find_func(tree, "collect_lean")
        _find_func(tree, "main")

    def test_visualize_recognition_functions_exist(self) -> None:
        tree = _module_ast(VISUALIZE_PATH)
        _find_func(tree, "main")
        _find_func(tree, "resolve_production_config_overrides")

    def test_recognition_pipeline_load_default_exists(self) -> None:
        """visualize_recognition.py が呼ぶ RecognitionPipeline.load_default
        の実体が src/recognition_pipeline.py 側にまだ存在すること。"""
        tree = _module_ast(RECOGNITION_PIPELINE_PATH)
        _find_func(tree, "load_default")
