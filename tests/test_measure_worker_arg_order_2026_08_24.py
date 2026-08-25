"""並列ワーカ呼出しの位置引数順序ズレを機械検証するテスト (2026-08-24 Q-03是正)。

scripts/measure_stable_cell_acc.py の _collect_parallel は
ProcessPoolExecutor.submit(_process_video_worker, ...) を約50個の位置引数で
呼び出す。従来は「順序ズレ厳禁」というコメント (8箇所) だけが頼りで、
新フラグ追加時に _process_video_worker の仮引数リストと submit の実引数リストの
どちらか一方だけを更新すると壊れても気づけない構造だった。

本テストは:
  1. _process_video_worker の仮引数名リストを inspect.signature で取得する。
  2. scripts/measure_stable_cell_acc.py のソースを ast で解析し、
     _collect_parallel 内の executor.submit(_process_video_worker, ...) から
     実引数の変数名リストを抽出する (submit の第1引数=関数参照自体は除く)。
  3. 両者が完全一致することを assert する。

さらに「検出ロジック自体が壊れていないか」の確認として、合成関数で意図的に
順序を入れ替えたケースが検出されることも確認する (測定器自体の健全性チェック、
CLAUDE.md 「fail-silent 警戒」原則)。
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts import measure_stable_cell_acc as msca  # noqa: E402

_SOURCE_PATH = _PROJ_ROOT / "scripts" / "measure_stable_cell_acc.py"
_WORKER_FUNC_NAME = "_process_video_worker"
_COLLECT_PARALLEL_FUNC_NAME = "_collect_parallel"

# Q-03 是正 (2026-08-24) で末尾追加した4フラグ。宣言順を固定で明示することで
# 「末尾に紛れて消えていないか」の追加ガードにする。
_Q03_TAIL_FLAGS: list[str] = [
    "enable_chain_formula_read_verify",
    "enable_formula_chain_count_update",
    "enable_slide_exit_no_min_display",
    "enable_formula_step_interlude",
]

# submit 呼出しの先頭3引数 (vid, str(vpath), (vid in holdout_ids)) は、
# pickle 化のための str() 変換や holdout 判定式であり、worker 側の仮引数名
# (video_id, video_path_str, is_holdout) とは意図的に変数名が異なる。
# この区間だけは「変数名の完全一致」チェックの対象外とし、個数のみ確認する。
# それ以降 (max_frames 以降) は全て同名ローカル変数のそのまま渡しである
# ことがコード規約 (コメントで反復明記) になっており、ここが本テストの主眼。
_SUBMIT_FREEFORM_PREFIX_LEN = 3


def _worker_param_names() -> list[str]:
    """_process_video_worker の仮引数名を宣言順のリストで返す。"""
    sig = inspect.signature(msca._process_video_worker)
    return list(sig.parameters.keys())


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    """モジュール AST から指定名の関数定義ノードを 1 つ探す。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"関数定義 {name!r} がソース中に見つからない (関数名変更?)")


def _find_submit_call(fn_node: ast.FunctionDef, worker_func_name: str) -> ast.Call:
    """関数定義内から executor.submit(worker_func_name, ...) 呼出しノードを探す。"""
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "submit"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == worker_func_name
        ):
            return node
    raise AssertionError(
        f"executor.submit({worker_func_name}, ...) 呼出しが "
        f"{fn_node.name} 内に見つからない"
    )


def _submit_call_arg_names(
    submit_call: ast.Call, freeform_prefix_len: int = 0
) -> list[str]:
    """submit 呼出しの実引数 (第1引数=関数参照を除く) の変数名リストを返す。

    このプロジェクトの規約 (コメントで反復明記) では、submit へ渡す実引数の
    大半は呼出し元スコープの同名ローカル変数への単純参照であり、式や定数は
    使わない。ただし先頭 freeform_prefix_len 個は変換式 (pickle化用の str() や
    holdout 判定の比較式等) であることが正規に許容されており、その区間は
    プレースホルダ文字列を返す (個数の突合にのみ使い、名前比較の対象外)。
    それ以外の位置で単純な変数参照でない場合は構造異常として fail させる。
    """
    assert not submit_call.keywords, (
        "executor.submit にキーワード引数が含まれている。本測定器は全引数が"
        "位置引数で渡される前提で設計されている。"
        f"keywords={[kw.arg for kw in submit_call.keywords]}"
    )
    arg_names: list[str] = []
    for idx, arg_node in enumerate(submit_call.args[1:]):
        if isinstance(arg_node, ast.Name):
            arg_names.append(arg_node.id)
            continue
        assert idx < freeform_prefix_len, (
            f"submit の実引数 (0-indexed, 関数参照を除く) {idx} 番目が単純な変数"
            f"参照ではない (型={type(arg_node).__name__})。freeform 許容区間"
            f"(先頭{freeform_prefix_len}個) の外なので、変換式やリテラルの"
            "混入は意図しないズレの可能性が高い (要手動確認)。"
        )
        arg_names.append(f"<freeform-expr:{type(arg_node).__name__}@{idx}>")
    return arg_names


def _load_submit_arg_names(freeform_prefix_len: int = 0) -> list[str]:
    """scripts/measure_stable_cell_acc.py を ast 解析し、_collect_parallel 内の
    executor.submit(_process_video_worker, ...) の実引数名リストを返す。
    """
    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    collect_parallel_fn = _find_function_def(tree, _COLLECT_PARALLEL_FUNC_NAME)
    submit_call = _find_submit_call(collect_parallel_fn, _WORKER_FUNC_NAME)
    return _submit_call_arg_names(submit_call, freeform_prefix_len=freeform_prefix_len)


def test_worker_signature_matches_submit_call_order() -> None:
    """_process_video_worker の仮引数順と executor.submit の実引数順が完全一致することを検証する。

    先頭 _SUBMIT_FREEFORM_PREFIX_LEN 個 (vid / str(vpath) / holdout判定式) は
    意図的な変換式のため名前比較から除外し、それ以降 (max_frames 以降、大半を
    占める bool フラグ列) で完全一致を要求する。ズレた場合はどの位置で何と何が
    食い違っているかをメッセージに明示する。
    """
    worker_params = _worker_param_names()
    submit_args = _load_submit_arg_names(
        freeform_prefix_len=_SUBMIT_FREEFORM_PREFIX_LEN
    )

    assert len(worker_params) == len(submit_args), (
        "引数の個数が一致しない: "
        f"_process_video_worker は仮引数 {len(worker_params)} 個、"
        f"executor.submit は実引数 {len(submit_args)} 個 (関数参照を除く)。\n"
        f"worker params 末尾10個: {worker_params[-10:]}\n"
        f"submit args   末尾10個: {submit_args[-10:]}"
    )

    # freeform 区間 (先頭 _SUBMIT_FREEFORM_PREFIX_LEN 個) は名前比較の対象外。
    # それ以降 (max_frames 以降) は全位置で厳密一致を要求する。
    checked_worker = worker_params[_SUBMIT_FREEFORM_PREFIX_LEN:]
    checked_submit = submit_args[_SUBMIT_FREEFORM_PREFIX_LEN:]
    mismatches = [
        (pos, w, s)
        for pos, (w, s) in enumerate(zip(checked_worker, checked_submit))
        if w != s
    ]
    assert not mismatches, (
        "executor.submit の実引数順が _process_video_worker の仮引数順から"
        "ズレている (位置引数プロトコル違反、位置は freeform prefix 除外後基準)。\n"
        + "\n".join(
            f"  position {pos} (freeform prefix除外後): "
            f"worker param={w!r}  vs  submit arg={s!r}"
            for pos, w, s in mismatches
        )
    )


def test_q03_tail_flags_present_and_aligned_in_both() -> None:
    """Q-03 是正で追加した4フラグが、両者の末尾に同じ順序で存在することを確認する。

    「末尾追加」という運用規約そのものが守られているかの追加ガード
    (途中挿入によるズレは上記テストでも捕まるが、意図の明示のため個別に確認)。
    """
    worker_params = _worker_param_names()
    submit_args = _load_submit_arg_names(
        freeform_prefix_len=_SUBMIT_FREEFORM_PREFIX_LEN
    )
    n = len(_Q03_TAIL_FLAGS)

    assert worker_params[-n:] == _Q03_TAIL_FLAGS, (
        "_process_video_worker の末尾4仮引数が Q-03 是正フラグと一致しない: "
        f"実際={worker_params[-n:]} 期待={_Q03_TAIL_FLAGS}"
    )
    assert submit_args[-n:] == _Q03_TAIL_FLAGS, (
        "executor.submit の末尾4実引数が Q-03 是正フラグと一致しない: "
        f"実際={submit_args[-n:]} 期待={_Q03_TAIL_FLAGS}"
    )


def test_detection_logic_catches_synthetic_order_swap() -> None:
    """検出ロジック自体の健全性確認 (測定器事故対策)。

    意図的に引数順を入れ替えた合成関数・合成 submit 呼出しに対して、
    上記と同じ突合ロジックが不一致を検出できることを確認する。
    これが通らなければ、本ファイルの他のテストが「常に緑」の fail-silent な
    測定器になっている可能性がある。
    """

    def _worker_synth(a: int, b: int, c: int = 0) -> None:
        return None

    synthetic_source = (
        "def _collect_parallel_synth():\n"
        "    executor.submit(_worker_synth, a, c, b)\n"  # b, c をわざと入替
    )
    tree = ast.parse(synthetic_source)
    fn_node = _find_function_def(tree, "_collect_parallel_synth")
    submit_call = _find_submit_call(fn_node, "_worker_synth")
    submit_args = _submit_call_arg_names(submit_call)
    worker_params = list(inspect.signature(_worker_synth).parameters.keys())

    mismatches = [
        pos for pos, (w, s) in enumerate(zip(worker_params, submit_args)) if w != s
    ]
    assert mismatches, (
        "合成テストで意図的に仕込んだ順序ズレ (b, c 入替) が検出できなかった。"
        "突合ロジックが壊れている可能性が高い。"
    )


def test_detection_logic_passes_synthetic_correct_order() -> None:
    """検出ロジックの偽陽性確認: 正しい順序では不一致が出ないことを確認する。"""

    def _worker_synth(a: int, b: int, c: int = 0) -> None:
        return None

    synthetic_source = (
        "def _collect_parallel_synth():\n"
        "    executor.submit(_worker_synth, a, b, c)\n"
    )
    tree = ast.parse(synthetic_source)
    fn_node = _find_function_def(tree, "_collect_parallel_synth")
    submit_call = _find_submit_call(fn_node, "_worker_synth")
    submit_args = _submit_call_arg_names(submit_call)
    worker_params = list(inspect.signature(_worker_synth).parameters.keys())

    assert worker_params == submit_args, (
        "正しい順序の合成呼出しなのに不一致と判定された (突合ロジックの偽陽性)。"
    )
