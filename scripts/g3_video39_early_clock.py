"""video39の私有早期J/PB moduleだけを30fpsへ接続する。元ファイルは不変更。"""
from __future__ import annotations
import ast
import hashlib
import inspect
import textwrap
from pathlib import Path
from typing import Any, Callable
from scripts import g3_video39_next as G
from scripts import g3_reset_scope as R

READER_SHA = '8c61ea9577287eda888ccd113dfabd6e916be02c1ef0b73bbd4a29e7adf1e838'
HISTORY_SHA = 'cd9a36cd7361f44096806a6ce0b02b7afadf188d9878ff01331df77d24fe79bb'
JOURNAL_SHA = 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'
OLD_FPS, OLD_STRIDE = 60, 2


class Clock(ast.NodeTransformer):
    """SHA固定関数中の除数60だけを変換し、個数も固定する。"""
    def __init__(self) -> None:
        self.count = 0

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value == OLD_FPS:
            node.right = ast.copy_location(ast.Constant(G.FPS), node.right)
            self.count += 1
        return node


def compile_clock(module: Any, source: str, name: str, count: int) -> tuple[Callable, dict]:
    """生成関数も私有moduleのglobalsを所有。元source名へ偽装しない。"""
    tree, transform = ast.parse(textwrap.dedent(source)), Clock()
    transform.visit(tree)
    G.P.M.require(transform.count == count, 'early_clock_transform_count')
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    values: dict[str, Any] = {}
    exec(compile(text, str(Path(__file__).resolve()), 'exec'), vars(module), values)
    return values[name], dict(function=name, divisions_changed=count, source=text,
                              sha256=hashlib.sha256(text.encode()).hexdigest())


def verify(module: Any, digest: str) -> None:
    """関数を差し替える前に固定原ファイルを認証する。"""
    G.P.M.require(G.P.M.digest(Path(module.__file__)) == digest, 'early_clock_source_changed')


def prepare_journal(stack: Any, module: Any, replace: Callable) -> dict:
    """原Jの入口と保存検査を同時に移譲。元emit/段階/原票SHA検査は維持する。"""
    verify(module, JOURNAL_SHA)
    G.P.M.require((module.FIRST, module.LAST, module.STRIDE) == (29052, 36298, OLD_STRIDE),
                  'journal_original_bounds')
    proofs = []
    for owner, name in ((module.Recorder, 'scope'), (module, 'verify_rows')):
        original = getattr(owner, name)
        G.P.M.require(original.__globals__ is vars(module), 'journal_function_owner')
        value, proof = compile_clock(module, inspect.getsource(original), name, 1)
        replace(stack, owner, name, value)
        proofs.append(proof)
    for name, value in (('FIRST', 0), ('LAST', G.END - 1), ('STRIDE', G.STRIDE)):
        replace(stack, module, name, value)
    G.P.M.require(module.verify.__globals__ is vars(module), 'journal_verifier_globals')
    return dict(original_sha=JOURNAL_SHA, functions=proofs, first=0, end_exclusive=G.END,
                stride=G.STRIDE, quality_gate_clear=False)


def prepare(stack: Any, reader: Any, history: Any, observation: Any,
            replace: Callable) -> dict:
    """modules返却直後、Capture/EarlyHistory生成前だけに適用する。"""
    verify(reader, READER_SHA)
    verify(history, HISTORY_SHA)
    G.P.M.require(history.STRIDE == OLD_STRIDE, 'early_original_stride')
    receipt = dict(reader_original_sha=READER_SHA, history_original_sha=HISTORY_SHA,
                   stride_before=OLD_STRIDE, stride_after=G.STRIDE, fps_constant_unused=True,
                   functions=[], quality_gate_clear=False)
    for name in ('validate_row', 'read_pair'):
        original = getattr(reader, name)
        G.P.M.require(original.__globals__ is vars(reader), 'early_reader_owner')
        value, proof = compile_clock(reader, inspect.getsource(original), name, 1)
        replace(stack, reader, name, value)
        receipt['functions'].append(proof)
    replace(stack, history, 'STRIDE', G.STRIDE)
    if observation is not None:
        _, reset_proof = R.generated(observation)
        value, proof = compile_clock(observation, reset_proof['generated_source'], 'capture', 2)
        replace(stack, observation, 'capture', value)
        receipt['functions'].append(proof)
        receipt['reset_original_sha'] = reset_proof['original_sha256']
    return receipt


def bind(stack: Any, runtime: Any, replace: Callable, item: dict, receipt: dict) -> None:
    """実modules globalを所有置換し、同関数内の次のconstructorより前に接続する。"""
    G.validate(item)
    original = runtime.modules
    def modules(owner: Any, load: Any, *, probability: dict | None = None) -> tuple:
        G.P.M.require(not receipt, 'early_clock_duplicate_modules')
        witness, reader, history = original(owner, load, probability=probability)
        observation = None if probability is None else probability['observation']
        receipt.update(prepare(owner, reader, history, observation, replace))
        receipt.update(source_id=item['source_id'], time_base=[1, G.FPS])
        return witness, reader, history
    replace(stack, runtime, 'modules', modules)
