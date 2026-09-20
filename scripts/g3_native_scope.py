"""G3予測入力の採否窓 FIRST_FRAME だけを実走frameへ差し替える。

原因: `Witness.creation_in_scope` (`data/verify/g2_history_baseline_adoption_2026-09-10_v4/witness.py`)
は `adapter.native.FIRST_FRAME` / `LAST_FRAME` を採否窓として読むが、native module
(`scripts/next_enqueue_live_shadow_v1.py`) の FIRST_FRAME はG2時代の32494に
固定されたまま。窓の開始より前 (実測3812〜32194) の候補は全て
`outside_original_directional_scope` で却下される (v12実走で1,096件全件)。

本ファイルはFIRST_FRAMEだけを実走開始frameへ書き換え、scope終了で必ず元へ戻す。
LAST_FRAMEは `window_candidate.bind_native` が既に実走値へ拡張済みなので触らない
(1点だけ差し替える型を守る、.claude/rules/02-engineering-principles.md 原則3)。
凍結資産 (data/verify配下) は一切編集しない。実行時差し替えのみ。
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
from typing import Any

from scripts import g3_video38_entry as G

EXPECTED_FIRST = 32494  # window_candidate.NEXT_FIRST と同値。first以外の照合はしない。


def native_module(adapter: Any) -> Any:
    """window_candidateの検証済みnative()を経由し、同じsingletonだけを返す。

    再import で別インスタンスを作らない。adapter.OC.native() は
    sys.modules上の既存実体を identity ごと検証してから返す (window_candidate.py:22-29)。
    """
    return adapter.OC.native()


def install(stack: ExitStack, adapter: Any, entry: Any) -> dict:
    """元window.bind実行後の一点で、FIRST_FRAMEだけを実走開始frameへ差し替える。

    window.bind (= adapter.OC.ORIGINAL.bind) は既に window_candidate.install() で
    ラップ済みで、呼ばれた時点で native module の import と LAST_FRAME 拡張が
    完了している。ここではそのラップの外側にもう一段足し、native moduleが
    確定した直後にFIRST_FRAMEだけを差し替える。
    """
    window, replace = adapter.OC.ORIGINAL, adapter.A.A.A.V4.replace_owned
    original = window.bind
    fired: list[bool] = []
    receipt: dict[str, Any] = dict(before_first=None, before_last=None, after_first=None,
        replaced=0, restored=None, native_module=None, native_source_sha=None)

    def bind(owner_stack: Any, latest: Any, owner_replace: Any) -> dict:
        G.require(not fired, 'native_scope_duplicate_bind')
        values = original(owner_stack, latest, owner_replace)
        fired.append(True)
        _apply(owner_stack, owner_replace, adapter, entry, receipt)
        return values
    replace(stack, window, 'bind', bind)
    return receipt


def _apply(stack: Any, replace: Any, adapter: Any, entry: Any, receipt: dict) -> None:
    """native moduleのFIRST_FRAMEだけを検証つきで差し替え、票へ実測値を書く。"""
    value = native_module(adapter)
    receipt['native_module'] = getattr(value, '__name__', None) or type(value).__module__
    receipt['native_source_sha'] = _digest(Path(value.__file__).resolve())
    before_first, before_last = value.FIRST_FRAME, value.LAST_FRAME
    receipt['before_first'], receipt['before_last'] = before_first, before_last
    if before_first != EXPECTED_FIRST:
        raise ValueError('native_scope_unexpected:' + repr(before_first))
    after_first = entry.__globals__['K'].bounds()['first_frame']

    def check_restored() -> None:
        receipt['restored'] = value.FIRST_FRAME == before_first
    # ExitStackはLIFOで戻すため、復元確認は実差し替えより先に登録し、
    # 「復元済みの値」を読む側を後に(=実行順で後段に)まわす。
    stack.callback(check_restored)
    replace(stack, value, 'FIRST_FRAME', after_first)
    receipt['after_first'], receipt['replaced'] = after_first, 1


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """既存G3 video38入口へ本差し替えを追加接続する専用入口。"""
    args = G.arguments()
    output = Path(args.output)
    plan = G.read(args.plan)
    G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'native_scope_entry_pin')
    previous = G.protect_runtime
    state: dict = {}

    def protect(stack: ExitStack, adapter: Any, entry: Any, fixed: dict) -> None:
        previous(stack, adapter, entry, fixed)
        state.update(install(stack, adapter, entry))

        def saved() -> None:
            G.save(output / 'G3_NATIVE_SCOPE.json', state)
        stack.callback(saved)
    G.protect_runtime = protect
    try:
        return G.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
