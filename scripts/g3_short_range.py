"""終端の構造欠陥だけを短い区間で再現するための区間短縮接続。

終端以外 (復号・認識) は毎回同一なので、構造依存の欠陥に139分かけるのは無駄。
END を環境変数で縮め、closure/終了保存/親票/通知までを数分で通す。

使えないもの:
  内容依存の欠陥。effect はframe15794、M1窓はframe35160以降なので短区間では出ない。
  それらは全区間runか、保存済み成果物を使う g3_terminal_replay.py で見る。

この接続を使ったrunは品質合格ではない。区間が本番と違う。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts import g3_closure_diagnostic as E

G = E.G
VARIABLE = 'G3_SHORT_END'
DEFAULT_END = 2000
ORIGINAL_END = 36300


def target_end() -> int:
    """環境変数で終端を決める。本番の36300より小さいことを強制する。"""
    value = os.environ.get(VARIABLE)
    end = DEFAULT_END if value is None else int(value)
    G.require(end > 0 and end % 2 == 0, 'short_range_even_positive')
    G.require(end < ORIGINAL_END, 'short_range_must_be_shorter')
    return end


def install(stack: Any, end: int) -> dict:
    """bind_bounds が読む G.END だけを縮める。FIRST・STRIDE・認識設定は触らない。"""
    G.require(G.END == ORIGINAL_END, 'short_range_original_end')
    G.require(G.FIRST == 0, 'short_range_original_first')
    previous = G.END
    stack.callback(setattr, G, 'END', previous)
    G.END = end
    return dict(original_end=previous, short_end=end, first=G.FIRST, stride=G.STRIDE,
                frames=len(range(G.FIRST, end, G.STRIDE)),
                content_dependent_defects_not_covered=['effect(frame15794)', 'M1(frame35160-)'],
                short_range_not_quality_pass=True, quality_gate_clear=False)


def main() -> int:
    """closure診断の前段に設置する。原G2ファイルと認識codeは変更しない。"""
    args = G.arguments()
    output = Path(args.output)
    plan = G.read(args.plan)
    G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'short_range_entry_pin')
    end = target_end()
    previous = G.protect_runtime
    state: dict = {}

    def protect(stack: Any, adapter: Any, main_module: Any, fixed: dict) -> None:
        # bind_bounds より前に縮める必要があるため、previous の呼出前に設置する。
        state.update(install(stack, end))
        previous(stack, adapter, main_module, fixed)

        def saved() -> None:
            G.save(output / 'G3_SHORT_RANGE.json', state)
        stack.callback(saved)
    G.protect_runtime = protect
    try:
        return E.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
