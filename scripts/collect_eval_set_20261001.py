"""評価セット: zenchi セット1 の第1〜40試合 (パート1・2) を現本番の認識構成で記録する (2026-10-01)。

既存の第41〜57試合の記録 (exev logs/pending_expiry/full/records/zenchi) と同じ収集器・同じ手順:
run_pending_expiry_full.collect_with_both → collect_r1.collect(on) → collect_e34b.collect。
変えるのは区間 (--start-sec/--end-sec) と出力先だけ。認識フラグは production_config の
placement_reconcile_flags() から読み (手書きしない)、既存 zenchi 記録のコマンドと一致することを
起動前に確かめる (wiring_check)。

使い方 (cwd は exev の data/logs/models を参照する実行ディレクトリ /mnt/d/puyo_analyzer/evalset_run):
  python -m scripts.collect_eval_set_20261001 --part p1|p2|p3check
p3check は第41試合からの短区間で、既存の第3パート記録との先頭一致を確かめる測定器の検査用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Callable

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
OUT = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/collect')
BASE_SOURCE = 'zenchi'
EXISTING_RECORD_META = EXEV/'logs/pending_expiry/full/records/zenchi.jsonl.json'
# 区間 (秒)。パート境界はレビュー動画 manifest の resolved_start/end_sec。
PART3_START = 2580.566
CHECK_END = 2702.0   # 第41・42試合 (2579.066〜2701.266) を含む短区間
PARTS = dict(p1=(0.0, 1357.633), p2=(1357.633, PART3_START), p3check=(PART3_START, CHECK_END))
COLLECT_R1_FLAG = '--placement-signal-reconcile'   # collect_r1 が自分で末尾へ足すフラグ


def production_extra_flags() -> list[str]:
    """本番の置き完了合図照合フラグのうち、collect_r1 が足さない分 (順序は本番定義のまま)。"""
    from src import production_config
    flags = shlex.split(production_config.placement_reconcile_flags())
    assert COLLECT_R1_FLAG in flags
    return [flag for flag in flags if flag != COLLECT_R1_FLAG]


def set_interval(args: list[str], start: float, end: float) -> list[str]:
    """区間の2値だけを差し替える (他の引数は元のまま)。"""
    args = list(args)
    for flag, value in (('--start-sec', start), ('--end-sec', end)):
        args[args.index(flag)+1] = repr(float(value))
    return args


def part_command(original: Callable[[str], list[str]], name: str) -> list[str]:
    """e34c の収集コマンド (zenchi 系統の元記録コマンド) に区間と本番フラグを適用する。"""
    args = original(name) + production_extra_flags()
    if name in PARTS:
        args = set_interval(args, *PARTS[name])
    return args


def wiring_check(original: Callable[[str], list[str]]) -> dict:
    """zenchi 名・元区間で作ったコマンドが既存記録のコマンドと出力先以外で一致するか。"""
    from scripts.collect_e34c import OUTPUT_FLAGS
    built = part_command(original, BASE_SOURCE) + [COLLECT_R1_FLAG]
    recorded = json.loads(EXISTING_RECORD_META.read_text())['command']

    def strip(args: list[str]) -> list[str]:
        # 出力先の値だけ伏せる (入力動画・区間・全フラグは文字列として比べる)
        return ['<out>' if i and args[i-1] in OUTPUT_FLAGS else a for i, a in enumerate(args)]
    built_cmp, recorded_cmp = strip(built), strip(recorded)
    result = dict(identical=built_cmp == recorded_cmp, built=built_cmp, recorded=recorded_cmp)
    assert result['identical'], result
    return result


def install(part: str) -> None:
    """収集器のモジュール定数を、区間・出力先・参照元 (exev) だけ差し替える。"""
    from scripts import collect_e34b as capture
    from scripts import collect_e34c as checked
    from scripts import collect_r1, e34c_observations
    capture.ROOT = EXEV                      # 元記録の status.json と境界 TSV を exev から読む
    original, before = checked.command, checked.has_before_board
    check = wiring_check(original)
    checked.command = lambda name: part_command(original, name)
    checked.has_before_board = lambda name: before(BASE_SOURCE)
    # R1 の on 収集は全連鎖通知を画像窓の対象へ自分で加えるため、事前の発火一覧は空でよい
    # (p3check の既存記録との一致で、この差し替えが出力を変えないことを確かめる)。
    collect_r1.fire_keys = e34c_observations.fire_keys = lambda name: set()
    collect_r1.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{part}.wiring.json').write_text(json.dumps(check, ensure_ascii=False, indent=1))


def main() -> None:
    """1パートを1プロセスで収集する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=tuple(PARTS), required=True)
    args = parser.parse_args()
    from scripts import collect_r1, run_c65_guard_full
    assert json.loads(run_c65_guard_full.STAGE2_SUMMARY.read_text())['passed']
    install(args.part)
    collect_r1.collect(args.part, 'on')


if __name__ == '__main__':
    main()
