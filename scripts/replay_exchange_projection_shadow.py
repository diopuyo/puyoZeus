"""保存した評価入力 (inputs.jsonl.gz) だけで、撃ち合い後のフィールド予測の影ログを再生する。

評価器は使わない (認識結果・おじゃま会計・時刻・試合番号だけを影ログのラッパーへ流す)。
出力は JSONL。重い処理は BelowNormal で 1 本ずつ実行すること。
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.b3_timing import TimingTable  # noqa: E402
from src.exchange_event_record import decode  # noqa: E402
from src.exchange_projection import DEFAULT_BUDGET  # noqa: E402
from src.exchange_projection_shadow import ExchangeProjectionShadow, jsonl_sink  # noqa: E402


def updates(path: Path, games: Optional[set[int]] = None) -> Iterator[tuple]:
    """保存入力の update 行 (評価器へ渡した引数) を順に復元する。未完了 gzip の末尾は読まない。"""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        while True:
            try:
                line = stream.readline()
            except EOFError:
                return
            if not line:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return
            if row.get("kind") != "update":
                continue
            args = decode(row["args"])
            if games is None or args[4] in games:
                yield args


def ledger_pending(review_csv: Path):
    """review_data.csv の予告 (1P_incoming / 2P_incoming、得点台帳の値) を時刻で引く関数を作る。

    保存した評価入力のおじゃま会計は台帳の予告と食い違う (10/13 夜 g16 は 1,074 が 5 と記録されていた)
    ので、再生では記録済みの台帳値を使う。直前の行 (同じ試合) を保持する。
    """
    import csv
    from bisect import bisect_right
    rows = []
    with review_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            def num(value: str) -> int:
                return int(float(value)) if value not in ("", None) else 0
            rows.append((float(row["t_sec"]), int(float(row["game_idx"])), num(row["1P_incoming"]), num(row["2P_incoming"])))
    rows.sort()
    times = [r[0] for r in rows]

    def pending(_snapshot, t_sec: float, game_idx: int) -> tuple[int, int]:
        index = bisect_right(times, t_sec) - 1
        if index < 0 or rows[index][1] != game_idx:
            return 0, 0
        return rows[index][2], rows[index][3]
    return pending


def lower_priority() -> None:
    """重い処理は BelowNormal (CLAUDE.md の運用)。Windows 以外では何もしない。"""
    if sys.platform != "win32":
        return
    import ctypes
    below_normal = 0x00004000
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-table", type=Path, default=None)
    parser.add_argument("--games", type=int, nargs="*", default=None)
    parser.add_argument("--expected-hand-sec", type=float, default=None)
    parser.add_argument("--snapshot-pending", action="store_true", help="予告を保存入力のおじゃま会計から読む")
    parser.add_argument("--expected-table", type=Path, default=None, help="撃ち合いの窓で較正した手時間表 (期待値側)")
    parser.add_argument("--ideal-width", type=int, default=DEFAULT_BUDGET.ideal_beam_width)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_BUDGET.max_native_calls)
    args = parser.parse_args(argv)
    lower_priority()
    table = TimingTable.from_json(args.timing_table.read_text(encoding="utf-8")) if args.timing_table else None
    budget = DEFAULT_BUDGET.__class__(max_native_calls=args.max_calls, ideal_beam_width=args.ideal_width)
    if args.out.exists():
        args.out.unlink()
    sink = jsonl_sink(args.out)
    review = args.inputs.parent / "review_data.csv"
    pending_fn = ledger_pending(review) if review.exists() and not args.snapshot_pending else None
    expected = TimingTable.from_json(args.expected_table.read_text(encoding="utf-8")) if args.expected_table else None
    shadow = ExchangeProjectionShadow(sink, table, budget, args.expected_hand_sec, pending_fn=pending_fn,
                                      expected_table=expected)
    started, count, last_t = time.perf_counter(), 0, 0.0
    for result, snapshot, _counts, t_sec, game_idx, *_rest in updates(args.inputs, set(args.games) if args.games else None):
        shadow.observe(result, snapshot, t_sec, game_idx)
        count += 1
        last_t = t_sec
    shadow.finish_game(last_t)
    sink.close()  # type: ignore[attr-defined]
    print(json.dumps(dict(updates=count, seconds=round(time.perf_counter() - started, 1),
                          events=shadow.event_count), ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
