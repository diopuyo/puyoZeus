"""AIへ返す出力を制限し、元ログと終了コードを保持する試作。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CHARS = 6000
DEFAULT_LINES = 80
READ_CHUNK = 8192
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "logs" / "agent_output_budget"


def positive(value: str) -> int:
    """上限の無効化につながるゼロと負数を拒否する。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("正の整数を指定してください")
    return number


def preview(path: Path, limit: int) -> tuple[str, int]:
    """固定メモリで文字数を数え、先頭と末尾を取得する。"""
    head_size = (limit + 1) // 2
    tail_size = limit // 2
    head, tail, total = "", "", 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        while chunk := stream.read(READ_CHUNK):
            total += len(chunk)
            head = (head + chunk)[:head_size]
            tail = (tail + chunk)[-limit:]
    if total <= limit:
        return tail, total
    return head + "\n…中略。全文は保存ログを参照…\n" + (tail[-tail_size:] if tail_size else ""), total


def emit_preview(path: Path, limit: int, code: int) -> None:
    """構造化メタデータと抜粋を出し、合否は解釈しない。"""
    excerpt, total = preview(path, limit)
    print(json.dumps({"log": str(path.resolve()), "exit_code": code,
                      "decoded_chars": total, "preview_char_budget": limit,
                      "truncated": total > limit}, ensure_ascii=False))
    print(excerpt)


def run_command(argv: list[str], log_dir: Path, limit: int) -> int:
    """一度だけ実行する。出力は生バイトで保存し、終了コードを維持する。"""
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ValueError("-- の後に実行ファイルと引数を指定してください")
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", prefix="run_", suffix=".log",
                                     dir=log_dir, delete=False) as stream:
        path = Path(stream.name)
        try:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                    shell=False, check=False)
            code = result.returncode
        except OSError as error:
            stream.write(f"起動失敗: {error}\n".encode("utf-8"))
            code = 127
    emit_preview(path, limit, code)
    return code


def show_lines(path: Path, start: int, count: int, limit: int) -> None:
    """指定範囲を文字数上限つきで読む。省略を既読完了と扱わない。"""
    remaining = limit
    exhausted = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, start=1):
            if number < start:
                continue
            if number >= start + count:
                break
            piece = f"{number}: {line}"
            print(piece[:remaining], end="")
            remaining -= min(len(piece), remaining)
            if remaining == 0:
                exhausted = True
                break
    print("\n" + json.dumps({"source": str(path.resolve()), "start_line": start,
                             "requested_lines": count,
                             "char_limit_reached": exhausted}, ensure_ascii=False))


def main() -> int:
    """実行と部分読み取りの共通入口。トークン数ではなく文字数を制限する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run")
    run.add_argument("--chars", type=positive, default=DEFAULT_CHARS)
    run.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    run.add_argument("command", nargs=argparse.REMAINDER)
    show = modes.add_parser("show")
    show.add_argument("path", type=Path)
    show.add_argument("--start", type=positive, default=1)
    show.add_argument("--lines", type=positive, default=DEFAULT_LINES)
    show.add_argument("--chars", type=positive, default=DEFAULT_CHARS)
    args = parser.parse_args()
    if args.mode == "run":
        return run_command(args.command, args.log_dir, args.chars)
    show_lines(args.path, args.start, args.lines, args.chars)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
