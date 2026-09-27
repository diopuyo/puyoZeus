"""同一E8入力をE10で再生し、E9比較の登録済み全指標を保存する。"""
from __future__ import annotations

from pathlib import Path

from scripts import run_e9_exchange_replay_20260927 as runner
from scripts.replay_exchange_event_20260926 import replay


def replay_original(record: Path, out: Path, model_dir: Path | None = None) -> dict:
    """指標基準はE9、入力は変更していないE8記録を使う。"""
    source = record.relative_to(Path("logs/e9"))
    return replay(Path("logs/e8") / source, out, model_dir)


def main() -> None:
    """E9と同じ集計関数・除外規則を適用する。"""
    runner.OUT, runner.BEFORE = Path("logs/e10"), Path("logs/e9")
    runner.metrics.FLIP_RATIO = 1.2
    runner.replay = replay_original
    runner.main()


if __name__ == "__main__":
    main()
