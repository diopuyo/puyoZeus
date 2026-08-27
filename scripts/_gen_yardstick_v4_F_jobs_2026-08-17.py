"""物差しv4ジョブファイルを本番構成F (production_config単一情報源) で再生成する (2026-08-17)。

## 目的
旧 `scripts/_jobs_yardstick_v4_2026-08-05.txt` は 2026-08-05 時点の認識強化
フラグ列を手打ちで並べていた (`--enable-effect-gate` 以降5個)。2026-08-17
までに RECOGNITION_ADOPTED (+ COLLECT_ONLY_ADOPTED) は13フラグに増えており、
手打ちのままでは新規採用フラグが漏れる (過去に何度も踏んだ配線漏れ事故と
同型、`src/production_config.py` docstring 参照)。

本スクリプトは `src.production_config.collect_flags()` を単一情報源として、
旧ジョブ行から「production_config が管理する認識フラグ (RECOGNITION_ADOPTED
+ COLLECT_ONLY_ADOPTED)」だけを機械的に剥がし、現在の `collect_flags()` の
出力に丸ごと置き換える。video/out-npz/start-sec/max-sec/sample-interval/
--with-next/--no-normalize-fps-30 等の「認識強化フラグでない」部分は一切
変更せず素通しする (物差しの窓設定を変えない=公平比較の前提)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._gen_yardstick_v4_F_jobs_2026-08-17
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.production_config import COLLECT_ONLY_ADOPTED, RECOGNITION_ADOPTED, collect_flags  # noqa: E402

OLD_JOBS_PATH: Path = Path("scripts/_jobs_yardstick_v4_2026-08-05.txt")
NEW_JOBS_PATH: Path = Path("scripts/_jobs_yardstick_v4_F_2026-08-17.txt")
OLD_OUT_DIR: str = "data/verify/board_labels_v4_yardstick_2026-08-05"
NEW_OUT_DIR: str = "data/verify/board_labels_v4F_yardstick_2026-08-17"


def _managed_flag_takes_value() -> dict[str, bool]:
    """production_config が単一情報源として管理するフラグ名 -> 値を取るか。"""
    out: dict[str, bool] = {}
    for f in RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED:
        parts = f.flag.split()
        out[parts[0]] = len(parts) > 1
    return out


def _strip_managed_flags(tokens: "list[str]", managed: "dict[str, bool]") -> "list[str]":
    """旧行トークン列から production_config 管理フラグ (と値) を除去する。"""
    kept: "list[str]" = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in managed:
            i += 2 if managed[tok] else 1
            continue
        kept.append(tok)
        i += 1
    return kept


def convert_line(line: str, managed: "dict[str, bool]") -> str:
    """旧ジョブ1行を構成F (production_config単一情報源) の1行に変換する。"""
    tokens = line.split()
    kept = _strip_managed_flags(tokens, managed)
    kept = [t.replace(OLD_OUT_DIR, NEW_OUT_DIR) for t in kept]
    return " ".join(kept) + " " + collect_flags()


def main() -> None:
    managed = _managed_flag_takes_value()
    old_lines = [
        l for l in OLD_JOBS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    new_lines = [convert_line(l, managed) for l in old_lines]
    NEW_JOBS_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[gen] {len(new_lines)} ジョブ -> {NEW_JOBS_PATH}")
    print(f"[gen] 出力先: {NEW_OUT_DIR}")
    print(f"[gen] collect_flags(): {collect_flags()}")
    print("[gen] サンプル行:")
    print(new_lines[0])


if __name__ == "__main__":
    main()
