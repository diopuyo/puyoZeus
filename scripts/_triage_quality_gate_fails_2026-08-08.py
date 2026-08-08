"""品質ゲート FAIL 動画のトリアージ: 誤認の中身 (色・行・列・時刻) を分解する.

scripts/phase_l_video_quality_gate.py の検査1 (試合開始+1〜3秒の空盤面に
非空セルが出る率) で FAIL した動画について、 「何色に化けたか」「どの行列か」
「どの試合か」を分解し、 user 裁定 + Phase L CNN seed 設計の材料にする。

判別したい仮説:
  H1 背景誤認 (空クラス未学習): 特定色に強く偏る + row 上部/端列に集中
  H2 おじゃま誤認: color=9 が多い
  H3 実際に置かれていた (試合開始判定のズレ): 色が散らばり row 下部に集中

出力: data/verify/phase_l_quality_gate_2026-08-07/triage_fails_2026-08-08.md
      + 代表フレーム抽出用の時刻一覧 TSV
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.phase_l_video_quality_gate import (  # noqa: E402
    MATCH_START_OFFSET_HI_SEC,
    MATCH_START_OFFSET_LO_SEC,
    MATCH_START_ROW_HI_EXCLUSIVE,
    MATCH_START_ROW_LO,
    DEFAULT_NPZ_DIR,
    DEFAULT_OUT_DIR,
    load_video_arrays,
)
from src.board import COLOR_EMPTY  # noqa: E402

# 色 ID -> 表示名 (CLAUDE.md 盤面データ表現)
COLOR_NAMES: dict[int, str] = {
    1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "おじゃま", 10: "UNKNOWN",
}
# 代表フレームとして書き出す上位件数 (動画あたり)
TOP_FRAMES_PER_VIDEO: int = 5


def _scorecard_fails(scorecard: Path) -> list[str]:
    """scorecard.tsv から FAIL 動画 ID を読む."""
    out: list[str] = []
    for i, line in enumerate(scorecard.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            continue
        cols = line.split("\t")
        if len(cols) > 1 and cols[1] == "FAIL":
            out.append(cols[0])
    return out


def _collect_nonempty(arrays) -> tuple[Counter, Counter, Counter, list[tuple]]:
    """検査1 と同じ窓で非空セルを分解する.

    戻り値: (色 Counter, 行 Counter, 列 Counter, スナップショット別件数一覧)
        スナップショット別件数一覧の要素 = (非空セル数, side, t_sec, game_idx)
    """
    by_color: Counter = Counter()
    by_row: Counter = Counter()
    by_col: Counter = Counter()
    snaps: list[tuple] = []
    for g in np.unique(arrays.game_idx):
        game_mask = arrays.game_idx == g
        if not game_mask.any():
            continue
        start_sec = arrays.t_sec[game_mask].min()
        lo = start_sec + MATCH_START_OFFSET_LO_SEC
        hi = start_sec + MATCH_START_OFFSET_HI_SEC
        for s in ("1P", "2P"):
            m = (
                game_mask & (arrays.side == s)
                & (arrays.t_sec >= lo) & (arrays.t_sec <= hi)
            )
            if not m.any():
                continue
            grids = arrays.grids[m][
                :, MATCH_START_ROW_LO:MATCH_START_ROW_HI_EXCLUSIVE, :
            ]
            times = arrays.t_sec[m]
            for k in range(grids.shape[0]):
                idx = np.argwhere(grids[k] != COLOR_EMPTY)
                if idx.size == 0:
                    continue
                for r, c in idx:
                    color = int(grids[k][r, c])
                    by_color[color] += 1
                    by_row[int(r) + MATCH_START_ROW_LO] += 1
                    by_col[int(c)] += 1
                snaps.append((int(idx.shape[0]), s, float(times[k]), int(g)))
    return by_color, by_row, by_col, snaps


def _fmt_counter(c: Counter, total: int, namer=None) -> str:
    """Counter を「名前 件数 (割合)」の 1 行文字列にする."""
    if total <= 0:
        return "(なし)"
    parts = []
    for k, v in c.most_common():
        label = namer(k) if namer else str(k)
        parts.append(f"{label}={v} ({v / total:.1%})")
    return ", ".join(parts)


def main() -> int:
    scorecard = DEFAULT_OUT_DIR / "scorecard.tsv"
    if not scorecard.exists():
        print(f"scorecard が無い: {scorecard}")
        return 1
    fails = _scorecard_fails(scorecard)
    md: list[str] = [
        "# 品質ゲート FAIL 動画トリアージ (2026-08-08)",
        "",
        "検査1 (試合開始+1〜3秒の空盤面) で非空と判定されたセルの内訳。",
        "背景誤認なら特定色偏り + 上部/端列集中、おじゃま誤認なら色9集中、",
        "試合開始判定ズレなら色が散らばり下部行に集中する。",
        "",
    ]
    frames_tsv: list[str] = ["video_id\tside\tt_sec\tgame_idx\tnonempty_cells"]
    for vid in fails:
        npz = DEFAULT_NPZ_DIR / f"{vid}.npz"
        if not npz.exists():
            md.append(f"## {vid}\n\n- npz 不在: {npz}\n")
            continue
        arrays = load_video_arrays(npz)
        by_color, by_row, by_col, snaps = _collect_nonempty(arrays)
        total = sum(by_color.values())
        snaps.sort(reverse=True)
        md.append(f"## {vid}")
        md.append("")
        md.append(f"- 非空セル総数: {total}")
        md.append(
            f"- 色内訳: {_fmt_counter(by_color, total, lambda k: COLOR_NAMES.get(k, str(k)))}"
        )
        md.append(f"- 行内訳 (row1=可視最上段): {_fmt_counter(by_row, total)}")
        md.append(f"- 列内訳 (col0=左端): {_fmt_counter(by_col, total)}")
        md.append(f"- 該当スナップショット数: {len(snaps)}")
        md.append("- 代表 (非空セル数が多い順):")
        for n, side, t, g in snaps[:TOP_FRAMES_PER_VIDEO]:
            md.append(f"    - {side} t={t:.2f}s game={g} cells={n}")
            frames_tsv.append(f"{vid}\t{side}\t{t:.2f}\t{g}\t{n}")
        md.append("")
    out_md = DEFAULT_OUT_DIR / "triage_fails_2026-08-08.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    out_tsv = DEFAULT_OUT_DIR / "triage_fails_frames_2026-08-08.tsv"
    out_tsv.write_text("\n".join(frames_tsv) + "\n", encoding="utf-8")
    print(f"出力: {out_md}")
    print(f"出力: {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
