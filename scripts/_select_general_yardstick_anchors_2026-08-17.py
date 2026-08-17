"""一般分布ラベルセットの試合・時刻アンカーを選定する (2026-08-17)。

## 目的
構成F (本番採用構成) の「一般分布」正解率測定用に、難所条件で選別しない
無作為・層別 (動画×試合内進行率3分位×1P/2P) サンプルの元となる
「どの試合のどの時刻を見るか」を決定する。実際の収集ジョブ (構成F、
`src.production_config.collect_flags()` 単一情報源) もここで生成する。

## サンプリング方針
各動画から**確定境界 (confidence=strict) かつ十分な長さの試合を1本**選び、
その試合の [開始+マージン, 終了-マージン] 区間を早期/中盤/終盤の3分位に
均等分割した時刻を「アンカー」とする (両者STABLE前提の評価対象なので、
1P/2PそれぞれのSTABLE確定盤面を後段でこのアンカー付近から拾う)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._select_general_yardstick_anchors_2026-08-17
"""
from __future__ import annotations

import csv
import importlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.production_config import collect_flags  # noqa: E402

# 試合末尾の決着演出除外幅は既存の板 (_build_board_label_sheets_2026-07-31.py)
# と同じ値を再利用する (「決着局面を除外する」という定義を1箇所に保つ)。
_SHEETS = importlib.import_module("scripts._build_board_label_sheets_2026-07-31")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

VIDEO_IDS: "tuple[str, ...]" = tuple(f"c{i}" for i in range(10, 24))  # c10-c23
WINNERS_PANEL_DIR: Path = Path("data/verify/winners_panel_diff_2026-07-26")

# 試合選定条件: 短すぎる試合は3分位を置く余地が無い
MIN_GAME_DURATION_SEC: float = 45.0
# 試合開始直後の遷移演出を避けるマージン
START_MARGIN_SEC: float = 4.0
# 試合終了直前の決着演出は既存定義を再利用 (MATCH_END_EXCLUDE_SEC=8.0)
END_MARGIN_SEC: float = _SHEETS.MATCH_END_EXCLUDE_SEC

TERTILE_NAMES: "tuple[str, ...]" = ("early", "mid", "late")
TERTILE_FRACTIONS: "tuple[float, ...]" = (1.0 / 6.0, 1.0 / 2.0, 5.0 / 6.0)

RANDOM_SEED: int = 20260817

OUT_DIR: Path = Path("data/verify/board_labels_general_2026-08-17")
RAW_NPZ_DIR: Path = OUT_DIR / "raw_npz"
ANCHOR_PLAN_TSV: Path = OUT_DIR / "anchor_plan.tsv"
JOBS_FILE: Path = Path("scripts/_jobs_general_yardstick_F_2026-08-17.txt")


@dataclass(frozen=True)
class VideoGamePick:
    """1動画につき選ばれた1試合。"""

    video_id: str
    game_abs_idx: int
    start_sec: float
    end_sec: float


# =============================================================================
# 1. 試合選定
# =============================================================================


def _load_games(video_id: str) -> "list[dict]":
    path = WINNERS_PANEL_DIR / f"video_{video_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("games", [])


def pick_game_for_video(video_id: str, rng: random.Random) -> "VideoGamePick | None":
    """confidence=strict かつ十分な長さの試合から無作為に1本選ぶ。"""
    games = _load_games(video_id)
    cands = [
        g for g in games
        if g.get("confidence") == "strict"
        and float(g["end_sec"]) - float(g["start_sec"]) >= MIN_GAME_DURATION_SEC
    ]
    if not cands:
        return None
    g = rng.choice(cands)
    return VideoGamePick(video_id, int(g["game_abs_idx"]), float(g["start_sec"]), float(g["end_sec"]))


# =============================================================================
# 2. アンカー時刻の計算
# =============================================================================


def anchor_times_for_game(pick: VideoGamePick) -> "list[tuple[str, float]]":
    """[開始+マージン, 終了-マージン] を3分位した (tertile名, 時刻) のリスト。"""
    lo = pick.start_sec + START_MARGIN_SEC
    hi = pick.end_sec - END_MARGIN_SEC
    if hi <= lo:  # マージンを取れない短い試合はマージン無しにフォールバック
        lo, hi = pick.start_sec, pick.end_sec
    return [(name, lo + frac * (hi - lo)) for name, frac in zip(TERTILE_NAMES, TERTILE_FRACTIONS)]


# =============================================================================
# 3. 出力 (anchor_plan.tsv + 構成Fジョブファイル)
# =============================================================================


def _job_line(pick: VideoGamePick) -> str:
    """1動画1試合分の構成F収集コマンド (production_config単一情報源)。"""
    out_npz = RAW_NPZ_DIR / f"{pick.video_id}_g{pick.game_abs_idx}.npz"
    dur = pick.end_sec - pick.start_sec
    return (
        f"PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t "
        f"--video $HOME/frames/video_{pick.video_id}.mp4 --out-npz {out_npz} "
        f"--start-sec {pick.start_sec:.1f} --max-sec {dur:.1f} "
        f"--sample-interval 0 --with-next {collect_flags()}"
    )


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: "list[list[str]]" = []
    job_lines: "list[str]" = []
    n_no_candidate = 0
    for video_id in VIDEO_IDS:
        pick = pick_game_for_video(video_id, rng)
        if pick is None:
            n_no_candidate += 1
            print(f"[skip] {video_id}: strict かつ{MIN_GAME_DURATION_SEC}秒以上の試合が無い")
            continue
        job_lines.append(_job_line(pick))
        out_npz = RAW_NPZ_DIR / f"{pick.video_id}_g{pick.game_abs_idx}.npz"
        for tertile, t_sec in anchor_times_for_game(pick):
            rows.append([
                pick.video_id, str(pick.game_abs_idx), f"{pick.start_sec:.1f}",
                f"{pick.end_sec:.1f}", tertile, f"{t_sec:.2f}", str(out_npz),
            ])
    with ANCHOR_PLAN_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["video", "game_abs_idx", "game_start", "game_end", "tertile", "anchor_t_sec", "npz_path"])
        w.writerows(rows)
    JOBS_FILE.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    print(f"[1/2] アンカー計画: {len(rows)}行 ({len(job_lines)}動画) -> {ANCHOR_PLAN_TSV}")
    print(f"[2/2] 構成Fジョブ: {len(job_lines)}行 -> {JOBS_FILE}")
    if n_no_candidate:
        print(f"警告: {n_no_candidate}動画は試合候補無しでスキップされた (要確認)")


if __name__ == "__main__":
    main()
