"""勝敗パネル境界データから「処理すべき試合区間」を汎用に決定するユーティリティ
(2026-07-30、タスク1: 試合外画面を処理しない)。

## 背景・実測結果 (2026-07-30)
`data/verify/winners_panel_diff_2026-07-26/video_*.json` の試合境界を
combined66 (66動画) で調べたところ、**同一動画内で隣接する試合同士の
ギャップは全動画で 0.0 秒** (前の試合の end_sec == 次の試合の start_sec)
だった。つまり「試合間のメニュー/結果画面」を個別に切り出して飛ばせる余地は
実質無く、削減できるのは **動画冒頭 (最初の試合が始まる前のオープニング) と
動画末尾 (最後の試合が終わった後のエンディング)** の2箇所のみ。
実測 (combined66・66動画): 全尺 62.18 時間に対し試合区間 (全試合ベース)
57.02 時間 = 削減率 8.3%、strict試合限定ベースでは削減率 13.5%。
「3〜4割が試合外」という前提は本データでは支持されない (要 user 確認)。

## 状態機械への影響 (2026-07-30 評価)
本ユーティリティが生成する区間は「動画1本につき1区間」(冒頭末尾クリップの
み) であり、区間内部の試合間遷移は無編集のまま連続処理される。これは
現行の collect_indicators_v2.py 本番経路 (動画全体を1プロセスで連続処理し、
試合間は score-reset 検知 (`recognition_pipeline.py` の
`_maybe_reset_on_score_reset` 系, line ~2899) が RecognitionPipeline.reset()
を呼んで内部 state をクリアする設計) と全く同じ挙動であり、新規リスクは
無い。唯一の違いは「cold-start (STABLE_WARMUP_FRAMES 等の初期化コスト) が
発生する開始点が動画冒頭 (オープニング演出) から最初の試合開始秒に前倒しに
なる」点のみ。force_in_match=True 構成ではオープニング演出中も
is_match_active=True 扱いで CNN 推論が走り続けるため、この区間を丸ごと
スキップすることで「試合外画面の内容が history バッファ等を汚染してから
1試合目が始まる」経路自体が消える (副次的に品質改善の可能性、未検証)。

## 使い方
    python -m scripts.select_match_regions \\
        --video-ids-csv data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \\
        --out-report-csv data/verify/match_regions_2026-07-30/regions.csv \\
        --out-jobs-txt scripts/_jobs_match_regions_2026-07-30.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

# --- 定数 (マジックナンバー禁止のため明示) ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL_DIR = PROJECT_ROOT / "data/verify/winners_panel_diff_2026-07-26"
DEFAULT_FRAMES_DIR = "$HOME/frames"  # ext4 コピー先 (9p I/O ボトルネック回避)
DEFAULT_COLLECT_MODULE = "scripts._collect_lean_1t"

# 境界検知誤差を吸収する安全マージン (既定 0.0 = マージン無し、後方互換)。
DEFAULT_HEAD_MARGIN_SEC: float = 0.0
DEFAULT_TAIL_MARGIN_SEC: float = 0.0


# ============================
# 純粋関数 (stateless、CLAUDE.md 原則)
# ============================

def compute_video_match_span(
    games: list[dict[str, Any]],
    confidence_filter: Optional[set[str]] = None,
    head_margin_sec: float = DEFAULT_HEAD_MARGIN_SEC,
    tail_margin_sec: float = DEFAULT_TAIL_MARGIN_SEC,
) -> Optional[tuple[float, float, int]]:
    """1動画分の games 配列から処理すべき区間 [start_sec, end_sec] を決める。

    winners_panel_diff の 'games' 配列 (game_abs_idx 順である必要はない) を
    受け取り、対象試合の中で最小の start_sec と最大の end_sec を返す。
    これにより「動画冒頭のオープニング」「動画末尾のエンディング」だけを
    除外した1区間が得られる (試合間ギャップは実測上ほぼ0秒のため、個々の
    試合をさらに分割してもスキップできる時間はほぼ増えない、2026-07-30 実測)。

    Args:
        games: winners_panel_diff JSON の 'games' 配列。各要素は最低限
            'start_sec' / 'end_sec' / 'confidence' キーを持つ dict。
        confidence_filter: 指定した confidence 値集合の試合のみを対象にする
            (例: {"strict"})。省略時 (None) は全試合を対象にする
            (= 削減幅が最大、boundary confidence が低くても実際の試合時間
            であることに変わりはないため既定はこちら)。
        head_margin_sec: 先頭に残す安全マージン秒 (既定0.0、後方互換)。
        tail_margin_sec: 末尾に残す安全マージン秒 (既定0.0、後方互換)。

    Returns:
        (start_sec, end_sec, 対象試合数)。対象試合が無ければ None。
    """
    eligible = games if confidence_filter is None else [
        g for g in games if g.get("confidence") in confidence_filter
    ]
    if not eligible:
        return None
    start = max(0.0, min(g["start_sec"] for g in eligible) - head_margin_sec)
    end = max(g["end_sec"] for g in eligible) + tail_margin_sec
    return start, end, len(eligible)


def merge_contiguous_games(
    games: list[dict[str, Any]],
    max_gap_sec: float = 0.0,
) -> list[tuple[float, float, list[int]]]:
    """試合間ギャップが max_gap_sec 以下なら1区間に併合する (汎用フォールバック)。

    combined66 では試合間ギャップが実測 0.0 秒だったため既定 (max_gap_sec=0.0)
    では動画1本が丸ごと1区間になる (compute_video_match_span と等価)。
    将来ギャップが実際に存在する動画が見つかった場合に備え、区間ごとに
    プロセスを分けたい用途向けに汎用実装として残す (2026-07-30 追加)。

    Args:
        games: game_abs_idx 昇順の games 配列。
        max_gap_sec: この秒数以下のギャップは同一区間とみなして併合する。

    Returns:
        [(start_sec, end_sec, [game_abs_idx, ...]), ...] (時系列順)。
    """
    ordered = sorted(games, key=lambda g: g["start_sec"])
    if not ordered:
        return []
    spans: list[tuple[float, float, list[int]]] = []
    cur_start = ordered[0]["start_sec"]
    cur_end = ordered[0]["end_sec"]
    cur_idxs = [ordered[0]["game_abs_idx"]]
    for g in ordered[1:]:
        gap = g["start_sec"] - cur_end
        if gap <= max_gap_sec:
            cur_end = max(cur_end, g["end_sec"])
            cur_idxs.append(g["game_abs_idx"])
        else:
            spans.append((cur_start, cur_end, cur_idxs))
            cur_start, cur_end, cur_idxs = (
                g["start_sec"], g["end_sec"], [g["game_abs_idx"]],
            )
    spans.append((cur_start, cur_end, cur_idxs))
    return spans


# ============================
# I/O ヘルパ
# ============================

def load_games_for_video(panel_dir: Path, video_id: str) -> list[dict[str, Any]]:
    """1動画分の winners_panel_diff JSON を読み込み games 配列を返す。"""
    path = panel_dir / f"{video_id}.json"
    with path.open(encoding="utf-8") as f:
        panel = json.load(f)
    return panel["games"]


def load_video_ids_from_csv(csv_path: Path) -> list[str]:
    """labeled_win_*.csv の video_id 列からユニーク一覧を得る (昇順)。"""
    video_ids: set[str] = set()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.add(row["video_id"])
    return sorted(video_ids)


# ============================
# レポート・ジョブ生成
# ============================

def build_region_rows(
    video_ids: list[str],
    panel_dir: Path,
    confidence_filter: Optional[set[str]] = None,
    head_margin_sec: float = DEFAULT_HEAD_MARGIN_SEC,
    tail_margin_sec: float = DEFAULT_TAIL_MARGIN_SEC,
) -> tuple[list[dict[str, Any]], list[str]]:
    """全動画について区間を決定し、(レポート行, 決定不能動画一覧) を返す。"""
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for vid in video_ids:
        games = load_games_for_video(panel_dir, vid)
        span = compute_video_match_span(
            games, confidence_filter, head_margin_sec, tail_margin_sec,
        )
        if span is None:
            skipped.append(vid)
            continue
        start, end, n_games = span
        rows.append({
            "video_id": vid,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "dur_sec": round(end - start, 3),
            "n_games": n_games,
        })
    return rows, skipped


def write_report_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """区間決定結果を CSV で書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "start_sec", "end_sec", "dur_sec", "n_games"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jobs_txt(
    rows: list[dict[str, Any]],
    out_path: Path,
    out_npz_dir: str,
    frames_dir: str = DEFAULT_FRAMES_DIR,
    collect_module: str = DEFAULT_COLLECT_MODULE,
    extra_flags: str = "--enable-chain-tracker --with-next",
) -> None:
    """動画1本=1区間=1ジョブのコマンド行を書き出す (既存 collect_boards_lean
    の --start-sec / --max-sec をそのまま再利用、コア収集スクリプトへの変更は
    不要)。

    Args:
        rows: build_region_rows の戻り値。
        out_path: ジョブ定義ファイル出力先。
        out_npz_dir: 出力 npz を置くディレクトリ (video_id ごとに1ファイル)。
        frames_dir: 入力動画を探すディレクトリ (既定は ext4 コピー先)。
        collect_module: 呼び出す収集モジュール (既定 _collect_lean_1t)。
        extra_flags: 収集コマンドに付与する追加フラグ。
    """
    lines = []
    for r in rows:
        vid = r["video_id"]
        video_path = f"{frames_dir}/{vid}.mp4"
        out_npz = f"{out_npz_dir}/{vid}.npz"
        cmd = (
            f"PYTHONPATH=. ./venv/bin/python -u -m {collect_module} "
            f"--video {video_path} --out-npz {out_npz} "
            f"--start-sec {r['start_sec']} --max-sec {r['dur_sec']} "
            f"{extra_flags}"
        )
        lines.append(cmd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # LF 固定 (feedback_crlf_churn_before_push: CRLF 混入で bash 側 argparse が壊れる事故対策)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ============================
# CLI エントリポイント
# ============================

def main() -> int:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-ids-csv", type=Path, required=True,
        help="video_id 列を含む labeled_win_*.csv (例: combined66)",
    )
    parser.add_argument(
        "--panel-dir", type=Path, default=DEFAULT_PANEL_DIR,
        help="winners_panel_diff JSON が置かれたディレクトリ",
    )
    parser.add_argument(
        "--out-report-csv", type=Path, required=True,
        help="区間決定結果レポート CSV の出力先",
    )
    parser.add_argument(
        "--out-jobs-txt", type=Path, default=None,
        help="ジョブ定義ファイルの出力先 (省略時は生成しない)",
    )
    parser.add_argument(
        "--out-npz-dir", type=str, default="data/indicators_v2/boards_lean_regions",
        help="--out-jobs-txt 指定時の npz 出力先ディレクトリ",
    )
    parser.add_argument(
        "--strict-only", action="store_true",
        help="confidence=strict の試合のみを対象にする (既定は全confidence対象、削減幅最大)",
    )
    parser.add_argument(
        "--head-margin-sec", type=float, default=DEFAULT_HEAD_MARGIN_SEC,
        help="先頭に残す安全マージン秒 (既定0.0)",
    )
    parser.add_argument(
        "--tail-margin-sec", type=float, default=DEFAULT_TAIL_MARGIN_SEC,
        help="末尾に残す安全マージン秒 (既定0.0)",
    )
    args = parser.parse_args()

    video_ids = load_video_ids_from_csv(args.video_ids_csv)
    confidence_filter = {"strict"} if args.strict_only else None
    rows, skipped = build_region_rows(
        video_ids, args.panel_dir, confidence_filter,
        args.head_margin_sec, args.tail_margin_sec,
    )
    write_report_csv(rows, args.out_report_csv)
    if args.out_jobs_txt is not None:
        write_jobs_txt(rows, args.out_jobs_txt, args.out_npz_dir)

    total_clipped_sec = sum(r["dur_sec"] for r in rows)
    print(f"[select_match_regions] 動画数: {len(video_ids)} (区間決定 {len(rows)} / 決定不能 {len(skipped)})")
    print(f"[select_match_regions] 総試合区間秒数: {total_clipped_sec:.1f} 秒")
    if skipped:
        print(f"[select_match_regions][WARN] 区間決定不能な動画: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
