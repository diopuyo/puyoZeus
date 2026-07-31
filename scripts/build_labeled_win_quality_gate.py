"""#43 c 系 93 本の勝敗 JSON (WIN★パネル差分方式) にラベル品質ゲートを適用する。

## 背景
data/verify/winners_panel_diff_2026-07-26/video_cNN.json は
scripts.extract_match_winners --panel-diff-mode の出力 (GameRecord 互換)。
2先 (星印) 形式の動画はパネル差分イベントがほぼ出ないため、素通しすると
labeled_win への寄与がゼロ近くになったり、稀に出る誤検出増分がノイズに
なる。本スクリプトは動画単位・試合単位で品質ゲートをかけ、
scripts.label_win_from_winners がそのまま読める互換 JSON (winner=None で
無効化した試合を含む) を別ディレクトリへ書き出す。

## ゲート基準 (マジックナンバーは全て定数化)
1. 動画単位: 検出試合数が STAR_FORMAT_MAX_GAMES 以下 -> 2先(星印)形式疑い
   として "excluded_star_format" 扱い (星個数カウントの実装は行わない、
   user 指示によりこの段階では除外リスト記録のみで十分)。
2. 試合単位: 試合長が [MIN_GAME_DURATION_SEC, MAX_GAME_DURATION_SEC] の
   範囲外 -> UNKNOWN (winner=None に差し替え)。
3. 試合単位: confidence=="asymmetric" (extract_match_winners._resolve_confidence
   で「片側が支配的に増分していない」と判定された試合 = 両側同時増分疑いの
   代理指標) -> UNKNOWN (winner=None に差し替え)。
   既存の confidence 判定ロジックを再利用するだけで新規の閾値は追加しない。

## 出力
- 動画別ゲート済み JSON: <out-dir>/video_cNN.json (label_win_from_winners.py
  の --winners-dir にそのまま渡せるスキーマ)
- レポート (Markdown): 動画別ステータス (ok / excluded_star_format) +
  試合単位 UNKNOWN 件数

## 使い方
    python -m scripts.build_labeled_win_quality_gate \\
        --winners-diff-dir data/verify/winners_panel_diff_2026-07-26 \\
        --out-dir data/verify/winners_panel_diff_gated_2026-07-26 \\
        --report data/verify/labeled_win_c20_2026-07-26/quality_gate_report.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# 2先(星印)形式疑い: 検出試合数がこれ以下の動画は除外リスト行き
STAR_FORMAT_MAX_GAMES: int = 2

# 試合長の許容範囲 (秒)。範囲外は UNKNOWN 化。
MIN_GAME_DURATION_SEC: float = 20.0
MAX_GAME_DURATION_SEC: float = 600.0

# 両側同時増分疑いの代理指標として扱う confidence ラベル
# (extract_match_winners._resolve_confidence が付与する既存分類を流用)
AMBIGUOUS_CONFIDENCE_LABELS: frozenset[str] = frozenset({"asymmetric"})

DEFAULT_WINNERS_DIFF_DIR: str = "data/verify/winners_panel_diff_2026-07-26"
DEFAULT_OUT_DIR: str = "data/verify/winners_panel_diff_gated_2026-07-26"
DEFAULT_REPORT_PATH: str = "data/verify/labeled_win_c20_2026-07-26/quality_gate_report.md"
# 選定スクリプト (select_labeled_win_videos.py) が読む機械可読サマリ (TSV)
DEFAULT_MANIFEST_PATH: str = "data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv"


@dataclass
class VideoGateResult:
    """1 動画分のゲート結果。"""
    video_id: str
    status: str                # "ok" / "excluded_star_format"
    n_games_total: int
    n_games_unknown: int       # UNKNOWN 化された試合数 (status=="ok" の場合のみ意味を持つ)
    n_games_duration_bad: int
    n_games_ambiguous_conf: int


def _game_is_duration_bad(game: dict) -> bool:
    """試合長が許容範囲外なら True。"""
    duration = float(game["end_sec"]) - float(game["start_sec"])
    return duration < MIN_GAME_DURATION_SEC or duration > MAX_GAME_DURATION_SEC


def _game_is_ambiguous_confidence(game: dict) -> bool:
    """confidence が両側同時増分疑いに該当すれば True。"""
    return str(game.get("confidence", "")) in AMBIGUOUS_CONFIDENCE_LABELS


def gate_video(video_id: str, games: list[dict]) -> tuple[VideoGateResult, list[dict]]:
    """1 動画の games リストにゲートを適用し、(結果, ゲート済みgamesリスト) を返す。"""
    n_total = len(games)
    if n_total <= STAR_FORMAT_MAX_GAMES:
        result = VideoGateResult(
            video_id=video_id, status="excluded_star_format",
            n_games_total=n_total, n_games_unknown=n_total,
            n_games_duration_bad=0, n_games_ambiguous_conf=0,
        )
        # 除外動画も winner=None で書き出す (label_win 側はラベル無しとして自然に無視する)
        gated = [dict(g, winner=None, gate_reason="excluded_star_format") for g in games]
        return result, gated

    gated_games: list[dict] = []
    n_duration_bad = 0
    n_ambiguous = 0
    n_unknown = 0
    for game in games:
        g = dict(game)
        reason = None
        if _game_is_duration_bad(game):
            reason = "duration_out_of_range"
            n_duration_bad += 1
        elif _game_is_ambiguous_confidence(game):
            reason = "ambiguous_confidence"
            n_ambiguous += 1
        if reason is not None:
            g["winner"] = None
            g["gate_reason"] = reason
            n_unknown += 1
        gated_games.append(g)

    result = VideoGateResult(
        video_id=video_id, status="ok",
        n_games_total=n_total, n_games_unknown=n_unknown,
        n_games_duration_bad=n_duration_bad, n_games_ambiguous_conf=n_ambiguous,
    )
    return result, gated_games


def process_all(winners_diff_dir: Path, out_dir: Path) -> list[VideoGateResult]:
    """winners_diff_dir 内の全 JSON にゲートを適用して out_dir へ書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[VideoGateResult] = []
    for json_path in sorted(winners_diff_dir.glob("video_c*.json")):
        video_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        games = data.get("games", [])
        result, gated_games = gate_video(video_id, games)
        results.append(result)
        out_payload = {"video_id": video_id, "games": gated_games}
        out_path = out_dir / f"{video_id}.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(out_payload, fp, ensure_ascii=False, indent=2)
    return results


def write_report(results: list[VideoGateResult], report_path: Path) -> None:
    """動画別ステータスをまとめた Markdown レポートを書き出す。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(1 for r in results if r.status == "ok")
    n_star = sum(1 for r in results if r.status == "excluded_star_format")
    lines = [
        "# labeled_win 品質ゲート結果 (#43 c系)",
        "",
        f"- 処理動画数: {len(results)}",
        f"- 正常 (ok): {n_ok}",
        f"- 星形式疑い除外 (excluded_star_format): {n_star}",
        "",
        "| video_id | status | 試合数 | UNKNOWN化 | 試合長異常 | 同時増分疑い |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x.video_id):
        lines.append(
            f"| {r.video_id} | {r.status} | {r.n_games_total} | "
            f"{r.n_games_unknown} | {r.n_games_duration_bad} | {r.n_games_ambiguous_conf} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(results: list[VideoGateResult], manifest_path: Path) -> None:
    """選定スクリプト向けの機械可読 TSV サマリを書き出す (追加出力、既存出力は変更しない)。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header = "video_id\tstatus\tn_games_total\tn_games_unknown\tn_games_duration_bad\tn_games_ambiguous_conf"
    rows = [header]
    for r in sorted(results, key=lambda x: x.video_id):
        rows.append(
            f"{r.video_id}\t{r.status}\t{r.n_games_total}\t"
            f"{r.n_games_unknown}\t{r.n_games_duration_bad}\t{r.n_games_ambiguous_conf}"
        )
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="labeled_win 品質ゲート (#43 c系)")
    parser.add_argument("--winners-diff-dir", default=DEFAULT_WINNERS_DIFF_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    winners_diff_dir = Path(args.winners_diff_dir)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    manifest_path = Path(args.manifest)

    results = process_all(winners_diff_dir, out_dir)
    write_report(results, report_path)
    write_manifest(results, manifest_path)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_star = sum(1 for r in results if r.status == "excluded_star_format")
    print(f"[gate] 処理動画数: {len(results)}  正常: {n_ok}  星形式疑い除外: {n_star}")
    print(f"[gate] レポート: {report_path}")
    print(f"[gate] ゲート済みJSON出力先: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
