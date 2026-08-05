"""長時間劣化 規模測定 Lv0+Lv1: 列凍結異常のスキャン (2026-08-06、追加収集ゼロ)。

docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §Lv0/Lv1。
`data/indicators_v2/boards_lean_regen_2026-07-31/*.npz` (既存収集済みデータ、
新規動画再取得なし) を走査し、c22型「列凍結」(OnlineHsvCalibrator 較正が
初回inject後に凍結し、以後ある列が実際は積まれているのに空と誤認され続ける
症状の仮説) の疑わしい区間を検出する。

## 検出定義
1. **列凍結 (主)**: (video, side, game_idx) 単位で、ある列 c が rows1-12
   (隠し段row0は対象外、確定盤面ではないため) 全て EMPTY のまま連続する
   最大区間を探す。区間が
     (a) 持続 >= COLUMN_FREEZE_MIN_DURATION_SEC (30秒、coordinator指定の
         目安) を満たす、または
     (b) 区間内で他列合計セル数が COLUMN_FREEZE_BURST_GROWTH_THRESHOLD
         以上増加した (短時間で急激に積みが進んだ、目安未満でも疑わしい)
   のいずれかを満たし、かつ両ケース共通で他列合計が
   COLUMN_FREEZE_MIN_OTHER_GROWTH 以上増えている (=「他列には積みが進行
   している」の直接条件) 場合に検出する。
2. **重力逆転 (副)**: 同一列内で、浅い行 (row小=盤面上部) に非空セルが
   あるのに、その直下の行が EMPTY という物理的にあり得ない状態
   (`project_gravity_violation_regen_lead_2026-07-30` の浮きぷよ定義と
   同じ) が GRAVITY_VIOLATION_MIN_PERSIST_SEC 秒以上連続する区間。

## false positive の既知の懸念 (docstring内で先に自己申告)
- 試合開始直後の「まだ誰もその列に置いていない」自然な空列を、直後の
  急速な連鎖清算 (他列が一時的に減る) と組み合わせて誤検出しうる。
- 連鎖の「清算→再構築」自体が一時的な列の不均衡を生むため、
  COLUMN_FREEZE_BURST_GROWTH_THRESHOLD が低すぎると通常プレイでも
  ヒットする恐れがある (本スクリプトの数値は物理量からの見積りであり、
  個別シーンからの逆算ではないが、Lv2の3-5動画再構築での精度校正が
  別途必要 = 本スキャンは「規模の目安」であり確定判定ではない)。
- npz は STABLE 確定snapshotのみのため、frame間の間引きにより短い凍結が
  スナップショット密度不足で見えなくなる可能性がある (過小検出方向)。

## 既存資産の再利用
なし (新規の走査ロジックのため純粋に numpy ベースで実装、既存の突合系
スクリプトとはデータ形式が異なるため流用対象がない)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._scan_longrun_column_freeze_2026-08-06
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUTPUT_DIR: Path = Path("data/verify/longrun_degradation_2026-08-06")
INTERVALS_CSV: Path = OUTPUT_DIR / "column_freeze_intervals.csv"
GRAVITY_CSV: Path = OUTPUT_DIR / "gravity_violation_intervals.csv"
VIDEO_SUMMARY_CSV: Path = OUTPUT_DIR / "video_summary.csv"

BOARD_ROWS: int = 13
BOARD_COLS: int = 6
VISIBLE_ROWS: range = range(1, BOARD_ROWS)  # row1-12、隠し段row0は対象外
COLOR_EMPTY: int = 0

# 列凍結 主条件 (coordinator指定の目安をそのまま採用)。
COLUMN_FREEZE_MIN_DURATION_SEC: float = 30.0
# 短時間でも急激に他列が積まれた場合の代替条件 (6手分=12個相当を目安、
# 1手=2個×典型6手の物理量見積り、個別シーンからの逆算ではない)。
COLUMN_FREEZE_BURST_GROWTH_THRESHOLD: int = 12
# 両条件共通の必須ゲート:「他列に積みが進行している」の最小量
# (ノイズ的な±1個変動を除外する目安)。
COLUMN_FREEZE_MIN_OTHER_GROWTH: int = 2

# 重力逆転 (浮きぷよ) の持続最小秒数 (瞬間的な遷移ノイズを除外)。
GRAVITY_VIOLATION_MIN_PERSIST_SEC: float = 1.0

# 開幕アーティファクト除外 (実測で発覚、2026-08-06): 試合開始直後は
# 「その列にまだ誰も置いていない」だけの自然な空列が数十〜200秒続くことが
# あり、これが列凍結条件 (a) を素朴に満たしてしまう (実測: 高信頼ヒット
# 395件中246件がgame_idx=0、うち209件がgame開始5秒以内に始まる区間 =
# 62.3%が開幕アーティファクトと推定される)。区間の開始時刻がgame内の
# 最初のsnapshotからこの秒数未満なら「開幕アーティファクト濃厚」として
# 別集計する (検出そのものは除外しない、集計を分けるのみ)。
OPENING_ARTIFACT_MAX_START_OFFSET_SEC: float = 5.0


# =============================================================================
# データ構造
# =============================================================================


@dataclass(frozen=True)
class ColumnFreezeInterval:
    """列凍結の疑わしい区間 1件。"""

    video: str
    side: str
    game_idx: int
    column: int
    start_t: float
    end_t: float
    duration_sec: float
    other_growth: int
    trigger: str  # "duration" / "burst_growth" / "both"
    game_start_offset_sec: float  # 区間開始 - このgameの最初のsnapshot時刻

    @property
    def is_opening_artifact(self) -> bool:
        """開幕直後 (=このgame自体の最初のsnapshotから間もない) 区間か。"""
        return self.game_start_offset_sec < OPENING_ARTIFACT_MAX_START_OFFSET_SEC


@dataclass(frozen=True)
class GravityViolationInterval:
    """重力逆転 (浮きぷよ) の持続区間 1件。"""

    video: str
    side: str
    game_idx: int
    column: int
    start_t: float
    end_t: float
    duration_sec: float


# =============================================================================
# 1. 列凍結検出 (1 group = 1 video×side×game_idx分)
# =============================================================================


def _col_nonempty_counts(grids: "np.ndarray") -> "np.ndarray":
    """(N,13,6) グリッド列から (N,6) の列別非空セル数 (row1-12限定) を返す。"""
    visible = grids[:, list(VISIBLE_ROWS), :]  # (N, 12, 6)
    return (visible != COLOR_EMPTY).sum(axis=1)  # (N, 6)


def _find_freeze_runs_for_column(
    col_counts: "np.ndarray", other_counts: "np.ndarray", t_secs: "np.ndarray",
) -> list[tuple[int, int, int]]:
    """1列分: 連続 EMPTY (count==0) の最大区間を全て (start_idx,end_idx,growth) で返す。"""
    is_empty = col_counts == 0
    runs: list[tuple[int, int, int]] = []
    start: "int | None" = None
    for i, empty in enumerate(is_empty):
        if empty and start is None:
            start = i
        elif not empty and start is not None:
            growth = int(other_counts[start:i].max() - other_counts[start])
            runs.append((start, i - 1, growth))
            start = None
    if start is not None:
        growth = int(other_counts[start:].max() - other_counts[start])
        runs.append((start, len(is_empty) - 1, growth))
    return runs


def scan_column_freeze_for_group(
    video: str, side: str, game_idx: int, grids: "np.ndarray", t_secs: "np.ndarray",
) -> list[ColumnFreezeInterval]:
    """1 (video,side,game_idx) 分の列凍結区間を全列について検出する。"""
    counts = _col_nonempty_counts(grids)  # (N, 6)
    total = counts.sum(axis=1)
    out: list[ColumnFreezeInterval] = []
    for c in range(BOARD_COLS):
        other = total - counts[:, c]
        for start_i, end_i, growth in _find_freeze_runs_for_column(counts[:, c], other, t_secs):
            if growth < COLUMN_FREEZE_MIN_OTHER_GROWTH:
                continue
            duration = float(t_secs[end_i] - t_secs[start_i])
            ok_duration = duration >= COLUMN_FREEZE_MIN_DURATION_SEC
            ok_burst = growth >= COLUMN_FREEZE_BURST_GROWTH_THRESHOLD
            if not (ok_duration or ok_burst):
                continue
            trigger = "both" if (ok_duration and ok_burst) else (
                "duration" if ok_duration else "burst_growth"
            )
            offset = float(t_secs[start_i] - t_secs[0])
            out.append(ColumnFreezeInterval(
                video, side, game_idx, c,
                float(t_secs[start_i]), float(t_secs[end_i]), duration, growth, trigger,
                offset,
            ))
    return out


# =============================================================================
# 2. 重力逆転 (浮きぷよ持続) 検出
# =============================================================================


def _floating_columns_per_snapshot(grids: "np.ndarray") -> "np.ndarray":
    """(N,13,6) から (N,6) の bool: 各snapshot×列で浮きぷよがあるか。

    隣接する2行 (row=r, r+1、r+1がより深い=下) のどこかで
    「上が非空・直下が空」ならその列は物理的にあり得ない状態
    (`project_gravity_violation_regen_lead_2026-07-30` の浮きぷよ定義)。
    """
    visible = grids[:, list(VISIBLE_ROWS), :]  # (N, 12, 6) row1(浅)..row12(深)
    is_empty = visible == COLOR_EMPTY
    out = np.zeros((grids.shape[0], BOARD_COLS), dtype=bool)
    for r in range(len(VISIBLE_ROWS) - 1):
        above_nonempty = ~is_empty[:, r, :]
        below_empty = is_empty[:, r + 1, :]
        out |= above_nonempty & below_empty
    return out


def scan_gravity_violation_for_group(
    video: str, side: str, game_idx: int, grids: "np.ndarray", t_secs: "np.ndarray",
) -> list[GravityViolationInterval]:
    """1 (video,side,game_idx) 分の重力逆転持続区間を全列について検出する。"""
    floating = _floating_columns_per_snapshot(grids)  # (N, 6)
    out: list[GravityViolationInterval] = []
    for c in range(BOARD_COLS):
        col_flag = floating[:, c]
        start: "int | None" = None
        for i, flag in enumerate(col_flag):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                _append_gravity_run(out, video, side, game_idx, c, t_secs, start, i - 1)
                start = None
        if start is not None:
            _append_gravity_run(out, video, side, game_idx, c, t_secs, start, len(col_flag) - 1)
    return out


def _append_gravity_run(
    out: list[GravityViolationInterval], video: str, side: str, game_idx: int,
    column: int, t_secs: "np.ndarray", start_i: int, end_i: int,
) -> None:
    """持続秒数がフィルタ以上の重力逆転区間のみ追加する (共通ヘルパ)。"""
    duration = float(t_secs[end_i] - t_secs[start_i])
    if duration >= GRAVITY_VIOLATION_MIN_PERSIST_SEC:
        out.append(GravityViolationInterval(
            video, side, game_idx, column, float(t_secs[start_i]), float(t_secs[end_i]), duration,
        ))


# =============================================================================
# 3. 1動画分の走査 (video×side×game_idx へグルーピング)
# =============================================================================


def scan_one_video(npz_path: Path) -> "tuple[list[ColumnFreezeInterval], list[GravityViolationInterval]]":
    """1 npz (1動画) 分を (video,side,game_idx) にグルーピングして両検出を走査する。"""
    data = np.load(npz_path, allow_pickle=True)
    video = npz_path.stem
    grids, sides, game_idxs, t_secs = data["grids"], data["side"], data["game_idx"], data["t_sec"]
    freeze_out: list[ColumnFreezeInterval] = []
    gravity_out: list[GravityViolationInterval] = []
    for side in ("1P", "2P"):
        side_mask = sides == side
        for game_idx in sorted(set(game_idxs[side_mask].tolist())):
            mask = side_mask & (game_idxs == game_idx)
            idx_sorted = np.argsort(t_secs[mask])
            g = grids[mask][idx_sorted]
            t = t_secs[mask][idx_sorted]
            if len(t) < 2:
                continue
            freeze_out.extend(scan_column_freeze_for_group(video, side, int(game_idx), g, t))
            gravity_out.extend(scan_gravity_violation_for_group(video, side, int(game_idx), g, t))
    return freeze_out, gravity_out


def scan_all_videos(npz_dir: Path) -> "tuple[list[ColumnFreezeInterval], list[GravityViolationInterval]]":
    """npz_dir 配下の全動画を走査する (fail-silent回避、読込失敗はログ出力)。"""
    all_freeze: list[ColumnFreezeInterval] = []
    all_gravity: list[GravityViolationInterval] = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        try:
            freeze, gravity = scan_one_video(npz_path)
        except Exception as e:
            print(f"  [警告] 読込失敗 {npz_path.name}: {e}")
            continue
        all_freeze.extend(freeze)
        all_gravity.extend(gravity)
    return all_freeze, all_gravity


# =============================================================================
# 4. Lv1: game_idx (実行時間の代理) との相関
# =============================================================================


def build_game_idx_correlation_report(freeze: list[ColumnFreezeInterval], tier_label: str) -> str:
    """video×side×game_idx 単位の検出件数を game_idx と相関させる (全体+層別)。"""
    counts: "dict[tuple[str, str, int], int]" = {}
    for f in freeze:
        key = (f.video, f.side, f.game_idx)
        counts[key] = counts.get(key, 0) + 1
    game_idxs = np.array([k[2] for k in counts])
    n_events = np.array([v for v in counts.values()])
    lines = [f"--- Lv1: game_idx相関 [{tier_label}] (検出済み区間を含む group={len(counts)}件) ---"]
    lines.append(_corr_line("全体", game_idxs, n_events))
    for side in ("1P", "2P"):
        idx = [i for i, k in enumerate(counts) if k[1] == side]
        if idx:
            lines.append(_corr_line(
                f"side={side}", game_idxs[idx], n_events[idx],
            ))
    return "\n".join(lines)


def _corr_line(label: str, x: "np.ndarray", y: "np.ndarray") -> str:
    """相関係数1行分 (n<3はデータ不足として明示)。"""
    if len(x) < 3:
        return f"  {label}: n={len(x)} (相関計算に不足)"
    r, p = stats.pearsonr(x, y)
    return f"  {label}: n={len(x)} pearson_r={r:.3f} (p={p:.3f})"


# =============================================================================
# 5. 出力 (CSV + サマリレポート)
# =============================================================================


def write_freeze_csv(freeze: list[ColumnFreezeInterval], out_path: Path) -> None:
    """列凍結区間をCSVに書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "side", "game_idx", "column", "start_t", "end_t", "duration_sec", "other_growth", "trigger", "is_opening_artifact"])
        for iv in freeze:
            writer.writerow([iv.video, iv.side, iv.game_idx, iv.column, f"{iv.start_t:.3f}", f"{iv.end_t:.3f}", f"{iv.duration_sec:.3f}", iv.other_growth, iv.trigger, iv.is_opening_artifact])


def write_gravity_csv(gravity: list[GravityViolationInterval], out_path: Path) -> None:
    """重力逆転区間をCSVに書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "side", "game_idx", "column", "start_t", "end_t", "duration_sec"])
        for iv in gravity:
            writer.writerow([iv.video, iv.side, iv.game_idx, iv.column, f"{iv.start_t:.3f}", f"{iv.end_t:.3f}", f"{iv.duration_sec:.3f}"])


def write_video_summary_csv(freeze: list[ColumnFreezeInterval], out_path: Path) -> None:
    """動画別の列凍結件数集計をCSVに書き出す。"""
    counts: "dict[str, int]" = {}
    for f_iv in freeze:
        counts[f_iv.video] = counts.get(f_iv.video, 0) + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "n_freeze_intervals"])
        for video, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            writer.writerow([video, n])


def _top_n_by_video(items: list[ColumnFreezeInterval], n: int) -> list[tuple[str, int]]:
    """video別件数の上位n件を返す共通ヘルパ。"""
    by_video: "dict[str, int]" = {}
    for f_iv in items:
        by_video[f_iv.video] = by_video.get(f_iv.video, 0) + 1
    return sorted(by_video.items(), key=lambda kv: -kv[1])[:n]


def build_summary_report(
    freeze: list[ColumnFreezeInterval], gravity: list[GravityViolationInterval], n_videos: int,
) -> str:
    """走査動画分の全体サマリ (trigger tier + 開幕アーティファクトで3層に層別)。"""
    high_conf = [f_iv for f_iv in freeze if f_iv.trigger in ("duration", "both")]
    low_conf = [f_iv for f_iv in freeze if f_iv.trigger == "burst_growth"]
    opening = [f_iv for f_iv in high_conf if f_iv.is_opening_artifact]
    clean = [f_iv for f_iv in high_conf if not f_iv.is_opening_artifact]
    lines = [
        f"--- Lv0: 列凍結スキャン結果 (走査動画数={n_videos}) ---",
        f"[高信頼] duration>=30s条件 (単独/both): 合計{len(high_conf)}件 "
        f"→ うち開幕アーティファクト濃厚(開始<{OPENING_ARTIFACT_MAX_START_OFFSET_SEC}s)="
        f"{len(opening)}件 ({len(opening)/max(len(high_conf),1)*100:.1f}%) / "
        f"クリーン(要注視候補)={len(clean)}件",
        f"[低信頼] burst_growth条件のみ (通常の速い連鎖と混同しうる): 合計{len(low_conf)}件",
        f"重力逆転(浮きぷよ)持続区間: 合計{len(gravity)}件",
        "上位10動画 (クリーンな高信頼のみ):",
    ]
    for video, n in _top_n_by_video(clean, 10):
        lines.append(f"  {video}: {n}件")
    lines.append("[クリーンな高信頼] 全件明細:")
    for f_iv in clean:
        lines.append(
            f"  {f_iv.video} {f_iv.side} game_idx={f_iv.game_idx} col{f_iv.column}: "
            f"{f_iv.start_t:.1f}-{f_iv.end_t:.1f}s ({f_iv.duration_sec:.1f}s) "
            f"growth={f_iv.other_growth} trigger={f_iv.trigger}"
        )
    c22_hits = [f_iv for f_iv in freeze if f_iv.video == "c22"]
    c22_clean = [f_iv for f_iv in c22_hits if f_iv.trigger in ("duration", "both") and not f_iv.is_opening_artifact]
    lines.append(
        f"[検出器健全性] c22の検出件数: 全体{len(c22_hits)}件 "
        f"(クリーン高信頼{len(c22_clean)}件・残りは開幕アーティファクトor低信頼)"
    )
    return "\n".join(lines)


# =============================================================================
# 6. main
# =============================================================================


def main() -> None:
    freeze, gravity = scan_all_videos(NPZ_DIR)
    n_videos = len(list(NPZ_DIR.glob("*.npz")))
    print(f"[1/3] 走査完了: {n_videos}動画")
    write_freeze_csv(freeze, INTERVALS_CSV)
    write_gravity_csv(gravity, GRAVITY_CSV)
    write_video_summary_csv(freeze, VIDEO_SUMMARY_CSV)
    print(f"[出力] {INTERVALS_CSV} / {GRAVITY_CSV} / {VIDEO_SUMMARY_CSV}")
    print("\n[2/3] " + build_summary_report(freeze, gravity, n_videos))
    high_conf = [f_iv for f_iv in freeze if f_iv.trigger in ("duration", "both")]
    clean = [f_iv for f_iv in high_conf if not f_iv.is_opening_artifact]
    print("\n[3/3] " + build_game_idx_correlation_report(high_conf, "高信頼(開幕アーティファクト含む,生)"))
    print("\n" + build_game_idx_correlation_report(clean, "高信頼クリーン(開幕アーティファクト除外)"))
    print("\n" + build_game_idx_correlation_report(freeze, "全件(burst_growth含む)"))


if __name__ == "__main__":
    main()
