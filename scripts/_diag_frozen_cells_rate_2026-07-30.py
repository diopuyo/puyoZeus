"""列/セル単位「値変化後フリーズ」検出器 + corpus全体の発生率測定 (2026-07-30)。

背景: E事例 (c21, 1P, g2, t=292.6s) で col0 の13行すべてが認識値0(空)になったが、
実映像では赤/赤/赤/緑が連続した実体として写っていた。連鎖が切れる主犯として
最有力だが n=1 で発生率が未知だった (memory project_column_recognition_loss_confirmed_2026-07-30)。

near-miss法は精度が低く却下済み。本スクリプトは「値が変化してから一定時間
フリーズし続けているセル/列」を直接抽出する専用検出器。

検出器設計:
1. 列単位の消失(最優先): ある列の非空セル数がMIN_STACK_BEFORE_COLLAPSE以上
   だった状態から1ステップで非空セル数が0に転落した事象を列崩壊候補とする。
   正常な連鎖消去との区別はスコア整合性で行う。src/scoring.pyの得点公式
   (BASE_SCORE_PER_PUYO=10, bonus_multiplier>=1)から、n個の色ぷよが本当に
   消えたなら得点は最低でも10*n増えるという理論下限が導ける。列崩壊時の
   スコア増分がこの理論下限を下回れば、スコアで説明できない消失としてフリーズ
   疑いとする。おじゃまぷよは自身の消去では得点を生まないため下限計算には
   含めない。
2. セル単位のフリーズ: 色からおじゃまが空に反転した個々のセルについて、
   次に非空に戻るまでの経過秒数を全数計測する。列崩壊のスコア疑惑判定を
   同一transitionのセル単位イベントにも継承させる(単独セルの得点整合性
   テストは緩すぎて使えないため。4連結の最小消去でも40点あり1セル分の
   下限10点は常にクリアしてしまう。これがnear-miss法の教訓と同じ罠であり
   単独セル閾値の新設は避けた)。
3. 正常ケースの除外の妥当性: npzはSTABLE確定盤面のみを保持し直前と同一盤面
   なら間引く。本当に消えて空になった正常ケースは得点増分としてnpzのscore列
   に必ず反映される(連鎖進行中フレームは間引かれるが連鎖完了時点までに得点
   は確定しているため)。得点で説明できるかは連鎖が起きたかの直接証拠であり
   freshnessだけでは判別できない罠を回避できる。

閾値の根拠: E事例は11.4秒から16.8秒凍結した(定義により差異あり、後述)。
feedback_placement_reflection_8frames_2026-07-25の8フレーム基準は別現象
(設置反映)の受け入れ基準であり本現象とは性質が違うため直接の閾値には
採用しない。閾値非依存で分布(中央値・p90・最大)を報告し複数のカットライン
(1秒/2秒/5秒/11.4秒)での該当率を併記する。

使用データ世代:
npz: data/indicators_v2/boards_lean_fixed_regen_2026-07-28 (#51後、着弾CSVと同世代)
corpus(共起分析用、読み取り専用): data/verify/chain_count_ocr_full_corpus_2026-07-29.csv

npzで見えるか事前確認(2026-07-30実測): c21 1P g2のnpzを直接ダンプしたところ、
col0がt=287.6(色あり)からt=289.8(全0)、t=304.0(全0のまま)、t=306.6(別の値で
部分復活)と、E事例の凍結がnpzのSTABLE snapshotの並びだけで明確に再現できる
ことを確認した。npzで検出可能、動画への再認識は不要と判断した。

実行方法: WSL経由、nice -n 19、単一プロセス。
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
)
from src.scoring import BASE_SCORE_PER_PUYO, MIN_BONUS_MULTIPLIER  # noqa: E402
from src.chain_count_ocr import _ensure_1080p  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion  # noqa: E402
from scripts.measure_exchange_dynamics import NpzRecord, _load_npz, _subset  # noqa: E402

# ============================
# 定数
# ============================

NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
CORPUS_CSV: Path = PROJ_ROOT / "data" / "verify" / "chain_count_ocr_full_corpus_2026-07-29.csv"
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "frozen_cells_2026-07-30"

# score OCR失敗センチネル (collect_boards_lean.py の変換規則、-1=None)。
SCORE_UNKNOWN: int = -1

# 列崩壊トリガー: 崩壊前の非空セル数がこれ以上 (色+おじゃま、空/UNKNOWN除く)。
MIN_STACK_BEFORE_COLLAPSE: int = 2

# スコア理論下限 (src/scoring.py 由来、色ぷよ1個あたりの絶対最小得点)。
SCORE_FLOOR_PER_COLORED_PUYO: int = BASE_SCORE_PER_PUYO * MIN_BONUS_MULTIPLIER

# 色ぷよとみなす値集合 (おじゃま・空・UNKNOWNは除く)。
COLORED_VALUES: frozenset[int] = frozenset({1, 2, 3, 4, 5})
# 非空とみなす値集合 (色+おじゃま、空/UNKNOWN除く)。
NONEMPTY_VALUES: frozenset[int] = COLORED_VALUES | {COLOR_OJAMA}

# 報告用の凍結時間カットライン (秒)。Eの実測値11.4秒を基準に複数併記。
FREEZE_CUTLINES_SEC: tuple[float, ...] = (1.0, 2.0, 5.0, 11.4)

# 共起分析: 列崩壊時刻からcorpusのt_chain_startを探す許容窓 (秒)。
# E実測: 崩壊t=289.8 は t_chain_start=292.6 の2.8秒前だった。
CO_OCCUR_WINDOW_BEFORE_SEC: float = 5.0
CO_OCCUR_WINDOW_AFTER_SEC: float = 30.0

# 実フレーム裏取りのランダム抽出件数・シード (cherry-pick禁止)
FRAME_SAMPLE_N: int = 8
FRAME_SAMPLE_SEED: int = 20260730

BOARD_CROP_UPSCALE: int = 2
CELL_CROP_UPSCALE: int = 10

# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class ColumnCollapseEvent:
    """1列がまるごと非空から空に転落した事象。"""
    video_stem: str
    side: str
    game_idx: int
    col: int
    t_before: float          # 崩壊直前 (非空だった) snapshotの時刻
    t_collapse: float        # 崩壊後 (全0になった) snapshotの時刻
    prev_colored_count: int  # 崩壊前の色ぷよ数
    prev_ojama_count: int    # 崩壊前のおじゃま数
    score_before: int
    score_after: int
    score_delta: int
    expected_min_score: int
    score_suspicious: bool   # score_delta < expected_min_score (score_available時のみ有効)
    score_available: bool    # score_before/afterともにOCR成功していたか (-1センチネル除外)
    t_recovery: float | None
    freeze_duration_sec: float | None
    right_censored: bool
    phase_bucket: str
    row_band_summary: str


@dataclass(frozen=True)
class CellFreezeEvent:
    """1セルが色/おじゃまから空に転落した事象。"""
    video_stem: str
    side: str
    game_idx: int
    row: int
    col: int
    prev_value: int
    t_to_empty: float
    t_recovery: float | None
    freeze_duration_sec: float | None
    right_censored: bool
    is_column_collapse_suspect: bool
    phase_bucket: str
    row_band: str
    col_band: str

# ============================
# 補助関数 (帯域・位相)
# ============================


def _row_band(row: int) -> str:
    """行を4帯域に分類する (既存 _diag_chain_break_cells と同基準)。"""
    if row < HIDDEN_ROWS:
        return "隠し段(row0)"
    if row <= 3:
        return "上部(row1-3)"
    if row <= 8:
        return "中央(row4-8)"
    return "下部(row9-12)"


def _col_band(col: int) -> str:
    """列を端/中央に分類する。"""
    return "端(col0/5)" if col in (0, BOARD_COLS - 1) else "中央(col1-4)"


def _phase_bucket(t: float, t_min: float, t_max: float) -> str:
    """試合内相対進行率で位相を切る (project_win_eval_regen_2026-07-26 の確定知見)。"""
    span = t_max - t_min
    ratio = 0.5 if span <= 0 else (t - t_min) / span
    if ratio < 1.0 / 3.0:
        return "序盤"
    if ratio < 2.0 / 3.0:
        return "中盤"
    return "終盤"


def _nonempty_count(col_values: np.ndarray) -> int:
    """色+おじゃまの非空セル数を返す (空/UNKNOWN除く)。"""
    return int(np.isin(col_values, list(NONEMPTY_VALUES)).sum())

# ============================
# 列単位検出
# ============================


def _scan_column_recovery(
    grids: np.ndarray, start_idx: int, col: int,
) -> int | None:
    """start_idx+1 以降で列colが再び非空になる最初のindexを返す (無ければNone)。"""
    n = grids.shape[0]
    for idx in range(start_idx + 1, n):
        if _nonempty_count(grids[idx, :, col]) > 0:
            return idx
    return None


def _build_row_band_summary(prev_col: np.ndarray) -> str:
    """崩壊前に非空だった行の帯域をカンマ区切りでまとめる。"""
    bands = sorted({
        _row_band(r) for r in range(BOARD_ROWS)
        if prev_col[r] in NONEMPTY_VALUES
    })
    return ",".join(bands)


def _detect_column_collapses(
    g: NpzRecord, video_stem: str, side: str, game_idx: int,
) -> list[ColumnCollapseEvent]:
    """1 (side, game_idx) 時系列から列崩壊事象を全数検出する。"""
    events: list[ColumnCollapseEvent] = []
    n = g.grids.shape[0]
    if n < 2:
        return events
    t_min, t_max = float(g.t_sec[0]), float(g.t_sec[-1])
    for i in range(n - 1):
        for col in range(BOARD_COLS):
            prev_col = g.grids[i, :, col]
            curr_col = g.grids[i + 1, :, col]
            prev_n = _nonempty_count(prev_col)
            if prev_n < MIN_STACK_BEFORE_COLLAPSE or _nonempty_count(curr_col) != 0:
                continue
            prev_colored = int(np.isin(prev_col, list(COLORED_VALUES)).sum())
            prev_ojama = int((prev_col == COLOR_OJAMA).sum())
            score_before_v, score_after_v = int(g.score[i]), int(g.score[i + 1])
            score_delta = score_after_v - score_before_v
            expected_min = SCORE_FLOOR_PER_COLORED_PUYO * prev_colored
            score_available = score_before_v != SCORE_UNKNOWN and score_after_v != SCORE_UNKNOWN
            recovery_idx = _scan_column_recovery(g.grids, i + 1, col)
            events.append(ColumnCollapseEvent(
                video_stem=video_stem, side=side, game_idx=game_idx, col=col,
                t_before=float(g.t_sec[i]), t_collapse=float(g.t_sec[i + 1]),
                prev_colored_count=prev_colored, prev_ojama_count=prev_ojama,
                score_before=score_before_v, score_after=score_after_v,
                score_delta=score_delta, expected_min_score=expected_min,
                score_suspicious=score_available and score_delta < expected_min,
                score_available=score_available,
                t_recovery=None if recovery_idx is None else float(g.t_sec[recovery_idx]),
                freeze_duration_sec=(
                    None if recovery_idx is None
                    else float(g.t_sec[recovery_idx] - g.t_sec[i + 1])
                ),
                right_censored=recovery_idx is None,
                phase_bucket=_phase_bucket(float(g.t_sec[i + 1]), t_min, t_max),
                row_band_summary=_build_row_band_summary(prev_col),
            ))
    return events

# ============================
# セル単位検出
# ============================


def _scan_cell_recovery(
    grids: np.ndarray, start_idx: int, row: int, col: int,
) -> int | None:
    """start_idx+1 以降でセル(row,col)が再び非空になる最初のindexを返す。"""
    n = grids.shape[0]
    for idx in range(start_idx + 1, n):
        if grids[idx, row, col] != COLOR_EMPTY:
            return idx
    return None


def _collapsed_cols_at(
    collapses: list[ColumnCollapseEvent], t_collapse: float,
) -> set[int]:
    """同一transition (t_collapse一致) で疑惑フラグが立った列集合を返す。"""
    return {c.col for c in collapses if c.t_collapse == t_collapse and c.score_suspicious}


def _detect_cell_freezes(
    g: NpzRecord, video_stem: str, side: str, game_idx: int,
    column_events: list[ColumnCollapseEvent],
) -> list[CellFreezeEvent]:
    """1 (side, game_idx) 時系列からセル単位の色/おじゃまから空フリーズを全数検出する。"""
    events: list[CellFreezeEvent] = []
    n = g.grids.shape[0]
    if n < 2:
        return events
    t_min, t_max = float(g.t_sec[0]), float(g.t_sec[-1])
    for i in range(n - 1):
        t_collapse = float(g.t_sec[i + 1])
        suspect_cols = _collapsed_cols_at(column_events, t_collapse)
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                prev_v = int(g.grids[i, row, col])
                curr_v = int(g.grids[i + 1, row, col])
                if prev_v not in NONEMPTY_VALUES or curr_v != COLOR_EMPTY:
                    continue
                recovery_idx = _scan_cell_recovery(g.grids, i + 1, row, col)
                events.append(CellFreezeEvent(
                    video_stem=video_stem, side=side, game_idx=game_idx,
                    row=row, col=col, prev_value=prev_v,
                    t_to_empty=t_collapse,
                    t_recovery=None if recovery_idx is None else float(g.t_sec[recovery_idx]),
                    freeze_duration_sec=(
                        None if recovery_idx is None
                        else float(g.t_sec[recovery_idx] - g.t_sec[i + 1])
                    ),
                    right_censored=recovery_idx is None,
                    is_column_collapse_suspect=col in suspect_cols,
                    phase_bucket=_phase_bucket(t_collapse, t_min, t_max),
                    row_band=_row_band(row), col_band=_col_band(col),
                ))
    return events

# ============================
# 試合セグメント分割 (game_idx desync対策)
# ============================
#
# memory project_game_idx_desync_bug_2026-07-29 の通り、npzのgame_idx列は
# 1P/2P独立カウンタのズレで信頼できない場合がある。scoreが減少する遷移は
# 物理的にありえない(1試合中は単調非減少)ため、試合境界の実測シグナルとして
# 採用し、同一game_idxラベル内でも試合を再分割する。これを怠ると試合終了時の
# 勝敗演出(盤面がUIで覆われる等)や次試合の空盤面が「列崩壊」に誤分類される
# (2026-07-30実測: c11 1P g1でscoreが238から26へ減少する境界を確認、これを
# セグメント分割せずに検出した結果は誤検出だった)。


def _split_into_match_segments(g: NpzRecord) -> list[NpzRecord]:
    """scoreが減少する遷移を境界として、1つの試合区間に再分割する。

    score_before/afterのいずれかがSCORE_UNKNOWN(-1)の場合はOCR欠測であり
    真の減少と断定できないため分割トリガーにしない (境界の見誤りを避ける)。
    """
    n = g.grids.shape[0]
    if n < 2:
        return [g]
    boundaries: list[int] = [0]
    for i in range(n - 1):
        prev_s, next_s = int(g.score[i]), int(g.score[i + 1])
        if prev_s == SCORE_UNKNOWN or next_s == SCORE_UNKNOWN:
            continue
        if next_s < prev_s:
            boundaries.append(i + 1)
    boundaries.append(n)
    segments: list[NpzRecord] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        segments.append(NpzRecord(
            video_id=g.video_id, side=g.side,
            t_sec=g.t_sec[start:end], game_idx=g.game_idx[start:end],
            grids=g.grids[start:end], score=g.score[start:end],
        ))
    return segments


# ============================
# per-video 処理
# ============================


def _process_video(stem: str) -> tuple[list[ColumnCollapseEvent], list[CellFreezeEvent]]:
    """1動画分の全side・全game_idxを処理する。"""
    npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
    if not npz_path.exists():
        return [], []
    col_events: list[ColumnCollapseEvent] = []
    cell_events: list[CellFreezeEvent] = []
    for rec in _load_npz(npz_path):
        for gidx in sorted(set(rec.game_idx.tolist())):
            mask = rec.game_idx == gidx
            g = _subset(rec, mask)
            order = np.argsort(g.t_sec)
            g = NpzRecord(
                video_id=g.video_id, side=g.side,
                t_sec=g.t_sec[order], game_idx=g.game_idx[order],
                grids=g.grids[order], score=g.score[order],
            )
            # game_idx desyncバグ対策: scoreの減少箇所で再分割してから検出する
            for seg_idx, seg in enumerate(_split_into_match_segments(g)):
                col_ev = _detect_column_collapses(seg, stem, rec.side, int(gidx) * 1000 + seg_idx)
                col_events.extend(col_ev)
                cell_events.extend(
                    _detect_cell_freezes(seg, stem, rec.side, int(gidx) * 1000 + seg_idx, col_ev)
                )
    return col_events, cell_events

# ============================
# 共起分析 (連鎖過小評価との関連)
# ============================


def _load_corpus_lookup() -> pd.DataFrame:
    """corpus CSV を読み取り専用で開く (走行中に追記されるため書き込み禁止)。"""
    df = pd.read_csv(CORPUS_CSV)
    df["gap"] = df["screen_chain_count"] - df["new_chain_count"]
    return df


def _match_corpus_gap(
    corpus: pd.DataFrame, video_stem: str, side: str, game_idx: int, t_collapse: float,
) -> float | None:
    """列崩壊時刻に対応するcorpusイベントのgapを探す (許容窓内、最も近いt_chain_startを採用)。"""
    sub = corpus[
        (corpus["video_stem"] == video_stem) & (corpus["side"] == side)
        & (corpus["game_idx"] == game_idx)
    ]
    if sub.empty:
        return None
    lo, hi = t_collapse - CO_OCCUR_WINDOW_BEFORE_SEC, t_collapse + CO_OCCUR_WINDOW_AFTER_SEC
    sub = sub[(sub["t_chain_start"] >= lo) & (sub["t_chain_start"] <= hi)]
    if sub.empty:
        return None
    sub = sub.assign(dist=(sub["t_chain_start"] - t_collapse).abs())
    row = sub.sort_values("dist").iloc[0]
    return float(row["gap"]) if pd.notna(row["gap"]) else None


def _annotate_co_occurrence(df_col: pd.DataFrame) -> pd.DataFrame:
    """列崩壊イベントに対応corpusのgapを付与する (見つからなければNaN)。"""
    if df_col.empty:
        return df_col
    corpus = _load_corpus_lookup()
    gaps = [
        _match_corpus_gap(corpus, r.video_stem, r.side, int(r.game_idx), r.t_collapse)
        for r in df_col.itertuples()
    ]
    df_col = df_col.copy()
    df_col["matched_gap"] = gaps
    return df_col

# ============================
# 実フレーム裏取り
# ============================


def _crop_board_region(frame: np.ndarray, side: str) -> np.ndarray:
    region: BoardRegion = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    return frame[region.y:region.y + region.height, region.x:region.x + region.width].copy()


def _crop_cell_region(frame: np.ndarray, side: str, row: int, col: int) -> np.ndarray:
    region: BoardRegion = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    cx, cy = region.cell_center(row, col)
    half_w = max(1, int(region.cell_width / 2))
    half_h = max(1, int(region.cell_height / 2))
    y1, y2 = max(0, cy - half_h), cy + half_h
    x1, x2 = max(0, cx - half_w), cx + half_w
    return frame[y1:y2, x1:x2].copy()


def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def _grab_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    """指定時刻のフレームを取得し1080pに正規化する (c11等720p動画対策)。"""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _ensure_1080p(frame)


def _save_frame_pair(ev: ColumnCollapseEvent, label: str) -> None:
    """崩壊前後のA/Bフレームを保存する (userレビュー用、判読可能な解像度)。"""
    video_path = VIDEO_DIR / f"video_{ev.video_stem}.mp4"
    ev_dir = OUT_DIR / label
    ev_dir.mkdir(parents=True, exist_ok=True)
    for tag, t in (("before", ev.t_before), ("collapse", ev.t_collapse)):
        frame = _grab_frame(video_path, t)
        if frame is None:
            print(f"[WARN] frame取得失敗: {label} {tag} (t={t})")
            continue
        board_crop = _crop_board_region(frame, ev.side)
        cv2.imwrite(str(ev_dir / f"board_{tag}.png"), _upscale(board_crop, BOARD_CROP_UPSCALE))
        for row in range(BOARD_ROWS):
            crop = _crop_cell_region(frame, ev.side, row, ev.col)
            cv2.imwrite(
                str(ev_dir / f"cell_{tag}_r{row}_c{ev.col}.png"),
                _upscale(crop, CELL_CROP_UPSCALE),
            )
    print(f"[OK] 実フレーム保存: {label} -> {ev_dir}")

def _sample_and_save_frames(df_col: pd.DataFrame) -> None:
    """疑惑列崩壊からランダムに動画横断でサンプルし実フレームを保存する。"""
    suspects = df_col[df_col["score_suspicious"] & ~df_col["right_censored"]]  # 回復確認済みのみ(証拠が強い候補)
    if suspects.empty:
        print("[WARN] 疑惑列崩壊が0件、実フレーム裏取りをスキップ")
        return
    rng = random.Random(FRAME_SAMPLE_SEED)
    stems = sorted(suspects["video_stem"].unique().tolist())
    rng.shuffle(stems)
    picked_rows: list[pd.Series] = []
    for stem in stems:
        rows = suspects[suspects["video_stem"] == stem]
        picked_rows.append(rows.sample(n=1, random_state=FRAME_SAMPLE_SEED).iloc[0])
        if len(picked_rows) >= FRAME_SAMPLE_N:
            break
    for row in picked_rows:
        label = (
            f"{row.video_stem}_{row.side}_g{int(row.game_idx)}"
            f"_col{int(row.col)}_t{row.t_collapse:.1f}"
        )
        ev = ColumnCollapseEvent(
            video_stem=row.video_stem, side=row.side, game_idx=int(row.game_idx),
            col=int(row.col), t_before=float(row.t_before),
            t_collapse=float(row.t_collapse),
            prev_colored_count=int(row.prev_colored_count),
            prev_ojama_count=int(row.prev_ojama_count),
            score_before=int(row.score_before), score_after=int(row.score_after),
            score_delta=int(row.score_delta), expected_min_score=int(row.expected_min_score),
            score_suspicious=bool(row.score_suspicious),
            score_available=bool(row.score_available),
            t_recovery=row.t_recovery if pd.notna(row.t_recovery) else None,
            freeze_duration_sec=(
                row.freeze_duration_sec if pd.notna(row.freeze_duration_sec) else None
            ),
            right_censored=bool(row.right_censored),
            phase_bucket=row.phase_bucket, row_band_summary=row.row_band_summary,
        )
        _save_frame_pair(ev, label)

# ============================
# 集計・報告
# ============================


def _summarize_column(df_col: pd.DataFrame) -> None:
    """列崩壊イベントの層別集計を表示・保存する。"""
    if df_col.empty:
        print("[列崩壊] 0件")
        return
    n_total = len(df_col)
    n_susp = int(df_col["score_suspicious"].sum())
    n_unavail = int((~df_col["score_available"]).sum())
    n_censored_susp = int((df_col["score_suspicious"] & df_col["right_censored"]).sum())
    n_confirmed = n_susp - n_censored_susp
    print(f"\n[列崩壊] 全体={n_total}件, スコアOCR欠測(判定不能)={n_unavail}件")
    print(f"[列崩壊] スコア疑惑(フリーズ候補、右打ち切り含む)={n_susp}件 ({100*n_susp/n_total:.1f}%)")
    print(f"[列崩壊] うち同一試合内で回復確認済み(=より確度が高い候補)={n_confirmed}件")
    for axis in ("video_stem", "side", "col", "phase_bucket"):
        print(f"\n[列崩壊 x {axis}] (疑惑のみ)")
        print(df_col[df_col["score_suspicious"]].groupby(axis).size()
              .sort_values(ascending=False).to_string())
    dur = df_col[df_col["score_suspicious"] & df_col["freeze_duration_sec"].notna()][
        "freeze_duration_sec"
    ]
    if not dur.empty:
        print(f"\n[疑惑列崩壊の凍結時間(回復確認済みのみ, n={len(dur)})]")
        print(f"中央値={dur.median():.2f}s p90={dur.quantile(0.9):.2f}s 最大={dur.max():.2f}s")
        for cut in FREEZE_CUTLINES_SEC:
            frac = (dur >= cut).mean()
            print(f"  >= {cut}秒: {(dur >= cut).sum()}件 ({100*frac:.1f}%)")
    n_censored = int((df_col["score_suspicious"] & df_col["right_censored"]).sum())
    print(f"[右打ち切り(未回復のままゲーム終了)] {n_censored}/{n_susp}件")
    df_col.to_csv(OUT_DIR / "column_collapse_events.csv", index=False)


def _summarize_cell(df_cell: pd.DataFrame) -> None:
    """セル単位フリーズの層別集計を表示・保存する。"""
    if df_cell.empty:
        print("[セルフリーズ] 0件")
        return
    n_susp_cells = int(df_cell["is_column_collapse_suspect"].sum())
    print(f"\n[セルフリーズ] 全体={len(df_cell)}件, 列崩壊疑惑と同時発生={n_susp_cells}件")
    for axis in ("row_band", "col_band", "phase_bucket"):
        print(f"\n[セルフリーズ x {axis}]")
        print(df_cell.groupby(axis).size().sort_values(ascending=False).to_string())
    dur = df_cell[df_cell["freeze_duration_sec"].notna()]["freeze_duration_sec"]
    if not dur.empty:
        print(f"\n[全セルフリーズの凍結時間(回復確認済みのみ, n={len(dur)})]")
        print(f"中央値={dur.median():.2f}s p90={dur.quantile(0.9):.2f}s 最大={dur.max():.2f}s")
    dur_susp = df_cell[
        df_cell["is_column_collapse_suspect"] & df_cell["freeze_duration_sec"].notna()
    ]["freeze_duration_sec"]
    if not dur_susp.empty:
        print(f"\n[列崩壊疑惑同時発生セルの凍結時間 (n={len(dur_susp)})]")
        print(f"中央値={dur_susp.median():.2f}s p90={dur_susp.quantile(0.9):.2f}s"
              f" 最大={dur_susp.max():.2f}s")
    df_cell.to_csv(OUT_DIR / "cell_freeze_events.csv", index=False)


def _summarize_co_occurrence(df_col: pd.DataFrame) -> None:
    """疑惑列崩壊と連鎖過小評価(gap)の共起を報告する。"""
    matched = df_col[df_col["score_suspicious"] & df_col["matched_gap"].notna()]
    print(f"\n[共起分析] 疑惑列崩壊のうちcorpusイベントと紐付いたもの: {len(matched)}件")
    if matched.empty:
        return
    med = matched["matched_gap"].median()
    avg = matched["matched_gap"].mean()
    print(f"  紐付いたイベントのgap(screen-new) 中央値={med:.1f} 平均={avg:.2f}")
    all_gaps = _load_corpus_lookup()["gap"].dropna()
    print(f"  corpus全体のgap 中央値={all_gaps.median():.1f} 平均={all_gaps.mean():.2f}"
          f" (参考、対応なしも含む全数)")

# ============================
# main
# ============================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in NPZ_DIR_REGEN.glob("*.npz"))
    print(f"[準備] 対象動画: {len(stems)}本")
    print(stems)

    all_col: list[ColumnCollapseEvent] = []
    all_cell: list[CellFreezeEvent] = []
    for stem in stems:
        col_ev, cell_ev = _process_video(stem)
        all_col.extend(col_ev)
        all_cell.extend(cell_ev)
        print(f"[{stem}] 列崩壊候補={len(col_ev)}件 セルフリーズ候補={len(cell_ev)}件", flush=True)

    df_col = pd.DataFrame([e.__dict__ for e in all_col])
    df_cell = pd.DataFrame([e.__dict__ for e in all_cell])
    df_col = _annotate_co_occurrence(df_col)

    _summarize_column(df_col)
    _summarize_cell(df_cell)
    _summarize_co_occurrence(df_col)
    _sample_and_save_frames(df_col)

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
