"""「エフェクト以外の誤りの真因特定」用: 全93誤りセルの onset 特定 + 実画面
キャプチャシート生成 (2026-08-04、user承認済みタスク)。

`scripts/_diag_c_zero_effect_2026-08-04.py` は (c) npz (裏走行中、着弾待ち)
を対象にしていたが、本スクリプトは **アンカー npz
(`data/indicators_v2/boards_lean_regen_2026-07-31`、全31動画完備)** を直接
対象にする (誤りは元々 OFF=アンカーに焼き付いているため (c) の着弾を待つ
必要がない)。batch1+batch2 の fixed 盤面の誤りセル **全93個** (row 制限なし、
7〜12行の23セルも含む) の onset (焼き付いた瞬間) を特定し、user 目視レビュー
用の実画面キャプチャシート (feedback_review_actual_screen_frames 準拠) を
生成する。

## 再利用 (コピペ禁止指示)
- アンカー突合: `scripts/measure_effect_gate_c_2026-08-04.py` の
  `load_all_samples` / `_lookup_anchor_row` / `_load_npz_index` を
  importlib 経由で import (ファイル名にハイフンを含むため動的 import)。
- onset 遡りロジック: `scripts/_diag_c_zero_effect_2026-08-04.py` の
  `_find_value_timeline` / `_find_responsible_onset` / `_build_trigger_windows`
  / `_time_in_any_window` / `_side_game_row_indices` / `_load_chain_trigger_secs`
  / `_read_frame_at` をそのまま import して使う (二重実装しない)。
  時間窓の近似マージン定数 (`OPPONENT_WINDOW_*` / `OWN_WINDOW_*`) も同モジュール
  から再利用する。

## 本スクリプトで新規に実装するもの
- おじゃま増分判定 (`_ojama_counts_in_window` / `_max_step_delta`): _diag には
  存在しない新規ロジック (盤面のおじゃまセル数の1ステップ最大増分を onset
  ±1秒窓で検出)。
- ゲーム開始からの経過秒 (`_game_start_t`): 新規。
- 実画面キャプチャシート生成一式 (PIL によるトリミング・赤枠描画・キャプション
  合成・シート分割保存) と index.md 出力: 新規。

## 修正 (2026-08-05、コーディネータ検収指摘への対応): 循環測定の排除
初版の `_ojama_increase_near` は盤面全体のおじゃまセル数を単純カウントして
いたため、**診断対象の「色→おじゃま誤読」セル自身が onset で +1 として
数えられ、判定が自明に True になる循環測定**になっていた (誤り93件中45件が
color_to_ojama、同時 onset 群では複数セル分水増しされる)。
これを修正し、以下 2 種類の増分を分けて算出する:
    - `ojama_delta_raw`: 修正前と同じ、盤面全体 (誤読セルを含む) の1ステップ
      最大増分。
    - `ojama_delta_clean`: **その盤面 (同一 video/side/game_idx の同一ラベル
      サンプル) に属する全ラベル誤りセルの位置を除外したグリッド**で数えた
      1ステップ最大増分。誤読セル自身の寄与を排除した「実着弾らしさ」を測る。
実着弾の物理則 (`reference_ojama_landing_pattern`: 6列均等 floor(N/6)+端数)
を利用し、clean 増分が **6以上ならほぼ確実に実着弾**、**1〜5なら小規模着弾
か別の誤読で曖昧**、**0なら着弾なし (=旧版は循環でTrueだった疑い)** に分類
する (`OJAMA_CLEAN_BUCKET_*` 定数)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.build_error_onset_sheet_2026-08-04
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

# ファイル名にハイフンを含むため通常の `from ... import` は構文エラーになる
# (コピペ禁止指示への対応として動的 import で突合/onset ロジックを再利用する)。
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

OUTPUT_DIR: Path = Path("data/verify/error_onset_sheet_2026-08-04")
INDEX_MD_PATH: Path = OUTPUT_DIR / "index.md"

EXPECTED_TOTAL_ERROR_CELLS: int = 93  # batch1=47 / batch2=46 (measure_effect_gate_c と同一検収)

# おじゃま増加判定の時間窓 (onset ±この秒数)。
OJAMA_CHECK_WINDOW_SEC: float = 1.0

# clean増分の分類バケット閾値 (reference_ojama_landing_pattern: 6列均等
# floor(N/6)+端数で降るため、1ステップで6以上増えればほぼ確実に実着弾)。
OJAMA_CLEAN_BUCKET_LANDING_MIN: int = 6
OJAMA_CLEAN_BUCKET_AMBIGUOUS_MIN: int = 1

# キャプチャ3枚の時刻オフセット (onset 基準、ラベル時点は個別指定)。
CAPTURE_PRE_ONSET_SEC: float = -0.5
CAPTURE_POST_ONSET_SEC: float = 0.2

# シート画像レイアウト。
CROP_MARGIN_PX: int = 20
THUMB_WIDTH_PX: int = 220
CELLS_PER_SHEET: int = 10
THUMB_GAP_PX: int = 8
CAPTION_HEIGHT_PX: int = 34
ROW_GAP_PX: int = 10
RED_BOX_COLOR_BGR: tuple[int, int, int] = (0, 0, 255)
RED_BOX_THICKNESS_PX: int = 3
BG_COLOR_RGB: tuple[int, int, int] = (255, 255, 255)
TEXT_COLOR_RGB: tuple[int, int, int] = (0, 0, 0)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class CellOnsetRecord:
    """1 誤りセル分の onset 特定 + 時間文脈 + キャプチャシート参照。"""

    video: str
    side: str
    game_idx: int
    row: int
    col: int
    wrong_value: int
    correct_value: int
    label_t_sec: float
    onset_frame_idx: "int | None"
    onset_t_sec: "float | None"
    persistence_sec: "float | None"
    onset_count_before_label: int
    is_pre_existing: bool
    opponent_window_hit: "bool | None"
    own_chain_hit: "bool | None"
    ojama_delta_raw: "int | None"
    ojama_delta_clean: "int | None"
    elapsed_since_game_start_sec: "float | None"
    group_key: str
    group_size: int = 1
    sheet_file: str = ""


# =============================================================================
# 1. 誤りセル列挙 (row 制限なし版、_diag の row1-3 限定版とは意図的に別実装)
# =============================================================================


def mismatched_cells_all(
    off_grid: "np.ndarray", correct_grid: "np.ndarray",
) -> list[tuple[int, int, int, int]]:
    """全13行×6列を対象に誤りセルを (row, col, wrong, correct) で列挙する。"""
    out: list[tuple[int, int, int, int]] = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            cv = int(correct_grid[row, col])
            if cv == COLOR_UNKNOWN:
                continue
            wv = int(off_grid[row, col])
            if wv != cv:
                out.append((row, col, wv, cv))
    return out


# =============================================================================
# 2. 新規ロジック: おじゃま増加判定 + ゲーム開始経過秒
# =============================================================================


def _ojama_count(grid: "np.ndarray", exclude: "frozenset[tuple[int, int]]" = frozenset()) -> int:
    """盤面のおじゃまセル数。exclude で指定した (row, col) は数えない。

    2026-08-05 修正: exclude 引数を追加。診断対象セル自身が「色→おじゃま
    誤読」の場合、そのセル位置を除外しないと誤読セル自身の寄与でおじゃま
    数が自明に増えてしまう循環測定になる (コーディネータ検収指摘)。
    """
    mask = grid == COLOR_OJAMA
    if exclude:
        mask = mask.copy()
        for r, c in exclude:
            mask[r, c] = False
    return int(np.count_nonzero(mask))


def _ojama_counts_in_window(
    idx: "object", side: str, game_idx: int, exclude: "frozenset[tuple[int, int]]",
) -> list[tuple[float, int]]:
    """(t_sec, おじゃま数) の時系列 (frame_idx 昇順、exclude 除外済み)。"""
    rows = _DIAG._side_game_row_indices(idx, side, game_idx)
    return [(float(idx.t_secs[i]), _ojama_count(idx.grids[i], exclude)) for i in rows]


def _max_step_delta(counts: list[tuple[float, int]], onset_t: float, window_sec: float) -> int:
    """onset ±window_sec 内 (境界の1行外側含む) での1ステップ最大増分を返す。

    増加がなければ 0 (減少のみの場合も 0、着弾判定には増加のみが意味を持つ)。
    """
    in_window = [
        i for i, (t, _) in enumerate(counts) if onset_t - window_sec <= t <= onset_t + window_sec
    ]
    if not in_window:
        return 0
    lo = max(0, min(in_window) - 1)
    hi = min(len(counts) - 1, max(in_window) + 1)
    if lo >= hi:
        return 0
    return max(0, max(counts[i + 1][1] - counts[i][1] for i in range(lo, hi)))


def _ojama_deltas_near(
    idx: "object", side: str, game_idx: int, onset_t: float, window_sec: float,
    exclude_positions: "frozenset[tuple[int, int]]",
) -> tuple[int, int]:
    """(ojama_delta_raw, ojama_delta_clean) を1回のタイムライン走査で算出する。"""
    raw_counts = _ojama_counts_in_window(idx, side, game_idx, frozenset())
    clean_counts = _ojama_counts_in_window(idx, side, game_idx, exclude_positions)
    delta_raw = _max_step_delta(raw_counts, onset_t, window_sec)
    delta_clean = _max_step_delta(clean_counts, onset_t, window_sec)
    return delta_raw, delta_clean


def ojama_delta_bucket(delta_clean: "int | None") -> str:
    """clean増分を実着弾らしさのバケットに分類する (docstring の分類定義参照)。"""
    if delta_clean is None:
        return "N/A"
    if delta_clean >= OJAMA_CLEAN_BUCKET_LANDING_MIN:
        return "6+"
    if delta_clean >= OJAMA_CLEAN_BUCKET_AMBIGUOUS_MIN:
        return "1-5"
    return "0"


def _game_start_t(idx: "object", game_idx: int) -> float:
    """(1P/2P 両方の) 該当 game_idx の最初の観測時刻 (=ゲーム開始の近似)。"""
    mask = idx.game_idxs == game_idx
    return float(np.min(idx.t_secs[mask])) if mask.any() else 0.0


# =============================================================================
# 3. 診断本体 (onset 特定 + 時間文脈付記)
# =============================================================================


def _diagnose_one_cell(
    idx: "object", chain_trigger_secs: "np.ndarray", side: str, game_idx: int,
    label_frame_idx: int, label_t_sec: float, row: int, col: int, wrong: int, correct: int,
    exclude_positions: "frozenset[tuple[int, int]]",
) -> CellOnsetRecord:
    """1 誤りセルの onset を特定し、時間文脈 (相手/自連鎖窓・おじゃま増分・経過秒) を付記する。

    exclude_positions: 同一盤面 (同一ラベルサンプル) に属する全誤りセルの
        位置。おじゃま増分算出時にこれらを除外し (2026-08-05 修正)、
        診断対象セル自身の誤読 (色→おじゃま等) による循環測定を排除する。
    """
    timeline = _DIAG._find_value_timeline(idx, side, game_idx, row, col)
    onset = _DIAG._find_responsible_onset(timeline, wrong, label_frame_idx)
    base_kwargs = dict(
        video="", side=side, game_idx=game_idx, row=row, col=col,
        wrong_value=wrong, correct_value=correct, label_t_sec=label_t_sec,
        group_key="",
    )
    if onset is None:
        return CellOnsetRecord(
            **base_kwargs, onset_frame_idx=None, onset_t_sec=None, persistence_sec=None,
            onset_count_before_label=0, is_pre_existing=False,
            opponent_window_hit=None, own_chain_hit=None,
            ojama_delta_raw=None, ojama_delta_clean=None, elapsed_since_game_start_sec=None,
        )
    onset_fi, onset_t, is_pre, n_onsets = onset
    opp_side = "2P" if side == "1P" else "1P"
    opp_windows = _DIAG._build_trigger_windows(
        idx, chain_trigger_secs, opp_side, game_idx,
        _DIAG.OPPONENT_WINDOW_PRE_MARGIN_SEC, _DIAG.OPPONENT_WINDOW_POST_SEC,
    )
    own_windows = _DIAG._build_trigger_windows(
        idx, chain_trigger_secs, side, game_idx,
        _DIAG.OWN_WINDOW_PRE_MARGIN_SEC, _DIAG.OWN_WINDOW_POST_SEC,
    )
    delta_raw, delta_clean = _ojama_deltas_near(
        idx, side, game_idx, onset_t, OJAMA_CHECK_WINDOW_SEC, exclude_positions,
    )
    return CellOnsetRecord(
        **base_kwargs, onset_frame_idx=onset_fi, onset_t_sec=onset_t,
        persistence_sec=label_t_sec - onset_t, onset_count_before_label=n_onsets,
        is_pre_existing=is_pre,
        opponent_window_hit=_DIAG._time_in_any_window(onset_t, opp_windows),
        own_chain_hit=_DIAG._time_in_any_window(onset_t, own_windows),
        ojama_delta_raw=delta_raw, ojama_delta_clean=delta_clean,
        elapsed_since_game_start_sec=onset_t - _game_start_t(idx, game_idx),
    )


def _diagnose_one_sample(
    s: "object", anchor_idx: "object", chain_trigger_secs: "np.ndarray",
) -> list[CellOnsetRecord]:
    """1 ラベルサンプル分の全誤りセルを診断する (アンカー突合失敗は空リストで明示)。"""
    anchor = _MC._lookup_anchor_row(
        anchor_idx, s.side, s.t_sec, s.game_idx, s.anchor_recognized_grid,
    )
    if anchor is None:
        print(f"  [警告] アンカー突合失敗: {s.video_stem} t={s.t_sec} {s.side}")
        return []
    cells = mismatched_cells_all(anchor.grid, s.correct_grid)
    # 2026-08-05 修正: 同一盤面の全誤りセル位置を除外集合として渡し、
    # おじゃま増分の循環測定 (誤読セル自身の寄与) を排除する。
    exclude_positions = frozenset((row, col) for row, col, _, _ in cells)
    records = [
        _diagnose_one_cell(
            anchor_idx, chain_trigger_secs, s.side, s.game_idx,
            anchor.frame_idx, anchor.t_sec, row, col, wrong, correct,
            exclude_positions,
        )
        for row, col, wrong, correct in cells
    ]
    for r in records:
        r.video = s.video_stem
    return records


def diagnose_all_samples() -> list[CellOnsetRecord]:
    """batch1+batch2 fixed 盤面の全誤りセルを動画ごとにキャッシュしつつ診断する。"""
    samples = [s for s in _MC.load_all_samples() if s.status == "fixed"]
    anchor_cache: dict[str, "object"] = {}
    trigger_cache: dict[str, "np.ndarray"] = {}
    records: list[CellOnsetRecord] = []
    for s in samples:
        if s.video_stem not in anchor_cache:
            npz_path = _MC.ANCHOR_NPZ_DIR / f"{s.video_stem}.npz"
            anchor_cache[s.video_stem] = _MC._load_npz_index(npz_path)
            trigger_cache[s.video_stem] = _DIAG._load_chain_trigger_secs(npz_path)
        records.extend(
            _diagnose_one_sample(s, anchor_cache[s.video_stem], trigger_cache[s.video_stem])
        )
    _assign_groups(records)
    return records


def _assign_groups(records: list[CellOnsetRecord]) -> None:
    """(video, side, game_idx, onset_frame_idx) が同一のセルをグループ化する。"""
    counts: dict[tuple, int] = {}
    for r in records:
        key = (r.video, r.side, r.game_idx, r.onset_frame_idx)
        counts[key] = counts.get(key, 0) + 1
    for r in records:
        key = (r.video, r.side, r.game_idx, r.onset_frame_idx)
        r.group_key = f"{r.video}_{r.side}_g{r.game_idx}_f{r.onset_frame_idx}"
        r.group_size = counts[key]


# =============================================================================
# 4. 実画面キャプチャシート生成 (userレビュー用、feedback_review_actual_screen_frames 準拠)
# =============================================================================


def _region_for_side(side: str) -> "object":
    """side に対応する BoardRegion を返す (実際の gate 配線と同じ選択方式)。"""
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _cell_full_rect(region: "object", row: int, col: int) -> tuple[int, int, int, int]:
    """セル全体 (sample_rect より広い、視認性重視) の矩形を返す。"""
    cx, cy = region.cell_center(row, col)
    half_w = region.cell_width / 2
    half_h = region.cell_height / 2
    return (
        int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h),
    )


def _crop_with_box(frame_bgr: "np.ndarray", region: "object", row: int, col: int) -> "np.ndarray":
    """region 周辺を margin 付きで切り出し、対象セルに赤枠を描いて返す。"""
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, region.x - CROP_MARGIN_PX)
    y1 = max(0, region.y - CROP_MARGIN_PX)
    x2 = min(fw, region.x + region.width + CROP_MARGIN_PX)
    y2 = min(fh, region.y + region.height + CROP_MARGIN_PX)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    bx1, by1, bx2, by2 = _cell_full_rect(region, row, col)
    cv2.rectangle(
        crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1),
        RED_BOX_COLOR_BGR, RED_BOX_THICKNESS_PX,
    )
    return crop


def _placeholder_thumbnail(text: str) -> "Image.Image":
    """フレーム取得不能時 (シーク範囲外等) のプレースホルダ画像。"""
    height = int(THUMB_WIDTH_PX * 720 / 384)
    img = Image.new("RGB", (THUMB_WIDTH_PX, height), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((10, height // 2), text, fill=TEXT_COLOR_RGB)
    return img


def _crop_to_thumbnail(crop_bgr: "np.ndarray") -> "Image.Image":
    """BGR crop を RGB PIL 画像に変換し、THUMB_WIDTH_PX 幅にリサイズする。"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    ratio = THUMB_WIDTH_PX / img.width
    new_size = (THUMB_WIDTH_PX, max(1, int(img.height * ratio)))
    return img.resize(new_size)


def _caption_strip(text: str, width: int) -> "Image.Image":
    """幅 width・高さ CAPTION_HEIGHT_PX の白背景キャプション帯を作る。"""
    img = Image.new("RGB", (width, CAPTION_HEIGHT_PX), BG_COLOR_RGB)
    draw = ImageDraw.Draw(img)
    draw.text((4, 8), text, fill=TEXT_COLOR_RGB)
    return img


def _build_frame_thumbnail(
    cap: "cv2.VideoCapture", fps: float, region: "object", row: int, col: int,
    t_sec: float, caption: str,
) -> "Image.Image":
    """指定秒のフレームを取得し、赤枠+キャプション付きサムネイルを1枚作る。"""
    frame = _DIAG._read_frame_at(cap, fps, max(t_sec, 0.0))
    if frame is None:
        thumb = _placeholder_thumbnail("N/A")
    else:
        thumb = _crop_to_thumbnail(_crop_with_box(frame, region, row, col))
    cap_img = _caption_strip(caption, thumb.width)
    combined = Image.new("RGB", (thumb.width, thumb.height + CAPTION_HEIGHT_PX), BG_COLOR_RGB)
    combined.paste(thumb, (0, 0))
    combined.paste(cap_img, (0, thumb.height))
    return combined


def _info_caption_text(r: CellOnsetRecord) -> str:
    """1セル分の情報キャプション文字列 (英数字のみ、PIL既定フォント対応)。"""
    onset_s = "N/A" if r.onset_t_sec is None else f"{r.onset_t_sec:.2f}s"
    persist_s = "N/A" if r.persistence_sec is None else f"{r.persistence_sec:.2f}s"
    grp = f" [group x{r.group_size}]" if r.group_size > 1 else ""
    return (
        f"{r.video} {r.side} row{r.row}col{r.col} wrong={r.wrong_value}->correct={r.correct_value} "
        f"onset={onset_s} persist={persist_s}{grp}"
    )


def _build_cell_row_image(r: CellOnsetRecord, cap: "cv2.VideoCapture", fps: float) -> "Image.Image":
    """1セル分: 情報キャプション + 3枚サムネイル (onset-0.5s/onset+0.2s/label) の行画像。"""
    region = _region_for_side(r.side)
    onset_t = r.onset_t_sec if r.onset_t_sec is not None else r.label_t_sec
    times_labels = [
        (onset_t + CAPTURE_PRE_ONSET_SEC, "onset-0.5s"),
        (onset_t + CAPTURE_POST_ONSET_SEC, "onset+0.2s"),
        (r.label_t_sec, "label t"),
    ]
    thumbs = [
        _build_frame_thumbnail(cap, fps, region, r.row, r.col, t, f"{label} t={t:.2f}")
        for t, label in times_labels
    ]
    row_w = sum(t.width for t in thumbs) + THUMB_GAP_PX * (len(thumbs) - 1)
    row_h = max(t.height for t in thumbs)
    info_img = _caption_strip(_info_caption_text(r), row_w)
    combined = Image.new("RGB", (row_w, row_h + CAPTION_HEIGHT_PX), BG_COLOR_RGB)
    combined.paste(info_img, (0, 0))
    x = 0
    for t in thumbs:
        combined.paste(t, (x, CAPTION_HEIGHT_PX))
        x += t.width + THUMB_GAP_PX
    return combined


def _build_sheet_image(
    records: list[CellOnsetRecord], cap: "cv2.VideoCapture", fps: float,
) -> "Image.Image":
    """複数セルの行画像を縦に連結してシート1枚を作る (最大 CELLS_PER_SHEET 件)。"""
    rows = [_build_cell_row_image(r, cap, fps) for r in records]
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + ROW_GAP_PX * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), BG_COLOR_RGB)
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + ROW_GAP_PX
    return sheet


def generate_sheets_for_video(video_stem: str, records: list[CellOnsetRecord]) -> int:
    """1動画分のセルをCELLS_PER_SHEETごとに分割してシートPNGを保存する。

    Returns:
        生成したシート枚数。
    """
    video_path = _DIAG.VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    chunks = [
        records[i:i + CELLS_PER_SHEET] for i in range(0, len(records), CELLS_PER_SHEET)
    ]
    for chunk_idx, chunk in enumerate(chunks):
        sheet = _build_sheet_image(chunk, cap, fps)
        filename = f"sheet_{video_stem}_{chunk_idx + 1:02d}.png"
        sheet.save(OUTPUT_DIR / filename)
        for r in chunk:
            r.sheet_file = filename
    cap.release()
    return len(chunks)


def generate_all_sheets(records: list[CellOnsetRecord]) -> None:
    """動画ごとにグループ化してシート生成を行う (video 単位でまとめる指示に対応)。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[CellOnsetRecord]] = {}
    for r in records:
        by_video.setdefault(r.video, []).append(r)
    for video_stem, recs in sorted(by_video.items()):
        recs.sort(key=lambda r: (r.side, r.label_t_sec, r.row, r.col))
        n_sheets = generate_sheets_for_video(video_stem, recs)
        print(f"  {video_stem}: {len(recs)} セル → シート {n_sheets} 枚")


# =============================================================================
# 5. index.md + 集計レポート
# =============================================================================


def _index_row(r: CellOnsetRecord) -> str:
    """index.md 用の1セル分の markdown テーブル行。"""
    onset_s = "N/A" if r.onset_t_sec is None else f"{r.onset_t_sec:.2f}"
    persist_s = "N/A" if r.persistence_sec is None else f"{r.persistence_sec:.2f}"
    elapsed_s = (
        "N/A" if r.elapsed_since_game_start_sec is None
        else f"{r.elapsed_since_game_start_sec:.2f}"
    )
    grp = f"x{r.group_size}" if r.group_size > 1 else ""
    bucket = ojama_delta_bucket(r.ojama_delta_clean)
    return (
        f"| {r.video} | {r.side} | {r.label_t_sec:.1f} | {r.row} | {r.col} | "
        f"{r.wrong_value}→{r.correct_value} | {onset_s} | {persist_s} | "
        f"{r.opponent_window_hit} | {r.own_chain_hit} | {r.ojama_delta_raw} | "
        f"{r.ojama_delta_clean} | {bucket} | "
        f"{elapsed_s} | {r.is_pre_existing} | {grp} | {r.sheet_file} |"
    )


def write_index_md(records: list[CellOnsetRecord], out_path: Path) -> None:
    """全誤りセルの一覧表 markdown を出力する。"""
    header = (
        "| video | side | label_t | row | col | wrong→correct | onset_t | "
        "persist | opp_window | own_chain | ojama_delta_raw | ojama_delta_clean | "
        "ojama_bucket | elapsed_from_start | pre_existing | group | sheet |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(_index_row(r) for r in sorted(
        records, key=lambda r: (r.video, r.side, r.label_t_sec, r.row, r.col)
    ))
    out_path.write_text(header + body + "\n", encoding="utf-8")


def build_summary(records: list[CellOnsetRecord]) -> str:
    """検収 (93一致) + onset特定成功率 + 時間文脈の自動分類集計。"""
    lines = ["--- 検収 + 集計 ---"]
    n = len(records)
    verdict = "合格" if n == EXPECTED_TOTAL_ERROR_CELLS else "★不合格★"
    lines.append(f"[検収] 誤りセル総数={n} (期待={EXPECTED_TOTAL_ERROR_CELLS}) → {verdict}")
    n_onset_ok = sum(1 for r in records if r.onset_t_sec is not None)
    lines.append(f"onset特定成功: {n_onset_ok}/{n} 件 ({n_onset_ok / n * 100:.1f}%)")
    n_pre = sum(1 for r in records if r.is_pre_existing)
    n_opp = sum(1 for r in records if r.opponent_window_hit)
    n_own = sum(1 for r in records if r.own_chain_hit)
    no_window_records = _no_time_window_records(records)
    n_no_window = len(no_window_records)
    lines.append(f"pre_existing (最初から誤値): {n_pre}/{n} 件")
    lines.append(f"相手連鎖窓内 (opponent_window_hit): {n_opp}/{n} 件")
    lines.append(f"自連鎖窓内 (own_chain_hit): {n_own}/{n} 件")
    lines.append(
        f"[重要] no_time_window (pre_existing でも相手連鎖窓内でもない) 相当: "
        f"{n_no_window}/{n} 件 ({n_no_window / n * 100:.1f}%) ※全93セル確定版"
    )
    lines.append("")
    lines.append(_ojama_bucket_report("全93セル", records))
    lines.append(_ojama_bucket_report(f"no_time_window ({n_no_window}件)", no_window_records))
    return "\n".join(lines)


def _no_time_window_records(records: list[CellOnsetRecord]) -> list[CellOnsetRecord]:
    """pre_existing でも相手連鎖窓内でもない (=no_time_window相当) の部分集合。"""
    return [
        r for r in records
        if r.onset_t_sec is not None and not r.is_pre_existing and not r.opponent_window_hit
    ]


def _ojama_bucket_report(label: str, records: list[CellOnsetRecord]) -> str:
    """clean増分バケット (0/1-5/6+) の分布を1行で報告する (2026-08-05 修正版)。"""
    n = len(records)
    counts = {"0": 0, "1-5": 0, "6+": 0, "N/A": 0}
    for r in records:
        counts[ojama_delta_bucket(r.ojama_delta_clean)] += 1
    pct = {k: (v / n * 100 if n else 0.0) for k, v in counts.items()}
    return (
        f"[clean増分分布] {label} (n={n}): "
        f"0={counts['0']}({pct['0']:.1f}%) / "
        f"1-5={counts['1-5']}({pct['1-5']:.1f}%) / "
        f"6+={counts['6+']}({pct['6+']:.1f}%) / N/A={counts['N/A']}"
    )


# =============================================================================
# 6. main
# =============================================================================


def main() -> None:
    print("[1/3] batch1+batch2 fixed 盤面の誤りセル onset 特定 (アンカー npz 使用)")
    records = diagnose_all_samples()
    print(f"  診断対象セル総数: {len(records)} 件")

    print("\n[2/3] 実画面キャプチャシート生成")
    generate_all_sheets(records)
    write_index_md(records, INDEX_MD_PATH)
    print(f"  index.md: {INDEX_MD_PATH}")

    print("\n[3/3] 集計")
    print(build_summary(records))


if __name__ == "__main__":
    main()
