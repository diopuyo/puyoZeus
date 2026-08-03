"""満杯盤面 人手ラベルシート 準備スクリプト (2026-08-02)。

背景誤検出 (でっち上げ) 頻度測定器の較正が 0/3 で失敗し、「満杯盤面・終盤で
盲目」と判明した (memory project_background_fp_meter_blind_on_full_boards_2026-08-02)。
原因は満杯に近い盤面の正解ラベル不足。本スクリプトは user (ぷよ有段者) が直接
ラベル付けするための材料シートを機械的に準備する (ラベル付け自体は行わない)。

## 候補選定ロジック
1. data/indicators_v2/boards_lean_regen_2026-07-31/*.npz の各 (video, side, game_idx)
   について、非空セル数 (13行×6列=78マス中) が最大になる 1 スナップショット
   ("そのゲームでの最満杯瞬間") を候補として抽出する。
   - primary tier: 非空セル数 >= PRIMARY_MIN_OCCUPANCY (60)
   - secondary tier: PRIMARY_MIN_OCCUPANCY 未満 SECONDARY_MIN_OCCUPANCY (55) 以上
2. 「位相」はそのゲーム (video_id, game_idx) の t_sec 範囲内での相対進行率で決める
   (終盤/中盤/序盤)。セル数の多さと位相は独立な軸 (中盤でも積み上げが早いと
   高セル数になりうるため)。
3. 動画1本あたり上限 MAX_CANDIDATES_PER_VIDEO 枚、終盤優先だが中盤も
   MID_PHASE_TARGET_FRACTION 程度混ぜて TARGET_TOTAL_CANDIDATES 件を選ぶ。

## 出力 (data/verify/full_board_label_sheet_2026-08-02/)
    frames/<video>_t<t_sec>_<side>_full.png       実画面フルフレーム (1920x1080)
    frames/<video>_t<t_sec>_<side>_compare.png    実画面盤面切り出し + 認識grid可視化 の比較画像
    labeling_sheet.csv                            user記入用シート (correct_grid, メモ列あり)
    label_sheet.md                                説明書き付き一覧 (Windowsパスリンク)

⚠️ 認識grid列は「正解の初期値」ではなく「疑ってみるための参考値」。満杯盤面では
認識が誤っている可能性が高いことを前提にラベル付けすること (label_sheet.md にも明記)。

Usage:
    PYTHONPATH=. python -m scripts.build_full_board_label_sheet
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS,
)
from scripts.extract_exchange_event_frames import (  # noqa: E402
    grab_frame, resolve_cached_video_path, to_windows_path,
)
from scripts.visualize_recognition import (  # noqa: E402
    CELL_H, CELL_W, COLOR_BGR, COLOR_SYMBOLS, P1_ROI_X, P1_ROI_Y,
    P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W,
)

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUTPUT_DIR: Path = Path("data/verify/full_board_label_sheet_2026-08-02")
FRAMES_SUBDIR_NAME: str = "frames"

TOTAL_CELLS: int = BOARD_ROWS * BOARD_COLS  # 13*6=78
# 非空セル数の閾値 (満杯78マス中)
PRIMARY_MIN_OCCUPANCY: int = 60
SECONDARY_MIN_OCCUPANCY: int = 55

# 選定対象の非空セル数「帯」の既定値 (--occupancy-min/--occupancy-max の既定値、
# 未指定なら旧来と完全に同じ挙動になる = 後方互換)。バッチ2 (準満杯帯 65-72) は
# CLIでこれらを上書きして使う (memory project_full_board_error_taxonomy_2026-08-02
# 「次バッチは occupancy 65-72 の準満杯帯」)。
DEFAULT_OCCUPANCY_MIN: int = SECONDARY_MIN_OCCUPANCY
DEFAULT_OCCUPANCY_MAX: int = TOTAL_CELLS

# 選定バグ修正 (2026-08-03): 満杯盤面ラベル第1バッチの skip13件中12件が
# game_idx=0付近の試合前演出画面 (色鮮やかな非ゲーム画面) だった。「そのゲームに
# 複数スナップショットが存在する」「ゲーム内相対時刻が一定以上」のいずれかを
# 満たさない候補は演出画面の疑いとして除外する (memory
# project_full_board_error_taxonomy_2026-08-02)。
MIN_SNAPSHOTS_PER_GAME: int = 5
# 上級者でも60セル埋めるには相応の手数(30組以上)を要するため、ゲーム開始から
# この秒数未満で高occupancyに達するのは物理的に非ゲーム画面の疑いが濃い。
MIN_GAME_ELAPSED_SEC_FOR_CUTSCENE_FILTER: float = 15.0

# 選定件数・偏り抑制パラメータ
TARGET_TOTAL_CANDIDATES: int = 40
MAX_CANDIDATES_PER_VIDEO: int = 2
# 終盤優先だが中盤もこの割合程度混ぜる (user仕様「2割程度」)
MID_PHASE_TARGET_FRACTION: float = 0.2

# ゲーム内の相対進行率 (0=ゲーム開始, 1=ゲーム最後の観測) による位相分類閾値
LATE_PHASE_FRAC_MIN: float = 0.66
MID_PHASE_FRAC_MIN: float = 0.33
PHASE_EARLY: str = "序盤"
PHASE_MID: str = "中盤"
PHASE_LATE: str = "終盤"

TIER_PRIMARY: str = "primary"
TIER_SECONDARY: str = "secondary"

VIDEO_ID_PREFIX: str = "video_"

# 比較画像レイアウト用
TITLE_BAR_HEIGHT_PX: int = 36
PANEL_GAP_PX: int = 8
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_TITLE: float = 0.7
FONT_SCALE_PANEL_LABEL: float = 0.55

# 日本語ラベル描画用 (cv2.putText の Hershey フォントは ASCII のみ対応で日本語が
# 文字化けするため、PIL 経由で描画する。scripts/visualize_advantage_overlay.py の
# 既存パターン (_font/FONT_CANDIDATES) を踏襲)。
JP_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc",
)
JP_FONT_SIZE_TITLE: int = 20
JP_FONT_SIZE_LABEL: int = 16


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class GameCandidate:
    """1 (video, side, game_idx) 分の「最満杯瞬間」候補。"""

    video_id: str          # "video_c10"
    side: str              # "1P" / "2P"
    game_idx: int
    frame_idx: int
    t_sec: float
    grid: np.ndarray       # (13,6) int8、行0=隠し段
    occupancy: int
    tier: str              # "primary" / "secondary"
    phase: str             # "序盤" / "中盤" / "終盤"
    game_progress_frac: float

    @property
    def video_stem(self) -> str:
        """"video_c10" -> "c10"。"""
        if self.video_id.startswith(VIDEO_ID_PREFIX):
            return self.video_id[len(VIDEO_ID_PREFIX):]
        return self.video_id


# =============================================================================
# 1. 候補抽出 (npz -> ゲームごとの最満杯瞬間)
# =============================================================================


def compute_occupancy(grid: np.ndarray) -> int:
    """盤面 (13行×6列) の非空セル数を数える (隠し段含む全78マス中)。"""
    return int(np.count_nonzero(grid != COLOR_EMPTY))


def classify_tier(
    occupancy: int,
    occupancy_min: int = DEFAULT_OCCUPANCY_MIN,
    occupancy_max: int = DEFAULT_OCCUPANCY_MAX,
    primary_min: int = PRIMARY_MIN_OCCUPANCY,
) -> "str | None":
    """非空セル数から tier を判定する ([occupancy_min, occupancy_max] 帯外は None)。

    occupancy_min/occupancy_max は既定値なら旧来 (>=55、上限なし) と完全に
    同じ挙動になる (後方互換)。バッチ2 (準満杯帯 65-72) 等は呼び出し側で
    上書きする。primary_min は帯内での tier 表示ラベルの区切り (旧来の
    PRIMARY_MIN_OCCUPANCY=60 の役割を維持)。
    """
    if occupancy < occupancy_min or occupancy > occupancy_max:
        return None
    return TIER_PRIMARY if occupancy >= primary_min else TIER_SECONDARY


def build_game_time_bounds(
    video_ids: np.ndarray, game_idxs: np.ndarray, t_secs: np.ndarray,
) -> dict[tuple[str, int], tuple[float, float]]:
    """(video_id, game_idx) ごとの t_sec 範囲 (両side合算) を求める。"""
    bounds: dict[tuple[str, int], tuple[float, float]] = {}
    for vid, gidx, t in zip(video_ids, game_idxs, t_secs):
        key = (str(vid), int(gidx))
        lo, hi = bounds.get(key, (float("inf"), float("-inf")))
        bounds[key] = (min(lo, float(t)), max(hi, float(t)))
    return bounds


def classify_phase(t_sec: float, lo: float, hi: float) -> tuple[str, float]:
    """ゲーム内相対進行率を計算し位相ラベルを返す (単一観測ゲームは終盤扱い)。"""
    if hi <= lo:
        return PHASE_LATE, 1.0
    frac = min(1.0, max(0.0, (t_sec - lo) / (hi - lo)))
    if frac >= LATE_PHASE_FRAC_MIN:
        return PHASE_LATE, frac
    if frac >= MID_PHASE_FRAC_MIN:
        return PHASE_MID, frac
    return PHASE_EARLY, frac


def count_snapshots_per_group(sides: np.ndarray, game_idxs: np.ndarray) -> dict[tuple[str, int], int]:
    """(side, game_idx) ごとの全npzスナップショット数 (tier閾値に関わらず全件) を数える。

    「そのゲームに複数スナップショットが存在する」判定の元データ (選定バグ修正用)。
    """
    counts: dict[tuple[str, int], int] = {}
    for side, gidx in zip(sides, game_idxs):
        key = (str(side), int(gidx))
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_probable_cutscene_snapshot(
    t_sec: float, game_lo_t_sec: float, side: str, game_idx: int,
    snapshot_counts: dict[tuple[str, int], int],
) -> bool:
    """試合前演出画面 (色鮮やかな非ゲーム画面) の誤爆候補かどうかを判定する。

    memory project_full_board_error_taxonomy_2026-08-02: 満杯盤面ラベル第1バッチの
    skip13件中12件が game_idx=0 付近の試合前演出画面だった (選定ロジックのバグ)。
    「そのゲームに複数スナップショットが存在する」「ゲーム内相対時刻が一定以上」の
    いずれかを満たさない場合は演出画面の疑いとして除外する。
    """
    elapsed_sec = t_sec - game_lo_t_sec
    too_early = elapsed_sec < MIN_GAME_ELAPSED_SEC_FOR_CUTSCENE_FILTER
    too_few_snapshots = snapshot_counts.get((side, game_idx), 0) < MIN_SNAPSHOTS_PER_GAME
    return too_early or too_few_snapshots


def extract_game_peak_candidates(
    npz_path: Path,
    occupancy_min: int = DEFAULT_OCCUPANCY_MIN,
    occupancy_max: int = DEFAULT_OCCUPANCY_MAX,
) -> list[GameCandidate]:
    """1動画分の npz から (side, game_idx) ごとの最満杯瞬間の候補を抽出する。

    occupancy_min/occupancy_max は既定値なら旧来と同じ挙動 (後方互換)。
    試合前演出画面の疑いがある候補は is_probable_cutscene_snapshot で除外する。
    """
    data = np.load(npz_path, allow_pickle=True)
    grids, video_ids = data["grids"], data["video_id"]
    sides, game_idxs = data["side"], data["game_idx"]
    t_secs, frame_idxs = data["t_sec"], data["frame_idx"]
    occupancies = np.count_nonzero(grids != COLOR_EMPTY, axis=(1, 2))
    bounds = build_game_time_bounds(video_ids, game_idxs, t_secs)
    snapshot_counts = count_snapshots_per_group(sides, game_idxs)

    best_by_group: dict[tuple[str, int], int] = {}
    for i in range(len(grids)):
        tier = classify_tier(int(occupancies[i]), occupancy_min, occupancy_max)
        if tier is None:
            continue
        lo, _hi = bounds[(str(video_ids[i]), int(game_idxs[i]))]
        if is_probable_cutscene_snapshot(
            float(t_secs[i]), lo, str(sides[i]), int(game_idxs[i]), snapshot_counts,
        ):
            continue
        key = (str(sides[i]), int(game_idxs[i]))
        cur = best_by_group.get(key)
        if cur is None or occupancies[i] > occupancies[cur]:
            best_by_group[key] = i

    candidates: list[GameCandidate] = []
    for (side, gidx), i in best_by_group.items():
        lo, hi = bounds[(str(video_ids[i]), gidx)]
        phase, frac = classify_phase(float(t_secs[i]), lo, hi)
        candidates.append(GameCandidate(
            video_id=str(video_ids[i]), side=side, game_idx=gidx,
            frame_idx=int(frame_idxs[i]), t_sec=float(t_secs[i]),
            grid=grids[i].astype(np.int64), occupancy=int(occupancies[i]),
            tier=classify_tier(int(occupancies[i]), occupancy_min, occupancy_max) or TIER_SECONDARY,
            phase=phase, game_progress_frac=frac,
        ))
    return candidates


def load_candidate_pool(
    npz_dir: Path,
    occupancy_min: int = DEFAULT_OCCUPANCY_MIN,
    occupancy_max: int = DEFAULT_OCCUPANCY_MAX,
) -> list[GameCandidate]:
    """npz_dir 配下の全 npz から候補プールを構築する (occupancy帯は呼び出し側指定)。"""
    pool: list[GameCandidate] = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        pool.extend(extract_game_peak_candidates(npz_path, occupancy_min, occupancy_max))
    return pool


# =============================================================================
# 2. 選定 (動画上限・位相配分)
# =============================================================================


def _sort_key(c: GameCandidate) -> tuple[int, int]:
    """優先度キー: primary tier優先、同tier内は非空セル数が多い順。"""
    tier_rank = 0 if c.tier == TIER_PRIMARY else 1
    return (tier_rank, -c.occupancy)


def select_candidates(
    pool: list[GameCandidate],
    target_total: int = TARGET_TOTAL_CANDIDATES,
    max_per_video: int = MAX_CANDIDATES_PER_VIDEO,
    mid_phase_target_fraction: float = MID_PHASE_TARGET_FRACTION,
) -> list[GameCandidate]:
    """動画上限・位相配分 (終盤優先+中盤を一定割合混ぜる) を満たしつつ候補を選ぶ。

    ラウンドロビン方式: 動画ごとに1本ずつ最良候補を選ぶ周回を繰り返し、
    1動画への偏りを避ける。位相は「中盤」枠を先に target 件数分満たしてから
    残りを「終盤」優先で埋める (中盤の方がプールで希少なため先取りしないと
    ラウンドロビンの過程で終盤に食われてしまう)。
    """
    mid_target = round(target_total * mid_phase_target_fraction)
    mid_pool = sorted((c for c in pool if c.phase == PHASE_MID), key=_sort_key)
    other_pool = sorted((c for c in pool if c.phase != PHASE_MID), key=_sort_key)

    selected: list[GameCandidate] = []
    n_by_video: dict[str, int] = {}
    selected += _round_robin_pick(mid_pool, mid_target, max_per_video, n_by_video)
    remaining = target_total - len(selected)
    selected += _round_robin_pick(other_pool, remaining, max_per_video, n_by_video)
    return selected


def _round_robin_pick(
    ordered_pool: list[GameCandidate], n_want: int,
    max_per_video: int, n_by_video: dict[str, int],
) -> list[GameCandidate]:
    """優先度順プールから、動画あたり max_per_video 件までの制約でラウンドロビン抽出する。"""
    picked: list[GameCandidate] = []
    remaining = list(ordered_pool)
    while len(picked) < n_want and remaining:
        used_this_round: set[str] = set()
        next_remaining: list[GameCandidate] = []
        for c in remaining:
            stem = c.video_stem
            if len(picked) >= n_want:
                next_remaining.append(c)
                continue
            if stem in used_this_round or n_by_video.get(stem, 0) >= max_per_video:
                next_remaining.append(c)
                continue
            picked.append(c)
            used_this_round.add(stem)
            n_by_video[stem] = n_by_video.get(stem, 0) + 1
        if not used_this_round:
            break  # これ以上ラウンドロビンで拾える候補がない
        remaining = next_remaining
    return picked


# =============================================================================
# 3. 画像生成 (実画面フルフレーム + 比較画像)
# =============================================================================


def encode_grid_string(grid: np.ndarray) -> str:
    """13行×6列のgridを1行文字列にエンコードする (行区切り'/'、不明='U')。"""
    rows = []
    for row in range(BOARD_ROWS):
        chars = []
        for col in range(BOARD_COLS):
            v = int(grid[row, col])
            chars.append("U" if v == COLOR_UNKNOWN else str(v))
        rows.append("".join(chars))
    return "/".join(rows)


def _jp_font(size: int) -> ImageFont.ImageFont:
    """日本語表示用フォントを取得する (meiryo優先、無ければPIL既定フォント)。"""
    for p in JP_FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_japanese_text(
    image: np.ndarray, text: str, xy: tuple[int, int], size: int, fill: tuple[int, int, int],
) -> np.ndarray:
    """cv2 BGR画像に日本語テキストを描画して返す (PIL往復、cv2.putTextはASCII専用のため)。"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_img).text(xy, text, font=_jp_font(size), fill=fill)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _roi_origin(side: str) -> tuple[int, int]:
    """side から実画面 ROI 原点 (x, y) を返す。"""
    return (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)


def _build_hidden_row_strip(width: int) -> np.ndarray:
    """隠し段 (画面外) を示す黒帯ストリップを作る (実画面クロップの上に載せる)。"""
    strip = np.zeros((CELL_H, width, 3), dtype=np.uint8)
    return draw_japanese_text(
        strip, "隠し段(画面外)", (4, CELL_H - 26), JP_FONT_SIZE_LABEL, (150, 150, 150),
    )


def crop_board_region_padded(frame: np.ndarray, side: str) -> np.ndarray:
    """実画面から盤面ROIを切り出し、npz grid(13行)と高さを揃えるため隠し段帯を上に足す。"""
    x, y = _roi_origin(side)
    crop = frame[y:y + ROI_H, x:x + ROI_W].copy()
    return np.vstack([_build_hidden_row_strip(ROI_W), crop])


def render_grid_panel(grid: np.ndarray) -> np.ndarray:
    """npz grid (13行×6列) を色パッチ+記号で可視化したパネル画像を作る。"""
    panel = np.full((BOARD_ROWS * CELL_H, BOARD_COLS * CELL_W, 3), 30, dtype=np.uint8)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            _draw_grid_cell(panel, grid, row, col)
    cv2.line(panel, (0, HIDDEN_ROWS * CELL_H), (panel.shape[1], HIDDEN_ROWS * CELL_H), (0, 0, 255), 2)
    return panel


def _draw_grid_cell(panel: np.ndarray, grid: np.ndarray, row: int, col: int) -> None:
    """パネル上の1セル分の色パッチ+記号+枠線を描画する (render_grid_panel の分割)。"""
    v = int(grid[row, col])
    x0, y0 = col * CELL_W, row * CELL_H
    bgr = COLOR_BGR.get(v, (0, 0, 0))
    if v != COLOR_EMPTY:
        cv2.rectangle(panel, (x0 + 2, y0 + 2), (x0 + CELL_W - 2, y0 + CELL_H - 2), bgr, -1)
    cv2.rectangle(panel, (x0, y0), (x0 + CELL_W, y0 + CELL_H), (90, 90, 90), 1)
    symbol = COLOR_SYMBOLS.get(v, "?")
    if symbol:
        cv2.putText(
            panel, symbol, (x0 + CELL_W // 2 - 8, y0 + CELL_H // 2 + 8),
            FONT, FONT_SCALE_TITLE, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            panel, symbol, (x0 + CELL_W // 2 - 8, y0 + CELL_H // 2 + 8),
            FONT, FONT_SCALE_TITLE, (255, 255, 255), 2, cv2.LINE_AA,
        )


def _add_titled_column(image: np.ndarray, title: str) -> np.ndarray:
    """画像の上にタイトルバー (帯+日本語文字) を足す。"""
    bar = np.full((TITLE_BAR_HEIGHT_PX, image.shape[1], 3), 50, dtype=np.uint8)
    bar = draw_japanese_text(bar, title, (6, 4), JP_FONT_SIZE_TITLE, (255, 255, 255))
    return np.vstack([bar, image])


def build_comparison_image(frame: np.ndarray, candidate: GameCandidate) -> np.ndarray:
    """実画面盤面切り出し + npz grid可視化パネルを左右に並べた比較画像を作る。"""
    left = _add_titled_column(crop_board_region_padded(frame, candidate.side), "実画面")
    right = _add_titled_column(render_grid_panel(candidate.grid), "認識grid(npz、要検証)")
    gap = np.full((left.shape[0], PANEL_GAP_PX, 3), 0, dtype=np.uint8)
    return np.hstack([left, gap, right])


# =============================================================================
# 4. 出力ファイル生成
# =============================================================================


def _frame_basename(candidate: GameCandidate) -> str:
    """ファイル名の共通部分 (video/t_sec/side を含む)。"""
    return f"{candidate.video_stem}_t{candidate.t_sec:.1f}_{candidate.side}"


def save_candidate_images(
    candidate: GameCandidate, video_path: Path, frames_dir: Path,
) -> tuple["Path | None", "Path | None"]:
    """1候補分の実画面フルフレームPNG + 比較画像PNGを保存する (失敗時はNone)。"""
    frame = grab_frame(video_path, candidate.t_sec)
    if frame is None:
        return None, None
    base = _frame_basename(candidate)
    full_path = frames_dir / f"{base}_full.png"
    compare_path = frames_dir / f"{base}_compare.png"
    cv2.imwrite(str(full_path), frame)
    cv2.imwrite(str(compare_path), build_comparison_image(frame, candidate))
    return full_path, compare_path


CSV_HEADER: tuple[str, ...] = (
    "video_id", "t_sec", "side", "game_idx", "occupancy", "tier", "phase",
    "image_full_frame", "image_comparison", "recognized_grid", "correct_grid", "memo",
)


def write_labeling_csv(rows: list[dict], out_path: Path) -> None:
    """user記入用 labeling_sheet.csv を書き出す (correct_grid/メモは空列、Excel想定でカンマ区切り)。"""
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEADER))
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in CSV_HEADER})


def _format_row_line(r: dict) -> str:
    """label_sheet.md の1候補分の行を組み立てる。"""
    return (
        f"- **{r['video_id']} {r['side']} t={r['t_sec']}秒** "
        f"(位相:{r['phase']}, 非空セル{r['occupancy']}/{TOTAL_CELLS}, tier:{r['tier']}) — "
        f"実画面: {r['image_full_frame']} / 比較画像: {r['image_comparison']}"
    )


def write_label_sheet_md(rows: list[dict], out_path: Path) -> Path:
    """説明書き付きの label_sheet.md を書き出す。"""
    header = [
        "# 満杯盤面 人手ラベルシート (2026-08-02)",
        "",
        "背景誤検出(でっち上げ)頻度測定器が満杯盤面・終盤で盲目と判明したため、"
        "満杯に近い盤面の正解ラベルを集めます。",
        "",
        "## お願い",
        "- 各項目の「比較画像」を見て、認識結果(右側パネル)が実画面(左側)と"
        "**本当に一致しているか疑いながら**確認してください。満杯盤面では"
        "認識が間違っている可能性が高いことがわかっています。",
        "- 間違いがあれば labeling_sheet.csv の correct_grid 列に正しい配置を"
        "記入してください (書式は recognized_grid 列と同じ、行を'/'区切り、"
        "各セル1文字: 0=空, 1=赤, 2=青, 3=緑, 4=黄, 5=紫, 9=おじゃま, U=見えない/不明)。",
        "- 気づいたことは memo 列に自由記述してください。",
        "- ⚠️ ごく一部の候補は「実画面」が対戦画面ではなく VS画面/選手紹介/演出オーバーレイ"
        "等の**非ゲーム画面**になっていることがあります (試合開始直前の game_idx=0 付近で"
        "確認済み)。その場合は correct_grid を書かず memo 欄に「非ゲーム画面」と記入して"
        "スキップしてください (認識grid自体が無効な参考値のため訂正不要)。",
        f"- 総候補数: {len(rows)} 件",
        "",
        "## 候補一覧",
        "",
    ]
    body = [_format_row_line(r) for r in rows]
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    return out_path


def _row_from_candidate(
    candidate: GameCandidate, full_path: "Path | None", compare_path: "Path | None",
) -> dict:
    """CSV/md 出力用の1候補分の辞書を組み立てる (画像パスはWindows形式)。"""
    return {
        "video_id": candidate.video_id, "t_sec": f"{candidate.t_sec:.1f}", "side": candidate.side,
        "game_idx": candidate.game_idx, "occupancy": candidate.occupancy, "tier": candidate.tier,
        "phase": candidate.phase,
        "image_full_frame": to_windows_path(full_path) if full_path else "(取得失敗)",
        "image_comparison": to_windows_path(compare_path) if compare_path else "(取得失敗)",
        "recognized_grid": encode_grid_string(candidate.grid), "correct_grid": "", "memo": "",
    }


# =============================================================================
# 5. 集計レポート (選定結果の分布)
# =============================================================================


def summarize_selection(selected: list[GameCandidate]) -> str:
    """選定結果の分布 (動画別/位相別/side別) を平易な日本語でまとめる。"""
    by_phase: dict[str, int] = {}
    by_side: dict[str, int] = {}
    by_video: dict[str, int] = {}
    for c in selected:
        by_phase[c.phase] = by_phase.get(c.phase, 0) + 1
        by_side[c.side] = by_side.get(c.side, 0) + 1
        by_video[c.video_stem] = by_video.get(c.video_stem, 0) + 1
    lines = [
        f"選定候補数: {len(selected)} 件 (動画数: {len(by_video)} 本)",
        f"位相別: {dict(sorted(by_phase.items()))}",
        f"side別: {dict(sorted(by_side.items()))}",
    ]
    return "\n".join(lines)


# =============================================================================
# メイン
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="満杯盤面 人手ラベルシート準備")
    parser.add_argument("--npz-dir", type=Path, default=NPZ_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL_CANDIDATES)
    parser.add_argument(
        "--occupancy-min", type=int, default=DEFAULT_OCCUPANCY_MIN,
        help="選定対象の非空セル数下限 (既定55=旧来と同じ)",
    )
    parser.add_argument(
        "--occupancy-max", type=int, default=DEFAULT_OCCUPANCY_MAX,
        help="選定対象の非空セル数上限 (既定78=旧来と同じ上限なし)",
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理: 候補抽出 -> 選定 -> 画像生成 -> CSV/md出力。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    print(f"[1/4] 候補抽出: {args.npz_dir} (occupancy帯: {args.occupancy_min}-{args.occupancy_max})")
    pool = load_candidate_pool(args.npz_dir, args.occupancy_min, args.occupancy_max)
    print(f"  候補プール(ゲームごとの最満杯瞬間): {len(pool)} 件")

    print("[2/4] 選定 (動画上限2枚・位相配分)")
    selected = select_candidates(pool, target_total=args.target_total)
    print("  " + summarize_selection(selected).replace("\n", "\n  "))

    print("[3/4] 画像生成")
    frames_dir = args.out_dir / FRAMES_SUBDIR_NAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for c in selected:
        video_path = resolve_cached_video_path(c.video_id)
        if video_path is None:
            print(f"  [WARN] {c.video_id}: ローカルキャッシュ無し、スキップ")
            continue
        full_path, compare_path = save_candidate_images(c, video_path, frames_dir)
        rows.append(_row_from_candidate(c, full_path, compare_path))
    print(f"  画像生成完了: {len(rows)} 件")

    print("[4/4] CSV/md 出力")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_labeling_csv(rows, args.out_dir / "labeling_sheet.csv")
    sheet_path = write_label_sheet_md(rows, args.out_dir / "label_sheet.md")
    print(f"  出力: {sheet_path}")
    print(f"\n[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
