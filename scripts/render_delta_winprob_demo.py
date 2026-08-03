"""ΔWinProb 発火直後速報 「動画版デモ」レンダラー。

## 目的 (user フィードバック対処)
「従来 = 連鎖が終わってから勝率が動く」 vs 「新 = 発火の瞬間に速報が出る」 を、
実ゲーム画面に重ねた合成動画で体感できるようにする (静止画では分かりにくい)。

## 設計方針 (既存資産の再利用・再実装禁止)
- 勝率モデル・STABLE 推移タイムライン・ΔWinProb イベント計算は本ファイルで
  再実装せず `scripts.compute_exchange_delta_winprob` の関数を import して使う
  (`train_winprob_models` / `_load_video_npz` / `_build_stable_timeline`)。
- ΔWinProb イベント本体は同スクリプトの本走行で既に全66動画分計算済み
  (`data/verify/exchange_delta_winprob_step3_2026-08-02/exchange_delta_winprob.csv`)
  のため、既定では **このCSVを対象動画分だけフィルタして読む** (66本の再計算
  不要)。CSV に対象動画が無い場合のみ `--recompute` で
  `load_aug_with_stacking_predictions` + `compute_all_delta_winprob` を
  **対象動画の行だけに絞ってから** 呼び出す (全66本の再計算はしない)。
- 音声トラック結合は `src.video_compositer.VideoCompositor._mux_audio` を
  再利用するが、この既存ヘルパーは「音声源動画の先頭から」結合する前提の
  引数構成 (offset 非対応) のため、対象区間が動画中盤から始まる本ユースケース
  では使えない (先頭からの音声とズレる)。そのため音声区間切り出しのみ
  `VideoCompositor._resolve_ffmpeg_bin()` (ffmpeg バイナリ解決ロジック) を
  再利用しつつ本ファイルで組み、切り出し済み音声クリップを
  `_mux_audio` にそのまま渡して最終結合する (mux 本体は再利用、offset 対応
  部分のみ追加、既存 API は無改変)。

## 近似の限界 (正直な注記、fail-silent回避)
発火「検知」の実時刻(掛け算式表示等)は npz に保存されていないため、
ΔWinProb イベント CSV の t_sec (連鎖終了時点、_detect_fire_events 参照) から
`src.indicators_v2.estimate_chain_anim_duration_sec(approx_fire_chains)`
(CHAIN_ANIM_PER_STEP_SEC=0.4秒/連鎖、23動画418イベント実測ベース) を逆算して
近似する。この近似値であることを画面上に明記する (CLAUDE.md
「マジックナンバー禁止」「fail-silent回避」原則に基づき、精度を誇張しない)。

## 使い方
    PYTHONPATH=. python -m scripts.render_delta_winprob_demo \\
        --video-id c61 --game-idx 16 \\
        --out-dir data/verify/delta_winprob_demo_2026-08-03
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.chain import ChainSimulator
from src.video_compositer import VideoCompositor
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_AUG_CSV,
    DEFAULT_LABELED_WIN_CSV,
    DEFAULT_MODEL_D_DIR,
    DEFAULT_NPZ_DIR,
    DEFAULT_OUT_DIR as DEFAULT_DELTA_WINPROB_DIR,
    VIDEO_ID_NPZ_PREFIX,
    ChainInProgressWindow,
    PhaseWinprobModel,
    _build_stable_timeline,
    _load_video_npz,
    _npz_stem_from_video_id,
    build_chain_in_progress_windows,
    compute_all_delta_winprob,
    ignition_time_for_event,
    load_aug_with_stacking_predictions,
    train_winprob_models,
)
from scripts.visualize_advantage_overlay import _font

# =============================================================================
# 定数定義 (マジックナンバー禁止)
# =============================================================================

DEFAULT_VIDEO_ID: str = "c61"
DEFAULT_GAME_IDX: int = 16
DEFAULT_VIDEO_MP4_DIR = Path("data/frames")
DEFAULT_OUT_DIR = Path("data/verify/delta_winprob_demo_2026-08-03")
DEFAULT_DELTA_WINPROB_CSV = DEFAULT_DELTA_WINPROB_DIR / "exchange_delta_winprob.csv"

# 試合区間の前後バッファ (前試合の勝敗画面/ロード演出・本試合の勝敗画面を含める)
MATCH_START_BUFFER_SEC: float = 8.0
MATCH_END_BUFFER_SEC: float = 8.0
# 隣接試合との最小マージン (前後試合の認識区間へ被らないようにする安全マージン)
NEIGHBOR_GAME_GUARD_SEC: float = 1.0

# 速報バッジ (テキスト表示) の表示継続秒数 (連鎖の長さに関わらず一定)
BADGE_TEXT_DISPLAY_SEC: float = 3.0

# 出力キャンバスの寸法
OUT_W: int = 1280
OUT_H: int = 720
TOP_H: int = 130     # 上部情報パネル (バー・バッジ)
BOTTOM_H: int = 170  # 下部グラフ帯
CANVAS_H: int = TOP_H + OUT_H + BOTTOM_H
DEFAULT_FPS: float = 30.0

# 上部パネル レイアウト
PANEL_TITLE_Y: int = 6
PANEL_BAR_TOP: int = 34
PANEL_BAR_H: int = 30
PANEL_BAR_W: int = 760
PANEL_NOTE_Y: int = PANEL_BAR_TOP + PANEL_BAR_H + 10
PANEL_BADGE_Y: int = PANEL_NOTE_Y + 24

# 色 (1P=青、2P=赤。既存 visualize_advantage_overlay の配色を踏襲)
COLOR_1P = (90, 140, 220)
COLOR_2P = (210, 90, 90)
COLOR_BADGE_BG = (40, 40, 20)

# 下部グラフ帯レイアウト
GRAPH_MARGIN_X: int = 40
GRAPH_MARGIN_TOP: int = 26
GRAPH_MARGIN_BOTTOM: int = 12


# =============================================================================
# 1. 対象区間の決定 (試合開始〜終了、前後バッファ込み)
# =============================================================================

def select_video_segment(
    cache, game_idx: int,
) -> tuple[float, float]:
    """対象 game_idx の動画内区間 (開始秒, 終了秒) をバッファ込みで決める。

    前後試合の認識区間 (game_idx±1) へ食い込まないよう
    NEIGHBOR_GAME_GUARD_SEC を最小マージンとして確保する
    (feedback_review_video_full_match: 試合は開始〜終了まで完結させる)。
    """
    gi = cache.r1p.game_idx
    t = cache.r1p.t_sec
    cur_mask = gi == game_idx
    if not cur_mask.any():
        raise ValueError(f"game_idx={game_idx} のフレームが npz に存在しません")
    first_t, last_t = float(t[cur_mask].min()), float(t[cur_mask].max())

    start_sec = max(0.0, first_t - MATCH_START_BUFFER_SEC)
    prev_mask = gi == (game_idx - 1)
    if prev_mask.any():
        prev_last_t = float(t[prev_mask].max())
        start_sec = max(start_sec, prev_last_t + NEIGHBOR_GAME_GUARD_SEC)

    end_sec = last_t + MATCH_END_BUFFER_SEC
    next_mask = gi == (game_idx + 1)
    if next_mask.any():
        next_first_t = float(t[next_mask].min())
        end_sec = min(end_sec, next_first_t - NEIGHBOR_GAME_GUARD_SEC)

    return start_sec, end_sec


# =============================================================================
# 2. 発火イベント (速報バッジ + バー即時ジャンプ) の構築
# =============================================================================

@dataclass(frozen=True)
class FireEventView:
    """1発火イベント分の描画用ビュー (1P視点に正規化済み)。"""
    ignition_sec: float       # 発火検知(近似)時刻
    fire_end_sec: float       # 連鎖終了(t_sec、旧方式が追いつく時刻)
    fire_side: str            # "1P" or "2P"
    winprob_before_1p: float  # 発火直前の1P視点勝率(0-100%)
    winprob_after_1p: float   # 発火後(予測)の1P視点勝率(0-100%)
    delta_winprob: float      # 発火側視点のΔ (attacker視点、符号そのまま)


def _to_1p_view(value: float, fire_side: str) -> float:
    """attacker(発火側)視点の値を1P視点へ変換する (2P発火なら補数を取る)。"""
    return value if fire_side == "1P" else 100.0 - value


def build_fire_event_views(events_df: pd.DataFrame) -> list[FireEventView]:
    """ΔWinProb イベント DataFrame (1試合分) から描画用ビュー一覧を作る。

    ignition_sec 昇順にソートして返す (表示ロジック側の探索を単純化するため)。
    """
    views: list[FireEventView] = []
    for _, ev in events_df.iterrows():
        ignition_sec = ignition_time_for_event(
            float(ev["t_sec"]), float(ev["approx_fire_chains"]))
        fire_side = str(ev["fire_side"])
        views.append(FireEventView(
            ignition_sec=ignition_sec,
            fire_end_sec=float(ev["t_sec"]),
            fire_side=fire_side,
            winprob_before_1p=_to_1p_view(float(ev["winprob_before"]), fire_side),
            winprob_after_1p=_to_1p_view(float(ev["winprob_after"]), fire_side),
            delta_winprob=float(ev["delta_winprob"]),
        ))
    views.sort(key=lambda v: v.ignition_sec)
    return views


# =============================================================================
# 3. 表示値の計算 (従来推移 vs 速報ジャンプ、stateless)
# =============================================================================

def stable_value_at(timeline_t: np.ndarray, timeline_v: np.ndarray, t: float) -> "float | None":
    """STABLE 推移タイムラインの t 時点の値を前方保持(forward-fill)で返す。

    t が最初の STABLE 観測より前なら None (=STABLE待ち、未確定表示用)。
    """
    if len(timeline_t) == 0 or t < timeline_t[0]:
        return None
    idx = int(np.searchsorted(timeline_t, t, side="right")) - 1
    idx = max(0, min(idx, len(timeline_t) - 1))
    return float(timeline_v[idx])


def _latest_event_at_or_before(events: list[FireEventView], t: float) -> "FireEventView | None":
    """t 時点で最も直近に発火検知された(ignition_sec<=t)イベントを返す。"""
    latest: "FireEventView | None" = None
    for ev in events:
        if ev.ignition_sec <= t:
            latest = ev
        else:
            break
    return latest


@dataclass(frozen=True)
class DisplayState:
    """1フレーム分の描画状態。"""
    winprob_1p: float                  # 描画するべき1P視点勝率(0-100%)
    waiting: bool                      # STABLE待ち(未確定)
    badge: "FireEventView | None"       # 速報バッジ表示対象 (Noneなら非表示)
    jump_active: bool                  # バーが速報側にジャンプ中か


def compute_display_state(
    events: list[FireEventView], timeline_t: np.ndarray, timeline_v: np.ndarray, t: float,
) -> DisplayState:
    """t 時点の表示状態を求める (純関数、state を一切持たない)。

    「即時ジャンプ」: 発火検知(ignition_sec)〜連鎖終了(fire_end_sec)の間は
    予測後勝率(winprob_after_1p)を即座に表示し続ける(旧方式はこの間まだ
    STABLE待ちで動かない=対比が生まれる)。連鎖終了後は実測 STABLE 推移
    (stable_value_at) にそのまま切り替える(データ駆動の自然な収束、
    合成的なブレンドは行わない)。
    """
    stable_v = stable_value_at(timeline_t, timeline_v, t)
    waiting = stable_v is None
    display_v = stable_v if stable_v is not None else 50.0

    ev = _latest_event_at_or_before(events, t)
    jump_active = ev is not None and t < ev.fire_end_sec
    if jump_active:
        display_v = ev.winprob_after_1p
        waiting = False

    badge = (
        ev if ev is not None and t <= ev.ignition_sec + BADGE_TEXT_DISPLAY_SEC else None
    )
    return DisplayState(winprob_1p=display_v, waiting=waiting, badge=badge, jump_active=jump_active)


# =============================================================================
# 4. 描画 (上部パネル + 下部グラフ帯)
# =============================================================================

def _draw_bar(d: "ImageDraw.ImageDraw", state: DisplayState, cx: int, x0: int) -> None:
    """1P/2P 勝率バー本体を描画する。"""
    top, bar_w, bar_h = PANEL_BAR_TOP, PANEL_BAR_W, PANEL_BAR_H
    if state.waiting:
        d.rectangle([x0, top, x0 + bar_w, top + bar_h], outline=(255, 255, 255), width=2)
        d.text((cx - 90, top + 4), "STABLE 待ち", font=_font(20), fill=(255, 255, 255))
        return
    split_x = int(x0 + (state.winprob_1p / 100.0) * bar_w)
    d.rectangle([x0, top, split_x, top + bar_h], fill=(*COLOR_1P, 200))
    d.rectangle([split_x, top, x0 + bar_w, top + bar_h], fill=(*COLOR_2P, 200))
    d.rectangle([x0, top, x0 + bar_w, top + bar_h], outline=(255, 255, 255), width=2)
    label = f"1P 勝率 {state.winprob_1p:.0f}%   /   2P 勝率 {100.0 - state.winprob_1p:.0f}%"
    d.text((cx - 140, top + 5), label, font=_font(18), fill=(255, 255, 255))
    if state.jump_active:
        d.rectangle([split_x - 3, top - 5, split_x + 3, top + bar_h + 5], fill=(255, 255, 0))


def _draw_badge(d: "ImageDraw.ImageDraw", badge: "FireEventView | None", x0: int) -> None:
    """速報バッジ (「1P/2P 発火速報 Δ+XX%」) を描画する。badge=None なら何も描かない。"""
    if badge is None:
        return
    color = COLOR_1P if badge.fire_side == "1P" else COLOR_2P
    text = (f"[速報] {badge.fire_side} が発火 → 予測勝率 "
            f"{badge.delta_winprob:+.0f}pt 変化 (連鎖終了前に先行表示・近似値)")
    d.rectangle([x0 - 6, PANEL_BADGE_Y - 4, x0 + 760, PANEL_BADGE_Y + 22],
                fill=(*COLOR_BADGE_BG, 210))
    d.text((x0, PANEL_BADGE_Y), text, font=_font(16), fill=color)


def _draw_top_panel(frame_canvas: "Image.Image", state: DisplayState) -> None:
    """上部情報パネル (タイトル・バー・注記・バッジ) を描画する。"""
    d = ImageDraw.Draw(frame_canvas, "RGBA")
    cx = OUT_W // 2
    x0 = cx - PANEL_BAR_W // 2
    d.text((x0, PANEL_TITLE_Y), "ΔWinProb 発火直後速報デモ (青=1P 赤=2P)",
           font=_font(18), fill=(255, 255, 0))
    _draw_bar(d, state, cx, x0)
    d.text((x0, PANEL_NOTE_Y),
           "従来: 連鎖終了(両者STABLE)まで勝率は動かない → 新: 発火検知の瞬間に速報",
           font=_font(14), fill=(200, 200, 200))
    _draw_badge(d, state.badge, x0)


def _draw_graph_strip(
    frame_canvas: "Image.Image", history: list[tuple[float, float]],
    events: list[FireEventView], t_rel: float, total: float,
) -> None:
    """下部グラフ帯 (1P視点勝率の推移 + 発火イベントの縦線) を描画する。"""
    d = ImageDraw.Draw(frame_canvas, "RGBA")
    gy_top = TOP_H + OUT_H + GRAPH_MARGIN_TOP
    gy_bot = CANVAS_H - GRAPH_MARGIN_BOTTOM
    gx0, gx1 = GRAPH_MARGIN_X, OUT_W - GRAPH_MARGIN_X
    total = max(total, 1.0)

    def _px(t: float) -> int:
        return int(gx0 + (t / total) * (gx1 - gx0))

    def _py(v: float) -> int:
        return int(gy_bot - (max(0.0, min(100.0, v)) / 100.0) * (gy_bot - gy_top))

    d.rectangle([gx0 - 4, gy_top - 20, gx1 + 4, gy_bot + 4], fill=(0, 0, 0, 150))
    d.text((gx0, gy_top - 20), "1P視点 勝率推移 (0〜100%)", font=_font(14), fill=(255, 255, 255))
    d.line([(gx0, _py(50.0)), (gx1, _py(50.0))], fill=(150, 150, 150), width=1)
    for ev in events:
        ex = _px(ev.ignition_sec)
        col = COLOR_1P if ev.fire_side == "1P" else COLOR_2P
        d.line([(ex, gy_top), (ex, gy_bot)], fill=(*col, 140), width=1)
    if len(history) >= 2:
        pts = [(_px(t), _py(v)) for t, v in history]
        d.line(pts, fill=(255, 255, 255), width=2)
    ph = _px(t_rel)
    d.line([(ph, gy_top), (ph, gy_bot)], fill=(255, 255, 0), width=2)
    d.rectangle([gx0, gy_top, gx1, gy_bot], outline=(255, 255, 255), width=1)


def compose_frame(
    frame_bgr: np.ndarray, state: DisplayState, history: list[tuple[float, float]],
    events_rel: list[FireEventView], t_rel: float, total: float,
) -> np.ndarray:
    """1フレーム分の合成画像 (上部パネル + ゲーム画面 + 下部グラフ帯) を作る。

    events_rel: グラフ帯の横軸 (経過秒、区間開始基準) に変換済みのイベント一覧。
    """
    canvas = Image.new("RGB", (OUT_W, CANVAS_H), (12, 12, 16))
    canvas.paste(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), (0, TOP_H))
    _draw_top_panel(canvas, state)
    _draw_graph_strip(canvas, history, events_rel, t_rel, total)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


# =============================================================================
# 5. 音声トラック結合 (区間切り出し + 既存 mux ヘルパー再利用)
# =============================================================================

def _extract_audio_segment(
    video_path: Path, start_sec: float, duration_sec: float, out_path: Path,
) -> bool:
    """元動画から対象区間の音声のみを切り出す (ffmpeg -ss/-t、ローカルファイル操作)。

    VideoCompositor._mux_audio は「音声源動画の先頭から」結合する引数構成
    (offset 非対応) のため、対象区間が動画中盤から始まる本ユースケースでは
    そのまま使えない。ffmpeg バイナリ解決のみ VideoCompositor を再利用し、
    区間切り出し自体は本関数で行う (モジュール冒頭の設計方針を参照)。
    """
    ffmpeg_bin = VideoCompositor._resolve_ffmpeg_bin()
    if ffmpeg_bin is None:
        return False
    cmd = [ffmpeg_bin, "-y", "-ss", str(start_sec), "-t", str(duration_sec),
           "-i", str(video_path), "-vn", "-c:a", "aac", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path.exists()
    except subprocess.CalledProcessError:
        return False


def mux_audio_for_segment(
    silent_video_path: Path, source_video_path: Path,
    start_sec: float, duration_sec: float, out_path: Path, tmp_audio_path: Path,
) -> bool:
    """区間切り出し音声 + 無音レンダー動画 を結合する (成功で True)。"""
    if not _extract_audio_segment(source_video_path, start_sec, duration_sec, tmp_audio_path):
        print("[audio] 音声区間切り出しに失敗 (ffmpeg 不在 or エラー) -> 無音のまま出力")
        return False
    ok = VideoCompositor._mux_audio(silent_video_path, tmp_audio_path, out_path)
    tmp_audio_path.unlink(missing_ok=True)
    return ok


# =============================================================================
# 6. メイン レンダーループ
# =============================================================================

def render_video(
    video_path: Path, out_silent_path: Path, start_sec: float, end_sec: float,
    events: list[FireEventView], timeline_t: np.ndarray, timeline_v: np.ndarray,
) -> int:
    """対象区間を読み込み、合成フレームを書き出す。書き出しフレーム数を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_silent_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_silent_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (OUT_W, CANVAS_H))
    total_dur = max(1.0, end_sec - start_sec)
    # グラフ帯描画用: イベント発火時刻を区間開始基準の相対秒へ変換 (1回だけ)
    events_rel = [
        FireEventView(ignition_sec=ev.ignition_sec - start_sec, fire_end_sec=ev.fire_end_sec - start_sec,
                     fire_side=ev.fire_side, winprob_before_1p=ev.winprob_before_1p,
                     winprob_after_1p=ev.winprob_after_1p, delta_winprob=ev.delta_winprob)
        for ev in events
    ]
    history: list[tuple[float, float]] = []
    written = 0
    for fi in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_abs = fi / fps
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        state = compute_display_state(events, timeline_t, timeline_v, t_abs)
        t_rel = t_abs - start_sec
        if not state.waiting:
            history.append((t_rel, state.winprob_1p))
        writer.write(compose_frame(frame, state, history, events_rel, t_rel, total_dur))
        written += 1
        if written % 300 == 0:
            print(f"  ... {written} frames (t={t_rel:.1f}s winprob_1p={state.winprob_1p:.0f}%)")
    cap.release()
    writer.release()
    return written


# =============================================================================
# 7. イベントデータの読込 (既定=既存CSVフィルタ、--recompute=対象動画だけ計算)
# =============================================================================

def load_events_for_video(
    video_id: str, game_idx: int, delta_winprob_csv: Path,
    recompute: bool, npz_dir: Path, aug_csv: Path, model_d_dir: Path,
    models: dict[str, PhaseWinprobModel],
) -> pd.DataFrame:
    """対象動画・対象試合の ΔWinProb イベント行を返す。

    既定 (recompute=False): 既存の全66動画分計算済みCSVを対象動画分だけ
    フィルタして読む (66本の再計算不要)。CSV に対象動画が無い/--recompute
    指定時のみ、aug_csv を対象動画の行だけに絞ってから
    `compute_all_delta_winprob` を呼ぶ (この場合も対象動画分だけの計算)。
    """
    full_video_id = video_id if video_id.startswith(VIDEO_ID_NPZ_PREFIX) else VIDEO_ID_NPZ_PREFIX + video_id
    if not recompute and delta_winprob_csv.exists():
        df = pd.read_csv(delta_winprob_csv)
        sub = df.loc[(df["video_id"] == full_video_id) & (df["game_idx"] == game_idx)
                     & (~df["match_failed"])].copy()
        if len(sub) > 0:
            print(f"[events] 既存CSVから読込: {delta_winprob_csv} ({len(sub)}行)")
            return sub.sort_values("t_sec").reset_index(drop=True)
        print(f"[events] 既存CSVに {full_video_id} game={game_idx} の行が無いため再計算します")

    print("[events] 対象動画分のみ ΔWinProb を再計算します (66本再計算ではない)")
    merged = load_aug_with_stacking_predictions(aug_csv, model_d_dir, n_folds=5)
    merged = merged.loc[merged["video_id"] == full_video_id].reset_index(drop=True)
    if len(merged) == 0:
        raise ValueError(f"aug_csv に {full_video_id} の行がありません: {aug_csv}")
    delta_df = compute_all_delta_winprob(merged, models, npz_dir, n_workers=1)
    sub = delta_df.loc[(delta_df["game_idx"] == game_idx) & (~delta_df["match_failed"])].copy()
    return sub.sort_values("t_sec").reset_index(drop=True)


# =============================================================================
# 8. main
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="ΔWinProb 発火直後速報デモ動画レンダラー")
    parser.add_argument("--video-id", type=str, default=DEFAULT_VIDEO_ID)
    parser.add_argument("--game-idx", type=int, default=DEFAULT_GAME_IDX)
    parser.add_argument("--video-path", type=Path, default=None,
                        help="元動画パス省略時は data/frames/video_<id>.mp4")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--labeled-win-csv", type=Path, default=DEFAULT_LABELED_WIN_CSV)
    parser.add_argument("--delta-winprob-csv", type=Path, default=DEFAULT_DELTA_WINPROB_CSV)
    parser.add_argument("--aug-csv", type=Path, default=DEFAULT_AUG_CSV)
    parser.add_argument("--model-d-dir", type=Path, default=DEFAULT_MODEL_D_DIR)
    parser.add_argument("--recompute", action="store_true",
                        help="既存CSVを使わず対象動画分だけΔWinProbを再計算する")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video_path or (DEFAULT_VIDEO_MP4_DIR / f"video_{args.video_id}.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"元動画が見つかりません: {video_path}")

    print("=== 1. npz 読込 + 対象区間決定 ===")
    cache = _load_video_npz(args.video_id, args.npz_dir)
    if cache is None:
        raise FileNotFoundError(f"npz が見つかりません: {args.npz_dir}/{_npz_stem_from_video_id(args.video_id)}.npz")
    start_sec, end_sec = select_video_segment(cache, args.game_idx)
    print(f"[segment] video={args.video_id} game_idx={args.game_idx} "
          f"区間=({start_sec:.1f}s, {end_sec:.1f}s) 長さ={end_sec - start_sec:.1f}s")

    print("\n=== 2. 勝率モデル学習 (board-only指標 + 位相別isotonic校正) ===")
    models = train_winprob_models(args.labeled_win_csv)
    sim = ChainSimulator()

    print("\n=== 3. ΔWinProb イベント読込 ===")
    events_df = load_events_for_video(
        args.video_id, args.game_idx, args.delta_winprob_csv,
        args.recompute, args.npz_dir, args.aug_csv, args.model_d_dir, models)
    events = build_fire_event_views(events_df)
    print(f"[events] 速報バッジ対象イベント数: {len(events)}")

    print("\n=== 4. STABLE 従来推移タイムライン構築 (連鎖中は仮想盤面, Fix B) ===")
    chain_windows: "list[ChainInProgressWindow]" = []
    if "stack_net_ojama_after_pred" in events_df.columns:
        chain_windows = build_chain_in_progress_windows(events_df, cache, sim)
    else:
        print("[warn] events_df に stack_net_ojama_after_pred が無いため"
              "連鎖中仮想盤面差し替え(Fix B)をスキップします")
    timeline_df = _build_stable_timeline(cache, args.game_idx, models, sim, chain_windows)
    if len(timeline_df) == 0:
        raise RuntimeError(f"{args.video_id} game={args.game_idx} のタイムライン生成に失敗")
    timeline_t = timeline_df["t_sec"].values.astype(float)
    timeline_v = timeline_df["winprob_1p"].values.astype(float)

    print("\n=== 5. 動画レンダー (無音) ===")
    silent_path = args.out_dir / f"delta_winprob_demo_{args.video_id}_g{args.game_idx}_silent.mp4"
    written = render_video(video_path, silent_path, start_sec, end_sec, events, timeline_t, timeline_v)
    print(f"[render] {written} frames -> {silent_path}")

    print("\n=== 6. 音声結合 ===")
    final_path = args.out_dir / f"delta_winprob_demo_{args.video_id}_g{args.game_idx}.mp4"
    tmp_audio_path = args.out_dir / f"_tmp_audio_{args.video_id}_g{args.game_idx}.m4a"
    ok = mux_audio_for_segment(silent_path, video_path, start_sec, end_sec - start_sec,
                               final_path, tmp_audio_path)
    if ok:
        silent_path.unlink(missing_ok=True)
        print(f"[done] 音声付き出力: {final_path}")
    else:
        print(f"[done] 音声結合失敗、無音動画のまま: {silent_path}")
    print(f"\n発火イベント数(速報表示対象)={len(events)}  区間長={end_sec - start_sec:.1f}s")


if __name__ == "__main__":
    main()
