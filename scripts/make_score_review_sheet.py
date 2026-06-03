"""score OCR 目視レビュー素材生成スクリプト。

機能D (掛け算式検知) の基盤健全性を「目で見て」判断するための画像を生成する。
対象フレームを以下のカテゴリに分類し、スコア ROI を切り出してカテゴリ別 PNG を出力する。

カテゴリ:
    1. STABLE 中 score=None (機能D誤発火リスクの核心)
    2. 単調性違反 (score が直前読取り値より減少 = 誤読確証)
    3. 低 confidence (score != None だが conf < LOW_CONF_THRESHOLD)

各疑わしいフレームについて:
    - score ROI (1P / 2P) の実画素
    - t_sec / side / OCR 読取り値 / confidence / state / 分類 を注記

出力: data/verify/score_review/ に カテゴリ別 PNG + テキストサマリ

使い方:
    python scripts/make_score_review_sheet.py
    python scripts/make_score_review_sheet.py --sample-interval 0.1
    python scripts/make_score_review_sheet.py --smoke  # v89_match01 のみ
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline
from src.score_ocr import (
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    SCORE_ROI_INK_RATIO_MIN,
    ScoreOcr,
    ScoreReadResult,
    _crop_score_roi,
    compute_score_roi_ink_ratio,
)

# ============================
# 定数
# ============================

# デフォルトサンプル間隔 (秒)
DEFAULT_SAMPLE_INTERVAL_SEC: float = 0.1

# 低 confidence 閾値: score != None だがこれ未満を「低信頼」とみなす
LOW_CONF_THRESHOLD: float = 0.70

# カテゴリ別最大出力フレーム数 (超えた場合は等間隔サブサンプル)
MAX_FRAMES_PER_CATEGORY: int = 40

# 出力ディレクトリ
OUTPUT_DIR: Path = Path("data/verify/score_review")

# contact sheet のレイアウト
SHEET_COLS: int = 4           # 横に並べるパネル数
ROI_DISPLAY_SCALE: float = 2.0  # ROI を拡大して見やすくする倍率
HEADER_HEIGHT: int = 90        # アノテーションヘッダの高さ (px)
SEPARATOR_H: int = 4           # パネル間の縦セパレータ幅
SEPARATOR_V: int = 6           # 行間のセパレータ高さ

# フォント設定
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.38
FONT_SCALE_MAIN = 0.42
FONT_THICKNESS = 1

# カラー定義
COLOR_OK = (80, 220, 80)      # 正常 (BGRで緑)
COLOR_WARN = (40, 200, 255)   # 警告 (BGRでオレンジ)
COLOR_ERR = (60, 60, 255)     # エラー (BGRで赤)
COLOR_INFO = (200, 200, 200)  # 情報テキスト (グレー)
COLOR_TITLE = (255, 255, 255) # タイトル (白)
COLOR_BG = (30, 30, 30)       # 背景 (濃グレー)
COLOR_SEP = (60, 60, 60)      # セパレータ

# 対象動画 (デフォルト)
DEFAULT_TARGETS: list[tuple[str, int]] = [
    ("v89", 1),
    ("v70", 2),
    ("v95", 1),
]

# match_clips ディレクトリ
MATCH_CLIPS_DIR: Path = Path("data/match_clips")


# ============================
# データクラス
# ============================

@dataclass
class SuspiciousFrame:
    """怪しいと判定したフレームの記録。"""
    t_sec: float
    side: str           # "1P" or "2P"
    score: int | None
    confidence: float
    digits: tuple[int | None, ...]
    state_name: str
    ink_ratio: float
    category: str       # "stable_none" / "monotonic_viol" / "low_conf"
    prev_score: int | None  # 単調性違反用: 直前スコア
    roi_img: np.ndarray  # score ROI 実画素 (BGR)


@dataclass
class VideoSummary:
    """1 動画の集計。"""
    video_id: str
    match_num: int
    total_frames: int = 0
    readable_1p: int = 0
    readable_2p: int = 0
    stable_none_1p: int = 0
    stable_none_2p: int = 0
    monotonic_viol_1p: int = 0
    monotonic_viol_2p: int = 0
    low_conf_1p: int = 0
    low_conf_2p: int = 0
    stable_frames_1p: int = 0
    stable_frames_2p: int = 0
    suspicious: list[SuspiciousFrame] = field(default_factory=list)

    # 状態追跡 (stateless 集計の外部 wrapper)
    _last_score_1p: int | None = field(default=None, repr=False)
    _last_score_2p: int | None = field(default=None, repr=False)

    def stable_none_rate_1p(self) -> float:
        if self.stable_frames_1p == 0:
            return 0.0
        return self.stable_none_1p / self.stable_frames_1p

    def stable_none_rate_2p(self) -> float:
        if self.stable_frames_2p == 0:
            return 0.0
        return self.stable_none_2p / self.stable_frames_2p

    def read_rate_1p(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.readable_1p / self.total_frames

    def read_rate_2p(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.readable_2p / self.total_frames


# ============================
# ROI 切り出し + アノテーション
# ============================

def _crop_roi_display(frame: np.ndarray, side: str) -> np.ndarray:
    """score ROI を切り出して表示用にスケールアップする。"""
    roi = _crop_score_roi(frame, side)  # type: ignore[arg-type]
    if roi is None or roi.size == 0:
        return np.zeros((40, 200, 3), dtype=np.uint8)
    h, w = roi.shape[:2]
    nh = int(h * ROI_DISPLAY_SCALE)
    nw = int(w * ROI_DISPLAY_SCALE)
    return cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_NEAREST)


def _category_color(category: str) -> tuple[int, int, int]:
    """カテゴリ別テキスト色を返す。"""
    if category == "stable_none":
        return COLOR_ERR
    elif category == "monotonic_viol":
        return COLOR_WARN
    else:
        return COLOR_INFO


def _build_panel(sf: SuspiciousFrame, panel_width: int) -> np.ndarray:
    """SuspiciousFrame 1 件を 1 パネル画像に変換する。

    パネル構成: HEADER(アノテーション) + ROI実画素
    """
    roi_disp = cv2.resize(
        sf.roi_img,
        (panel_width, int(sf.roi_img.shape[0] * panel_width / max(sf.roi_img.shape[1], 1))),
        interpolation=cv2.INTER_NEAREST,
    )
    roi_h, roi_w = roi_disp.shape[:2]
    total_h = HEADER_HEIGHT + roi_h
    panel = np.full((total_h, panel_width, 3), COLOR_BG, dtype=np.uint8)
    # ROI を下半分に配置
    panel[HEADER_HEIGHT:HEADER_HEIGHT + roi_h, :roi_w] = roi_disp[:, :roi_w]

    # ヘッダにアノテーションを描く
    cat_color = _category_color(sf.category)
    cat_label = {
        "stable_none": "STABLE-None",
        "monotonic_viol": "単調性違反",
        "low_conf": "低Conf",
    }.get(sf.category, sf.category)

    lines = [
        (f"t={sf.t_sec:.2f}s {sf.side}  [{cat_label}]", cat_color),
        (
            f"OCR: {'None' if sf.score is None else sf.score}  "
            f"conf={sf.confidence:.3f}  state={sf.state_name}",
            COLOR_TITLE,
        ),
    ]
    # 単調性違反の場合: 前スコアも表示
    if sf.category == "monotonic_viol" and sf.prev_score is not None:
        lines.append((f"prev={sf.prev_score} → cur={sf.score}  DECREASE!", COLOR_WARN))
    # digits 表示
    dstr = "".join(("?" if d is None else str(d)) for d in sf.digits)
    lines.append((f"digits: {dstr}  ink={sf.ink_ratio:.3f}", COLOR_INFO))

    y = 14
    for text, color in lines:
        cv2.putText(panel, text, (4, y), FONT, FONT_SCALE_SMALL, color, FONT_THICKNESS,
                    cv2.LINE_AA)
        y += 18
    return panel


def _build_sheet(
    frames: list[SuspiciousFrame],
    title: str,
    total_count: int,
    panel_width: int = 320,
) -> np.ndarray | None:
    """SuspiciousFrame リストから contact sheet PNG を生成する。

    Args:
        frames: 表示対象フレーム (サブサンプル済)
        title: シート上部タイトル
        total_count: 実際の総件数 (サブサンプル前)
        panel_width: 1 パネルの横幅 (px)

    Returns:
        contact sheet 画像 (BGR)、フレームが空なら None
    """
    if not frames:
        return None

    panels = [_build_panel(sf, panel_width) for sf in frames]
    # 全パネルの高さを揃える (最大に合わせる)
    max_h = max(p.shape[0] for p in panels)
    unified: list[np.ndarray] = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.full((max_h - p.shape[0], panel_width, 3), COLOR_BG, dtype=np.uint8)
            p = np.vstack([p, pad])
        unified.append(p)

    # SHEET_COLS 列に並べる
    rows: list[np.ndarray] = []
    sep_h_img = np.full((SEPARATOR_V, panel_width * SHEET_COLS + SEPARATOR_H * (SHEET_COLS - 1), 3),
                        COLOR_SEP, dtype=np.uint8)
    sep_v_img = np.full((max_h, SEPARATOR_H, 3), COLOR_SEP, dtype=np.uint8)

    for row_start in range(0, len(unified), SHEET_COLS):
        row_panels = unified[row_start:row_start + SHEET_COLS]
        # 不足分を空白で補填
        while len(row_panels) < SHEET_COLS:
            row_panels.append(np.full((max_h, panel_width, 3), COLOR_BG, dtype=np.uint8))
        row_img_parts: list[np.ndarray] = []
        for i, rp in enumerate(row_panels):
            row_img_parts.append(rp)
            if i < SHEET_COLS - 1:
                row_img_parts.append(sep_v_img)
        rows.append(np.hstack(row_img_parts))
        rows.append(sep_h_img)

    body = np.vstack(rows[:-1]) if rows else np.zeros((1, 1, 3), dtype=np.uint8)

    # タイトルバー
    title_bar_h = 40
    sheet_w = body.shape[1]
    title_bar = np.full((title_bar_h, sheet_w, 3), (50, 50, 50), dtype=np.uint8)
    display = f"{title}  [表示:{len(frames)}/{total_count}件]"
    cv2.putText(title_bar, display, (8, 28), FONT, FONT_SCALE_MAIN,
                COLOR_TITLE, FONT_THICKNESS, cv2.LINE_AA)

    return np.vstack([title_bar, body])


# ============================
# フレームループ処理
# ============================

def _state_name(state: BoardState) -> str:
    """BoardState を文字列ラベルに変換する。"""
    return state.name if hasattr(state, "name") else str(state)


def _update_summary_one_side(
    summary: VideoSummary,
    side: str,
    t_sec: float,
    score: int | None,
    confidence: float,
    digits: tuple[int | None, ...],
    state: BoardState,
    ink_ratio: float,
    roi_img: np.ndarray,
) -> None:
    """1 フレーム・1 サイドの集計と怪しいフレームの記録を行う (50行以内)。"""
    is_1p = (side == "1P")
    st_name = _state_name(state)
    is_stable = (st_name == "STABLE")

    # stable フレーム数カウント
    if is_stable:
        if is_1p:
            summary.stable_frames_1p += 1
        else:
            summary.stable_frames_2p += 1

    # readable カウント
    if score is not None:
        if is_1p:
            summary.readable_1p += 1
        else:
            summary.readable_2p += 1

    # カテゴリ分類 & 記録
    last = summary._last_score_1p if is_1p else summary._last_score_2p
    category: str | None = None

    if is_stable and score is None:
        category = "stable_none"
        if is_1p:
            summary.stable_none_1p += 1
        else:
            summary.stable_none_2p += 1

    elif score is not None and last is not None and score < last:
        category = "monotonic_viol"
        if is_1p:
            summary.monotonic_viol_1p += 1
        else:
            summary.monotonic_viol_2p += 1

    elif score is not None and confidence < LOW_CONF_THRESHOLD:
        category = "low_conf"
        if is_1p:
            summary.low_conf_1p += 1
        else:
            summary.low_conf_2p += 1

    if category is not None:
        summary.suspicious.append(SuspiciousFrame(
            t_sec=t_sec,
            side=side,
            score=score,
            confidence=confidence,
            digits=digits,
            state_name=st_name,
            ink_ratio=ink_ratio,
            category=category,
            prev_score=last if category == "monotonic_viol" else None,
            roi_img=roi_img.copy(),
        ))

    # 最終スコア更新
    if score is not None:
        if is_1p:
            summary._last_score_1p = score
        else:
            summary._last_score_2p = score


def _process_video(
    video_path: Path,
    video_id: str,
    match_num: int,
    sample_interval_sec: float,
) -> VideoSummary:
    """1 動画をフレームループで処理して VideoSummary を返す。"""
    summary = VideoSummary(video_id=video_id, match_num=match_num)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 動画を開けない: {video_path}")
        return summary

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_raw / fps
    print(f"  動画: {video_path.name}  fps={fps:.1f}  duration={duration:.1f}s")
    cap.release()

    # pipeline と OCR を初期化
    try:
        pipeline = RecognitionPipeline.load_default()
        pipeline.set_video_id(video_id)
    except Exception as exc:
        print(f"  [ERROR] pipeline 初期化失敗: {exc}")
        return summary

    ocr = ScoreOcr.load_default()
    interval_frames = max(1, int(fps * sample_interval_sec))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    sample_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % interval_frames == 0:
            t_sec = frame_idx / fps
            _process_one_frame(frame, pipeline, ocr, sample_idx, t_sec, summary)
            sample_idx += 1
        frame_idx += 1

    cap.release()
    summary.total_frames = sample_idx
    return summary


def _process_one_frame(
    frame: np.ndarray,
    pipeline: RecognitionPipeline,
    ocr: ScoreOcr,
    sample_idx: int,
    t_sec: float,
    summary: VideoSummary,
) -> None:
    """1 フレームを処理して summary を更新する。"""
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

    pr = pipeline.update(sample_idx, t_sec, frame)
    ocr_res: ScoreReadResult = ocr.read(frame)

    for side in ("1P", "2P"):
        is_1p = (side == "1P")
        score = ocr_res.score_1p if is_1p else ocr_res.score_2p
        conf = ocr_res.confidence_1p if is_1p else ocr_res.confidence_2p
        digits = ocr_res.digits_1p if is_1p else ocr_res.digits_2p
        state = pr.p1.state if is_1p else pr.p2.state
        roi = _crop_score_roi(frame, side)  # type: ignore[arg-type]
        ink = compute_score_roi_ink_ratio(roi) if roi is not None else 0.0
        if roi is None:
            roi = np.zeros((40, 200, 3), dtype=np.uint8)

        _update_summary_one_side(
            summary=summary,
            side=side,
            t_sec=t_sec,
            score=score,
            confidence=conf,
            digits=digits,
            state=state,
            ink_ratio=ink,
            roi_img=roi,
        )


# ============================
# サブサンプル + シート出力
# ============================

def _subsample(items: list, max_n: int) -> list:
    """items を等間隔サブサンプルして最大 max_n 件に絞る。"""
    if len(items) <= max_n:
        return items
    step = len(items) / max_n
    indices = [int(i * step) for i in range(max_n)]
    return [items[i] for i in indices]


def _save_category_sheet(
    summary: VideoSummary,
    category: str,
    output_dir: Path,
) -> Path | None:
    """カテゴリ別フレームを contact sheet として保存する。"""
    frames = [sf for sf in summary.suspicious if sf.category == category]
    total = len(frames)
    if total == 0:
        return None

    sampled = _subsample(frames, MAX_FRAMES_PER_CATEGORY)
    cat_short = {
        "stable_none": "stable_none",
        "monotonic_viol": "mono_viol",
        "low_conf": "low_conf",
    }.get(category, category)

    title = (
        f"{summary.video_id}_match{summary.match_num:02d}"
        f" | {cat_short}"
        f" | stable_none={summary.stable_none_rate_1p():.3f}(1P)"
        f"/{summary.stable_none_rate_2p():.3f}(2P)"
    )
    sheet = _build_sheet(sampled, title, total)
    if sheet is None:
        return None

    fname = f"{summary.video_id}_match{summary.match_num:02d}_{cat_short}.png"
    out_path = output_dir / fname
    cv2.imwrite(str(out_path), sheet)
    return out_path


def _print_video_summary(summary: VideoSummary) -> None:
    """コンソールに動画集計を出力する。"""
    vid = f"{summary.video_id}_m{summary.match_num:02d}"
    print(f"\n  [{vid}] frames={summary.total_frames}")
    print(f"    read_rate: 1P={summary.read_rate_1p():.3f}  2P={summary.read_rate_2p():.3f}")
    print(
        f"    STABLE-None: 1P={summary.stable_none_1p}件 "
        f"(rate={summary.stable_none_rate_1p():.4f})  "
        f"2P={summary.stable_none_2p}件 "
        f"(rate={summary.stable_none_rate_2p():.4f})"
    )
    print(
        f"    単調性違反: 1P={summary.monotonic_viol_1p}  "
        f"2P={summary.monotonic_viol_2p}"
    )
    print(
        f"    低Conf: 1P={summary.low_conf_1p}  "
        f"2P={summary.low_conf_2p}"
    )


# ============================
# エントリポイント
# ============================

def _build_targets(smoke: bool) -> list[tuple[str, int]]:
    """処理対象リストを返す。"""
    if smoke:
        return [("v89", 1)]
    return DEFAULT_TARGETS


def _find_video_path(video_id: str, match_num: int) -> Path | None:
    """match_clips ディレクトリから動画ファイルを探す。"""
    clip_dir = MATCH_CLIPS_DIR / video_id
    candidates = [
        clip_dir / f"{video_id}_match{match_num:02d}.mp4",
        clip_dir / f"match_{video_id}_{match_num:02d}.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # glob フォールバック
    found = list(clip_dir.glob(f"*match*{match_num:02d}*.mp4"))
    return found[0] if found else None


def main(argv: list[str] | None = None) -> int:
    """スクリプトエントリポイント。"""
    parser = argparse.ArgumentParser(
        description="score OCR 目視レビュー素材生成",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC,
        help=f"サンプリング間隔 (秒, default={DEFAULT_SAMPLE_INTERVAL_SEC})",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="スモーク: v89_match01 のみ実行",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"出力ディレクトリ (default={OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    targets = _build_targets(smoke=args.smoke)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"score OCR 目視レビュー素材生成")
    print(f"sample_interval={args.sample_interval}s  対象={len(targets)}動画")
    print(f"出力先: {out_dir.resolve()}")

    all_summaries: list[VideoSummary] = []
    generated_pngs: list[Path] = []

    for video_id, match_num in targets:
        video_path = _find_video_path(video_id, match_num)
        if video_path is None:
            print(f"\n[SKIP] {video_id} match{match_num:02d}: ファイル見つからず")
            continue

        print(f"\n処理中: {video_id} match{match_num:02d} ...")
        summary = _process_video(video_path, video_id, match_num, args.sample_interval)
        all_summaries.append(summary)
        _print_video_summary(summary)

        # カテゴリ別シート保存
        for cat in ("stable_none", "monotonic_viol", "low_conf"):
            p = _save_category_sheet(summary, cat, out_dir)
            if p is not None:
                generated_pngs.append(p)
                count = len([sf for sf in summary.suspicious if sf.category == cat])
                print(f"    [{cat}] {count}件 → {p.name}")
            else:
                print(f"    [{cat}] 0件 (画像なし)")

    # テキストサマリを出力
    summary_txt = out_dir / "summary.txt"
    _write_summary_txt(all_summaries, summary_txt, args.sample_interval)

    # 結果報告
    print("\n" + "=" * 60)
    print("生成した PNG (Windows パス):")
    proj_root_win = str(out_dir.resolve()).replace("/mnt/c/", "C:\\").replace("/", "\\")
    for p in generated_pngs:
        win_path = str(p.resolve()).replace("/mnt/c/", "C:\\").replace("/", "\\")
        print(f"  {win_path}")
    print(f"\nサマリ: {summary_txt.resolve()}")
    print("=" * 60)

    _print_final_report(all_summaries)
    return 0


def _write_summary_txt(
    summaries: list[VideoSummary],
    out_path: Path,
    sample_interval: float,
) -> None:
    """集計テキストファイルを書き出す。"""
    lines: list[str] = [
        "score OCR 信頼性 目視レビュー 集計サマリ",
        f"sample_interval={sample_interval}s",
        f"LOW_CONF_THRESHOLD={LOW_CONF_THRESHOLD}",
        "",
        f"{'動画':<20} {'read_rate_1P':>12} {'read_rate_2P':>12} "
        f"{'stable_none_1P':>14} {'stable_none_2P':>14} "
        f"{'単調性違反1P':>10} {'単調性違反2P':>10} "
        f"{'低conf1P':>8} {'低conf2P':>8}",
        "-" * 100,
    ]
    for s in summaries:
        label = f"{s.video_id}_m{s.match_num:02d}"
        lines.append(
            f"{label:<20}"
            f" {s.read_rate_1p():>12.4f}"
            f" {s.read_rate_2p():>12.4f}"
            f" {s.stable_none_rate_1p():>14.4f}"
            f" {s.stable_none_rate_2p():>14.4f}"
            f" {s.monotonic_viol_1p:>10}"
            f" {s.monotonic_viol_2p:>10}"
            f" {s.low_conf_1p:>8}"
            f" {s.low_conf_2p:>8}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _print_final_report(summaries: list[VideoSummary]) -> None:
    """最終集計をコンソールに出力する。"""
    print("\n[最終集計]")
    print(
        f"{'動画':<20} {'read_1P':>8} {'read_2P':>8}"
        f" {'stbNone1P':>10} {'stbNone2P':>10}"
        f" {'mono1P':>7} {'mono2P':>7}"
        f" {'lconf1P':>7} {'lconf2P':>7}"
    )
    print("-" * 90)
    for s in summaries:
        label = f"{s.video_id}_m{s.match_num:02d}"
        print(
            f"{label:<20}"
            f" {s.read_rate_1p():>8.4f}"
            f" {s.read_rate_2p():>8.4f}"
            f" {s.stable_none_rate_1p():>10.4f}"
            f" {s.stable_none_rate_2p():>10.4f}"
            f" {s.monotonic_viol_1p:>7}"
            f" {s.monotonic_viol_2p:>7}"
            f" {s.low_conf_1p:>7}"
            f" {s.low_conf_2p:>7}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
