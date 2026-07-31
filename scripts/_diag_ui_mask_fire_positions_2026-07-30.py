"""UI マスク (UiMaskMatcher.is_ui) の発火位置を実測する診断スクリプト (2026-07-30)。

背景:
    cv2.matchTemplate が認識時間の 69.4% を占め、そのうち UI マスク判定
    (src/ui_mask.py:96) が単独で全体の 45.3% (memory
    `project_recognition_profile_matchtemplate_2026-07-30`)。
    src/ui_mask.py の docstring は「r01 c2 の×マーク」と特定セル位置を示唆
    しており、発火位置が限定的なら位置ゲートで大半を削減できる可能性がある。
    本スクリプトは is_ui が実際にどの (side, row, col) で発火するか、
    どのテンプレートで発火するか、NCC スコア分布はどうかを実測する。

厳守事項:
    - src/ は一切変更しない。UiMaskMatcher.match と BoardRegion.cell_sample_rect
      をスクリプト内でモンキーパッチし、with ブロックで必ず元に戻す。
    - is_ui(bgr_patch) には (row, col) が引数として渡されない。呼び出し元
      (src/hybrid_classifier.py:196-200 の HybridClassifier.classify_batch)
      でも patch は元 frame への view のままなので、patch の生ポインタと
      frame の生ポインタの差分から (x1, y1) を逆算し、cell_sample_rect が
      記録した (row, col) -> (x1, y1) 表と突き合わせて位置を復元する
      (ジオメトリ照合)。復元できなければ「unresolved」として集計し、
      位置分布には含めない (推測で埋めない)。

実行例 (WSL, 他エージェントの video_c60 1800-1890s 収集と競合しない時間帯を使用):
    nice -n 19 env PYTHONPATH=. ./venv/bin/python \
        -m scripts._diag_ui_mask_fire_positions_2026-07-30 --smoke-test

    nice -n 19 env PYTHONPATH=. ./venv/bin/python \
        -m scripts._diag_ui_mask_fire_positions_2026-07-30
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.board import HIDDEN_ROWS
from src.image_reader import BoardRegion
from src.recognition_pipeline import RecognitionPipeline
from src.ui_mask import UiMaskMatcher, UiMatchResult

# --- 定数 (マジックナンバー禁止のため名前を与える) ---
TARGET_W: int = 1920
TARGET_H: int = 1080
IMG_MID_X: int = TARGET_W // 2   # P1(x≈282)/P2(x≈1258) を判別する閾値
TOLERANCE_PX: int = 3            # 境界クリップ等での逆算誤差許容 (px)
MAX_SAVED_PER_TEMPLATE: int = 2  # 視覚裏取り用に保存するクロップ数の上限/(テンプレ×side)
SCORE_BUCKET_WIDTH: float = 0.05  # 発火しなかった NCC スコアのヒストグラム粒度
OUT_DIR: Path = Path("data/verify/ui_mask_fire_2026-07-30")
CONTEXT_MARGIN_PX: int = 120      # 発火セル周辺を見せるための可視化マージン (px)
BOX_COLOR_BGR: tuple[int, int, int] = (0, 0, 255)  # 発火セル矩形のハイライト色 (赤)

# 計測区間: (video_stem, start_sec, n_frames) のリスト。
# video_c60 の 1800-1890s は他エージェントが収集中のため避け、時間帯をずらした。
# 2026-07-30 追記: コーディネータ指示によりuserドメイン知識で発火位置が
# (row=1,col=2) 単セルと確定したため、「位置網羅性」の優先度を下げ、
# 「(1,2)近傍の滲み有無」の確認に絞って区間数・総フレーム数を縮小
# (他エージェントの並列ジョブとの CPU 競合回避)。3動画×1区間×400フレーム。
DEFAULT_WINDOWS: tuple[tuple[str, float, int], ...] = (
    ("video_c60", 1500.0, 400),
    ("video_c56", 1500.0, 400),
    ("video_c65", 1500.0, 400),
)


@dataclass(frozen=True)
class RectRecord:
    """cell_sample_rect が返した1セル分の矩形記録。"""

    side: str
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class FireRecord:
    """is_ui=True で発火した1回分の記録。位置復元できなければ None。"""

    side: str | None
    row: int | None
    col: int | None
    visible_row: int | None
    template: str
    score: float
    video: str
    t_sec: float


def _resolve_position(
    patch: np.ndarray, frame: np.ndarray | None, rect_log: list[RectRecord],
) -> RectRecord | None:
    """patch の frame 内オフセットを逆算し、記録済み矩形と突き合わせる。

    patch は frame[y1:y2, x1:x2] の view (コピーなし) である前提。
    view でなければ (=ポインタ差分が範囲外) None を返し unresolved 扱いにする。
    """
    if frame is None or patch.size == 0:
        return None
    try:
        frame_ptr = frame.__array_interface__["data"][0]
        patch_ptr = patch.__array_interface__["data"][0]
    except (AttributeError, KeyError, TypeError):
        return None
    offset = patch_ptr - frame_ptr
    if offset < 0 or frame.ndim != 3:
        return None
    row_stride, col_stride = frame.strides[0], frame.strides[1]
    y1, rem = divmod(offset, row_stride)
    x1 = rem // col_stride
    if y1 >= frame.shape[0] or x1 >= frame.shape[1]:
        return None
    best: RectRecord | None = None
    best_dist = TOLERANCE_PX + 1
    for rec in rect_log:
        dist = abs(rec.x1 - x1) + abs(rec.y1 - y1)
        if dist < best_dist:
            best_dist, best = dist, rec
    return best if best_dist <= TOLERANCE_PX else None


def _make_context_crop(frame: np.ndarray, rec: RectRecord) -> np.ndarray:
    """発火セル周辺を切り出し、実際にサンプルされた矩形を赤枠で示す。

    タイル 32x30px 程度の生パッチだけでは×印か赤ぷよ誤検出かの目視判定が
    困難なため、周辺セルも写る大きさで保存する (視覚的裏取り用)。
    """
    h, w = frame.shape[:2]
    cx1 = max(0, rec.x1 - CONTEXT_MARGIN_PX)
    cy1 = max(0, rec.y1 - CONTEXT_MARGIN_PX)
    cx2 = min(w, rec.x2 + CONTEXT_MARGIN_PX)
    cy2 = min(h, rec.y2 + CONTEXT_MARGIN_PX)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    box_pt1 = (rec.x1 - cx1, rec.y1 - cy1)
    box_pt2 = (rec.x2 - cx1, rec.y2 - cy1)
    cv2.rectangle(crop, box_pt1, box_pt2, BOX_COLOR_BGR, 1)
    return crop


class CellRectRecorder:
    """BoardRegion.cell_sample_rect をモンキーパッチし (row,col)->(x1,y1) を記録する。

    with ブロックを抜けると必ず元の実装に戻す。挙動 (戻り値) は一切変えない。
    """

    def __init__(self, rect_log: list[RectRecord]) -> None:
        self._original = BoardRegion.cell_sample_rect
        self._rect_log = rect_log

    def __enter__(self) -> "CellRectRecorder":
        original, rect_log = self._original, self._rect_log

        def patched(
            region_self: BoardRegion, row: int, col: int,
        ) -> tuple[int, int, int, int]:
            x1, y1, x2, y2 = original(region_self, row, col)
            side = "P1" if region_self.x < IMG_MID_X else "P2"
            rect_log.append(RectRecord(
                side=side, row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2,
            ))
            return x1, y1, x2, y2

        BoardRegion.cell_sample_rect = patched
        return self

    def __exit__(self, *_exc: object) -> None:
        BoardRegion.cell_sample_rect = self._original


class UiMaskRecorder:
    """UiMaskMatcher.match をモンキーパッチし発火位置・テンプレ・スコアを記録する。

    with ブロックを抜けると必ず元の実装に戻す。挙動 (戻り値) は一切変えない。
    """

    def __init__(self) -> None:
        self._original = UiMaskMatcher.match
        self.rect_log: list[RectRecord] = []
        self.current_frame: np.ndarray | None = None
        self.current_video: str = ""
        self.current_t_sec: float = 0.0
        self.fired_records: list[FireRecord] = []
        self.crops_to_save: list[tuple[FireRecord, np.ndarray]] = []
        self.save_count_by_key: Counter[str] = Counter()
        self.template_fire_count: Counter[str] = Counter()
        self.best_template_count: Counter[str] = Counter()
        self.score_hist_not_fired: dict[int, int] = defaultdict(int)
        self.score_hist_fired: list[float] = []
        self.n_calls: int = 0
        self.n_resolved: int = 0
        self.n_unresolved: int = 0

    def __enter__(self) -> "UiMaskRecorder":
        original = self._original

        def patched(matcher_self: UiMaskMatcher, bgr_patch: np.ndarray) -> UiMatchResult:
            result = original(matcher_self, bgr_patch)
            self._record(bgr_patch, result)
            return result

        UiMaskMatcher.match = patched
        return self

    def __exit__(self, *_exc: object) -> None:
        UiMaskMatcher.match = self._original

    def _record(self, bgr_patch: np.ndarray, result: UiMatchResult) -> None:
        """1回の match() 呼び出しを集計する (発火有無に関わらず)。"""
        self.n_calls += 1
        tmpl = result.template_name or "(none)"
        self.best_template_count[tmpl] += 1
        if not result.is_ui:
            bucket = int(round(result.score / SCORE_BUCKET_WIDTH))
            self.score_hist_not_fired[bucket] += 1
            return
        self.score_hist_fired.append(result.score)
        self.template_fire_count[tmpl] += 1
        pos = _resolve_position(bgr_patch, self.current_frame, self.rect_log)
        if pos is None:
            self.n_unresolved += 1
        else:
            self.n_resolved += 1
        rec = FireRecord(
            side=pos.side if pos else None,
            row=pos.row if pos else None,
            col=pos.col if pos else None,
            visible_row=(pos.row - HIDDEN_ROWS) if pos else None,
            template=tmpl, score=result.score,
            video=self.current_video, t_sec=self.current_t_sec,
        )
        self.fired_records.append(rec)
        if pos is not None and self.current_frame is not None:
            # テンプレ×サイド単位で上限管理 (P1/P2 両方の視覚裏取り例を残すため)
            save_key = f"{tmpl}:{pos.side}"
            self.save_count_by_key[save_key] += 1
            if self.save_count_by_key[save_key] <= MAX_SAVED_PER_TEMPLATE:
                context = _make_context_crop(self.current_frame, pos)
                self.crops_to_save.append((rec, context))


def _ensure_target_size(frame: np.ndarray) -> np.ndarray:
    """フレームを 1920x1080 に揃える (本番同値)。"""
    if frame.shape[:2] != (TARGET_H, TARGET_W):
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return frame


def build_pipeline() -> RecognitionPipeline:
    """本番 (レンダ) と同じ設定でパイプラインを作る (課題指定の設定値)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
    )


def run_window(
    recorder: UiMaskRecorder, video_path: Path, start_sec: float, n_frames: int,
) -> int:
    """1区間 (n_frames 枚) を処理する。

    is_ui は STABLE 確定前の生セル分類でも呼ばれるため、位置復元用の
    current_frame / rect_log は全フレームで正しく更新する (旧実装バグの
    修正: 以前は先頭 WARMUP_FRAMES 分だけ更新をスキップしており、その間の
    発火が「前区間の最終フレーム」という不整合な frame 参照で解決を試み
    られ unresolved になっていた。fps 計測用の助走は本スクリプトの主目的
    である位置分布には不要なため撤去し、全フレームを計測対象にする)。
    """
    pipeline = build_pipeline()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    done = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = _ensure_target_size(frame)
        fi = start_frame + i
        t_sec = fi / fps
        recorder.rect_log.clear()
        recorder.current_frame = frame
        recorder.current_video = video_path.stem
        recorder.current_t_sec = t_sec
        try:
            pipeline.update(fi, t_sec, frame)
        except Exception as exc:  # noqa: BLE001 - 診断用、1フレーム失敗で全体を止めない
            print(f"  [警告] {video_path.stem} t={t_sec:.1f}s で例外: {exc}")
            continue
        done += 1
    cap.release()
    return done


@dataclass
class WindowResult:
    """1区間の処理結果サマリ。"""

    video: str
    start_sec: float
    n_frames: int
    done_frames: int
    wall_sec: float


def _save_crops(recorder: UiMaskRecorder) -> list[Path]:
    """発火セルの実フレーム切り抜きを保存する (×印か赤ぷよ誤検出かの目視裏取り用)。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, (rec, patch) in enumerate(recorder.crops_to_save):
        pos_label = (
            f"{rec.side}_r{rec.row}_c{rec.col}" if rec.side else "unresolved"
        )
        name = f"{rec.video}_t{rec.t_sec:.1f}_{pos_label}_{rec.template}_{idx}.png"
        path = OUT_DIR / name
        cv2.imwrite(str(path), patch)
        saved.append(path)
    return saved


def _print_position_breakdown(recorder: UiMaskRecorder) -> None:
    """side x row x col の発火分布を表示する。"""
    counts: Counter[tuple[str | None, int | None, int | None]] = Counter()
    for rec in recorder.fired_records:
        counts[(rec.side, rec.row, rec.col)] += 1
    total = len(recorder.fired_records)
    print(f"\n=== 発火位置分布 (総発火回数 {total}) ===")
    for (side, row, col), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        visible_row = (row - HIDDEN_ROWS) if row is not None else None
        label = (
            f"side={side} row(raw)={row} visible_row={visible_row} col={col}"
            if side is not None else "unresolved (位置復元不可)"
        )
        pct = n / total * 100 if total > 0 else 0.0
        print(f"  {n:>6}回 ({pct:5.1f}%)  {label}")


def _print_template_breakdown(recorder: UiMaskRecorder) -> None:
    """テンプレート別の発火回数・argmax勝利回数 (telop/match_end 4枚含む) を表示する。"""
    print("\n=== テンプレート別内訳 ===")
    all_templates = set(recorder.best_template_count) | set(recorder.template_fire_count)
    for tmpl in sorted(all_templates):
        fired = recorder.template_fire_count.get(tmpl, 0)
        best = recorder.best_template_count.get(tmpl, 0)
        print(f"  {tmpl:<28} 発火 {fired:>6}回 / argmax勝利 {best:>6}回 "
              f"(全 match() 呼出 {recorder.n_calls} 回中)")


def _print_score_distribution(recorder: UiMaskRecorder) -> None:
    """NCC スコア分布 (発火/非発火) と閾値との余裕を表示する。"""
    fired = np.array(recorder.score_hist_fired, dtype=np.float64)
    print("\n=== NCC スコア分布 ===")
    if fired.size > 0:
        pcts = np.percentile(fired, [0, 25, 50, 75, 100])
        print(f"  発火した側 (is_ui=True, n={fired.size}): "
              f"min={pcts[0]:.3f} p25={pcts[1]:.3f} p50={pcts[2]:.3f} "
              f"p75={pcts[3]:.3f} max={pcts[4]:.3f}")
    else:
        print("  発火した側: 記録なし (発火ゼロ)")
    not_fired_total = sum(recorder.score_hist_not_fired.values())
    near_miss = sum(
        n for bucket, n in recorder.score_hist_not_fired.items()
        if 0.60 <= bucket * SCORE_BUCKET_WIDTH < 0.75
    )
    max_bucket = max(recorder.score_hist_not_fired) if recorder.score_hist_not_fired else 0
    print(f"  発火しなかった側: n={not_fired_total}, "
          f"うち閾値未満0.15以内 (0.60-0.75) の近接ミス={near_miss} "
          f"({near_miss / not_fired_total * 100 if not_fired_total else 0:.2f}%), "
          f"観測最大スコア≈{max_bucket * SCORE_BUCKET_WIDTH:.3f}")


def main() -> None:
    """全区間を処理し、位置・テンプレ・スコアの集計結果を報告する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="data/frames")
    ap.add_argument("--smoke-test", action="store_true",
                     help="各区間を30フレームのみ処理する疎通確認モード")
    args = ap.parse_args()

    windows = DEFAULT_WINDOWS
    if args.smoke_test:
        windows = tuple((v, s, 30) for v, s, _ in DEFAULT_WINDOWS)

    print(f"UI マスク閾値 = {UiMaskMatcher.load_default()._threshold}")
    results: list[WindowResult] = []
    with UiMaskRecorder() as ui_rec, CellRectRecorder(ui_rec.rect_log):
        for video_name, start_sec, n_frames in windows:
            video_path = Path(args.video_dir) / f"{video_name}.mp4"
            print(f"\n>>> {video_path} start={start_sec}s frames={n_frames}")
            t0 = time.perf_counter()
            done = run_window(ui_rec, video_path, start_sec, n_frames)
            wall = time.perf_counter() - t0
            print(f"    完了 {done}/{n_frames} フレーム, {wall:.1f}秒 "
                  f"({done / wall if wall > 0 else 0:.2f}fps)")
            results.append(WindowResult(video_name, start_sec, n_frames, done, wall))

        _print_position_breakdown(ui_rec)
        _print_template_breakdown(ui_rec)
        _print_score_distribution(ui_rec)
        print(f"\n=== 位置復元率 === resolved={ui_rec.n_resolved} "
              f"unresolved={ui_rec.n_unresolved} "
              f"(発火全体 {len(ui_rec.fired_records)} 回中)")
        saved = _save_crops(ui_rec)
        print(f"\n=== 保存したクロップ ({len(saved)}枚) ===")
        for p in saved:
            print(f"  {p.resolve()}")

    total_done = sum(r.done_frames for r in results)
    total_wall = sum(r.wall_sec for r in results)
    print(f"\n=== 総計 === フレーム数 {total_done}, 総時間 {total_wall:.1f}秒, "
          f"平均 {total_done / total_wall if total_wall > 0 else 0:.2f}fps")

    summary_path = OUT_DIR / "summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "n_calls": ui_rec.n_calls,
        "n_fired": len(ui_rec.fired_records),
        "n_resolved": ui_rec.n_resolved,
        "n_unresolved": ui_rec.n_unresolved,
        "template_fire_count": dict(ui_rec.template_fire_count),
        "best_template_count": dict(ui_rec.best_template_count),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nサマリ JSON: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
