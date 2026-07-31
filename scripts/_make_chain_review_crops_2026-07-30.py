"""連鎖数OCR食い違い事例(9件)の「連鎖数が読める最小限の切り抜き」を生成する。

## 背景 (userタスク指定 2026-07-30)

data/verify/chain_count_disagreement_2026-07-29/ の contact_sheet.png は
9事例のsummary.pngを縦連結したもので、1事例あたりが小さくなり焼き込んだ
文字が読めない失敗作だった (userがスマホで見て指摘)。本スクリプトは
文字情報を画像に一切焼き込まず、判断に必要な「N れんさ!」ポップアップ
部分だけを大きく切り抜く (1事例1枚、必要な場合のみ前後を追加)。

## 事前調査 (本スクリプト作成前に実施、scratchpad限定の探索スクリプトで実施)

9事例それぞれについて、低閾値(0.35)OCRで候補時刻を洗い出し→時間クラスタ
リング→実フレームを目視確認、という手順で最大連鎖数の表示時刻を特定した。
この過程で **2件、既存の目視判定 (memory `project_chain_count_both_
untrustworthy_2026-07-30` 記載) と食い違う結果**を得た:
  - B (c11 1P g3): 旧判定=11 → 実際は「12」まで進行 (t≈461.6s に実フレーム
    目視で確認、旧判定の11はその1ステップ前)
  - G (c22 2P g6): 旧判定=13 → 実際は「12」で終了 (t≈654.3s が最終、13は
    未観測。旧判定の13はOCR誤検出由来と推測されるが未検証)
他7件 (A/E/H/I/C2/D2/F2) は目視再確認の結果、旧判定と一致した (Hのみ
「単一の最大値」という問いが成立しない事例と再確認、詳細はdocstring末尾)。

## 切り出し方式

`src.chain_count_ocr` の `_ensure_1080p()` (解像度正規化、video_c11は実解像度
1280x720のため必須) → `_crop_search_roi()` (盤面+検索マージンのROI) を経由し、
production の `ChainCountOcr.read_side()` が返す検出位置 (`location`) を中心に
生成マージン分だけ広げてサブクロップする。2桁表示 (10-19連鎖) は本番の
2桁結合ロジックがこの1フレームだけでは失敗する場合があるため、production
閾値(0.60)で検出できない事例は探索時に確認済みの低閾値検出位置を手動で
指定する (G のみ、本ファイル末尾 `_MANUAL_LOCATION_OVERRIDES` 参照)。

## 出力

data/verify/chain_review_crops_2026-07-30/ に `<事例名>_max.jpg` (必須) と
`<事例名>_prev.jpg` / `_next.jpg` (数字が明瞭に読める場合は省略、本タスクでは
B・G・H のみ追加: B/Gは食い違い発覚事例の裏取り証跡として1つ前のステップを、
Hは「別々の2つの1連鎖」であることを示すため2件目を追加)。
manifest.json に事例ごとのメタ情報・readable判定・生成ファイル一覧を書く。

本番の src/ scripts/ 既存コードは一切変更しない (読み取り専用の追加スクリプト)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent

from src.chain_count_ocr import (  # noqa: E402
    ChainCountOcr,
    _crop_search_roi,
    _ensure_1080p,
)

Side = Literal["1P", "2P"]

VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "chain_review_crops_2026-07-30"

# 検索ROI内でのサブクロップ余白 (px、_crop_search_roi() が返すROI座標系)。
# 数字グリフ (代表110x70px) + "れんさ!" テキストが右に続くぶんだけ右側を
# 広めに取る (2桁表示・単桁表示のどちらでも収まるよう実測で調整した値)。
CROP_PAD_LEFT_PX: int = 20
CROP_PAD_TOP_PX: int = 30
CROP_PAD_RIGHT_PX: int = 320
CROP_PAD_BOTTOM_PX: int = 140

# 見やすさのための最終拡大倍率 (サブクロップ後、JPEG保存前に適用)。
FINAL_UPSCALE: float = 2.5

# JPEG品質・目標サイズ (userタスク指定: 1枚80KB目安、全体2MB以内)。
JPEG_QUALITY: int = 85
TARGET_MAX_BYTES: int = 80_000
MIN_JPEG_QUALITY: int = 40


@dataclass(frozen=True)
class CropTarget:
    """1枚の切り抜き画像の生成指定。"""

    event_label: str
    suffix: Literal["max", "prev", "next"]
    video_stem: str
    side: Side
    t_sec: float
    manual_location: tuple[int, int] | None = None  # production閾値で未検出の場合の手動指定


# 事前調査 (探索スクリプト+目視確認) で確定した生成対象。
# 各時刻・座標は本ファイル冒頭docstringの調査結果に基づく。
CROP_TARGETS: tuple[CropTarget, ...] = (
    CropTarget("A", "max", "c11", "1P", 586.0),
    CropTarget("B", "prev", "c11", "1P", 460.4),
    CropTarget("B", "max", "c11", "1P", 461.6),
    CropTarget("E", "max", "c21", "1P", 304.7),
    CropTarget("G", "prev", "c22", "2P", 653.0),
    CropTarget("G", "max", "c22", "2P", 654.3, manual_location=(347, 585)),
    CropTarget("H", "max", "c22", "2P", 696.1),
    CropTarget("H", "next", "c22", "2P", 702.9),
    CropTarget("I", "max", "c22", "1P", 290.8),
    CropTarget("C2", "max", "c21", "2P", 693.5),
    # D2: production OCR (閾値0.60) は左端の演出アイコンを "4" の誤検出として
    # 拾ってしまう(NG位置 (30,297))ため手動で正しい位置を指定する
    # (2026-07-30 発見、既知の digit_2/3/4 信頼度不安定の一事例、
    # src/chain_count_ocr.py モジュールdocstring既知課題と符合)。
    CropTarget("D2", "max", "c16", "2P", 853.9, manual_location=(43, 391)),
    CropTarget("F2", "max", "c21", "2P", 586.8),
)


def _read_frame_1080p(video_stem: str, t_sec: float) -> np.ndarray | None:
    """動画から指定時刻のフレームを読み1080pに正規化する。"""
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _ensure_1080p(frame)


def _locate_popup(
    frame_1080p: np.ndarray, side: Side, manual_location: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """検索ROIを切り出し、ポップアップ位置 (ROI座標系) を決定する。

    manual_location が指定されている場合はそれを最優先する (2026-07-30 発見:
    production閾値0.60でも digit_2/3/4 等が無関係の演出アイコンを誤検出する
    ケースがあり、目視確認済みの正しい位置を上書きで使う必要があるため)。
    未指定の場合のみ production の `ChainCountOcr.read_side()` 検出位置を使う。
    """
    roi = _crop_search_roi(frame_1080p, side)
    if roi is None:
        return np.zeros((10, 10, 3), dtype=np.uint8), None
    if manual_location is not None:
        return roi, manual_location
    ocr = ChainCountOcr.load_default()
    result = ocr.read_side(frame_1080p, side)
    return roi, result.location


def _subcrop_and_upscale(roi: np.ndarray, location: tuple[int, int]) -> np.ndarray:
    """検出位置を中心に余白付きでサブクロップし、拡大する。"""
    loc_x, loc_y = location
    h, w = roi.shape[:2]
    x1 = max(0, loc_x - CROP_PAD_LEFT_PX)
    y1 = max(0, loc_y - CROP_PAD_TOP_PX)
    x2 = min(w, loc_x + CROP_PAD_RIGHT_PX)
    y2 = min(h, loc_y + CROP_PAD_BOTTOM_PX)
    sub = roi[y1:y2, x1:x2]
    new_size = (int(sub.shape[1] * FINAL_UPSCALE), int(sub.shape[0] * FINAL_UPSCALE))
    return cv2.resize(sub, new_size, interpolation=cv2.INTER_CUBIC)


def _save_jpeg_under_budget(img: np.ndarray, out_path: Path) -> int:
    """80KB以下になるようJPEG品質を段階的に下げながら保存し、バイト数を返す。"""
    quality = JPEG_QUALITY
    while quality >= MIN_JPEG_QUALITY:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and buf.nbytes <= TARGET_MAX_BYTES:
            out_path.write_bytes(buf.tobytes())
            return buf.nbytes
        quality -= 10
    # 最低品質でも収まらない場合はそのまま保存 (正直に大きいまま出す)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, MIN_JPEG_QUALITY])
    out_path.write_bytes(buf.tobytes())
    return buf.nbytes


def _process_target(target: CropTarget) -> tuple[str, int] | None:
    """1件のCropTargetを処理し、(ファイル名, バイト数) を返す (失敗時 None)。"""
    frame = _read_frame_1080p(target.video_stem, target.t_sec)
    if frame is None:
        print(f"[MISS] {target.event_label}_{target.suffix}: フレーム取得失敗")
        return None
    roi, location = _locate_popup(frame, target.side, target.manual_location)
    if location is None:
        print(f"[MISS] {target.event_label}_{target.suffix}: 検出位置なし")
        return None
    cropped = _subcrop_and_upscale(roi, location)
    out_name = f"{target.event_label}_{target.suffix}.jpg"
    out_path = OUT_DIR / out_name
    n_bytes = _save_jpeg_under_budget(cropped, out_path)
    print(f"[OK] {out_name}: {n_bytes/1024:.1f}KB")
    return out_name, n_bytes


@dataclass(frozen=True)
class EventMeta:
    """事例ごとのメタ情報 (マニフェスト出力用)。"""

    label: str
    video_id: str
    side: Side
    game_idx: int
    t_max_sec: float
    delta_score: int
    simulate_chain_count: int
    screen_chain_count: int
    my_judged_true_value: int | str
    prev_recorded_true_value: int | str | None  # 前回(2026-07-29)の目視判定と食い違う場合のみ設定
    readable: bool
    readable_note: str


# 前回目視判定 (memory `project_chain_count_both_untrustworthy_2026-07-30`) と
# 本タスクでの再確認結果。B・G は食い違いを検出した (docstring冒頭参照)。
EVENT_META: tuple[EventMeta, ...] = (
    EventMeta("A", "c11", "1P", 6, 586.0, 327, 1, 2, 2, None,
              True, "2 れんさ! が明瞭"),
    EventMeta("B", "c11", "1P", 3, 461.6, 38220, 10, 9, 12, 11,
              True, "12 が明瞭。t=460.4に11の1つ前ステップも確認 (B_prev.jpg)"),
    EventMeta("E", "c21", "1P", 2, 304.7, 20540, 2, 8, 9, None,
              True, "9 れんさ! が明瞭"),
    EventMeta("G", "c22", "2P", 6, 654.3, 46540, 13, 9, 12, 13,
              True, "12 が明瞭かつ最終 (13は未観測)。t=653.0に11の1つ前ステップも確認 (G_prev.jpg)"),
    EventMeta("H", "c22", "2P", 7, 696.1, 4213, 9, 4,
              "判定不能(別々の1連鎖が2回)",
              "判定不能(全消し演出を挟んで別連鎖が続く形跡)",
              False,
              "1 れんさ! 自体は明瞭(H_max/H_next とも)だが2回とも独立した1連鎖であり、"
              "単一の最大値という問い自体が成立しない(前回判定を再確認・支持)"),
    EventMeta("I", "c22", "1P", 1, 290.8, 34220, 4, 9, 10, None,
              True, "10 れんさ! が明瞭"),
    EventMeta("C2", "c21", "2P", 8, 693.5, 7922, 3, 6, 5, None,
              True, "5 れんさ! が明瞭"),
    EventMeta("D2", "c16", "2P", 9, 853.9, 5910, 3, 5, 4, None,
              True, "4 れんさ! が明瞭。production OCR(閾値0.60)は無関係の演出アイコンを"
              "誤検出したため手動で正しい位置を指定した(本ファイル docstring参照)"),
    EventMeta("F2", "c21", "2P", 6, 586.8, 14620, 1, 7, 8, None,
              True, "8 れんさ! が明瞭"),
)


def _files_for_label(label: str) -> list[str]:
    """指定事例に属する生成済みファイル名一覧を返す (存在するもののみ)。"""
    return [
        f"{label}_{suffix}.jpg"
        for suffix in ("max", "prev", "next")
        if (OUT_DIR / f"{label}_{suffix}.jpg").exists()
    ]


def _build_manifest() -> list[dict]:
    """EVENT_META と実際に生成されたファイルからマニフェスト辞書列を作る。"""
    manifest: list[dict] = []
    for meta in EVENT_META:
        manifest.append({
            "event_label": meta.label,
            "video_id": meta.video_id,
            "side": meta.side,
            "game_idx": meta.game_idx,
            "t_max_sec": meta.t_max_sec,
            "delta_score": meta.delta_score,
            "simulate_chain_count": meta.simulate_chain_count,
            "screen_chain_count": meta.screen_chain_count,
            "my_judged_true_value": meta.my_judged_true_value,
            "prev_recorded_true_value_if_different": meta.prev_recorded_true_value,
            "files": _files_for_label(meta.label),
            "readable": meta.readable,
            "readable_note": meta.readable_note,
        })
    return manifest


def _write_manifest() -> None:
    manifest = _build_manifest()
    out_path = OUT_DIR / "manifest.json"
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[OK] manifest.json -> {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, int]] = []
    for target in CROP_TARGETS:
        r = _process_target(target)
        if r is not None:
            results.append(r)
    total = sum(b for _, b in results)
    print(f"\n[完了] {len(results)}枚, 合計 {total/1024/1024:.2f}MB → {OUT_DIR}")
    _write_manifest()


if __name__ == "__main__":
    main()
