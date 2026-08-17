"""ラベル規約v2の共通ユーティリティ (W8根治、2026-08-17)。

## 背景
`docs/KNOWN_WEAKNESSES.md` W8: 物差しラベルが絶対 frame_idx 依存だったため、
動画再DLで大半が「どのフレームを指しているか分からない」状態に陥った
(52盤面中46盤面がアンカー不能)。根治は「参照画像のハッシュ+周辺数フレームの
視覚シグネチャ」を主キーにし、frame_idx を補助キーへ降格すること。

本モジュールは新規ラベル作成スクリプト (`_build_board_label_sheets_general_
2026-08-17.py` 等) と将来の測定スクリプトの両方から使われる共通部品を提供する。

## 既存資産の再利用 (コピペ禁止指示への対応)
ピクセル照合による再アンカリング (NCC走査、標準解像度への正規化) は
`scripts/_reanchor_yardstick_labels_2026-08-14.py` に実装済みのものを
importlib 経由でそのまま呼び出す (video_props / scan_video_window /
region_for_side / STD_WIDTH / STD_HEIGHT)。ここでは「ハッシュ計算」と
「周辺フレームシグネチャの保存/再アンカリング」という新規部分のみを追加する。
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ファイル名に日付ハイフンを含むため動的import (既存規約に準拠)
_RA = importlib.import_module("scripts._reanchor_yardstick_labels_2026-08-14")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# 平均ハッシュのグリッド辺 (16x16=256bit。8x8=64bitより盤面内の細かい配色差を
# 区別できる。ぷよ盤面はセルの繰り返しパターンが多く、粗いハッシュだと
# 異なる盤面同士が衝突しやすいため厚めに取る)。
HASH_GRID_SIZE: int = 16

# 周辺フレームシグネチャに含めるオフセット (0=対象フレーム自身)。
# ±2フレームまでを保存し、再DL後の1-2フレームのタイミングずれでも
# 再アンカリングできるようにする。
NEIGHBOR_FRAME_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2)

# シグネチャ用縮小サムネイルの (幅, 高さ)。盤面ROI比 384:720 を保った小サイズ。
SIGNATURE_THUMB_SIZE: tuple[int, int] = (48, 90)

STD_WIDTH: int = _RA.STD_WIDTH
STD_HEIGHT: int = _RA.STD_HEIGHT


# =============================================================================
# データ構造
# =============================================================================


@dataclass(frozen=True)
class AnchorKey:
    """1ラベルの内容ベース主キー。"""

    hash_hex: str  # average_hash の16進文字列
    signature: np.ndarray  # shape=(len(NEIGHBOR_FRAME_OFFSETS), H, W) uint8


# =============================================================================
# 1. フレーム読み出し (標準解像度へ正規化)
# =============================================================================


def read_frame_at(video_path: Path, frame_idx: int) -> "np.ndarray | None":
    """指定フレームを標準解像度 (1920x1080) の BGR 画像として返す。"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def crop_gray(frame: np.ndarray, region: "tuple[int, int, int, int]") -> np.ndarray:
    """標準解像度画像から ROI を切り出してグレースケール化する。"""
    x, y, w, h = region
    crop = frame[y : y + h, x : x + w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 2. 平均ハッシュ (主キーの一部)
# =============================================================================


def average_hash(gray: np.ndarray, grid_size: int = HASH_GRID_SIZE) -> str:
    """グレースケール画像の平均ハッシュを16進文字列で返す (再DL後の一致判定用)。"""
    small = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    bits = (small.astype(np.float64) > small.mean()).astype(np.uint8).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    hex_digits = grid_size * grid_size // 4
    return format(value, f"0{hex_digits}x")


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """2つの16進ハッシュ文字列のハミング距離 (ビット単位)。"""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


# =============================================================================
# 3. 周辺フレームシグネチャ (再アンカリング用テンプレート)
# =============================================================================


def build_neighbor_signature(
    video_path: Path, frame_idx: int, region: "tuple[int, int, int, int]",
    offsets: "tuple[int, ...]" = NEIGHBOR_FRAME_OFFSETS,
    thumb_size: "tuple[int, int]" = SIGNATURE_THUMB_SIZE,
) -> np.ndarray:
    """frame_idx±offsets の縮小グレースケール画像列を返す (shape=(len(offsets),H,W))。"""
    cap = cv2.VideoCapture(str(video_path))
    thumbs: "list[np.ndarray]" = []
    for off in offsets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, frame_idx + off)))
        ok, frame = cap.read()
        if not ok or frame is None:
            thumbs.append(np.zeros(thumb_size[::-1], dtype=np.uint8))
            continue
        if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
            frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
        gray = crop_gray(frame, region)
        thumbs.append(cv2.resize(gray, thumb_size, interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(thumbs, axis=0)


def build_anchor_key(
    video_path: Path, frame_idx: int, region: "tuple[int, int, int, int]",
) -> AnchorKey:
    """指定フレームの主キー (ハッシュ+周辺シグネチャ) を構築する。"""
    frame = read_frame_at(video_path, frame_idx)
    if frame is None:
        raise ValueError(f"フレーム読込失敗: {video_path} f{frame_idx}")
    gray = crop_gray(frame, region)
    return AnchorKey(
        hash_hex=average_hash(gray),
        signature=build_neighbor_signature(video_path, frame_idx, region),
    )


def save_anchor_sidecar(
    path: Path, key: AnchorKey, candidate_grid: "np.ndarray | None" = None,
) -> None:
    """AnchorKey を npz サイドカーとして保存する (主キーの永続化)。

    `candidate_grid` (省略可、後方互換の optional 引数) はラベリング時点で
    表示した候補グリッド (どの構成の出力か、`base_config` は labels.tsv 側に
    別記録) を保存する。正解再構成 (候補グリッド + wrong_cells 上書き) に使う。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: "dict[str, np.ndarray]" = dict(
        hash_hex=np.array(key.hash_hex),
        signature=key.signature, offsets=np.array(NEIGHBOR_FRAME_OFFSETS),
    )
    if candidate_grid is not None:
        kwargs["candidate_grid"] = np.asarray(candidate_grid, dtype=np.int8)
    np.savez_compressed(str(path), **kwargs)


def load_anchor_sidecar(path: Path) -> "tuple[AnchorKey, np.ndarray | None]":
    """npz サイドカーから (AnchorKey, candidate_grid) を復元する (grid無ければNone)。"""
    d = np.load(str(path), allow_pickle=False)
    key = AnchorKey(hash_hex=str(d["hash_hex"]), signature=np.asarray(d["signature"]))
    grid = np.asarray(d["candidate_grid"]) if "candidate_grid" in d.files else None
    return key, grid


# =============================================================================
# 4. 再アンカリング (動画が再DLされても参照画像そのもので位置を再特定する)
# =============================================================================


def reanchor_by_signature(
    video_path: Path, region: "tuple[int, int, int, int]", key: AnchorKey,
    hint_t_sec: float, window_sec: float = _RA.SEARCH_WINDOW_SEC,
) -> "tuple[float, float]":
    """シグネチャの中心フレーム (offset=0) を参照画像として NCC 再走査する。

    `_reanchor_yardstick_labels_2026-08-14.scan_video_window` を再利用する
    (コピペ禁止指示への対応)。戻り値は (best_t_sec, best_ncc_score)。
    """
    center_idx = NEIGHBOR_FRAME_OFFSETS.index(0)
    ref_gray = key.signature[center_idx]
    best_t, best_score, _zero = _RA.scan_video_window(
        video_path, region, ref_gray, hint_t_sec, window_sec,
    )
    return best_t, best_score
