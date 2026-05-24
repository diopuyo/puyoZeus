"""
CNN による色分類を人の目で検証するための可視化ツール。

使い方:
    ./venv/bin/python scripts/verify_color_classification.py
        → data/frames/sample/frame_*.png の全フレームを走査して
          data/verify/color_review_<stem>.png に注釈付きグリッドを書き出す。

    ./venv/bin/python scripts/verify_color_classification.py path/to/frame.png
        → 指定フレームだけ処理する。

学習への影響回避:
    CUDA_VISIBLE_DEVICES="" で CPU 推論を強制するため、GPU 学習と干渉しない。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.board import Board
from src.board_rules import apply_gravity, diff_boards
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.physics_sanity import PhysicsSanityChecker, ViolationKind

# ============================
# 定数
# ============================

# 予測クラス → 日本語ラベル + 表示用 BGR 色
CLASS_DISPLAY: dict[int, tuple[str, tuple[int, int, int]]] = {
    COLOR_EMPTY: ("空", (64, 64, 64)),
    COLOR_RED: ("赤", (0, 0, 220)),
    COLOR_BLUE: ("青", (220, 80, 0)),
    COLOR_GREEN: ("緑", (0, 180, 0)),
    COLOR_YELLOW: ("黄", (0, 200, 220)),
    COLOR_PURPLE: ("紫", (180, 0, 180)),
    COLOR_OJAMA: ("お", (200, 200, 200)),
}

# パッチ表示サイズ
TILE_SIZE: int = 48
LABEL_HEIGHT: int = 16
SIDE_GAP: int = 32
OUTPUT_DIR: Path = Path("data/verify")


def _classify_region(
    gated: GatedCnnClassifier,
    frame: np.ndarray,
    region: BoardRegion,
) -> tuple[list[list[int]], list[list[np.ndarray]]]:
    """
    1 領域 (1P or 2P) の全セルをパッチ切り出し + 分類する。

    Args:
        gated: 色分類器 (空判定ゲート + CNN)。
        frame: BGR フレーム画像 (1920×1080)。
        region: 対象盤面領域。

    Returns:
        予測ラベル行列 (13×6) と パッチ画像行列 (13×6、BGR)。
    """
    labels: list[list[int]] = []
    patches: list[list[np.ndarray]] = []
    # 隠し段 (row < HIDDEN_ROWS) は画面外で halo/UI が混入するのでスキップし、
    # 表示では「空」扱いにする。可視行のみ CNN に通す。
    for row in range(BOARD_ROWS):
        row_labels: list[int] = []
        row_patches: list[np.ndarray] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                patch = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                row_labels.append(COLOR_EMPTY)
                row_patches.append(patch)
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            # 画像外 (隠し段等) はゼロ埋めパッチ扱いにする
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                patch = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                label = COLOR_EMPTY
            else:
                patch = frame[y1c:y2c, x1c:x2c].copy()
                label = gated.classify(patch)
            row_labels.append(label)
            row_patches.append(patch)
        labels.append(row_labels)
        patches.append(row_patches)
    return labels, patches


def _render_tile(
    patch: np.ndarray,
    label: int,
    violation_kinds: set[ViolationKind] | None = None,
) -> np.ndarray:
    """
    パッチ画像と予測ラベルを合成したタイルを作る。
    violation_kinds 指定があれば赤枠で違反マークを付ける。
    """
    tile = np.zeros((TILE_SIZE + LABEL_HEIGHT, TILE_SIZE, 3), dtype=np.uint8)
    if patch.size > 0:
        resized = cv2.resize(patch, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
        tile[:TILE_SIZE, :, :] = resized
    name, color = CLASS_DISPLAY.get(label, ("?", (128, 128, 128)))
    tile[TILE_SIZE:, :, :] = color
    cv2.putText(
        tile,
        name,
        (4, TILE_SIZE + LABEL_HEIGHT - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # 物理サニティ違反があれば赤枠で強調
    if violation_kinds:
        border_color = (0, 0, 255)  # BGR: 赤
        thickness = 3
        cv2.rectangle(
            tile,
            (0, 0),
            (TILE_SIZE - 1, TILE_SIZE + LABEL_HEIGHT - 1),
            border_color,
            thickness,
        )
        # 違反種別マーカー (左上に小さい文字)
        markers: list[str] = []
        if ViolationKind.AIRBORNE in violation_kinds:
            markers.append("浮")
        if ViolationKind.UNRESOLVED_CHAIN in violation_kinds:
            markers.append("4+")
        if markers:
            cv2.putText(
                tile,
                "/".join(markers),
                (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return tile


def _labels_to_board(labels: list[list[int]]) -> Board:
    """予測ラベル行列から Board オブジェクトを作る。"""
    board = Board()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board.set(row, col, labels[row][col])
    return board


def _count_4plus_groups(labels: list[list[int]]) -> int:
    """4+ 同色連結（おじゃま除く）の数を返す。"""
    from src.chain import ChainSimulator, MIN_ERASE_COUNT
    from src.board import COLOR_OJAMA

    sim = ChainSimulator()
    board = _labels_to_board(labels)
    return sum(
        1 for g in sim.find_groups(board)
        if g.size >= MIN_ERASE_COUNT and g.color != COLOR_OJAMA
    )


def _maybe_apply_gravity(
    raw_labels: list[list[int]],
    raw_patches: list[list[np.ndarray]],
) -> tuple[list[list[int]], list[list[np.ndarray]], set[tuple[int, int]], str]:
    """
    安全条件を満たすときだけ重力補正を適用する。

    条件:
        - raw に 4+ 連結が無い（連鎖中ではない）
        - gravity 適用後も 4+ 連結が増えない（誤った下詰めで偽連結を作らない）

    戻り値:
        (ラベル, パッチ, 変化セル集合, tag文字列)
        tag: "chain" / "misclass" / "applied" / "noop"
    """
    raw_4plus = _count_4plus_groups(raw_labels)
    if raw_4plus > 0:
        return raw_labels, raw_patches, set(), "chain"
    cand_labels, cand_patches, cand_changed = _apply_gravity_with_patches(
        raw_labels, raw_patches,
    )
    cand_4plus = _count_4plus_groups(cand_labels)
    if cand_4plus > raw_4plus:
        # 重力が偽連結を作った → 色誤認疑い、生を採用
        return raw_labels, raw_patches, set(), "misclass"
    if not cand_changed:
        return raw_labels, raw_patches, set(), "noop"
    return cand_labels, cand_patches, cand_changed, "applied"


def _apply_gravity_with_patches(
    labels: list[list[int]],
    patches: list[list[np.ndarray]],
) -> tuple[list[list[int]], list[list[np.ndarray]], set[tuple[int, int]]]:
    """
    ラベルとパッチ画像を同時に列重力で下詰めする。

    戻り値:
        (補正後ラベル, 補正後パッチ, 変化セルの (row, col) 集合)

    HIDDEN_ROWS は触らない。COLOR_UNKNOWN は位置固定。
    空セルのパッチは元位置の画像を残す（bg を見せる）。
    """
    before = _labels_to_board(labels)
    after = apply_gravity(before)
    changed = {(c.row, c.col) for c in diff_boards(before, after)}

    new_labels: list[list[int]] = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # パッチは元のものをコピーで初期化（空セル表示用に背景パッチを残す）
    new_patches: list[list[np.ndarray]] = [
        [patches[r][c] for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)
    ]

    for col in range(BOARD_COLS):
        # 列方向に (元 row, label, patch) を収集（上→下の順、非空のみ）
        items: list[tuple[int, int, np.ndarray]] = []
        for row in range(BOARD_ROWS):
            lbl = labels[row][col]
            if row < HIDDEN_ROWS:
                # 隠し段は触らず、そのまま new_labels へ
                new_labels[row][col] = lbl
                continue
            if lbl == COLOR_EMPTY:
                continue
            items.append((row, lbl, patches[row][col]))

        # 下から詰める
        write_row = BOARD_ROWS - 1
        for _src_row, lbl, patch in reversed(items):
            if write_row < HIDDEN_ROWS:
                break
            new_labels[write_row][col] = lbl
            new_patches[write_row][col] = patch
            write_row -= 1

    return new_labels, new_patches, changed


def _compose_board_grid(
    labels: list[list[int]],
    patches: list[list[np.ndarray]],
    side_name: str,
    violations_by_cell: dict[tuple[int, int], set[ViolationKind]] | None = None,
) -> np.ndarray:
    """1 側 (1P or 2P) の 6×13 グリッドを合成する。
    violations_by_cell が与えられると、該当セルに赤枠 + 違反マーカーを描く。
    """
    tile_h = TILE_SIZE + LABEL_HEIGHT
    header_h = 44
    grid_w = BOARD_COLS * TILE_SIZE
    grid_h = header_h + BOARD_ROWS * tile_h
    canvas = np.full((grid_h, grid_w, 3), 32, dtype=np.uint8)
    cv2.putText(
        canvas,
        side_name,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # サマリ (違反件数)
    if violations_by_cell is not None:
        n_air = sum(1 for v in violations_by_cell.values() if ViolationKind.AIRBORNE in v)
        n_chain = sum(1 for v in violations_by_cell.values() if ViolationKind.UNRESOLVED_CHAIN in v)
        summary = f"violations: 浮遊={n_air} 4+={n_chain}"
        cv2.putText(
            canvas,
            summary,
            (60, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )
    violations_by_cell = violations_by_cell or {}
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            violations = violations_by_cell.get((row, col))
            tile = _render_tile(patches[row][col], labels[row][col], violations)
            y0 = header_h + row * tile_h
            x0 = col * TILE_SIZE
            canvas[y0:y0 + tile_h, x0:x0 + TILE_SIZE, :] = tile
    return canvas


def _count_classes(labels: list[list[int]]) -> dict[int, int]:
    """クラス別カウントを返す。"""
    counts: dict[int, int] = {}
    for row in labels:
        for label in row:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _format_counts(counts: dict[int, int]) -> str:
    """クラス別カウントを人間可読の一行に整形する。"""
    parts: list[str] = []
    for cls in (
        COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
        COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
    ):
        name = CLASS_DISPLAY[cls][0]
        parts.append(f"{name}={counts.get(cls, 0)}")
    return " ".join(parts)


def process_frame(
    frame_path: Path,
    gated: GatedCnnClassifier,
    p1_region: BoardRegion,
    p2_region: BoardRegion,
    out_dir: Path,
) -> Path | None:
    """1 フレームを処理して検証用 PNG を書き出し、出力パスを返す。"""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"{frame_path.name}: 読み込み失敗")
        return None
    if frame.shape[:2] != (1080, 1920):
        print(f"{frame_path.name}: スキップ (shape={frame.shape})")
        return None

    raw_labels_1p, raw_patches_1p = _classify_region(gated, frame, p1_region)
    raw_labels_2p, raw_patches_2p = _classify_region(gated, frame, p2_region)

    # 重力適用条件: CNN 生に 4+ 連結が無く、かつ gravity 適用後も 4+ 連結を
    # 新しく作らない場合のみ適用。さもなければ CNN 生をそのまま表示する。
    # 理由:
    #   - CNN 生で 4+ があるサイド = 連鎖中（puyo が中空落下中）→ gravity 不適切
    #   - gravity が 4+ を新造 = 色誤認シグナル（連鎖済みなので絶対にあり得ない状態）
    labels_1p, patches_1p, changed_1p, gravity_1p = _maybe_apply_gravity(
        raw_labels_1p, raw_patches_1p,
    )
    labels_2p, patches_2p, changed_2p, gravity_2p = _maybe_apply_gravity(
        raw_labels_2p, raw_patches_2p,
    )

    # 物理サニティチェック（補正後でも残っている違反を可視化）
    checker = PhysicsSanityChecker()
    board_1p = _labels_to_board(labels_1p)
    board_2p = _labels_to_board(labels_2p)
    viol_1p = checker.check(board_1p)
    viol_2p = checker.check(board_2p)
    viol_map_1p: dict[tuple[int, int], set[ViolationKind]] = {}
    viol_map_2p: dict[tuple[int, int], set[ViolationKind]] = {}
    for v in viol_1p:
        viol_map_1p.setdefault((v.row, v.col), set()).add(v.kind)
    for v in viol_2p:
        viol_map_2p.setdefault((v.row, v.col), set()).add(v.kind)

    grid_1p = _compose_board_grid(labels_1p, patches_1p, "1P", viol_map_1p)
    grid_2p = _compose_board_grid(labels_2p, patches_2p, "2P", viol_map_2p)

    gap = np.full((grid_1p.shape[0], SIDE_GAP, 3), 32, dtype=np.uint8)
    combined = np.hstack([grid_1p, gap, grid_2p])

    counts_1p = _count_classes(labels_1p)
    counts_2p = _count_classes(labels_2p)
    print(f"\n=== {frame_path.name} ===")
    tag_map = {
        "chain": "連鎖中 (gravity skip)",
        "misclass": "色誤認疑い (gravity revert)",
        "applied": f"重力補正={len(changed_1p)}セル",
        "noop": "重力補正=0セル",
    }
    tag_1p = tag_map.get(gravity_1p, gravity_1p)
    if gravity_1p == "applied":
        tag_1p = f"重力補正={len(changed_1p)}セル"
    tag_2p = tag_map.get(gravity_2p, gravity_2p)
    if gravity_2p == "applied":
        tag_2p = f"重力補正={len(changed_2p)}セル"
    print(f"  1P: {_format_counts(counts_1p)}  違反={len(viol_1p)} (浮遊={sum(1 for v in viol_1p if v.kind == ViolationKind.AIRBORNE)}, 4+={sum(1 for v in viol_1p if v.kind == ViolationKind.UNRESOLVED_CHAIN)}) {tag_1p}")
    print(f"  2P: {_format_counts(counts_2p)}  違反={len(viol_2p)} (浮遊={sum(1 for v in viol_2p if v.kind == ViolationKind.AIRBORNE)}, 4+={sum(1 for v in viol_2p if v.kind == ViolationKind.UNRESOLVED_CHAIN)}) {tag_2p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"color_review_{frame_path.stem}.png"
    cv2.imwrite(str(out_path), combined)
    print(f"  → {out_path}")
    return out_path


def main() -> None:
    """エントリーポイント。"""
    # holdout ベストを保護した cnn_global_best.pt を既定で使う。
    # 学習進行中の cnn_best.pt は Cycle 内で上書きされるため、推論の品質は不安定。
    cnn_path = Path("models/cnn_global_best.pt")
    if not cnn_path.exists():
        cnn_path = Path("models/cnn_best.pt")
    calib_path = Path("models/calibration_video01.json")

    cnn = CnnPatchClassifier.load(cnn_path)
    config = CalibratedConfig.load(calib_path)
    gated = GatedCnnClassifier(color_classifier=cnn)

    if len(sys.argv) > 1:
        target_paths = [Path(p) for p in sys.argv[1:]]
    else:
        sample_dir = Path("data/frames/sample")
        target_paths = sorted(
            p for p in sample_dir.glob("frame_*.png") if "debug" not in p.name
        )

    if not target_paths:
        print("対象フレームが見つかりません")
        return

    print(f"CNN: {cnn_path}")
    print(f"Calibration: {calib_path}")
    print(f"対象: {len(target_paths)}フレーム")

    generated: list[Path] = []
    for fp in target_paths:
        result = process_frame(fp, gated, config.p1_region, config.p2_region, OUTPUT_DIR)
        if result is not None:
            generated.append(result)

    print(f"\n生成完了: {len(generated)} 枚")
    for p in generated:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
