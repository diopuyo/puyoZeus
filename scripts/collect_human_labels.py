"""
人による CNN 色分類ラベル訂正収集ツール。

使い方:
    # ① 訂正対象フレームを export (CSV + パッチ画像)
    ./venv/bin/python scripts/collect_human_labels.py export data/frames/sample/frame_0300s.png

    # ② 出力 CSV を開いて true_label 列を編集（CNN 予測と違うセルだけ書き換え）
    #     CSV: data/verify/human_labels/frame_0300s_<ts>.csv
    #     画像: data/verify/human_labels/frame_0300s_<ts>_patches/<side>_<row>_<col>.png

    # ③ 編集済み CSV を import (jsonl に変換)
    ./venv/bin/python scripts/collect_human_labels.py import \\
        data/verify/human_labels/frame_0300s_<ts>.csv

CSV 形式:
    side,row,col,cnn_predicted,true_label,patch_file
    1P,1,0,empty,,patches/1P_01_00.png
    1P,1,1,empty,red,patches/1P_01_01.png   ← ここを red に修正した例
    ...

    true_label 列は:
      - 空欄のまま → CNN 予測を正解扱い（訂正不要）
      - empty/red/blue/green/yellow/purple/ojama の 7 値のいずれか → 訂正
      - skip → 判定困難で除外 (学習に使わない)
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
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
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier

# 正解ラベルの文字列 <-> 整数コード
LABEL_TO_CODE: dict[str, int] = {
    "empty": COLOR_EMPTY,
    "red": COLOR_RED,
    "blue": COLOR_BLUE,
    "green": COLOR_GREEN,
    "yellow": COLOR_YELLOW,
    "purple": COLOR_PURPLE,
    "ojama": COLOR_OJAMA,
}
CODE_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_CODE.items()}
SKIP_LABEL: str = "skip"

OUTPUT_DIR: Path = Path("data/verify/human_labels")
# holdout best を保護した cnn_global_best.pt を優先する
# (cnn_best.pt は学習中 Cycle 内で上書きされ、劣化版になりうる)
_GLOBAL_BEST: Path = Path("models/cnn_global_best.pt")
_LATEST: Path = Path("models/cnn_best.pt")
DEFAULT_CNN: Path = _GLOBAL_BEST if _GLOBAL_BEST.exists() else _LATEST
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")


def _extract_cell(frame: np.ndarray, region: BoardRegion, row: int, col: int) -> np.ndarray:
    """1 セルの BGR パッチを取り出す。可視範囲外は空配列。"""
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(frame.shape[1], x2)
    y2c = min(frame.shape[0], y2)
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    return frame[y1c:y2c, x1c:x2c].copy()


def _predicted_label(classifier: GatedCnnClassifier, patch: np.ndarray) -> str:
    """パッチを分類して文字列ラベルで返す。"""
    code = classifier.classify(patch)
    return CODE_TO_LABEL.get(code, "empty")


def export_frame(
    frame_path: Path,
    classifier: GatedCnnClassifier,
    p1_region: BoardRegion,
    p2_region: BoardRegion,
) -> Path:
    """
    フレームを CSV + パッチ画像群に export する。

    Returns:
        生成した CSV のパス。
    """
    frame = cv2.imread(str(frame_path))
    if frame is None or frame.shape[:2] != (1080, 1920):
        raise RuntimeError(f"不正フレーム: {frame_path}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = frame_path.stem
    out_root = OUTPUT_DIR / f"{stem}_{ts}"
    patches_dir = out_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root.with_suffix(".csv")
    rows: list[list[str]] = []

    sides: list[tuple[str, BoardRegion]] = [("1P", p1_region), ("2P", p2_region)]
    for side_name, region in sides:
        for row in range(HIDDEN_ROWS, BOARD_ROWS):  # 隠し段は除外
            for col in range(BOARD_COLS):
                patch = _extract_cell(frame, region, row, col)
                predicted = _predicted_label(classifier, patch)
                patch_file = f"patches/{side_name}_{row:02d}_{col:02d}.png"
                cv2.imwrite(str(out_root / patch_file), patch)
                rows.append([side_name, str(row), str(col), predicted, "", patch_file])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "row", "col", "cnn_predicted", "true_label", "patch_file"])
        writer.writerows(rows)

    meta_path = out_root / "_meta.json"
    meta = {
        "frame": str(frame_path),
        "export_ts": ts,
        "cnn_model": str(DEFAULT_CNN),
        "calibration": str(DEFAULT_CALIB),
        "cell_count": len(rows),
        "csv": str(csv_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"export 完了:")
    print(f"  CSV: {csv_path}")
    print(f"  パッチ: {patches_dir}/ ({len(rows)} 枚)")
    print(f"  メタ: {meta_path}")
    print()
    print(f"次の手順:")
    print(f"  1) {csv_path} を開き、true_label 列を編集")
    print(f"     空欄のまま = CNN 予測を正解扱い")
    print(f"     empty/red/blue/green/yellow/purple/ojama = 訂正")
    print(f"     skip = 判定困難で除外")
    print(f"  2) ./venv/bin/python scripts/collect_human_labels.py import {csv_path}")
    return csv_path


def _validate_label(label: str) -> str | None:
    """true_label セルを検証し、正規化したラベルを返す。不正なら None。"""
    label = label.strip().lower()
    if label == "" or label == SKIP_LABEL:
        return label
    if label in LABEL_TO_CODE:
        return label
    return None


def import_csv(csv_path: Path) -> Path:
    """
    編集済み CSV を読んで訂正 jsonl を出力する。

    Returns:
        生成した jsonl のパス。
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    meta_path = csv_path.parent / "_meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    jsonl_path = csv_path.with_suffix(".jsonl")
    total = 0
    corrections = 0
    skipped = 0
    invalid = 0

    with csv_path.open("r", newline="", encoding="utf-8") as fin, \
         jsonl_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        # 1 行目: メタ情報
        meta_entry = {
            "kind": "meta",
            "import_ts": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S JST"),
            "csv_source": str(csv_path),
            "frame": meta.get("frame"),
            "cnn_model": meta.get("cnn_model"),
        }
        fout.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

        for row in reader:
            total += 1
            true_label_raw = row.get("true_label", "")
            true_label = _validate_label(true_label_raw)
            if true_label is None:
                invalid += 1
                print(f"  [WARN] 不正なラベル: {true_label_raw} (side={row['side']} row={row['row']} col={row['col']})")
                continue
            if true_label == "":
                # 訂正なし = CNN 予測を正解扱い
                continue
            if true_label == SKIP_LABEL:
                skipped += 1
                continue

            # 実際の訂正
            corrections += 1
            entry = {
                "kind": "correction",
                "frame": meta.get("frame"),
                "side": row["side"],
                "row": int(row["row"]),
                "col": int(row["col"]),
                "cnn_predicted": row["cnn_predicted"],
                "true_label": true_label,
                "patch_file": str(csv_path.parent / row["patch_file"]),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"import 完了:")
    print(f"  対象セル: {total}")
    print(f"  訂正件数: {corrections}")
    print(f"  skip: {skipped}")
    print(f"  不正ラベル: {invalid}")
    print(f"  出力 jsonl: {jsonl_path}")
    return jsonl_path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "export":
        if len(sys.argv) < 3:
            print("使い方: export <frame.png>")
            sys.exit(1)
        cnn = CnnPatchClassifier.load(DEFAULT_CNN)
        config = CalibratedConfig.load(DEFAULT_CALIB)
        classifier = GatedCnnClassifier(color_classifier=cnn)
        for frame_path_str in sys.argv[2:]:
            export_frame(
                Path(frame_path_str), classifier,
                config.p1_region, config.p2_region,
            )
    elif cmd == "import":
        if len(sys.argv) < 3:
            print("使い方: import <csv>")
            sys.exit(1)
        for csv_path_str in sys.argv[2:]:
            import_csv(Path(csv_path_str))
    else:
        print(f"未知のコマンド: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
