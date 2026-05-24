"""
ターミナル上で色分類結果をコピペ編集できるレビュー UI。

使い方:
    # ① フレームの推論結果をターミナルに表示（編集テンプレ付き）
    ./venv/bin/python scripts/terminal_review.py show data/frames/sample/frame_0600s.png

    # ② 表示された TEMPLATE ブロックをエディタにコピー、間違いを修正
    #    1P / 2P 各 12 行 × 6 列 を 1 文字ずつに編集する
    #    空行やコメント行 (#) は保持される
    #    編集後テキストをファイル保存 (例: edited_0600s.txt)

    # ③ 修正後のテキストをインポートして labels.jsonl に書く
    ./venv/bin/python scripts/terminal_review.py apply \
        data/frames/sample/frame_0600s.png edited_0600s.txt

文字コード:
    . / _   = 空
    R       = 赤
    B       = 青
    G       = 緑
    Y       = 黄
    P       = 紫
    O       = おじゃま
    ?       = スキップ（学習除外、判定困難セル用）

列は 6 列固定。空白区切りでも連続でも可（" R . B . . ." でも ".RB..." でも解釈）。
隠し段 (row 0) は表示・編集対象外（画面外のため）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
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
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)
from src.board_rules import apply_gravity, diff_boards
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.physics_sanity import PhysicsSanityChecker

# ============================
# 定数
# ============================

# 色コード ↔ 1 文字記号
CODE_TO_CHAR: dict[int, str] = {
    COLOR_EMPTY:  ".",
    COLOR_RED:    "R",
    COLOR_BLUE:   "B",
    COLOR_GREEN:  "G",
    COLOR_YELLOW: "Y",
    COLOR_PURPLE: "P",
    COLOR_OJAMA:  "O",
    COLOR_UNKNOWN: "?",
}
CHAR_TO_CODE: dict[str, int] = {
    ".": COLOR_EMPTY,
    "_": COLOR_EMPTY,
    "R": COLOR_RED,
    "B": COLOR_BLUE,
    "G": COLOR_GREEN,
    "Y": COLOR_YELLOW,
    "P": COLOR_PURPLE,
    "O": COLOR_OJAMA,
}
# '?' は「学習除外」シグナル（CHAR_TO_CODE には含めない、別途扱う）
SKIP_CHAR: str = "?"

# ANSI 色
CODE_TO_ANSI: dict[int, str] = {
    COLOR_EMPTY:   "\033[90m",      # 暗グレー
    COLOR_RED:     "\033[91m",      # 明赤
    COLOR_BLUE:    "\033[94m",      # 明青
    COLOR_GREEN:   "\033[92m",      # 明緑
    COLOR_YELLOW:  "\033[93m",      # 明黄
    COLOR_PURPLE:  "\033[95m",      # 明紫
    COLOR_OJAMA:   "\033[97m",      # 明白
    COLOR_UNKNOWN: "\033[37m",      # グレー
}
ANSI_RESET: str = "\033[0m"
ANSI_VIOLATION_BG: str = "\033[41m"  # 赤背景（違反マーク）

# 実行時に無効化できる（--no-color）
_USE_COLOR: bool = True

# 優先モデル（学習中の cnn_best より holdout 保護版）
_GLOBAL_BEST: Path = Path("models/cnn_global_best.pt")
_LATEST: Path = Path("models/cnn_best.pt")
DEFAULT_CNN: Path = _GLOBAL_BEST if _GLOBAL_BEST.exists() else _LATEST
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")

OUTPUT_DIR: Path = Path("data/verify/human_labels")


# ============================
# データ構造
# ============================


@dataclass
class SideResult:
    """1 側 (1P/2P) の CNN 結果と補正結果。"""
    side: str                                # "1P" or "2P"
    raw_labels: list[list[int]]              # CNN 生予測
    corrected_labels: list[list[int]]        # 重力補正後
    violations: set[tuple[int, int]]         # 浮遊セル座標（raw 基準）


# ============================
# 推論
# ============================


def _classify_region(
    gated: GatedCnnClassifier,
    frame: np.ndarray,
    region: BoardRegion,
) -> list[list[int]]:
    """1 側の全セルを CNN で分類する。"""
    labels: list[list[int]] = []
    for row in range(BOARD_ROWS):
        row_labels: list[int] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_labels.append(COLOR_EMPTY)
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                row_labels.append(COLOR_EMPTY)
                continue
            patch = frame[y1c:y2c, x1c:x2c]
            row_labels.append(gated.classify(patch))
        labels.append(row_labels)
    return labels


def _labels_to_board(labels: list[list[int]]) -> Board:
    board = Board()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board.set(row, col, labels[row][col])
    return board


def _board_to_labels(board: Board) -> list[list[int]]:
    return [[board.get(r, c) for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]


def _airborne_cells(labels: list[list[int]]) -> set[tuple[int, int]]:
    """浮遊セル（非空で直下が空）の (row, col) を返す。"""
    result: set[tuple[int, int]] = set()
    for row in range(HIDDEN_ROWS, BOARD_ROWS - 1):
        for col in range(BOARD_COLS):
            if labels[row][col] == COLOR_EMPTY:
                continue
            if labels[row + 1][col] == COLOR_EMPTY:
                result.add((row, col))
    return result


def _infer_sides(
    frame_path: Path,
    cnn_path: Path,
    calib_path: Path,
) -> tuple[SideResult, SideResult]:
    """フレーム 1 枚から 1P/2P の生予測と重力補正結果を得る。"""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"フレーム読込失敗: {frame_path}")
    if frame.shape[:2] != (1080, 1920):
        raise RuntimeError(f"解像度不一致: {frame.shape}, 期待 (1080, 1920)")

    cnn = CnnPatchClassifier.load(cnn_path)
    config = CalibratedConfig.load(calib_path)
    gated = GatedCnnClassifier(color_classifier=cnn)

    raw_1p = _classify_region(gated, frame, config.p1_region)
    raw_2p = _classify_region(gated, frame, config.p2_region)
    corr_1p = _board_to_labels(apply_gravity(_labels_to_board(raw_1p)))
    corr_2p = _board_to_labels(apply_gravity(_labels_to_board(raw_2p)))

    return (
        SideResult("1P", raw_1p, corr_1p, _airborne_cells(raw_1p)),
        SideResult("2P", raw_2p, corr_2p, _airborne_cells(raw_2p)),
    )


# ============================
# 表示
# ============================


def _colored_char(code: int) -> str:
    """1 セルを色付き（または無色）1 文字で返す。"""
    ch = CODE_TO_CHAR.get(code, "?")
    if not _USE_COLOR:
        return ch
    ansi = CODE_TO_ANSI.get(code, "")
    return f"{ansi}{ch}{ANSI_RESET}"


def _format_visible_rows(labels: list[list[int]], marks: set[tuple[int, int]] | None = None) -> list[str]:
    """可視行（row 1 ～ 12）を色付きテーブル文字列で返す。"""
    marks = marks or set()
    lines: list[str] = []
    header = "     " + " ".join(f"c{c}" for c in range(BOARD_COLS))
    lines.append(header)
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        cells: list[str] = []
        for col in range(BOARD_COLS):
            code = labels[row][col]
            txt = _colored_char(code)
            if (row, col) in marks:
                ch = CODE_TO_CHAR.get(code, "?")
                if _USE_COLOR:
                    txt = f"{ANSI_VIOLATION_BG}{ch}{ANSI_RESET}"
                else:
                    txt = f"[{ch}]"  # カラー無効時は角括弧で違反マーク
            cells.append(f" {txt}")
        lines.append(f"r{row:02d}:" + "".join(cells))
    return lines


def _print_side_by_side(result: SideResult) -> None:
    """1 側について「CNN 生」と「重力補正後」を並べて表示する。"""
    print(f"\n==== {result.side}  浮遊数={len(result.violations)} ====")
    print("CNN 生 (浮遊=赤背景)                    重力補正後")
    raw_lines = _format_visible_rows(result.raw_labels, marks=result.violations)
    corr_lines = _format_visible_rows(result.corrected_labels)
    for a, b in zip(raw_lines, corr_lines):
        print(f"{a}      {b}")


def _class_counts_str(labels: list[list[int]]) -> str:
    """クラス別カウントを 1 行で返す。"""
    counts: dict[int, int] = {}
    for row in labels:
        for code in row:
            counts[code] = counts.get(code, 0) + 1
    order = (COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
             COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA)
    return " ".join(f"{CODE_TO_CHAR[c]}={counts.get(c, 0)}" for c in order)


# ============================
# 編集テンプレ
# ============================


def _emit_template(result_1p: SideResult, result_2p: SideResult, source: str) -> str:
    """
    ユーザが編集する編集テンプレをテキストで返す。
    デフォルト値は「重力補正後」の盤面。CNN 誤認を訂正したい部分だけ書き換える。
    """
    lines: list[str] = []
    lines.append(f"# source: {source}")
    lines.append(f"# generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("# 文字: . =空  R=赤 B=青 G=緑 Y=黄 P=紫 O=おじゃま  ?=除外")
    lines.append("# 各行 6 文字（空白任意）。# 行・空行はコメント扱い。")
    lines.append("# 隠し段 (row 0) は省略（画面外）。")
    lines.append("")
    for result in (result_1p, result_2p):
        lines.append(f"=== {result.side} ===")
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            cells = "".join(
                CODE_TO_CHAR.get(result.corrected_labels[row][col], "?")
                for col in range(BOARD_COLS)
            )
            lines.append(f"r{row:02d}: {cells}")
        lines.append("")
    return "\n".join(lines)


# ============================
# 編集テンプレのパース
# ============================


@dataclass(frozen=True)
class ParsedCell:
    """編集テンプレから抽出した 1 セルの人手値。skip=True なら学習除外。"""
    side: str
    row: int
    col: int
    code: int | None   # skip のとき None
    skip: bool


def _parse_template(text: str) -> list[ParsedCell]:
    """編集済みテンプレ文字列をセル単位リストにパースする。"""
    cells: list[ParsedCell] = []
    side: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("==="):
            inner = line.strip("= ").strip()
            if inner in ("1P", "2P"):
                side = inner
            continue
        if side is None:
            continue
        # "r07: .RB.Y." や "r07: . R B . Y ." 形式を受け付ける
        if ":" not in line:
            continue
        head, _, body = line.partition(":")
        head = head.strip()
        if not head.startswith("r"):
            continue
        try:
            row = int(head[1:])
        except ValueError:
            continue
        compact = "".join(body.split())
        if len(compact) != BOARD_COLS:
            raise ValueError(
                f"{side} {head}: 6 文字必要ですが {len(compact)} 文字です: '{body.strip()}'"
            )
        for col, ch in enumerate(compact):
            ch_up = ch.upper()
            if ch_up == SKIP_CHAR:
                cells.append(ParsedCell(side=side, row=row, col=col, code=None, skip=True))
                continue
            if ch_up not in CHAR_TO_CODE:
                raise ValueError(f"{side} {head} 列{col}: 不明な文字 '{ch}'")
            cells.append(
                ParsedCell(
                    side=side, row=row, col=col,
                    code=CHAR_TO_CODE[ch_up], skip=False,
                )
            )
    return cells


# ============================
# インポート
# ============================


def _import_labels(
    frame_path: Path,
    edited_text: str,
    cnn_path: Path,
    calib_path: Path,
) -> Path:
    """
    編集済みテンプレから訂正 jsonl とパッチ画像を書き出す。

    collect_human_labels.py と互換のフォーマットで出力:
        data/verify/human_labels/<frame_stem>_<ts>/
            ├ patches/<side>_<row>_<col>.png   ← 訂正分のみ（npz 取り込み可能に）
        data/verify/human_labels/<frame_stem>_<ts>.jsonl
            1 行目: meta  {"kind": "meta", ...}
            以降:    correction {"kind": "correction", "frame", "side", "row",
                                  "col", "cnn_predicted", "true_label", "patch_file"}
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"フレーム読込失敗: {frame_path}")

    parsed = _parse_template(edited_text)
    result_1p, result_2p = _infer_sides(frame_path, cnn_path, calib_path)
    results: dict[str, SideResult] = {"1P": result_1p, "2P": result_2p}

    # cell_sample_rect 参照用に region を再取得
    config = CalibratedConfig.load(calib_path)
    regions = {"1P": config.p1_region, "2P": config.p2_region}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = OUTPUT_DIR / f"{frame_path.stem}_{ts}"
    patches_dir = out_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / f"{frame_path.stem}_{ts}.jsonl"

    # コード → 文字ラベル（"empty", "red" 等）。human_labels_to_npz が期待する形式
    code_to_str: dict[int, str] = {
        COLOR_EMPTY: "empty", COLOR_RED: "red", COLOR_BLUE: "blue",
        COLOR_GREEN: "green", COLOR_YELLOW: "yellow", COLOR_PURPLE: "purple",
        COLOR_OJAMA: "ojama",
    }

    n_diff = 0
    n_skip = 0
    n_same = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        # メタエントリ
        meta = {
            "kind": "meta",
            "import_ts": ts,
            "frame": str(frame_path),
            "cnn_model": str(cnn_path),
            "tool": "terminal_review.py",
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for cell in parsed:
            side_result = results[cell.side]
            cnn_code = side_result.raw_labels[cell.row][cell.col]

            if cell.skip:
                n_skip += 1
                continue
            if cell.code == cnn_code:
                n_same += 1
                continue

            # 訂正: パッチを抽出・保存
            region = regions[cell.side]
            x1, y1, x2, y2 = region.cell_sample_rect(cell.row, cell.col)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            patch = frame[y1c:y2c, x1c:x2c]
            patch_rel = f"patches/{cell.side}_{cell.row:02d}_{cell.col:02d}.png"
            patch_abs = (out_root / patch_rel).resolve()
            cv2.imwrite(str(patch_abs), patch)

            entry = {
                "kind": "correction",
                "frame": str(frame_path),
                "side": cell.side,
                "row": cell.row,
                "col": cell.col,
                "cnn_predicted": code_to_str.get(cnn_code, str(cnn_code)),
                "true_label": code_to_str.get(cell.code, str(cell.code)),
                # 絶対パスで書く。_gather_corrections が相対を jsonl.parent に結合する
                # バグ（二重プレフィックス）を回避。
                "patch_file": str(patch_abs),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_diff += 1

    print(f"\nインポート完了: {jsonl_path}")
    print(f"  パッチ: {patches_dir}/ ({n_diff} 枚)")
    print(f"  訂正={n_diff}  確認済={n_same}  skip={n_skip}  合計={len(parsed)}")
    return jsonl_path


# ============================
# エントリーポイント
# ============================


def cmd_show(args: argparse.Namespace) -> int:
    frame_path = Path(args.frame)
    if not frame_path.exists():
        print(f"フレームが存在しません: {frame_path}", file=sys.stderr)
        return 1
    cnn_path = Path(args.cnn) if args.cnn else DEFAULT_CNN
    calib_path = Path(args.calib) if args.calib else DEFAULT_CALIB

    print(f"CNN       : {cnn_path}")
    print(f"Calib     : {calib_path}")
    print(f"Frame     : {frame_path}")

    result_1p, result_2p = _infer_sides(frame_path, cnn_path, calib_path)
    # 違反物理サニティ件数もサマリ
    checker = PhysicsSanityChecker()
    v1 = checker.check(_labels_to_board(result_1p.raw_labels))
    v2 = checker.check(_labels_to_board(result_2p.raw_labels))
    print(f"Violations: 1P={len(v1)}  2P={len(v2)} (浮遊+未消去連結)")
    print(f"1P counts : {_class_counts_str(result_1p.raw_labels)}")
    print(f"2P counts : {_class_counts_str(result_2p.raw_labels)}")

    _print_side_by_side(result_1p)
    _print_side_by_side(result_2p)

    template = _emit_template(result_1p, result_2p, str(frame_path))
    print("\n================ 編集テンプレ（↓コピーして編集） ================")
    print(template)
    print("================ 編集テンプレ 終端 ================\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(template, encoding="utf-8")
        print(f"編集テンプレを書き出しました: {out}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    frame_path = Path(args.frame)
    edited = Path(args.edited)
    if not frame_path.exists():
        print(f"フレームが存在しません: {frame_path}", file=sys.stderr)
        return 1
    if not edited.exists():
        print(f"編集済みテキストが存在しません: {edited}", file=sys.stderr)
        return 1
    cnn_path = Path(args.cnn) if args.cnn else DEFAULT_CNN
    calib_path = Path(args.calib) if args.calib else DEFAULT_CALIB
    _import_labels(frame_path, edited.read_text(encoding="utf-8"), cnn_path, calib_path)
    return 0


def main() -> int:
    global _USE_COLOR
    parser = argparse.ArgumentParser(description="ターミナル色分類レビュー")
    parser.add_argument("--no-color", action="store_true", help="ANSI カラーを無効化")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="フレームの推論結果をターミナル表示")
    p_show.add_argument("frame", help="フレーム PNG (1920×1080)")
    p_show.add_argument("--cnn", help="CNN モデルパス (default: cnn_global_best.pt)")
    p_show.add_argument("--calib", help="キャリブレーション JSON")
    p_show.add_argument("--out", help="編集テンプレの書き出し先ファイル")
    p_show.set_defaults(func=cmd_show)

    p_apply = sub.add_parser("apply", help="編集済みテンプレを jsonl に書き出す")
    p_apply.add_argument("frame", help="フレーム PNG")
    p_apply.add_argument("edited", help="編集済みテンプレ .txt")
    p_apply.add_argument("--cnn", help="CNN モデルパス")
    p_apply.add_argument("--calib", help="キャリブレーション JSON")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    if args.no_color or os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        _USE_COLOR = False
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
