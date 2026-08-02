"""満杯盤面 人手ラベル付け用 クリック操作HTMLツール 生成スクリプト (2026-08-02)。

scripts/build_full_board_label_sheet.py が準備した labeling_sheet.csv +
frames/*_full.png (40候補) から、user が「13x6文字列の手書き」をせずに
**クリックだけ**でラベル付けを完結できるローカルHTMLツールを生成する。

## 座標系 (scripts/visualize_recognition.py 由来、既存と完全一致させること)
- 盤面ROI: P1=(282,160), P2=(1258,160)、幅384px×高さ720px (可視12行分)
- 1セル = CELL_W(64px) × CELL_H(60px)、6列×12行 (隠し段 row0 は画面外)
- 隠し段 (row0) は実画面に写らないため、可視12行の画像とは別に上部の
  独立パネルとして表示する (クリック自体は同じサイクル関数を共有)

## 出力
    data/verify/full_board_label_sheet_2026-08-02/label_tool.html
    data/verify/full_board_label_sheet_2026-08-02/frames/<base>_board_crop.png
        (既存 *_full.png から可視12行だけを切り出した新規画像、グリッド重畳の土台)

## 使い方 (生成)
    PYTHONPATH=. python -m scripts.build_full_board_label_tool

## 使い方 (ラベル付け、user向け)
    label_tool.html を Windows Explorer からダブルクリックしてブラウザで開く。
    セルをクリックで 空→赤→青→緑→黄→紫→おじゃま→不明 と循環 (右クリックで逆循環)。
    各候補の3ボタンのいずれかを押すと次候補へ自動スクロール。
    「結果をダウンロード」で labeling_result.csv が保存される。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS,
)
from scripts.build_full_board_label_sheet import (  # noqa: E402
    OUTPUT_DIR, VIDEO_ID_PREFIX,
)
from scripts.visualize_recognition import (  # noqa: E402
    CELL_H, CELL_W, COLOR_BGR, COLOR_SYMBOLS, N_VISIBLE_ROWS, P1_ROI_X,
    P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W,
)

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

LABELING_CSV_NAME: str = "labeling_sheet.csv"
HTML_FILENAME: str = "label_tool.html"
FRAMES_SUBDIR_NAME: str = "frames"
FULL_FRAME_SUFFIX: str = "_full.png"
BOARD_CROP_SUFFIX: str = "_board_crop.png"

# クリックで循環させる色順 (user仕様: 空→赤→青→緑→黄→紫→おじゃま→不明→空)
CYCLE_ORDER: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
    COLOR_PURPLE, COLOR_OJAMA, COLOR_UNKNOWN,
)

# localStorage キー接頭辞 (出力ディレクトリ名を末尾に付けて世代分離)
STORAGE_KEY_PREFIX: str = "puyo_full_board_label_tool::"

# 表示用の見やすさ設定 (座標系には影響しない、CSS表示幅のみ)
DISPLAY_WIDTH_PX: int = 480
# 表示スケール (実ROI幅 -> 表示幅) と、隠し段パネルの表示高さ (アスペクト比維持)
_DISPLAY_SCALE: float = DISPLAY_WIDTH_PX / ROI_W
HIDDEN_PANEL_HEIGHT_PX: int = round(CELL_H * HIDDEN_ROWS * _DISPLAY_SCALE)

STATUS_OK: str = "ok"
STATUS_FIXED: str = "fixed"
STATUS_SKIP: str = "skip"

RESULT_CSV_HEADER: tuple[str, ...] = (
    "video_id", "t_sec", "side", "status", "correct_grid", "changed_cells",
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class ToolCandidate:
    """label_tool.html に埋め込む1候補分のデータ。"""

    key: str                  # "video_c17|101.3|1P" (再読込時の状態復元キー)
    video_id: str
    t_sec: str                 # 表示・CSV往復用に文字列のまま保持
    side: str
    game_idx: str
    occupancy: str
    tier: str
    phase: str
    image_rel_path: str        # "frames/xxx_board_crop.png" (常にposix区切り)
    init_grid: list            # 13行×6列 int のネストリスト


# =============================================================================
# 1. grid文字列デコード (encode_grid_string の逆変換)
# =============================================================================


def decode_grid_string(encoded: str) -> np.ndarray:
    """"U..."/形式の文字列を (13,6) int配列に戻す (encode_grid_stringの逆)。"""
    rows = encoded.split("/")
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64)
    for r, row_str in enumerate(rows):
        for c, ch in enumerate(row_str):
            grid[r, c] = COLOR_UNKNOWN if ch == "U" else int(ch)
    return grid


# =============================================================================
# 2. 実画面クロップ (可視12行のみ、隠し段の合成帯なし)
# =============================================================================


def crop_visible_board_region(frame: np.ndarray, side: str) -> np.ndarray:
    """実画面から盤面ROIの可視12行分だけを切り出す (隠し段は含まない)。

    scripts/build_full_board_label_sheet.crop_board_region_padded と異なり
    隠し段の合成黒帯は付けない (HTMLツール側で別パネル表示するため)。
    """
    x, y = (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)
    return frame[y:y + ROI_H, x:x + ROI_W].copy()


def frame_basename_from_row(row: dict) -> str:
    """CSV1行分から、既存フレームPNGのファイル名共通部分を復元する。"""
    video_id = row["video_id"]
    stem = video_id[len(VIDEO_ID_PREFIX):] if video_id.startswith(VIDEO_ID_PREFIX) else video_id
    return f"{stem}_t{float(row['t_sec']):.1f}_{row['side']}"


def build_board_crop_image(frames_dir: Path, base: str) -> "Path | None":
    """既存 <base>_full.png から可視12行クロップを切り出し保存する (失敗時None)。"""
    full_path = frames_dir / f"{base}{FULL_FRAME_SUFFIX}"
    if not full_path.exists():
        return None
    frame = cv2.imread(str(full_path))
    if frame is None:
        return None
    side = "1P" if "_1P" in base else "2P"
    crop = crop_visible_board_region(frame, side)
    out_path = frames_dir / f"{base}{BOARD_CROP_SUFFIX}"
    cv2.imwrite(str(out_path), crop)
    return out_path


# =============================================================================
# 3. 候補データ組み立て
# =============================================================================


def load_labeling_rows(csv_path: Path) -> list[dict]:
    """既存 labeling_sheet.csv を読み込む (BOM対応)。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_tool_candidate(row: dict, frames_dir: Path) -> "ToolCandidate | None":
    """CSV1行 -> ToolCandidate (画像未生成/欠損時はNone)。"""
    base = frame_basename_from_row(row)
    crop_path = build_board_crop_image(frames_dir, base)
    if crop_path is None or not row.get("recognized_grid"):
        return None
    grid = decode_grid_string(row["recognized_grid"])
    rel_path = (Path(FRAMES_SUBDIR_NAME) / crop_path.name).as_posix()
    return ToolCandidate(
        key=f"{row['video_id']}|{row['t_sec']}|{row['side']}",
        video_id=row["video_id"], t_sec=row["t_sec"], side=row["side"],
        game_idx=row.get("game_idx", ""), occupancy=row.get("occupancy", ""),
        tier=row.get("tier", ""), phase=row.get("phase", ""),
        image_rel_path=rel_path, init_grid=grid.tolist(),
    )


def build_tool_candidates(rows: list[dict], frames_dir: Path) -> list[ToolCandidate]:
    """全CSV行から ToolCandidate リストを組み立てる (欠損行はスキップ+警告)。"""
    candidates: list[ToolCandidate] = []
    for row in rows:
        cand = build_tool_candidate(row, frames_dir)
        if cand is None:
            print(f"  [WARN] 画像/認識grid欠損のためスキップ: {row.get('video_id')} "
                  f"t={row.get('t_sec')} {row.get('side')}")
            continue
        candidates.append(cand)
    return candidates


# =============================================================================
# 4. 色パレット (COLOR_BGR -> CSS hex、既存凡例と一致させる)
# =============================================================================


def _bgr_to_css_hex(bgr: tuple[int, int, int]) -> str:
    """cv2 BGRタプルをCSS用 "#rrggbb" 文字列に変換する。"""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def build_color_palette() -> dict:
    """JS側に埋め込む色パレット (色id -> {hex, symbol}) を組み立てる。"""
    return {
        str(color_id): {
            "hex": _bgr_to_css_hex(COLOR_BGR[color_id]),
            "symbol": COLOR_SYMBOLS.get(color_id, "?"),
        }
        for color_id in CYCLE_ORDER
    }


# =============================================================================
# 5. HTML 生成
# =============================================================================


def _geometry_json() -> str:
    """座標系定数 (行数・列数) をJSに渡すJSON文字列を組み立てる。"""
    return json.dumps({
        "boardCols": BOARD_COLS, "visibleRows": N_VISIBLE_ROWS,
        "hiddenRows": HIDDEN_ROWS, "cellW": CELL_W, "cellH": CELL_H,
        "displayWidthPx": DISPLAY_WIDTH_PX,
    })


def render_html_document(
    candidates: list[ToolCandidate], storage_key: str,
) -> str:
    """label_tool.html 全体を組み立てる (CSS/JSは _HTML_CSS/_HTML_JS 定数を利用)。"""
    cand_json = json.dumps([c.__dict__ for c in candidates], ensure_ascii=False)
    palette_json = json.dumps(build_color_palette(), ensure_ascii=False)
    cycle_json = json.dumps(list(CYCLE_ORDER))
    body = _render_candidate_sections_html(candidates)
    script = _HTML_JS_TEMPLATE.format(
        candidates_json=cand_json, palette_json=palette_json,
        cycle_json=cycle_json, geometry_json=_geometry_json(),
        storage_key=json.dumps(storage_key),
        status_ok=json.dumps(STATUS_OK), status_fixed=json.dumps(STATUS_FIXED),
        status_skip=json.dumps(STATUS_SKIP),
        result_csv_header=json.dumps(",".join(RESULT_CSV_HEADER)),
    )
    return _HTML_DOCUMENT_TEMPLATE.format(
        css=_HTML_CSS, body=body, script=script, total=len(candidates),
    )


def _render_candidate_sections_html(candidates: list[ToolCandidate]) -> str:
    """各候補1件分の <section> HTML断片を連結する。"""
    sections = [_render_one_candidate_html(i, c) for i, c in enumerate(candidates)]
    return "\n".join(sections)


def _render_one_candidate_html(index: int, c: ToolCandidate) -> str:
    """1候補分の <section> (見出し+隠し段パネル+画像+グリッド+ボタン群) を組み立てる。"""
    title = (
        f"#{index + 1} {c.video_id} {c.side} t={c.t_sec}秒 "
        f"(位相:{c.phase} / 非空セル:{c.occupancy} / tier:{c.tier})"
    )
    return f'''
<section class="candidate" id="cand-{index}" data-key="{_html_escape(c.key)}">
  <h2>{_html_escape(title)}</h2>
  <div class="hidden-row-note">隠し段(行0・画面外): 見えないため「不明」のままでOKです</div>
  <div class="hidden-row-panel" data-index="{index}"
       style="width:{DISPLAY_WIDTH_PX}px;height:{HIDDEN_PANEL_HEIGHT_PX}px"></div>
  <div class="board-wrap" style="width:{DISPLAY_WIDTH_PX}px">
    <img src="{_html_escape(c.image_rel_path)}" alt="board crop">
    <div class="grid-overlay" data-index="{index}"></div>
  </div>
  <div class="controls">
    <label class="opacity-toggle">
      <input type="checkbox" class="opacity-chk" data-index="{index}" checked> グリッド表示
    </label>
    <button class="btn-ok" data-index="{index}">認識通りでOK</button>
    <button class="btn-fixed" data-index="{index}">修正完了</button>
    <button class="btn-skip" data-index="{index}">非ゲーム画面 (スキップ)</button>
    <span class="status-badge" data-index="{index}">未処理</span>
  </div>
</section>'''


def _html_escape(text: str) -> str:
    """HTML属性/本文用の最小エスケープ (& < > " のみ)。"""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


# =============================================================================
# 6. 生成後の静的整合チェック (壊れたHTML/相対パス切れを検出)
# =============================================================================


def validate_generated_html(html_path: Path, candidates: list[ToolCandidate], out_dir: Path) -> None:
    """生成物の静的整合を確認する (JSONパース可否・画像存在・grid形状)。"""
    text = html_path.read_text(encoding="utf-8")
    start = text.index("const CANDIDATES = ") + len("const CANDIDATES = ")
    end = text.index(";", start)
    parsed = json.loads(text[start:end])
    assert len(parsed) == len(candidates), "埋め込みJSON件数が候補数と不一致"
    for c in candidates:
        img_path = out_dir / c.image_rel_path
        assert img_path.exists(), f"画像が見つからない: {img_path}"
        assert len(c.init_grid) == BOARD_ROWS, f"grid行数不正: {c.key}"
        assert all(len(r) == BOARD_COLS for r in c.init_grid), f"grid列数不正: {c.key}"
    print(f"  [OK] 静的整合チェック通過 ({len(candidates)}件)")


# =============================================================================
# メイン
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="満杯盤面ラベル付けHTMLツール生成")
    parser.add_argument("--csv", type=Path, default=OUTPUT_DIR / LABELING_CSV_NAME)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """メイン処理: CSV読込 -> クロップ画像生成 -> HTML生成 -> 整合チェック。"""
    args = _parse_args()
    print(f"[1/3] labeling_sheet.csv 読込: {args.csv}")
    rows = load_labeling_rows(args.csv)
    print(f"  {len(rows)} 件")

    print("[2/3] クロップ画像生成 + 候補データ組み立て")
    frames_dir = args.out_dir / FRAMES_SUBDIR_NAME
    candidates = build_tool_candidates(rows, frames_dir)
    print(f"  有効候補: {len(candidates)} 件")

    print("[3/3] label_tool.html 生成")
    storage_key = STORAGE_KEY_PREFIX + args.out_dir.name
    html = render_html_document(candidates, storage_key)
    html_path = args.out_dir / HTML_FILENAME
    html_path.write_text(html, encoding="utf-8")
    validate_generated_html(html_path, candidates, args.out_dir)
    print(f"\n[DONE] {html_path}")


# =============================================================================
# HTML/CSS/JS テンプレート (関数外の定数として保持、1関数50行制約を回避)
# =============================================================================

_HTML_CSS = """
body { font-family: "Meiryo", sans-serif; background: #1e1e1e; color: #eee; margin: 0; }
#storage-warning { background: #a52; color: #fff; padding: 8px 16px; font-weight: bold; }
header { position: sticky; top: 0; background: #111; padding: 10px 16px; z-index: 10;
  border-bottom: 2px solid #444; }
#progress { font-size: 1.1em; font-weight: bold; }
#download-btn { margin-left: 16px; padding: 6px 14px; background: #2a6; color: #fff;
  border: none; border-radius: 4px; cursor: pointer; }
.candidate { padding: 16px; border-bottom: 1px solid #444; }
.hidden-row-note { color: #999; font-size: 0.85em; margin-bottom: 4px; }
.hidden-row-panel { display: grid; gap: 1px; margin-bottom: 6px; width: fit-content; }
.board-wrap { position: relative; }
.board-wrap img { width: 100%; display: block; }
.grid-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; gap: 0; }
.cell, .hidden-cell { border: 1px solid rgba(255,255,255,0.35); cursor: pointer;
  display: flex; align-items: center; justify-content: center; font-weight: bold;
  color: #000; text-shadow: 0 0 2px #fff; user-select: none; box-sizing: border-box; }
.hidden-cell { width: 100%; height: 100%; }
.cell.changed { border: 3px solid #ff0; }
.grid-overlay.hidden-mode { display: none; }
.controls { margin-top: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.controls button { padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; }
.btn-ok { background: #2a6; color: #fff; }
.btn-fixed { background: #26a; color: #fff; }
.btn-skip { background: #888; color: #fff; }
.status-badge { padding: 4px 10px; border-radius: 12px; background: #444; }
.status-badge.st-ok { background: #2a6; }
.status-badge.st-fixed { background: #26a; }
.status-badge.st-skip { background: #a52; }
"""

_HTML_DOCUMENT_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>満杯盤面 ラベル付けツール</title>
<style>{css}</style>
</head>
<body>
<header>
  <span id="progress">完了 0/{total}</span>
  <button id="download-btn">結果をダウンロード</button>
</header>
<main id="main">{body}</main>
<script>{script}</script>
</body>
</html>
"""

# JS本体。{{ }} は文字列.format() のエスケープ (JS側の中括弧はそのまま使うため二重化)。
_HTML_JS_TEMPLATE = """
const CANDIDATES = {candidates_json};
const PALETTE = {palette_json};
const CYCLE = {cycle_json};
const GEOM = {geometry_json};
const STORAGE_KEY = {storage_key};
const STATUS_OK = {status_ok};
const STATUS_FIXED = {status_fixed};
const STATUS_SKIP = {status_skip};

let STATE = {{}};

// file:// で開いた場合、ブラウザによっては localStorage が opaque origin
// 扱いで使えないことがある (よくある既知の落とし穴)。例外を握りつぶさず
// 警告バナーで気づけるようにし、動作自体は継続する (fail-silent禁止)。
let storageWarned = false;

function warnStorageUnavailable() {{
  if (storageWarned) return;
  storageWarned = true;
  const bar = document.createElement("div");
  bar.id = "storage-warning";
  bar.textContent = "⚠ このブラウザではローカル自動保存(localStorage)が使えません。"
    + "閉じると入力内容が失われます。こまめに「結果をダウンロード」してください。";
  document.body.insertBefore(bar, document.body.firstChild);
}}

function loadState() {{
  let saved = {{}};
  try {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) saved = JSON.parse(raw);
  }} catch (e) {{
    warnStorageUnavailable();
  }}
  for (const c of CANDIDATES) {{
    const prior = saved[c.key];
    STATE[c.key] = prior ? prior : {{
      grid: c.init_grid.map(row => row.slice()), status: null,
    }};
  }}
}}

function saveState() {{
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE));
  }} catch (e) {{
    warnStorageUnavailable();
  }}
}}

function cellStyle(colorId) {{
  const p = PALETTE[String(colorId)];
  if (colorId === 0) return {{ background: "transparent", symbol: "" }};
  return {{ background: p.hex, symbol: p.symbol }};
}}

function buildGridCells(cand, container, rowOffset, rowCount, isHidden) {{
  container.style.gridTemplateColumns = `repeat(${{GEOM.boardCols}}, 1fr)`;
  container.style.gridTemplateRows = `repeat(${{rowCount}}, 1fr)`;
  for (let r = 0; r < rowCount; r++) {{
    for (let c = 0; c < GEOM.boardCols; c++) {{
      const div = document.createElement("div");
      div.className = isHidden ? "hidden-cell" : "cell";
      div.dataset.row = r + rowOffset;
      div.dataset.col = c;
      div.addEventListener("click", () => onCellClick(cand.key, r + rowOffset, c, 1));
      div.addEventListener("contextmenu", (e) => {{
        e.preventDefault();
        onCellClick(cand.key, r + rowOffset, c, -1);
      }});
      container.appendChild(div);
    }}
  }}
}}

function renderCell(cand, row, col) {{
  const val = STATE[cand.key].grid[row][col];
  const initVal = cand.init_grid[row][col];
  const selector = row < GEOM.hiddenRows
    ? `.hidden-row-panel[data-index="${{cand._index}}"] [data-row="${{row}}"][data-col="${{col}}"]`
    : `.grid-overlay[data-index="${{cand._index}}"] [data-row="${{row}}"][data-col="${{col}}"]`;
  const div = document.querySelector(selector);
  const style = cellStyle(val);
  div.style.background = style.background;
  div.textContent = style.symbol;
  div.classList.toggle("changed", val !== initVal);
}}

function renderAllCells(cand) {{
  for (let r = 0; r < GEOM.hiddenRows + GEOM.visibleRows; r++) {{
    for (let c = 0; c < GEOM.boardCols; c++) renderCell(cand, r, c);
  }}
}}

function onCellClick(key, row, col, direction) {{
  const cand = CANDIDATES.find(c => c.key === key);
  const grid = STATE[key].grid;
  const cur = grid[row][col];
  let idx = CYCLE.indexOf(cur);
  idx = (idx + direction + CYCLE.length) % CYCLE.length;
  grid[row][col] = CYCLE[idx];
  renderCell(cand, row, col);
  saveState();
}}

function setStatus(index, status) {{
  const cand = CANDIDATES[index];
  STATE[cand.key].status = status;
  saveState();
  updateBadge(index);
  updateProgress();
  scrollToNext(index);
}}

function updateBadge(index) {{
  const badge = document.querySelector(`.status-badge[data-index="${{index}}"]`);
  const status = STATE[CANDIDATES[index].key].status;
  const labels = {{ [STATUS_OK]: "認識通りでOK", [STATUS_FIXED]: "修正完了", [STATUS_SKIP]: "非ゲーム画面" }};
  badge.textContent = status ? labels[status] : "未処理";
  badge.className = "status-badge" + (status ? " st-" + status : "");
}}

function updateProgress() {{
  const done = CANDIDATES.filter(c => STATE[c.key].status).length;
  document.getElementById("progress").textContent = `完了 ${{done}}/${{CANDIDATES.length}}`;
}}

function scrollToNext(index) {{
  const next = document.getElementById(`cand-${{index + 1}}`);
  if (next) next.scrollIntoView({{ behavior: "smooth", block: "start" }});
}}

function encodeGridString(grid) {{
  return grid.map(row => row.map(v => v === 10 ? "U" : String(v)).join("")).join("/");
}}

function countChanged(cand) {{
  const grid = STATE[cand.key].grid;
  let n = 0;
  for (let r = 0; r < grid.length; r++) {{
    for (let c = 0; c < grid[r].length; c++) {{
      if (grid[r][c] !== cand.init_grid[r][c]) n++;
    }}
  }}
  return n;
}}

function buildResultCsv() {{
  const lines = [{result_csv_header}];
  for (const c of CANDIDATES) {{
    const st = STATE[c.key];
    const status = st.status || "";
    const correctGrid = status === STATUS_SKIP ? "" : encodeGridString(st.grid);
    lines.push([c.video_id, c.t_sec, c.side, status, correctGrid, countChanged(c)].join(","));
  }}
  return lines.join("\\n");
}}

function downloadResult() {{
  const csv = buildResultCsv();
  const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labeling_result.csv";
  a.click();
  URL.revokeObjectURL(url);
}}

function initCandidate(index) {{
  const cand = CANDIDATES[index];
  cand._index = index;
  const hiddenPanel = document.querySelector(`.hidden-row-panel[data-index="${{index}}"]`);
  const gridOverlay = document.querySelector(`.grid-overlay[data-index="${{index}}"]`);
  buildGridCells(cand, hiddenPanel, 0, GEOM.hiddenRows, true);
  buildGridCells(cand, gridOverlay, GEOM.hiddenRows, GEOM.visibleRows, false);
  renderAllCells(cand);
  updateBadge(index);
  document.querySelector(`.btn-ok[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_OK));
  document.querySelector(`.btn-fixed[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_FIXED));
  document.querySelector(`.btn-skip[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_SKIP));
  document.querySelector(`.opacity-chk[data-index="${{index}}"]`).addEventListener(
    "change", (e) => gridOverlay.classList.toggle("hidden-mode", !e.target.checked));
}}

function init() {{
  loadState();
  CANDIDATES.forEach((_, i) => initCandidate(i));
  updateProgress();
  document.getElementById("download-btn").addEventListener("click", downloadResult);
}}

// テスト用フック: 自動回帰テスト(jsdom)がconst/let束縛を参照できるよう
// window に公開する (挙動には影響しない、ブラウザ利用時は無視してよい)。
window.CANDIDATES = CANDIDATES;
window.STATE = STATE;
window.CYCLE = CYCLE;
window.GEOM = GEOM;
window.STATUS_OK = STATUS_OK;
window.STATUS_FIXED = STATUS_FIXED;
window.STATUS_SKIP = STATUS_SKIP;
window.buildResultCsv = buildResultCsv;

init();
"""


if __name__ == "__main__":
    main()
