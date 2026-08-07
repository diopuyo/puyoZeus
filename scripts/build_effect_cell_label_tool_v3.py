"""エフェクト有無 セルラベル付け用 クリック操作HTMLツール 第3弾 生成 (2026-08-04)。

scripts/build_effect_cell_label_sheet_v3.py が準備した labeling_sheet.csv +
frames/*_board_crop.png (~100候補、burst/smoke/telop_negative/zenkeshi/baseline
5層) から、user が盤面クロップ画像の上に直接重ねた透明グリッドをクリックする
だけで「エフェクトが被っているセル」をラベル付けできるローカルHTMLツールを
生成する。

## 第1弾ツール (scripts/build_effect_cell_label_tool.py) との差分
較正 (data/verify/effect_detector_calibration_2026-08-04/calibration_report.md §5)
で「連鎖数テロップの発光」との混同が唯一の失敗モードと判明した。第1弾では
テロップ表示中のフレームを「フレーム異常(スキップ)」ボタンで運用していたが、
これは「フレーム自体が壊れている」ケースと「対象外の演出が写っている」ケースを
区別できず、テロップ混同の頻度が正しく数えられなかった。本ツールは
**「対象外エフェクト」ボタンを新設**し、連鎖数テロップ/全消しテロップ/自連鎖
発光などの対象外演出を正式なラベル (status="out_of_scope") として記録する。
既存の B/S循環 (セルクリックでなし->バースト->煙)・エフェクトなしボタン・
結果ダウンロード・localStorage自動保存の UX はそのまま維持する。

## 出力
    data/verify/effect_cell_label_v3_2026-08-04/label_tool_v3.html

## 使い方 (ラベル付け、user向け)
    label_tool_v3.html を Windows Explorer からダブルクリックしてブラウザで開く。
    セルをクリックで なし→バースト→煙 と循環 (右クリックで逆循環)。
    エフェクトが無いフレームは「エフェクトなし」ボタンで即確定。
    連鎖数テロップ等の対象外演出が写っている場合は「対象外エフェクト」ボタン。
    「結果をダウンロード」で labeling_result.csv が保存される。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.build_effect_cell_label_tool_v3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_effect_cell_label_sheet_v3 import (  # noqa: E402
    LAYER_LABEL_JA, OUTPUT_DIR,
)
from scripts.visualize_recognition import CELL_H, CELL_W, N_VISIBLE_ROWS, ROI_W  # noqa: E402
from src.board import BOARD_COLS  # noqa: E402

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

LABELING_CSV_NAME: str = "labeling_sheet.csv"
HTML_FILENAME: str = "label_tool_v3.html"

# クリックで循環させる状態順 (なし -> バースト -> 煙 -> なし)。
# 第1弾からの互換維持のため値は変更しない。
EFFECT_STATE_NONE: int = 0
EFFECT_STATE_BURST: int = 1
EFFECT_STATE_SMOKE: int = 2
CYCLE_ORDER: tuple[int, ...] = (EFFECT_STATE_NONE, EFFECT_STATE_BURST, EFFECT_STATE_SMOKE)

# 状態ごとの表示色 (画像の上に重ねるため半透明、CSS rgba文字列)
STATE_PALETTE: dict[int, dict[str, str]] = {
    EFFECT_STATE_NONE: {"rgba": "transparent", "symbol": ""},
    EFFECT_STATE_BURST: {"rgba": "rgba(255,140,26,0.55)", "symbol": "B"},
    EFFECT_STATE_SMOKE: {"rgba": "rgba(176,176,176,0.55)", "symbol": "S"},
}

STORAGE_KEY_PREFIX: str = "puyo_effect_cell_label_tool_v3::"

# 表示用の見やすさ設定 (座標系には影響しない、CSS表示幅のみ)
DISPLAY_WIDTH_PX: int = 480
_DISPLAY_SCALE: float = DISPLAY_WIDTH_PX / ROI_W
BOARD_DISPLAY_HEIGHT_PX: int = round((N_VISIBLE_ROWS * CELL_H) * _DISPLAY_SCALE)
FULL_THUMB_WIDTH_PX: int = 220

STATUS_NO_EFFECT: str = "no_effect"
STATUS_MARKED: str = "marked"
STATUS_OUT_OF_SCOPE: str = "out_of_scope"  # 新設 (連鎖数テロップ/全消しテロップ等)
STATUS_SKIP: str = "skip"

RESULT_CSV_HEADER: tuple[str, ...] = (
    "video_id", "t_sec", "side", "layer", "note", "status",
    "effect_grid", "marked_cells",
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class EffectToolCandidate:
    """label_tool_v3.html に埋め込む1候補分のデータ。"""

    key: str
    video_id: str
    t_sec: str
    side: str
    layer: str
    note: str
    image_rel_path: str   # "frames/xxx_board_crop.png"
    full_rel_path: str    # "frames/xxx_full.png" (サムネイル用)


# =============================================================================
# 1. 候補データ組み立て
# =============================================================================


def load_labeling_rows(csv_path: Path) -> list[dict]:
    """labeling_sheet.csv を読み込む (BOM対応)。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _rel_path_under_frames(windows_path: str) -> str:
    """CSVのWindows絶対パスから、HTMLからの相対パス "frames/<ファイル名>" を組み立てる。"""
    name = Path(windows_path.replace("\\", "/")).name
    return (Path("frames") / name).as_posix()


def build_tool_candidate(row: dict, out_dir: Path) -> "EffectToolCandidate | None":
    """CSV1行 -> EffectToolCandidate (画像欠損時はNone)。"""
    image_rel = _rel_path_under_frames(row.get("image_board_crop", ""))
    full_rel = _rel_path_under_frames(row.get("image_full_frame", ""))
    if not (out_dir / image_rel).exists() or not (out_dir / full_rel).exists():
        return None
    return EffectToolCandidate(
        key=f"{row['video_id']}|{row['t_sec']}|{row['side']}|{row['layer']}",
        video_id=row["video_id"], t_sec=row["t_sec"], side=row["side"],
        layer=row["layer"], note=row.get("note", ""),
        image_rel_path=image_rel, full_rel_path=full_rel,
    )


def build_tool_candidates(rows: list[dict], out_dir: Path) -> list[EffectToolCandidate]:
    """全CSV行から EffectToolCandidate リストを組み立てる (欠損行はスキップ+警告)。"""
    candidates: list[EffectToolCandidate] = []
    for row in rows:
        cand = build_tool_candidate(row, out_dir)
        if cand is None:
            print(f"  [WARN] 画像欠損のためスキップ: {row.get('video_id')} "
                  f"t={row.get('t_sec')} {row.get('side')}")
            continue
        candidates.append(cand)
    return candidates


# =============================================================================
# 2. HTML 生成
# =============================================================================


def _geometry_json() -> str:
    """座標系定数をJSに渡すJSON文字列を組み立てる。"""
    return json.dumps({
        "boardCols": BOARD_COLS, "visibleRows": N_VISIBLE_ROWS,
        "cellW": CELL_W, "cellH": CELL_H, "displayWidthPx": DISPLAY_WIDTH_PX,
    })


def _palette_json() -> str:
    """状態パレットをJSに渡すJSON文字列を組み立てる。"""
    return json.dumps({str(k): v for k, v in STATE_PALETTE.items()}, ensure_ascii=False)


def render_html_document(candidates: list[EffectToolCandidate], storage_key: str) -> str:
    """label_tool_v3.html 全体を組み立てる。"""
    cand_json = json.dumps([c.__dict__ for c in candidates], ensure_ascii=False)
    body = "\n".join(_render_one_candidate_html(i, c) for i, c in enumerate(candidates))
    script = _HTML_JS_TEMPLATE.format(
        candidates_json=cand_json, palette_json=_palette_json(),
        cycle_json=json.dumps(list(CYCLE_ORDER)), geometry_json=_geometry_json(),
        storage_key=json.dumps(storage_key),
        status_no_effect=json.dumps(STATUS_NO_EFFECT), status_marked=json.dumps(STATUS_MARKED),
        status_out_of_scope=json.dumps(STATUS_OUT_OF_SCOPE), status_skip=json.dumps(STATUS_SKIP),
        result_csv_header=json.dumps(",".join(RESULT_CSV_HEADER)),
    )
    return _HTML_DOCUMENT_TEMPLATE.format(
        css=_HTML_CSS, body=body, script=script, total=len(candidates),
    )


def _render_one_candidate_html(index: int, c: EffectToolCandidate) -> str:
    """1候補分の <section> (見出し+サムネ+盤面クロップ+オーバーレイ+ボタン群) を組み立てる。"""
    layer_ja = LAYER_LABEL_JA.get(c.layer, c.layer)
    note = f" / {c.note}" if c.note else ""
    title = f"#{index + 1} {c.video_id} {c.side} t={c.t_sec}秒 ({layer_ja}{note})"
    return f'''
<section class="candidate" id="cand-{index}" data-key="{_html_escape(c.key)}">
  <h2>{_html_escape(title)}</h2>
  <a class="full-thumb-link" href="{_html_escape(c.full_rel_path)}" target="_blank"
     title="実画面フルショット (別タブで原寸表示)">
    <img class="full-thumb" src="{_html_escape(c.full_rel_path)}" alt="full frame"
         style="width:{FULL_THUMB_WIDTH_PX}px">
  </a>
  <div class="board-wrap" style="width:{DISPLAY_WIDTH_PX}px;height:{BOARD_DISPLAY_HEIGHT_PX}px">
    <img src="{_html_escape(c.image_rel_path)}" alt="board crop">
    <div class="grid-overlay" data-index="{index}"></div>
  </div>
  <div class="controls">
    <button class="btn-none" data-index="{index}">エフェクトなし</button>
    <button class="btn-marked" data-index="{index}">マーク完了</button>
    <button class="btn-outofscope" data-index="{index}">対象外エフェクト</button>
    <button class="btn-skip" data-index="{index}">フレーム異常(スキップ)</button>
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
# 3. 生成後の静的整合チェック
# =============================================================================


def validate_generated_html(html_path: Path, candidates: list[EffectToolCandidate], out_dir: Path) -> None:
    """生成物の静的整合を確認する (JSONパース可否・画像存在・4ボタン埋め込み)。"""
    text = html_path.read_text(encoding="utf-8")
    start = text.index("const CANDIDATES = ") + len("const CANDIDATES = ")
    end = text.index(";", start)
    parsed = json.loads(text[start:end])
    assert len(parsed) == len(candidates), "埋め込みJSON件数が候補数と不一致"
    assert "btn-outofscope" in text, "「対象外エフェクト」ボタンが生成HTMLに存在しない"
    for c in candidates:
        assert (out_dir / c.image_rel_path).exists(), f"画像が見つからない: {c.image_rel_path}"
        assert (out_dir / c.full_rel_path).exists(), f"フルショットが見つからない: {c.full_rel_path}"
    print(f"  [OK] 静的整合チェック通過 ({len(candidates)}件)")


# =============================================================================
# メイン
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="エフェクト有無セルラベル付けHTMLツール第3弾生成")
    parser.add_argument("--csv", type=Path, default=OUTPUT_DIR / LABELING_CSV_NAME)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """メイン処理: CSV読込 -> 候補データ組み立て -> HTML生成 -> 整合チェック。"""
    args = _parse_args()
    print(f"[1/3] labeling_sheet.csv 読込: {args.csv}")
    rows = load_labeling_rows(args.csv)
    print(f"  {len(rows)} 件")

    print("[2/3] 候補データ組み立て")
    candidates = build_tool_candidates(rows, args.out_dir)
    print(f"  有効候補: {len(candidates)} 件")

    print("[3/3] label_tool_v3.html 生成")
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
.full-thumb-link { display: inline-block; float: right; margin-left: 12px; }
.full-thumb { border: 1px solid #555; border-radius: 4px; display: block; }
.board-wrap { position: relative; overflow: hidden; clear: both; border: 1px solid #444; }
.board-wrap img { width: 100%; height: 100%; display: block; object-fit: cover; }
.grid-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; }
.cell { border: 1px solid rgba(255,255,255,0.25); cursor: pointer; display: flex;
  align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;
  color: #fff; text-shadow: 0 0 3px #000, 0 0 3px #000; user-select: none; box-sizing: border-box; }
.controls { margin-top: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.controls button { padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; }
.btn-none { background: #888; color: #fff; }
.btn-marked { background: #26a; color: #fff; }
.btn-outofscope { background: #b8860b; color: #fff; }
.btn-skip { background: #a52; color: #fff; }
.status-badge { padding: 4px 10px; border-radius: 12px; background: #444; }
.status-badge.st-no_effect { background: #888; }
.status-badge.st-marked { background: #26a; }
.status-badge.st-out_of_scope { background: #b8860b; }
.status-badge.st-skip { background: #a52; }
"""

_HTML_DOCUMENT_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>エフェクト有無 セルラベル付けツール 第3弾</title>
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
const STATUS_NO_EFFECT = {status_no_effect};
const STATUS_MARKED = {status_marked};
const STATUS_OUT_OF_SCOPE = {status_out_of_scope};
const STATUS_SKIP = {status_skip};

let STATE = {{}};
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

function emptyGrid() {{
  return Array.from({{ length: GEOM.visibleRows }}, () => Array(GEOM.boardCols).fill(0));
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
    STATE[c.key] = saved[c.key] ? saved[c.key] : {{ grid: emptyGrid(), status: null }};
  }}
}}

function saveState() {{
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE));
  }} catch (e) {{
    warnStorageUnavailable();
  }}
}}

function cellStyle(state) {{
  const p = PALETTE[String(state)];
  return {{ background: p.rgba, symbol: p.symbol }};
}}

function buildGridCells(cand, container) {{
  container.style.gridTemplateColumns = `repeat(${{GEOM.boardCols}}, 1fr)`;
  container.style.gridTemplateRows = `repeat(${{GEOM.visibleRows}}, 1fr)`;
  for (let r = 0; r < GEOM.visibleRows; r++) {{
    for (let c = 0; c < GEOM.boardCols; c++) {{
      const div = document.createElement("div");
      div.className = "cell";
      div.dataset.row = r;
      div.dataset.col = c;
      div.addEventListener("click", () => onCellClick(cand.key, r, c, 1));
      div.addEventListener("contextmenu", (e) => {{
        e.preventDefault();
        onCellClick(cand.key, r, c, -1);
      }});
      container.appendChild(div);
    }}
  }}
}}

function renderCell(cand, row, col) {{
  const val = STATE[cand.key].grid[row][col];
  const selector = `.grid-overlay[data-index="${{cand._index}}"] [data-row="${{row}}"][data-col="${{col}}"]`;
  const div = document.querySelector(selector);
  const style = cellStyle(val);
  div.style.background = style.background;
  div.textContent = style.symbol;
}}

function renderAllCells(cand) {{
  for (let r = 0; r < GEOM.visibleRows; r++) {{
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
  if (status === STATUS_NO_EFFECT || status === STATUS_OUT_OF_SCOPE) STATE[cand.key].grid = emptyGrid();
  STATE[cand.key].status = status;
  renderAllCells(cand);
  saveState();
  updateBadge(index);
  updateProgress();
  scrollToNext(index);
}}

function updateBadge(index) {{
  const badge = document.querySelector(`.status-badge[data-index="${{index}}"]`);
  const status = STATE[CANDIDATES[index].key].status;
  const labels = {{ [STATUS_NO_EFFECT]: "エフェクトなし", [STATUS_MARKED]: "マーク完了",
    [STATUS_OUT_OF_SCOPE]: "対象外エフェクト", [STATUS_SKIP]: "スキップ" }};
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
  return grid.map(row => row.join("")).join("/");
}}

function countMarked(cand) {{
  const grid = STATE[cand.key].grid;
  let n = 0;
  for (const row of grid) for (const v of row) if (v !== 0) n++;
  return n;
}}

function buildResultCsv() {{
  const lines = [{result_csv_header}];
  for (const c of CANDIDATES) {{
    const st = STATE[c.key];
    const status = st.status || "";
    const effectGrid = (status === STATUS_SKIP || status === STATUS_OUT_OF_SCOPE)
      ? "" : encodeGridString(st.grid);
    lines.push([c.video_id, c.t_sec, c.side, c.layer, c.note, status,
      effectGrid, countMarked(c)].join(","));
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
  const overlay = document.querySelector(`.grid-overlay[data-index="${{index}}"]`);
  buildGridCells(cand, overlay);
  renderAllCells(cand);
  updateBadge(index);
  document.querySelector(`.btn-none[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_NO_EFFECT));
  document.querySelector(`.btn-marked[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_MARKED));
  document.querySelector(`.btn-outofscope[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_OUT_OF_SCOPE));
  document.querySelector(`.btn-skip[data-index="${{index}}"]`).addEventListener(
    "click", () => setStatus(index, STATUS_SKIP));
}}

function init() {{
  loadState();
  CANDIDATES.forEach((_, i) => initCandidate(i));
  updateProgress();
  document.getElementById("download-btn").addEventListener("click", downloadResult);
}}

// テスト用フック: 自動回帰テスト(jsdom)がconst/let束縛を参照できるよう window に公開。
window.CANDIDATES = CANDIDATES;
window.STATE = STATE;
window.CYCLE = CYCLE;
window.GEOM = GEOM;
window.STATUS_NO_EFFECT = STATUS_NO_EFFECT;
window.STATUS_MARKED = STATUS_MARKED;
window.STATUS_OUT_OF_SCOPE = STATUS_OUT_OF_SCOPE;
window.STATUS_SKIP = STATUS_SKIP;
window.buildResultCsv = buildResultCsv;

init();
"""


if __name__ == "__main__":
    main()
