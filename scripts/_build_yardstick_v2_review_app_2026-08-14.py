"""物差し v2 の採点用 自己完結HTMLアプリ生成 (2026-08-14)。

## 踏襲元
`scripts/build_full_board_label_tool.py` (2026-08-02) のクリック循環UI
(空→赤→青→緑→黄→紫→おじゃま→不明→空、右クリックで逆循環) と localStorage
自動保存パターンをそのまま踏襲する。差分は user 指示 (2026-08-14 追加指示) の
「1盤面ずつのナビ (前へ/次へ/一覧) 」「コピー用JSON/TSVをテキストエリアに
出す」の2点を新規に追加すること。

置換方式について: 旧ツールは JS を `str.format()` に通すため `{{`/`}}` で
全JS波括弧を二重化していたが、本ツールはナビ/一覧/書き出しUIが増えて
JSが長く可読性が落ちるため、**プレースホルダトークンの `str.replace()`**
方式に変更する (JS側の波括弧はそのまま書ける)。

## 入力
`scripts._build_yardstick_v2_sheets_2026-08-14` が出力した
`data/verify/yardstick_v2_2026-08-14/manifest.json` (60盤面、init_grid付き)
と `anchors/*_roi.png` (盤面ROIの生画像、実画面表示に使う)。

## 出力
`data/verify/yardstick_v2_2026-08-14/review_app.html`
(単一HTML、外部CDN不使用、画像はローカル相対パス `anchors/...` 参照)。

## 使い方 (生成)
    PYTHONPATH=. ./venv/bin/python -m scripts._build_yardstick_v2_review_app_2026-08-14

## 使い方 (採点、user向け)
    review_app.html を Windows Explorer からダブルクリックしてブラウザで開く。
    セルをクリック (左クリック=順循環、右クリック=逆循環) で訂正。
    「OK/修正完了/非対局画面」のいずれかを押すと次の盤面へ進む。
    右上の「一覧」から任意の盤面へジャンプできる。
    採点結果は自動でブラウザに保存される (再開可能)。
    完了したら下部の「JSON/TSVダウンロード」または「コピー」で結果を取り出す。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SHEET = importlib.import_module("scripts._build_yardstick_v2_sheets_2026-08-14")

OUT_DIR: Path = _SHEET.OUT_DIR
MANIFEST_PATH: Path = OUT_DIR / "manifest.json"
HTML_PATH: Path = OUT_DIR / "review_app.html"

STORAGE_KEY_PREFIX: str = "puyo_yardstick_v2_review::"

# クリックで循環させる色順 (既存 build_full_board_label_tool.py と同一仕様)
CYCLE_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 9, 10)
COLOR_HEX: dict[int, str] = {
    0: "transparent", 1: "#dc3c3c", 2: "#3c78dc", 3: "#50c850",
    4: "#e6d23c", 5: "#c850c8", 9: "#aaaaaa", 10: "#000000",
}
COLOR_SYMBOL: dict[int, str] = {
    0: "", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "邪", 10: "?",
}

BOARD_ROWS: int = _SHEET._SHEET.BOARD_ROWS
BOARD_COLS: int = _SHEET._SHEET.BOARD_COLS
HIDDEN_ROWS: int = _SHEET._SHEET.HIDDEN_ROWS

STATUS_OK: str = "ok"
STATUS_FIXED: str = "fixed"
STATUS_NOT_A_BOARD: str = "not_a_board"


def load_manifest() -> list[dict]:
    """manifest.json を読み、anchor_status=="ok" の盤面のみ返す。"""
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ok_rows = [r for r in data if r.get("anchor_status") == "ok"]
    skipped = len(data) - len(ok_rows)
    if skipped:
        print(f"  [警告] anchor_status!=ok のため {skipped} 件除外")
    return ok_rows


def build_candidate_json(rows: list[dict]) -> str:
    """JS へ埋め込む CANDIDATES 配列のJSON文字列を組み立てる。"""
    out = []
    for r in rows:
        roi_name = Path(r["board_roi_png"]).name
        out.append({
            "key": r["sheet_id"], "videoId": r["video_id"], "side": r["side"],
            "frameIdxAux": r["frame_idx"], "tSec": r["t_sec"], "phase": r["phase"],
            "fillRatio": r["fill_ratio"], "hasOjama": r["has_ojama"],
            "roiRelPath": f"anchors/{roi_name}", "initGrid": r["init_grid"],
        })
    return json.dumps(out, ensure_ascii=False)


def build_palette_json() -> str:
    """色パレット (hex+記号) のJSON文字列を組み立てる。"""
    palette = {
        str(cid): {"hex": COLOR_HEX[cid], "symbol": COLOR_SYMBOL[cid]}
        for cid in CYCLE_ORDER
    }
    return json.dumps(palette, ensure_ascii=False)


def _replace_placeholders(template: str, values: dict[str, str]) -> str:
    """__TOKEN__ 形式のプレースホルダを str.replace で埋める (波括弧問題を回避)。"""
    out = template
    for k, v in values.items():
        out = out.replace(f"__{k}__", v)
    return out


def render_html(rows: list[dict]) -> str:
    """review_app.html 全体を組み立てる。"""
    script = _replace_placeholders(_JS_TEMPLATE, {
        "CANDIDATES_JSON": build_candidate_json(rows),
        "PALETTE_JSON": build_palette_json(),
        "CYCLE_JSON": json.dumps(list(CYCLE_ORDER)),
        "GEOMETRY_JSON": json.dumps({
            "boardRows": BOARD_ROWS, "boardCols": BOARD_COLS, "hiddenRows": HIDDEN_ROWS,
        }),
        "STORAGE_KEY": json.dumps(STORAGE_KEY_PREFIX + OUT_DIR.name),
        "STATUS_OK": json.dumps(STATUS_OK), "STATUS_FIXED": json.dumps(STATUS_FIXED),
        "STATUS_NOT_A_BOARD": json.dumps(STATUS_NOT_A_BOARD),
    })
    return _replace_placeholders(_HTML_TEMPLATE, {
        "CSS": _CSS, "SCRIPT": script, "TOTAL": str(len(rows)),
    })


def validate_generated_html(html_path: Path, rows: list[dict]) -> None:
    """生成物の静的整合を確認する (JSONパース可否・画像存在)。"""
    text = html_path.read_text(encoding="utf-8")
    start = text.index("const CANDIDATES = ") + len("const CANDIDATES = ")
    end = text.index(";\n", start)
    parsed = json.loads(text[start:end])
    assert len(parsed) == len(rows), "埋め込みJSON件数がmanifest件数と不一致"
    for r in rows:
        roi_path = OUT_DIR / r["board_roi_png"].split("/")[-1]
        assert (OUT_DIR / "anchors" / roi_path.name).exists(), f"ROI画像が無い: {roi_path.name}"
    print(f"  [OK] 静的整合チェック通過 ({len(rows)}件)")


def main() -> None:
    print(f"[1/3] manifest読込: {MANIFEST_PATH}")
    rows = load_manifest()
    print(f"  {len(rows)} 盤面")
    print("[2/3] HTML生成")
    html = render_html(rows)
    HTML_PATH.write_text(html, encoding="utf-8")
    print("[3/3] 静的整合チェック")
    validate_generated_html(HTML_PATH, rows)
    print(f"\n[DONE] {HTML_PATH}")


# =============================================================================
# HTML/CSS/JS テンプレート (関数外の定数、1関数50行制約回避のため分離)
# =============================================================================

_CSS = """
* { box-sizing: border-box; }
body { font-family: "Meiryo", system-ui, sans-serif; background: #14161a; color: #e6e8eb;
  margin: 0; }
#storage-warning { background: #a52; color: #fff; padding: 8px 16px; font-weight: bold; }
header { position: sticky; top: 0; background: #1e2127; padding: 10px 16px; z-index: 10;
  border-bottom: 2px solid #2c313a; display: flex; align-items: center; gap: 14px;
  flex-wrap: wrap; }
header button, .controls button, .export-panel button {
  padding: 6px 14px; border-radius: 6px; border: none; cursor: pointer; font-size: 0.95em; }
#progress { font-weight: bold; }
#overview-btn { background: #456; color: #fff; }
#prev-btn, #next-btn { background: #345; color: #fff; }
main { max-width: 920px; margin: 0 auto; padding: 16px; }
.meta-line { color: #9aa2ad; font-size: 0.9em; margin-bottom: 8px; }
.pane-row { display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; }
.board-wrap { position: relative; overflow: hidden; border: 1px solid #2c313a; width: 480px; }
.board-wrap img { width: 100%; height: auto; display: block; }
.hidden-row-note { color: #9aa2ad; font-size: 0.85em; margin-bottom: 4px; }
/* board_roi_png は可視12行のみ (384x720 ネイティブ)。表示幅480pxに揃えると
   高さは 720*(480/384)=900px になる。右のグリッドも同じ縦横比で揃えないと
   左右の行の高さがズレて見比べられなくなる (1行=900/12=75px)。 */
.hidden-row-panel, .grid-overlay { display: grid; gap: 0; width: 480px; }
.hidden-row-panel { height: 75px; margin-bottom: 6px; }
.grid-overlay { height: 900px; }
.cell, .hidden-cell { border: 1px solid rgba(255,255,255,0.35); cursor: pointer;
  display: flex; align-items: center; justify-content: center; font-weight: bold;
  color: #000; text-shadow: 0 0 2px #fff; user-select: none; }
.cell.changed { border: 3px solid #ff0; }
.controls { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.btn-ok { background: #2a6; color: #fff; } .btn-fixed { background: #26a; color: #fff; }
.btn-skip { background: #a52; color: #fff; }
.status-badge { padding: 4px 10px; border-radius: 12px; background: #444; }
.status-badge.st-ok { background: #2a6; } .status-badge.st-fixed { background: #26a; }
.status-badge.st-not_a_board { background: #a52; }
#overview-panel { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9);
  z-index: 20; overflow: auto; padding: 20px; }
#overview-panel.show { display: block; }
.ov-grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; max-width: 700px;
  margin: 0 auto; }
.ov-cell { padding: 10px 0; text-align: center; border-radius: 6px; background: #333;
  cursor: pointer; font-size: 0.85em; }
.ov-cell.st-ok { background: #2a6; } .ov-cell.st-fixed { background: #26a; }
.ov-cell.st-not_a_board { background: #a52; }
.export-panel { max-width: 920px; margin: 24px auto; padding: 14px; border-top: 2px solid #2c313a; }
.export-panel textarea { width: 100%; height: 160px; background: #1e2127; color: #e6e8eb;
  border: 1px solid #2c313a; font-family: monospace; font-size: 0.8em; }
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head><meta charset="utf-8"><title>物差し v2 採点アプリ</title><style>__CSS__</style></head>
<body>
<header>
  <span id="progress">完了 0/__TOTAL__</span>
  <button id="prev-btn">← 前へ</button>
  <button id="next-btn">次へ →</button>
  <button id="overview-btn">一覧</button>
</header>
<main id="main"></main>
<div id="overview-panel">
  <div class="ov-grid" id="ov-grid"></div>
  <p style="text-align:center;margin-top:16px;">
    <button id="ov-close">閉じる</button>
  </p>
</div>
<div class="export-panel">
  <h3>採点結果の書き出し</h3>
  <div class="controls">
    <button id="show-json-btn">コピー用JSON表示</button>
    <button id="show-tsv-btn">コピー用TSV表示</button>
    <button id="copy-btn">テキストエリアをコピー</button>
    <button id="dl-json-btn">JSONファイルダウンロード</button>
    <button id="dl-tsv-btn">TSVファイルダウンロード (labels.tsv形式)</button>
  </div>
  <textarea id="export-area" readonly placeholder="上のボタンでJSON/TSVを表示"></textarea>
</div>
<script>__SCRIPT__</script>
</body>
</html>
"""

_JS_TEMPLATE = """
const CANDIDATES = __CANDIDATES_JSON__;
const PALETTE = __PALETTE_JSON__;
const CYCLE = __CYCLE_JSON__;
const GEOM = __GEOMETRY_JSON__;
const STORAGE_KEY = __STORAGE_KEY__;
const STATUS_OK = __STATUS_OK__;
const STATUS_FIXED = __STATUS_FIXED__;
const STATUS_NOT_A_BOARD = __STATUS_NOT_A_BOARD__;

let STATE = {};
let CURRENT = 0;
let storageWarned = false;

function warnStorageUnavailable() {
  if (storageWarned) return;
  storageWarned = true;
  const bar = document.createElement("div");
  bar.id = "storage-warning";
  bar.textContent = "\\u26a0 \\u3053\\u306e\\u30d6\\u30e9\\u30a6\\u30b6\\u3067\\u306f"
    + "\\u81ea\\u52d5\\u4fdd\\u5b58\\u304c\\u4f7f\\u3048\\u307e\\u305b\\u3093\\u3002"
    + "\\u3053\\u307e\\u3081\\u306b\\u30c0\\u30a6\\u30f3\\u30ed\\u30fc\\u30c9\\u3057"
    + "\\u3066\\u304f\\u3060\\u3055\\u3044\\u3002";
  document.body.insertBefore(bar, document.body.firstChild);
}

function loadState() {
  let saved = {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) saved = JSON.parse(raw);
  } catch (e) { warnStorageUnavailable(); }
  const savedStates = saved.states || {};
  for (const c of CANDIDATES) {
    const prior = savedStates[c.key];
    STATE[c.key] = prior ? prior : { grid: c.initGrid.map(row => row.slice()), status: null };
  }
  CURRENT = Number.isInteger(saved.current) && saved.current < CANDIDATES.length ? saved.current : 0;
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ current: CURRENT, states: STATE }));
  } catch (e) { warnStorageUnavailable(); }
}

function cellStyle(colorId) {
  const p = PALETTE[String(colorId)];
  return { background: colorId === 0 ? "transparent" : p.hex, symbol: p.symbol };
}

function boardMetaText(cand, idx) {
  const pct = Math.round(cand.fillRatio * 100);
  return `${cand.key} | video=${cand.videoId} side=${cand.side} phase=${cand.phase} `
    + `\\u5145\\u5145\\u5ea6=${pct}% \\u304a\\u3058\\u3083\\u307e=${cand.hasOjama} `
    + `frame_idx(\\u88dc\\u52a9)=${cand.frameIdxAux} t=${cand.tSec}s (${idx + 1}/${CANDIDATES.length})`;
}

function boardHtml(cand, idx) {
  return (
    '<div class="meta-line">' + boardMetaText(cand, idx) + '</div>'
    + '<div class="hidden-row-note">\\u96a0\\u3057\\u6bb5(\\u884c0\\u30fb\\u753b\\u9762\\u5916): '
    + '\\u898b\\u3048\\u306a\\u3044\\u305f\\u3081\\u300c\\u4e0d\\u660e\\u300d\\u306e\\u307e\\u307e\\u3067OK\\u3067\\u3059'
    + '</div>'
    + '<div class="hidden-row-panel" id="hidden-panel"></div>'
    + '<div class="pane-row">'
    + '<div class="board-wrap" id="board-wrap"><img src="' + cand.roiRelPath + '" alt="board"></div>'
    + '<div class="grid-overlay" id="grid-overlay"></div>'
    + '</div>'
    + '<div class="controls">'
    + '<button class="btn-ok" id="btn-ok">\\u8a8d\\u8b58\\u901a\\u308a\\u3067OK</button>'
    + '<button class="btn-fixed" id="btn-fixed">\\u4fee\\u6b63\\u5b8c\\u4e86</button>'
    + '<button class="btn-skip" id="btn-skip">\\u975e\\u5bfe\\u5c40\\u753b\\u9762(NOT_A_BOARD)</button>'
    + '<span class="status-badge" id="status-badge">\\u672a\\u51e6\\u7406</span>'
    + '</div>'
  );
}

function buildGridCells(container, cand, rowOffset, rowCount, isHidden) {
  container.innerHTML = "";
  container.style.gridTemplateColumns = `repeat(${GEOM.boardCols}, 1fr)`;
  container.style.gridTemplateRows = `repeat(${rowCount}, 1fr)`;
  for (let r = 0; r < rowCount; r++) {
    for (let c = 0; c < GEOM.boardCols; c++) {
      const div = document.createElement("div");
      div.className = isHidden ? "hidden-cell" : "cell";
      const row = r + rowOffset;
      div.dataset.row = row;
      div.dataset.col = c;
      div.addEventListener("click", () => onCellClick(cand.key, row, c, 1));
      div.addEventListener("contextmenu", (e) => { e.preventDefault(); onCellClick(cand.key, row, c, -1); });
      container.appendChild(div);
    }
  }
}

function renderCell(cand, row, col) {
  const val = STATE[cand.key].grid[row][col];
  const initVal = cand.initGrid[row][col];
  const isHidden = row < GEOM.hiddenRows;
  const scope = isHidden ? document.getElementById("hidden-panel") : document.getElementById("grid-overlay");
  const div = scope.querySelector(`[data-row="${row}"][data-col="${col}"]`);
  const style = cellStyle(val);
  div.style.background = style.background;
  div.textContent = style.symbol;
  div.classList.toggle("changed", val !== initVal);
}

function renderAllCells(cand) {
  for (let r = 0; r < GEOM.hiddenRows + (GEOM.boardRows - GEOM.hiddenRows); r++) {
    for (let c = 0; c < GEOM.boardCols; c++) renderCell(cand, r, c);
  }
}

function onCellClick(key, row, col, direction) {
  const cand = CANDIDATES.find(c => c.key === key);
  const grid = STATE[key].grid;
  let idx = CYCLE.indexOf(grid[row][col]);
  idx = (idx + direction + CYCLE.length) % CYCLE.length;
  grid[row][col] = CYCLE[idx];
  renderCell(cand, row, col);
  saveState();
}

function updateBadge() {
  const cand = CANDIDATES[CURRENT];
  const status = STATE[cand.key].status;
  const labels = {
    [STATUS_OK]: "\\u8a8d\\u8b58\\u901a\\u308a\\u3067OK", [STATUS_FIXED]: "\\u4fee\\u6b63\\u5b8c\\u4e86",
    [STATUS_NOT_A_BOARD]: "\\u975e\\u5bfe\\u5c40\\u753b\\u9762",
  };
  const badge = document.getElementById("status-badge");
  badge.textContent = status ? labels[status] : "\\u672a\\u51e6\\u7406";
  badge.className = "status-badge" + (status ? " st-" + status : "");
}

function updateProgress() {
  const done = CANDIDATES.filter(c => STATE[c.key].status).length;
  document.getElementById("progress").textContent = `\\u5b8c\\u4e86 ${done}/${CANDIDATES.length}`;
}

function renderBoard(index) {
  CURRENT = Math.max(0, Math.min(index, CANDIDATES.length - 1));
  const cand = CANDIDATES[CURRENT];
  document.getElementById("main").innerHTML = boardHtml(cand, CURRENT);
  buildGridCells(document.getElementById("hidden-panel"), cand, 0, GEOM.hiddenRows, true);
  buildGridCells(document.getElementById("grid-overlay"), cand, GEOM.hiddenRows,
    GEOM.boardRows - GEOM.hiddenRows, false);
  renderAllCells(cand);
  updateBadge();
  updateProgress();
  document.getElementById("btn-ok").addEventListener("click", () => setStatus(STATUS_OK));
  document.getElementById("btn-fixed").addEventListener("click", () => setStatus(STATUS_FIXED));
  document.getElementById("btn-skip").addEventListener("click", () => setStatus(STATUS_NOT_A_BOARD));
  saveState();
}

function setStatus(status) {
  const cand = CANDIDATES[CURRENT];
  STATE[cand.key].status = status;
  saveState();
  updateBadge();
  updateProgress();
  renderOverviewGrid();
  if (CURRENT < CANDIDATES.length - 1) renderBoard(CURRENT + 1);
}

function renderOverviewGrid() {
  const grid = document.getElementById("ov-grid");
  grid.innerHTML = "";
  CANDIDATES.forEach((c, i) => {
    const status = STATE[c.key].status;
    const cell = document.createElement("div");
    cell.className = "ov-cell" + (status ? " st-" + status : "");
    cell.textContent = String(i + 1);
    cell.title = c.key;
    cell.addEventListener("click", () => {
      renderBoard(i);
      document.getElementById("overview-panel").classList.remove("show");
    });
    grid.appendChild(cell);
  });
}

function countChanged(cand) {
  const grid = STATE[cand.key].grid;
  let n = 0;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) if (grid[r][c] !== cand.initGrid[r][c]) n++;
  }
  return n;
}

function wrongCellsList(cand) {
  const grid = STATE[cand.key].grid;
  const out = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) {
      if (grid[r][c] !== cand.initGrid[r][c]) out.push(`r${r}c${c}=${grid[r][c]}`);
    }
  }
  return out;
}

function wrongCellsField(cand) {
  const status = STATE[cand.key].status;
  if (status === STATUS_NOT_A_BOARD) return "NOT_A_BOARD";
  const changed = wrongCellsList(cand);
  return changed.length ? changed.join(",") : "ok";
}

function buildResultJson() {
  return JSON.stringify(CANDIDATES.map(c => ({
    sheet_id: c.key, video_id: c.videoId, side: c.side, frame_idx_aux: c.frameIdxAux,
    t_sec: c.tSec, status: STATE[c.key].status,
    corrected_grid: STATE[c.key].grid, wrong_cells: wrongCellsList(c),
    changed_count: countChanged(c),
  })), null, 2);
}

function buildResultTsv() {
  const lines = ["sheet\\tvideo\\tside\\tframe_idx_aux\\twrong_cells"];
  for (const c of CANDIDATES) {
    lines.push([c.key, c.videoId, c.side, c.frameIdxAux, wrongCellsField(c)].join("\\t"));
  }
  return lines.join("\\n");
}

function downloadBlob(text, filename, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function wireGlobalControls() {
  document.getElementById("prev-btn").addEventListener("click", () => renderBoard(CURRENT - 1));
  document.getElementById("next-btn").addEventListener("click", () => renderBoard(CURRENT + 1));
  document.getElementById("overview-btn").addEventListener("click", () => {
    renderOverviewGrid();
    document.getElementById("overview-panel").classList.add("show");
  });
  document.getElementById("ov-close").addEventListener("click", () =>
    document.getElementById("overview-panel").classList.remove("show"));
  document.getElementById("show-json-btn").addEventListener("click", () =>
    document.getElementById("export-area").value = buildResultJson());
  document.getElementById("show-tsv-btn").addEventListener("click", () =>
    document.getElementById("export-area").value = buildResultTsv());
  document.getElementById("dl-json-btn").addEventListener("click", () =>
    downloadBlob(buildResultJson(), "yardstick_v2_review_result.json", "application/json"));
  document.getElementById("dl-tsv-btn").addEventListener("click", () =>
    downloadBlob(buildResultTsv(), "labels.tsv", "text/tab-separated-values"));
  document.getElementById("copy-btn").addEventListener("click", async () => {
    const area = document.getElementById("export-area");
    area.select();
    try {
      await navigator.clipboard.writeText(area.value);
    } catch (e) {
      document.execCommand("copy");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") renderBoard(CURRENT - 1);
    if (e.key === "ArrowRight") renderBoard(CURRENT + 1);
  });
}

function init() {
  loadState();
  wireGlobalControls();
  renderBoard(CURRENT);
}

// テスト用フック: 自動回帰テスト(jsdom)が参照できるよう window に公開する。
window.CANDIDATES = CANDIDATES;
window.STATE = STATE;
window.buildResultJson = buildResultJson;
window.buildResultTsv = buildResultTsv;
window.renderBoard = renderBoard;
window.setStatus = setStatus;

init();
"""


if __name__ == "__main__":
    main()
