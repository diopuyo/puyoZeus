"""一般分布ラベルセット (2026-08-17) のHTML入力ツール生成器。

labels.tsv の手書き記入の代わりに、ブラウザ上でセルをクリックして修正を
記入できる自己完結HTML (`label_editor.html`) を生成する (user要望 2026-08-17)。

- 入力: `data/verify/board_labels_general_2026-08-17/labels.tsv` (行順) +
  `anchors/*.npz` の `candidate_grid` (13x6、行0=隠し段は表示しない)
- 出力: 同ディレクトリの `label_editor.html`。シートPNGは相対パス参照
  (file:// で開く前提)。進捗は localStorage に自動保存。
- エクスポート: `labels_filled.tsv` (sheet\twrong_cells 形式) をダウンロード。
  wrong_cells は既存規約 (`r3c2=1,r5c0=0` / `ok`) と同一。

使い方:
    python -m scripts._gen_label_editor_html_2026-08-17
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LABEL_DIR: Path = _ROOT / "data" / "verify" / "board_labels_general_2026-08-17"
VISIBLE_ROW_MIN: int = 1  # r1=最上段 (可視)。行0 (隠し段) は測定対象外
VISIBLE_ROW_MAX: int = 12  # r12=最下段
N_COLS: int = 6
SHEET_PNG_WIDTH: int = 1128  # 実測 (35枚均一)
SHEET_CROP_WIDTH: int = 768  # 左側の実画面クロップ幅


def _load_records() -> list[dict]:
    """labels.tsv の行順に、シートIDと認識グリッド (r1..r12) を読み込む。"""
    records: list[dict] = []
    tsv = LABEL_DIR / "labels.tsv"
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        sheet = cols[0]
        if not sheet.endswith(".png"):
            continue
        anchor_npz = LABEL_DIR / "anchors" / (sheet[:-4] + ".npz")
        grid = np.load(anchor_npz, allow_pickle=True)["candidate_grid"]
        rows = [
            [int(v) for v in grid[r]]
            for r in range(VISIBLE_ROW_MIN, VISIBLE_ROW_MAX + 1)
        ]
        records.append(
            {
                "sheet": sheet,
                "video": cols[3],
                "side": cols[4],
                "tertile": cols[7],
                "grid": rows,
            }
        )
    return records


def _build_html(records: list[dict]) -> str:
    """自己完結のラベル入力HTMLを組み立てる。"""
    data_json = json.dumps(records, ensure_ascii=False)
    img_scale_pct = SHEET_PNG_WIDTH / SHEET_CROP_WIDTH * 100.0
    # JSはテンプレ文字列の衝突を避けるため .format せず連結で埋め込む
    head = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>一般分布ラベル入力 (35盤面)</title>
<style>
:root { --bg:#14161a; --panel:#1e2128; --line:#3a3f4a; --txt:#e8e8e8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--txt);
       font-family:"Segoe UI","Yu Gothic UI",sans-serif; }
header { display:flex; align-items:center; gap:16px; padding:10px 16px;
         background:var(--panel); border-bottom:1px solid var(--line);
         position:sticky; top:0; z-index:10; flex-wrap:wrap; }
h1 { font-size:16px; margin:0; }
#progress { font-size:14px; color:#9fd49f; }
main { display:flex; gap:18px; padding:14px 16px; align-items:flex-start;
       flex-wrap:wrap; }
.imgbox { width:min(46vw,560px); aspect-ratio:768/1440; overflow:hidden;
          border:1px solid var(--line); border-radius:6px; }
.imgbox img { width:IMG_SCALE_PCT%; display:block; image-rendering:auto; }
.side { flex:1; min-width:420px; }
table.grid { border-collapse:collapse; }
table.grid td { width:52px; height:44px; border:1px solid #555;
                text-align:center; cursor:pointer; font-weight:bold;
                font-size:13px; user-select:none; position:relative; }
table.grid td.changed { outline:3px solid #ff9800; outline-offset:-3px; }
table.grid th { font-size:11px; color:#aaa; font-weight:normal; padding:2px 6px; }
.palette { display:flex; gap:6px; margin:10px 0; flex-wrap:wrap; }
.palette button { padding:8px 12px; border-radius:6px; border:2px solid transparent;
                  cursor:pointer; font-weight:bold; font-size:13px; }
.palette button.sel { border-color:#fff; box-shadow:0 0 6px #fff8; }
.controls { display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }
.controls button { padding:10px 14px; border-radius:6px; border:1px solid var(--line);
                   background:#2a2e37; color:var(--txt); cursor:pointer; font-size:14px; }
.controls button.primary { background:#2e7d32; }
.controls button.warn { background:#8d6e00; }
#wrongPreview { font-family:Consolas,monospace; font-size:13px; color:#ffcc80;
                min-height:1.4em; word-break:break-all; }
#sheetInfo { font-size:13px; color:#bbb; margin-bottom:6px; }
#jump { max-width:100%; background:#2a2e37; color:var(--txt);
        border:1px solid var(--line); padding:6px; border-radius:6px; }
.status-done { color:#9fd49f; } .status-fix { color:#ffb74d; } .status-todo { color:#e57373; }
#exportArea { width:100%; height:120px; background:#111; color:#ccc;
              font-family:Consolas,monospace; font-size:12px; margin-top:8px;
              border:1px solid var(--line); display:none; }
kbd { background:#333; border-radius:3px; padding:1px 5px; font-size:11px; }
.hint { font-size:12px; color:#999; margin-top:6px; line-height:1.7; }
</style></head><body>
"""
    head = head.replace("IMG_SCALE_PCT", f"{img_scale_pct:.4f}")
    body = """
<header>
  <h1>一般分布ラベル入力</h1>
  <span id="progress"></span>
  <select id="jump"></select>
  <button id="exportBtn" style="padding:8px 14px;border-radius:6px;cursor:pointer;">
    labels_filled.tsv をエクスポート</button>
</header>
<main>
  <div class="imgbox"><img id="sheetImg" alt="実画面クロップ"></div>
  <div class="side">
    <div id="sheetInfo"></div>
    <div class="palette" id="palette"></div>
    <table class="grid" id="gridTable"></table>
    <div style="margin-top:8px">誤り記入: <span id="wrongPreview"></span></div>
    <div class="controls">
      <button class="primary" id="okBtn">誤りなし=OK → 次へ</button>
      <button class="warn" id="fixBtn">修正を確定 → 次へ</button>
      <button id="resetBtn">この盤面をリセット</button>
      <button id="prevBtn">← 前へ</button>
      <button id="nextBtn">次へ →</button>
    </div>
    <div class="hint">
      使い方: 下のパレットで正しい色を選び、誤っているセルをクリックして塗り替える。
      右クリックでそのセルを元に戻す。<br>
      キー: <kbd>0</kbd>空 <kbd>1</kbd>赤 <kbd>2</kbd>青 <kbd>3</kbd>緑 <kbd>4</kbd>黄
      <kbd>5</kbd>紫 <kbd>9</kbd>おじゃま / <kbd>Enter</kbd>=OKで次へ /
      <kbd>←</kbd><kbd>→</kbd>=移動。<br>
      <b>判断根拠は必ず左の実画面</b> (右のグリッドは認識の下書き)。進捗は自動保存されます。
    </div>
    <textarea id="exportArea" readonly></textarea>
  </div>
</main>
<script>
const DATA = __DATA_JSON__;
const COLORS = {0:"#23262d",1:"#d33",2:"#39f",3:"#3b3",4:"#dc3",5:"#a4d",9:"#999"};
const NAMES  = {0:"空",1:"赤",2:"青",3:"緑",4:"黄",5:"紫",9:"おじゃま"};
const CODES  = [0,1,2,3,4,5,9];
const LS_KEY = "board_labels_general_2026-08-17";
let cur = 0, selColor = 9;
let state = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
// state[sheet] = {status:"ok"|"fix", cells:{"r-c":val}}

function save(){ localStorage.setItem(LS_KEY, JSON.stringify(state)); }
function cellsOf(i){ const s = state[DATA[i].sheet]; return (s && s.cells) || {}; }
function statusOf(i){ const s = state[DATA[i].sheet]; return s ? s.status : null; }

function wrongStr(i){
  const c = cellsOf(i), parts = [];
  Object.keys(c).sort().forEach(k => {
    const [r, col] = k.split("-");
    parts.push("r" + r + "c" + col + "=" + c[k]);
  });
  return parts.join(",");
}

function render(){
  const rec = DATA[cur];
  document.getElementById("sheetImg").src = "sheets/" + rec.sheet;
  const st = statusOf(cur);
  const stTxt = st === "ok" ? "確認済み(OK)" : st === "fix" ? "確認済み(修正あり)" : "未確認";
  document.getElementById("sheetInfo").innerHTML =
    "<b>" + (cur+1) + "/" + DATA.length + "</b> " + rec.sheet +
    " | " + rec.video + " " + rec.side + " (" + rec.tertile + ") | 状態: <b>" + stTxt + "</b>";
  const tbl = document.getElementById("gridTable");
  tbl.innerHTML = "";
  const hd = document.createElement("tr");
  hd.innerHTML = "<th></th>" + [0,1,2,3,4,5].map(c => "<th>c"+c+"</th>").join("");
  tbl.appendChild(hd);
  const cells = cellsOf(cur);
  rec.grid.forEach((row, ri) => {
    const r = ri + 1;
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = "r" + r; tr.appendChild(th);
    row.forEach((orig, c) => {
      const key = r + "-" + c;
      const val = (key in cells) ? cells[key] : orig;
      const td = document.createElement("td");
      td.style.background = COLORS[val];
      td.style.color = (val === 0 || val === 9) ? "#ddd" : "#fff";
      td.textContent = val === 0 ? "" : val;
      if (key in cells) td.classList.add("changed");
      td.onclick = () => {
        if (selColor === orig && (key in cells)) { delete cells[key]; }
        else if (selColor === orig) { return; }
        else { cells[key] = selColor; }
        state[DATA[cur].sheet] = state[DATA[cur].sheet] || {};
        state[DATA[cur].sheet].cells = cells;
        save(); render();
      };
      td.oncontextmenu = (e) => {
        e.preventDefault();
        if (key in cells) { delete cells[key]; save(); render(); }
      };
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
  document.getElementById("wrongPreview").textContent = wrongStr(cur) || "(なし)";
  const done = DATA.filter((_, i) => statusOf(i)).length;
  document.getElementById("progress").textContent = "確認済み " + done + "/" + DATA.length;
  const jump = document.getElementById("jump");
  jump.innerHTML = DATA.map((d, i) => {
    const s = statusOf(i);
    const mark = s === "ok" ? "✓ " : s === "fix" ? "✎ " : "・ ";
    return "<option value='" + i + "'" + (i === cur ? " selected" : "") + ">" +
           mark + d.sheet + "</option>";
  }).join("");
}

function renderPalette(){
  const pal = document.getElementById("palette");
  pal.innerHTML = "";
  CODES.forEach(code => {
    const b = document.createElement("button");
    b.textContent = code + " " + NAMES[code];
    b.style.background = COLORS[code];
    b.style.color = (code === 0 || code === 9) ? "#ddd" : "#fff";
    if (code === selColor) b.classList.add("sel");
    b.onclick = () => { selColor = code; renderPalette(); };
    pal.appendChild(b);
  });
}

function confirmSheet(kind){
  const cells = cellsOf(cur);
  if (kind === "ok" && Object.keys(cells).length > 0) {
    alert("修正セルがあります。「修正を確定」を使うか、リセットしてください。");
    return;
  }
  if (kind === "fix" && Object.keys(cells).length === 0) {
    alert("修正セルがありません。誤りが無ければ「誤りなし=OK」を使ってください。");
    return;
  }
  state[DATA[cur].sheet] = state[DATA[cur].sheet] || {};
  state[DATA[cur].sheet].status = kind;
  state[DATA[cur].sheet].cells = cells;
  save();
  if (cur < DATA.length - 1) cur++;
  render();
}

document.getElementById("okBtn").onclick = () => confirmSheet("ok");
document.getElementById("fixBtn").onclick = () => confirmSheet("fix");
document.getElementById("resetBtn").onclick = () => {
  if (state[DATA[cur].sheet]) { delete state[DATA[cur].sheet]; save(); render(); }
};
document.getElementById("prevBtn").onclick = () => { if (cur > 0) { cur--; render(); } };
document.getElementById("nextBtn").onclick = () => { if (cur < DATA.length - 1) { cur++; render(); } };
document.getElementById("jump").onchange = (e) => { cur = +e.target.value; render(); };

document.getElementById("exportBtn").onclick = () => {
  const todo = DATA.filter((_, i) => !statusOf(i)).length;
  if (todo > 0 && !confirm("未確認が " + todo + " 枚あります。エクスポートしますか?")) return;
  let out = "sheet\\twrong_cells\\n";
  DATA.forEach((d, i) => {
    const s = statusOf(i);
    const w = wrongStr(i);
    out += d.sheet + "\\t" + (s === "ok" ? "ok" : (w || "(未確認)")) + "\\n";
  });
  const area = document.getElementById("exportArea");
  area.style.display = "block"; area.value = out;
  const blob = new Blob([out], {type: "text/tab-separated-values"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_filled.tsv";
  a.click();
};

document.addEventListener("keydown", (e) => {
  if ("0123459".includes(e.key)) { selColor = +e.key; renderPalette(); }
  else if (e.key === "Enter") confirmSheet(Object.keys(cellsOf(cur)).length ? "fix" : "ok");
  else if (e.key === "ArrowLeft") { if (cur > 0) { cur--; render(); } }
  else if (e.key === "ArrowRight") { if (cur < DATA.length - 1) { cur++; render(); } }
});

renderPalette(); render();
</script></body></html>
"""
    return head + body.replace("__DATA_JSON__", data_json)


def main() -> None:
    records = _load_records()
    if not records:
        raise SystemExit("labels.tsv からレコードを読めませんでした")
    out = LABEL_DIR / "label_editor.html"
    out.write_text(_build_html(records), encoding="utf-8")
    print(f"生成: {out} ({len(records)}盤面)")


if __name__ == "__main__":
    main()
