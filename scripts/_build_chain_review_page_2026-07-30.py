"""連鎖数レビュー用の単一HTMLページを生成する (Artifact 公開用)。

user から「画像が見づらい、1枚でレビューに必要な情報が無いと見れない」との指摘を受け、
9事例のsummary.pngを縦連結した contact_sheet.png (文字が読めない) を作り直すもの。

設計方針:
  - 数値・メタ情報は画像に焼き込まず HTML テキストとして出す (拡大しても読める)
  - 画像は「N れんさ!」ポップアップの切り抜きだけ (判断に必要な最小限)
  - 画像は data URI で埋め込み自己完結させる (Artifact の CSP は外部取得を禁止)
  - スマホ1カラム、事例ごとに同意/違うをタップ記録し最後にコピー可能なテキストを生成
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

# --- パス定数 ---
CROP_DIR = Path("data/verify/chain_review_crops_2026-07-30")
MANIFEST_PATH = CROP_DIR / "manifest.json"
OUT_HTML = Path("data/verify/chain_review_crops_2026-07-30/review.html")

# --- 表示定数 ---
MAX_CHAIN_PICKER = 19  # 「違う」を選んだときに出す連鎖数ボタンの上限
SECONDARY_SUFFIXES = ("_prev.jpg", "_next.jpg")  # 補助画像の識別子


def encode_data_uri(path: Path) -> str:
    """画像を data URI 文字列に変換する。"""
    raw = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def signed(delta: int) -> str:
    """判定値との差を符号付きで返す (0 は一致表記)。"""
    if delta == 0:
        return "一致"
    return f"{delta:+d}"


def build_value_cell(label: str, value: int, judged: int | None) -> str:
    """simulate / 画面OCR の1セルを組む (判定値との差を併記)。"""
    if judged is None:
        diff_html = '<span class="diff diff-na">—</span>'
    else:
        delta = value - judged
        cls = "diff-zero" if delta == 0 else "diff-off"
        diff_html = f'<span class="diff {cls}">{signed(delta)}</span>'
    return (
        f'<div class="vcell"><span class="vlabel">{label}</span>'
        f'<span class="vnum">{value}</span>{diff_html}</div>'
    )


def build_secondary(entry: dict) -> str:
    """補助画像 (1つ前/次のステップ) のブロックを組む。無ければ空文字。"""
    extras = [f for f in entry["files"] if f.endswith(SECONDARY_SUFFIXES)]
    if not extras:
        return ""
    items = []
    for fname in extras:
        uri = encode_data_uri(CROP_DIR / fname)
        kind = "1つ前のステップ" if fname.endswith("_prev.jpg") else "次の連鎖"
        items.append(
            f'<figure class="sub"><img src="{uri}" alt="{kind}">'
            f"<figcaption>{kind}</figcaption></figure>"
        )
    return f'<div class="subrow">{"".join(items)}</div>'


def build_card(entry: dict) -> str:
    """1事例分のカードHTMLを組む。"""
    label = entry["event_label"]
    judged_raw = entry["my_judged_true_value"]
    judged = judged_raw if isinstance(judged_raw, int) else None
    prev = entry["prev_recorded_true_value_if_different"]

    main_file = next(f for f in entry["files"] if f.endswith("_max.jpg"))
    main_uri = encode_data_uri(CROP_DIR / main_file)

    # 訂正マーカー: 前回判定と違う場合のみ
    correction = ""
    if prev is not None and isinstance(prev, int):
        correction = f'<span class="tag tag-fix">訂正 {prev} → {judged}</span>'

    if judged is None:
        verdict = (
            '<div class="verdict verdict-na"><span class="vlabel">私の判定</span>'
            '<span class="vna">判定不能</span></div>'
        )
    else:
        verdict = (
            '<div class="verdict"><span class="vlabel">私の判定</span>'
            f'<span class="vjudged">{judged}</span></div>'
        )

    meta = (
        f'{entry["video_id"]} / {entry["side"]} / {entry["game_idx"]}試合目 '
        f'/ {entry["t_max_sec"]:.1f}秒 / 得点差 {entry["delta_score"]:,}'
    )

    picker = "".join(
        f'<button type="button" class="pick" data-case="{label}" data-val="{n}">{n}</button>'
        for n in range(1, MAX_CHAIN_PICKER + 1)
    )

    return f"""<article class="card" id="case-{label}">
  <header class="chead">
    <span class="chip">{label}</span>
    <div class="cmeta">{meta}</div>
    {correction}
  </header>
  <figure class="shot"><img src="{main_uri}" alt="{label} の連鎖数ポップアップ"></figure>
  <p class="note">{entry["readable_note"]}</p>
  {build_secondary(entry)}
  <div class="values">
    {build_value_cell("simulate", entry["simulate_chain_count"], judged)}
    {build_value_cell("画面OCR", entry["screen_chain_count"], judged)}
    {verdict}
  </div>
  <div class="actions" data-case="{label}">
    <button type="button" class="act ok" data-case="{label}" aria-pressed="false">
      この判定で合っている
    </button>
    <button type="button" class="act ng" data-case="{label}" aria-pressed="false">
      違う
    </button>
  </div>
  <div class="pickwrap" data-case="{label}" hidden>
    <span class="picklabel">本当の連鎖数</span>
    <div class="picks">{picker}<button type="button" class="pick pick-na"
      data-case="{label}" data-val="?">読めない</button></div>
  </div>
</article>"""


def build_html(entries: list[dict]) -> str:
    """ページ全体を組む。"""
    cards = "\n".join(build_card(e) for e in entries)
    labels = json.dumps([e["event_label"] for e in entries], ensure_ascii=False)
    return TEMPLATE.replace("{{CARDS}}", cards).replace("{{LABELS}}", labels)


TEMPLATE = r"""<title>連鎖数の読み取り確認</title>
<style>
:root{
  --ground:#eceef2; --surface:#ffffff; --mat:#0d1014;
  --ink:#161a20; --muted:#5f6a7d; --line:#d3d8e0;
  --gold:#b57500; --gold-bg:#fff6e2;
  --ok:#1c7d5e; --ng:#c04a34;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#101319; --surface:#191e26; --mat:#0a0c10;
    --ink:#e8ebf0; --muted:#8892a4; --line:#2b323d;
    --gold:#ffbe3d; --gold-bg:#2a2113;
    --ok:#3fb98f; --ng:#e8674f;
  }
}
:root[data-theme="dark"]{
  --ground:#101319; --surface:#191e26; --mat:#0a0c10;
  --ink:#e8ebf0; --muted:#8892a4; --line:#2b323d;
  --gold:#ffbe3d; --gold-bg:#2a2113;
  --ok:#3fb98f; --ng:#e8674f;
}
:root[data-theme="light"]{
  --ground:#eceef2; --surface:#ffffff; --mat:#0d1014;
  --ink:#161a20; --muted:#5f6a7d; --line:#d3d8e0;
  --gold:#b57500; --gold-bg:#fff6e2;
  --ok:#1c7d5e; --ng:#c04a34;
}
*{box-sizing:border-box}
body{
  margin:0; padding:0 16px 140px;
  background:var(--ground); color:var(--ink);
  font-family:"Hiragino Kaku Gothic ProN","Yu Gothic UI","Yu Gothic",
    "Noto Sans JP","Meiryo",system-ui,sans-serif;
  line-height:1.7; -webkit-text-size-adjust:100%;
}
.wrap{max-width:560px; margin:0 auto}
header.top{padding:28px 0 18px; border-bottom:1px solid var(--line)}
h1{margin:0 0 10px; font-size:1.5rem; line-height:1.35; text-wrap:balance; letter-spacing:.01em}
.lede{margin:0; color:var(--muted); font-size:.95rem}
.lede strong{color:var(--ink); font-weight:600}
.stack{display:flex; flex-direction:column; gap:22px; padding-top:22px}
.card{
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:16px; display:flex; flex-direction:column; gap:12px;
}
.chead{display:flex; align-items:center; gap:10px; flex-wrap:wrap}
.chip{
  font-size:1rem; font-weight:700; letter-spacing:.04em;
  background:var(--ink); color:var(--surface);
  min-width:2.2em; text-align:center; padding:2px 8px; border-radius:5px;
}
.cmeta{font-size:.82rem; color:var(--muted); font-variant-numeric:tabular-nums; flex:1 1 auto}
.tag{font-size:.72rem; padding:2px 8px; border-radius:99px; white-space:nowrap}
.tag-fix{background:var(--gold-bg); color:var(--gold); border:1px solid var(--gold)}
figure{margin:0}
.shot{background:var(--mat); border-radius:8px; padding:8px; overflow-x:auto}
.shot img{display:block; max-width:100%; height:auto; margin:0 auto; image-rendering:auto}
.note{margin:0; font-size:.82rem; color:var(--muted)}
.subrow{display:flex; gap:10px; flex-wrap:wrap}
.sub{flex:1 1 45%; min-width:130px}
.sub img{display:block; max-width:100%; height:auto; background:var(--mat);
  border-radius:6px; padding:5px}
.sub figcaption{font-size:.72rem; color:var(--muted); padding-top:4px}
.values{
  display:grid; grid-template-columns:1fr 1fr 1.2fr; gap:8px;
  border-top:1px solid var(--line); padding-top:12px;
}
.vcell,.verdict{display:flex; flex-direction:column; align-items:center; gap:1px}
.vlabel{font-size:.7rem; color:var(--muted); letter-spacing:.06em}
.vnum{font-size:1.7rem; font-weight:600; color:var(--muted); font-variant-numeric:tabular-nums}
.vjudged{font-size:2.3rem; font-weight:700; color:var(--gold);
  font-variant-numeric:tabular-nums; line-height:1.1}
.vna{font-size:1.05rem; font-weight:700; color:var(--gold); text-align:center}
.verdict{background:var(--gold-bg); border-radius:7px; padding:4px 2px}
.diff{font-size:.75rem; font-variant-numeric:tabular-nums}
.diff-off{color:var(--ng)} .diff-zero{color:var(--ok)} .diff-na{color:var(--muted)}
.actions{display:flex; gap:8px}
.act{
  flex:1 1 0; font:inherit; font-size:.9rem; padding:11px 6px; cursor:pointer;
  background:var(--surface); color:var(--ink);
  border:1px solid var(--line); border-radius:8px;
}
.act:hover{border-color:var(--muted)}
.act:focus-visible{outline:2px solid var(--gold); outline-offset:2px}
.act[aria-pressed="true"].ok{background:var(--ok); border-color:var(--ok); color:#fff}
.act[aria-pressed="true"].ng{background:var(--ng); border-color:var(--ng); color:#fff}
.pickwrap{display:flex; flex-direction:column; gap:6px}
.picklabel{font-size:.75rem; color:var(--muted)}
.picks{display:flex; flex-wrap:wrap; gap:5px}
.pick{
  font:inherit; font-size:.85rem; font-variant-numeric:tabular-nums;
  min-width:2.4em; padding:6px 7px; cursor:pointer;
  background:var(--surface); color:var(--ink);
  border:1px solid var(--line); border-radius:6px;
}
.pick:focus-visible{outline:2px solid var(--gold); outline-offset:2px}
.pick[aria-pressed="true"]{background:var(--ng); border-color:var(--ng); color:#fff}
.pick-na{min-width:auto}
.bar{
  position:fixed; left:0; right:0; bottom:0; z-index:5;
  background:var(--surface); border-top:1px solid var(--line);
  padding:10px 16px 12px;
}
.barin{max-width:560px; margin:0 auto; display:flex; flex-direction:column; gap:7px}
.count{font-size:.8rem; color:var(--muted); font-variant-numeric:tabular-nums}
.out{
  font:inherit; font-size:.82rem; width:100%; min-height:2.6em; resize:vertical;
  background:var(--ground); color:var(--ink);
  border:1px solid var(--line); border-radius:6px; padding:7px 8px;
}
.copy{
  font:inherit; font-size:.88rem; font-weight:600; padding:10px; cursor:pointer;
  background:var(--ink); color:var(--surface); border:none; border-radius:7px;
}
.copy:focus-visible{outline:2px solid var(--gold); outline-offset:2px}
footer.foot{padding:26px 0 10px; color:var(--muted); font-size:.8rem;
  border-top:1px solid var(--line); margin-top:24px}
footer.foot p{margin:0 0 8px}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
</style>

<div class="wrap">
<header class="top">
  <h1>連鎖数の読み取り確認</h1>
  <p class="lede">画面の「N れんさ!」を実フレームで追って読んだ値が正しいか見てほしい。
  各事例、<strong>金色の数字が私の読み</strong>で、左の2つは
  <strong>simulate</strong>(盤面から計算)と<strong>画面OCR</strong>(自動読み取り)。
  小さい符号付き数字は私の読みからのズレ。</p>
</header>

<div class="stack">
{{CARDS}}
</div>

<footer class="foot">
  <p>事例名が A・B・E・G・H・I・C2・D2・F2 と飛んでいるのは、当初のC・D・Fが
  試合外画面(ロゴや実況カットイン)しか写っておらず判定できず、別事例に差し替えたため。</p>
  <p>D2 では自動OCRが「4 れんさ!」ではなく画面左端の無関係な演出アイコンを4と誤検出していた
  (信頼度0.673で通過)。ここは目視で正しい位置を指定している。</p>
</footer>
</div>

<div class="bar"><div class="barin">
  <span class="count" id="count"></span>
  <textarea class="out" id="out" readonly aria-label="結果テキスト"></textarea>
  <button type="button" class="copy" id="copy">結果をコピー</button>
</div></div>

<script>
(function(){
  var LABELS = {{LABELS}};
  var state = {};

  function render(){
    var done = 0, parts = [];
    LABELS.forEach(function(k){
      var s = state[k];
      if(!s) return;
      done++;
      parts.push(s.ok ? (k + ":OK") : (k + ":違う→" + s.val));
    });
    document.getElementById("count").textContent =
      done + " / " + LABELS.length + " 件 回答済み";
    document.getElementById("out").value = parts.join("  ");
  }

  function clearPicks(caseId){
    document.querySelectorAll('.pick[data-case="' + caseId + '"]').forEach(function(b){
      b.setAttribute("aria-pressed", "false");
    });
  }

  document.querySelectorAll(".act").forEach(function(btn){
    btn.addEventListener("click", function(){
      var id = btn.dataset.case;
      var isOk = btn.classList.contains("ok");
      document.querySelectorAll('.act[data-case="' + id + '"]').forEach(function(b){
        b.setAttribute("aria-pressed", String(b === btn));
      });
      var wrap = document.querySelector('.pickwrap[data-case="' + id + '"]');
      if(isOk){
        state[id] = {ok:true};
        wrap.hidden = true;
        clearPicks(id);
      } else {
        wrap.hidden = false;
        if(!state[id] || state[id].ok) state[id] = null;
      }
      render();
    });
  });

  document.querySelectorAll(".pick").forEach(function(btn){
    btn.addEventListener("click", function(){
      var id = btn.dataset.case;
      clearPicks(id);
      btn.setAttribute("aria-pressed", "true");
      state[id] = {ok:false, val:btn.dataset.val};
      render();
    });
  });

  document.getElementById("copy").addEventListener("click", function(){
    var out = document.getElementById("out");
    var btn = this;
    var label = btn.textContent;
    function done(msg){
      btn.textContent = msg;
      setTimeout(function(){ btn.textContent = label; }, 1600);
    }
    if(!out.value){ done("まだ回答がありません"); return; }
    if(navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(out.value).then(function(){
        done("コピーしました");
      }, function(){
        out.select(); done("選択したので長押しでコピー");
      });
    } else {
      out.select(); done("選択したので長押しでコピー");
    }
  });

  render();
})();
</script>
"""


def main() -> None:
    """マニフェストを読み HTML を書き出す。"""
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    html = build_html(entries)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"wrote {OUT_HTML} ({size_kb:.0f} KB, {len(entries)} cases)")


if __name__ == "__main__":
    main()
