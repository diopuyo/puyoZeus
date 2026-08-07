"""スマホ版レビューHTML組み立て (2026-08-05、使い捨て)。images.json を差し込む。"""
import json
from pathlib import Path

IMAGES = json.loads(Path("data/verify/mobile_review_2026-08-05/images.json").read_text(encoding="utf-8"))
OUT = Path("data/verify/mobile_review_2026-08-05/review_mobile.html")

HTML = """<title>認識レビュー 8/5 — バーストガード採否</title>
<style>
:root {
  --bg: #14161d; --card: #1d2029; --line: #2b2f3d;
  --tx: #e9ebf2; --sub: #9aa1b5;
  --acc: #3ecf8e; --acc-tx: #0c1512;
  --warn: #e6a23c; --bad: #e06c5f;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #f4f5f8; --card: #ffffff; --line: #dde0e8; --tx: #22252e; --sub: #5d6475; --acc: #159f68; --acc-tx: #ffffff; }
}
:root[data-theme="dark"] { --bg: #14161d; --card: #1d2029; --line: #2b2f3d; --tx: #e9ebf2; --sub: #9aa1b5; --acc: #3ecf8e; --acc-tx: #0c1512; }
:root[data-theme="light"] { --bg: #f4f5f8; --card: #ffffff; --line: #dde0e8; --tx: #22252e; --sub: #5d6475; --acc: #159f68; --acc-tx: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--tx);
  font-family: "Hiragino Sans", "Noto Sans JP", "Yu Gothic UI", system-ui, sans-serif;
  font-size: 16px; line-height: 1.75; }
#bar { position: sticky; top: 0; z-index: 5; background: var(--bg); padding: 12px 16px 8px; }
#barin { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
#barfill { height: 100%; width: 0%; background: var(--acc); transition: width .25s; }
#step { font-size: 12px; color: var(--sub); margin-top: 6px; letter-spacing: .06em; }
#wrap { padding: 4px 16px 130px; max-width: 640px; margin: 0 auto; }
h2 { font-size: 19px; line-height: 1.5; margin: 10px 0 4px; text-wrap: balance; }
.hint { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; font-size: 14.5px; color: var(--tx); margin: 10px 0; }
.hint b { color: var(--acc); }
.hint .num { font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0; }
td, th { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; font-variant-numeric: tabular-nums; }
img.shot { width: 100%; border-radius: 8px; border: 1px solid var(--line); margin: 8px 0; }
.imgnote { font-size: 12px; color: var(--sub); margin: -4px 0 8px; }
#answers { position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg);
  border-top: 1px solid var(--line); padding: 10px 14px calc(12px + env(safe-area-inset-bottom)); }
#answers .row { display: flex; gap: 8px; max-width: 640px; margin: 0 auto; }
button.opt { flex: 1; font-size: 15px; padding: 14px 6px; border: 1px solid var(--line);
  border-radius: 12px; background: var(--card); color: var(--tx); cursor: pointer; }
button.opt.sel { background: var(--acc); color: var(--acc-tx); border-color: var(--acc); font-weight: 700; }
textarea.memo { width: 100%; min-height: 44px; background: var(--card); color: var(--tx);
  border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; font-size: 14px; margin-top: 8px; }
.nav { display: flex; gap: 8px; margin-top: 8px; max-width: 640px; margin-left: auto; margin-right: auto; }
.nav button { font-size: 14px; padding: 10px 16px; border-radius: 10px; border: 1px solid var(--line);
  background: none; color: var(--sub); cursor: pointer; }
.nav button#nx { margin-left: auto; color: var(--tx); }
#result { width: 100%; min-height: 220px; background: var(--card); color: var(--tx);
  border: 1px solid var(--line); border-radius: 10px; font-family: ui-monospace, monospace; font-size: 12px; padding: 10px; }
.big { font-size: 30px; font-weight: 800; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
.good { color: var(--acc); } .warncol { color: var(--warn); }
@media (prefers-reduced-motion: reduce) { #barfill { transition: none; } }
</style>
<div id="bar"><div id="barin"><div id="barfill"></div></div><div id="step"></div></div>
<div id="wrap"></div>
<div id="answers"></div>
<script>
"use strict";
const IMG = __IMAGES__;
const OPT_APPROVE = [["OK","承認する"],["HOLD","保留 (メモへ理由)"]];
const OPT_CONFIRM = [["OK","納得した"],["Q","疑問あり (メモへ)"]];
const OPT_READ = [["READ","読んだ"]];
const Q = [
 {t:"1/10 バーストガードの採用判断", opts:OPT_APPROVE, hint:`
   <p>昨夜〜今日で作った新しい誤読対策 (バーストガード) を、<b>データ収集・評価の既定構成として採用</b>してよいかの判断です。</p>
   <div class="big good">93 → 33 マス</div>
   <p class="num">人手で正解を付けた55盤面の既知誤読が <b>64.5%減</b> (31動画の全量再認識で実測)。</p>
   <table>
    <tr><td>通常のぷよ設置の反映</td><td><b>無傷</b> (遅延 中央値0コマ / 9割が0コマ)</td></tr>
    <tr><td>演出中の設置</td><td>9割は+2コマ以内。長い演出中のみ最大4.7秒の持ち越しあり</td></tr>
    <tr><td>テスト</td><td>4,242件 全パス / 機能はスイッチ式で従来動作と完全互換</td></tr>
   </table>
   <p>仕組み: 相手のおじゃま送付の光を画面から直接検知して窓を開き、窓の間は「物理的にあり得る変化」だけを盤面に通します (おじゃま落下中に9以外が湧く=誤読、など)。</p>`},
 {t:"2/10 旧方式 (案B) の正式見送り", opts:OPT_APPROVE, hint:`
   <p>昨夜のレビューで保留にした旧方式 (4条件ゲート) の扱いです。<b>93マスに対し改善ゼロ</b>が確定し、新方式が発想を引き継いだ上位互換になったため、正式に見送り (コードは無効のまま保管) を提案します。</p>`},
 {t:"3/10 改善例の確認 — 10マス同時誤読の現場", img:"c18_scene", opts:OPT_CONFIRM, hint:`
   <p>最大の改善例。相手の大連鎖の送付演出で<b>10マスが同時に化けた</b>瞬間です (各行=1マス、左=直前/中央=直後/右=数秒後)。新方式でこの盤面の誤りは <b class="num">10→2マス</b> になりました。中央のコマの光り方を見て、「これが原因」という説明に納得できるか確認してください。</p>`},
 {t:"4/10 改善例の確認 — おじゃま着弾の汚染", img:"c19_scene", opts:OPT_CONFIRM, hint:`
   <p>おじゃま79個級の着弾で下段まで汚染された例。この盤面の誤りは <b class="num">22→0マス</b> (全滅) になりました。</p>`},
 {t:"5/10 改善例の確認 — 中規模の例", img:"c15_scene", opts:OPT_CONFIRM, hint:`
   <p>中規模の改善例 (<b class="num">7→0マス</b>)。ここまでの3例で「直っている実感」に問題がないか確認してください。</p>`},
 {t:"6/10 残る33マスの正体① — 弱い光 (14マス)", img:"c29_scene", opts:[["YES","ラベル作業をやる (30分)"],["LATER","後日やる"],["NO","やらない"]], hint:`
   <p>残った33マスの最大勢力は「<b>光が弱くて検知の基準 (明るさ95.4%) に届かなかった</b>」14マスです (画像は代表例)。</p>
   <p>これを削るには検知基準を下げる較正が必要で、誤検知しない根拠を作るために<b>あなたの追加ラベル作業 (中間の明るさ帯の画像 約30枚、30分想定)</b> が必要です。やりますか?</p>`},
 {t:"7/10 残る33マスの正体② — 方針の分岐", opts:[["MEASURE","先に99.99%再測定 (推奨)"],["STAGE2","先に残りを潰す"]], hint:`
   <p>残りの内訳: 弱い光14 / 原理的に区別不能9 (光の誤読が「あり得る変化」に偽装するケース、根治は将来の光除去方式) / 煙6 / 隠し段の副作用 (対策実装済み・今夜全量検証)。</p>
   <p><b>選択肢A (推奨)</b>: 現状の成果で認識精度99.99%の再測定に進み、足りなければ戻って続きを潰す — 「もう十分か」を先に知る道。<br>
   <b>選択肢B</b>: 弱光較正+煙対策 (1〜2日) を先にやり切ってから測る道。</p>`},
 {t:"8/10 報告 — 今日ボツにした機構", opts:OPT_READ, hint:`
   <p>正直な報告です。「窓を閉じた直後の残光」対策 (クールダウン+連鎖延長) は、実装後の検証で<b>「長い連鎖の間、凍結が11秒連結し、正規の設置まで巻き込んで雪だるま式に盤面が古くなる」</b>欠陥を検出し、本日ボツにしました (3回の再走行はすべて中間検知で止めています)。残光対策は設計をやり直してStage2に持ち越します。</p>`},
 {t:"9/10 報告 — 副作用の実測", opts:OPT_READ, hint:`
   <p>採用判断の材料として副作用も正直に: ①通常の設置反映は無傷 (実測ゼロ差) ②隠し段 (13段目) に少数の新規誤りが出る副作用があり、対策 (1.5b) を実装済み・今夜全量検証中 ③長い演出中は盤面の更新が最大数秒持ち越されます (「演出が終わってから書き込む」方式の宿命で、リアルタイム表示では「演出中」表示で補う設計を予定)。</p>`},
 {t:"10/10 自由メモ", opts:[["DONE","完了"]], hint:`
   <p>気づいたこと・優先度の希望・質問など、何でもどうぞ (任意)。</p>`},
];
const ans = Q.map(() => ({c: null, m: ""}));
let cur = 0;
const wrap = document.getElementById("wrap");
const answers = document.getElementById("answers");
function render() {
  if (cur >= Q.length) { done(); return; }
  const q = Q[cur], a = ans[cur];
  document.getElementById("barfill").style.width = (cur / Q.length * 100) + "%";
  document.getElementById("step").textContent = "質問 " + (cur + 1) + " / " + Q.length;
  let h = "<h2>" + q.t + "</h2><div class='hint'>" + q.hint + "</div>";
  if (q.img && IMG[q.img]) h += "<img class='shot' src='" + IMG[q.img] + "' alt='実画面シート'><div class='imgnote'>ピンチで拡大できます。各行=1マス (左=誤読の直前 / 中央=直後 / 右=数秒後、赤枠が該当マス)</div>";
  h += "<textarea class='memo' placeholder='メモ (任意)' oninput='ans[" + cur + "].m=this.value'>" + a.m + "</textarea>";
  wrap.innerHTML = h;
  answers.innerHTML = "<div class='row'>" + q.opts.map(([k, l]) =>
    "<button class='opt " + (a.c === k ? "sel" : "") + "' onclick='pick(\\"" + k + "\\")'>" + l + "</button>").join("") +
    "</div><div class='nav'><button onclick='go(-1)' " + (cur === 0 ? "disabled" : "") + ">← 戻る</button>" +
    "<button id='nx' onclick='go(1)'>" + (a.c ? "次へ →" : "スキップ →") + "</button></div>";
  window.scrollTo(0, 0);
}
function pick(k) { ans[cur].c = k; go(1); }
function go(d) { cur = Math.max(0, cur + d); render(); }
function done() {
  document.getElementById("barfill").style.width = "100%";
  document.getElementById("step").textContent = "完了";
  const lines = Q.map((q, i) => "Q" + (i + 1) + " [" + q.t + "] => " + (ans[i].c || "(未回答)") + (ans[i].m ? " memo: " + ans[i].m : ""));
  wrap.innerHTML = "<h2>✅ 完了 — 下の結果をコピーしてClaudeに貼り付けてください</h2><textarea id='result' readonly>" + lines.join("\\n") + "</textarea>";
  answers.innerHTML = "<div class='row'><button class='opt sel' onclick='copyR()'>結果をコピー</button><button class='opt' onclick='cur=0;render()'>やり直す</button></div>";
}
function copyR() {
  const t = document.getElementById("result"); t.select();
  navigator.clipboard.writeText(t.value).then(() => alert("コピーしました。Claudeのチャットに貼り付けてください。"));
}
render();
</script>
"""


def main() -> None:
    html = HTML.replace("__IMAGES__", json.dumps(IMAGES))
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
