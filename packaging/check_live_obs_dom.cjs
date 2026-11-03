/* 実配信HTML/JS/DTOの3枠契約。ブラウザ実画面やpaintの検査ではない。 */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = process.argv[2];
const elements = new Map();
function element() {
  return {textContent:'',dataset:{},style:{setProperty(){}},setAttribute(){},removeAttribute(){},appendChild(node){
    if (node.id) elements.set(node.id,node);
  },set innerHTML(value){
    this.html=value;
    for (const match of value.matchAll(/id="([^"]+)"/g)) elements.set(match[1],element());
  }};
}
const html = fs.readFileSync(path.join(root,'served_overlay.html'),'utf8');
for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1],element());
assert(html.includes('overlay_ama.js') && html.includes('overlay_v49f.js'));
const document = {createElement:element,head:element(),getElementById(id){
  assert(elements.has(id),`Missing actual HTML element: ${id}`); return elements.get(id);
}};
// 実HTMLのB3描画も実行する。グラフ・装飾・アニメーションのブラウザ依存部分だけ代替する。
const window = {document,innerWidth:1920,innerHeight:1080,matchMedia:()=>({matches:true}),
  addEventListener(){},cancelAnimationFrame(){},clearFullHistory(){},updateFullHistory(){},
  resetThemeMotion(){},updateThemeMotion(){},renderContents(){}};
window.window=window;
const context=vm.createContext(window);
const inline=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]
  .map(match=>match[1]).find(script=>script.includes('function renderSnapshot(data)'));
assert(inline,'B3 renderer is missing');
vm.runInContext(inline,context);
for (const name of ['v49f','ama']) {
  vm.runInContext(fs.readFileSync(path.join(root,`served_overlay_${name}.js`),'utf8'),context);
}
const records = fs.readFileSync(path.join(root,'polls.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
const counts={b3:0,v49f:0,ama:0,all_three:0,ama_held:0,ama_waiting:0};
const fmt=p=>Number.isFinite(p)?(100*p).toFixed(p===0||p===1?0:1)+'%':'—';
for (const {snapshot} of records) {
  window.renderSnapshot(snapshot);
  const evaluations=snapshot.evaluations||{};
  if (evaluations.b3_model_label) assert.equal(elements.get('heading').textContent,'実戦勝率: B3 (M3)');
  const b3Shown=/^\d+$/.test(elements.get('p1').textContent);
  if (b3Shown) {
    const winner=window.confirmedWinner(snapshot);
    const p=winner?(winner==='1P'?1:0):evaluations.practical.p1_win_probability;
    assert.equal(elements.get('p1').textContent,String(Math.round(p*100)));
    assert.equal(elements.get('p2').textContent,String(100-Math.round(p*100)));
  }
  counts.b3+=b3Shown;
  for (const key of ['v49f','ama']) {
    const value=evaluations[key==='ama'?'ama_plain':key];
    assert.equal(elements.get(key+'-p1').textContent,fmt(value?.p1));
    assert.equal(elements.get(key+'-p2').textContent,fmt(value?.p2));
    counts[key]+=value?.p1!=null;
  }
  const ama=evaluations.ama_plain;
  counts.all_three+=b3Shown && evaluations.v49f?.p1!=null && ama?.p1!=null;
  counts.ama_waiting+=ama?.p1==null;
  if (ama?.age_sec>0) {
    assert(elements.get('ama-status').textContent.includes(ama.age_sec.toFixed(1)+'秒保持'));
    counts.ama_held++;
  }
}
assert(elements.has('v49f-card') && elements.has('ama-card'));
const passed=counts.all_three>0;
fs.writeFileSync(path.join(root,'dom_smoke.json'),JSON.stringify({passed,counts,
  method:'served HTML/JS and actual DTO; browser visual/paint not measured'},null,2));
console.log(counts);
assert(passed,'3枠すべての有効値を確認できない');
