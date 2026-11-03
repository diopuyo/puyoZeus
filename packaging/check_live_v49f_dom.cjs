/* 実際に配信されたJSと公開DTOを使った表示契約の検査。ブラウザの描画時間ではない。 */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.join(process.env.PUYO_DEV_WORK || path.join(__dirname, '..', '.local'), 'build_20261019a_stage', 'verification_r2') + path.sep;
const elements = new Map();
const element = () => ({textContent:'', innerHTML:'', dataset:{}, setAttribute(){}, appendChild(){}});
const document = {createElement: element, head: element(), getElementById(id) {
  if (!elements.has(id)) elements.set(id, element()); return elements.get(id);
}};
const window = {renderSnapshot() {}};
vm.runInNewContext(fs.readFileSync(root+'served_overlay_v49f.js','utf8'), {document, window});
const records = fs.readFileSync(root+'polls.jsonl','utf8').trim().split('\n').map(JSON.parse);
let displayed=0, waiting=0, held=0;
const examples=[];
for (const {snapshot} of records) {
  window.renderSnapshot(snapshot);
  const v = snapshot.evaluations?.v49f;
  assert.equal(document.getElementById('heading').textContent, 'B3 勝率');
  if (v?.p1 != null) {
    const expected = (v.p1 === 0 || v.p1 === 1 ? (100*v.p1).toFixed(0) : (100*v.p1).toFixed(1))+'%';
    assert.equal(document.getElementById('v49f-p1').textContent, expected);
    displayed++;
    if (v.age_sec>0) held++;
    if (examples.length<3) examples.push({p1:expected,p2:document.getElementById('v49f-p2').textContent,status:document.getElementById('v49f-status').textContent});
  } else { assert.equal(document.getElementById('v49f-p1').textContent, '—'); waiting++; }
}
assert(displayed>0 && waiting>0);
const result={displayed,waiting,held,examples,method:'Node VM DOM contract; not browser visual QA'};
fs.writeFileSync(root+'dom_smoke.json',JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
