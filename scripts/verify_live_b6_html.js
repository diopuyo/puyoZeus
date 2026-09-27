/* OBS用HTMLの状態表示と勝率抑止をNodeのDOM代替で検証し、撮影用fixtureを作る。 */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('src/phase_j/overlay.html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes={};
const get=id=>nodes[id]??=( {dataset:{},style:{},textContent:'',hidden:false,setAttribute(k,v){this[k]=v;}} );
const sandbox={document:{getElementById:get},setTimeout:()=>0,clearTimeout:()=>{},
  EventSource:class{addEventListener(){}}};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const data={display:{input_status:'ready',visibility:'visible',status:'live'},
 evaluations:{practical:{availability:'available',p1_win_probability:.673},
 display_layers:{includes_prediction:true},counter_search:{pending:false}}};
sandbox.renderSnapshot(data);
assert.equal(get('p1').textContent,'67.3');assert.equal(get('p2').textContent,'32.7');
assert.equal(get('prediction').hidden,false);assert.equal(get('state').textContent,'確定');
let checks=4;
for(const [phase,label] of [['verifying','入力確認中'],['no_puyo_screen','ぷよ画面なし'],['calibrating','色を較正中 42%']]){
 sandbox.renderSnapshot({...data,display:{...data.display,input_status:phase,calibration_progress:42}});
 assert.equal(get('state').textContent,label);assert.equal(get('p1').textContent,'—');checks+=2;
}
sandbox.renderSnapshot({...data,evaluations:{...data.evaluations,counter_search:{pending:true}}});
assert.equal(get('state').textContent,'計算中');checks++;
sandbox.renderSnapshot({...data,display:{...data.display,visibility:'hidden',message:'次試合待ち'}});
assert.equal(get('state').textContent,'次試合待ち');assert.equal(get('p1').textContent,'—');checks+=2;
sandbox.unavailable('接続待ち','offline');assert.equal(get('prediction').hidden,true);checks++;
const output='logs/live_b6_html';fs.mkdirSync(output,{recursive:true});
fs.writeFileSync(`${output}/overlay-preview.html`,html.replace('connect();',`renderSnapshot(${JSON.stringify(data)});`));
fs.writeFileSync(`${output}/verification.json`,JSON.stringify({checks,passed:checks,width:1920,height:1080},null,2));
console.log(`${checks} HTML assertions passed`);
