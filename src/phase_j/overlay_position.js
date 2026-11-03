'use strict';
// 実画面見本に合わせた保守的な注意領域。任意の位置への配置は禁止しない。
(()=>{
  const WIDTH=1920,HEIGHT=1080,SNAP=24,MARGIN=20;
  const RIBBON_SHADOW=8,STAMP_SHADOW=4;
  const regions=[{x:260,y:155,w:445,h:780},{x:1215,y:155,w:445,h:780},
    {x:700,y:160,w:130,h:390},{x:1090,y:160,w:130,h:390},
    {x:230,y:925,w:470,h:135},{x:1220,y:925,w:470,h:135},
    {x:700,y:0,w:520,h:175},{x:700,y:935,w:520,h:145}];
  const handle=get('drag-handle'),preview=get('preview');
  let dragging=null;
  /** @returns {object|null} */
  function panelBox() {
    const panel=get('overlay').contentDocument?.getElementById('panel');
    return panel?{x:panel.offsetLeft,y:panel.offsetTop,w:panel.offsetWidth,h:panel.offsetHeight}:null;
  }
  /** @returns {void} */
  function refresh() {
    const box=panelBox(),isL=readControls().layout.startsWith('l');handle.hidden=isL||!box;
    if(!box) return;
    Object.assign(handle.style,{left:`${box.x/WIDTH*100}%`,top:`${box.y/HEIGHT*100}%`,
      width:`${box.w/WIDTH*100}%`,height:`${box.h/HEIGHT*100}%`});
    const mascot=get('overlay').contentDocument.getElementById('mascot');
    const occupied=[box];
    if(mascot&&!mascot.hidden)occupied.push({x:mascot.offsetLeft,y:mascot.offsetTop,w:mascot.offsetWidth,h:mascot.offsetHeight});
    const child=get('overlay').contentWindow,stage=child.document.getElementById('stage');
    if(['battle-a','esports','normal'].includes(stage.dataset.theme)&&stage.dataset.battleLayout==='ribbon') {
      occupied[0]={...box,w:box.w+RIBBON_SHADOW,h:box.h+RIBBON_SHADOW};
      const scale=stage.getBoundingClientRect().width/WIDTH,origin=stage.getBoundingClientRect();
      for(const node of child.document.querySelectorAll('#battle-speech,.battle-state')) {
        if(!node.getClientRects().length)continue;
        const rect=node.getBoundingClientRect();
        occupied.push({x:(rect.left-origin.left)/scale,y:(rect.top-origin.top)/scale,
          w:rect.width/scale+STAMP_SHADOW,h:rect.height/scale+STAMP_SHADOW});
      }
    }
    const overlap=occupied.some(b=>regions.some(r=>b.x<r.x+r.w&&b.x+b.w>r.x&&b.y<r.y+r.h&&b.y+b.h>r.y));
    get('overlap-warning').textContent=isL?'':PuyoI18n.t(overlap?
      '注意：盤面・NEXT・得点・予告・WINと重なる可能性があります。実画面で確認してください。':
      '主要なゲーム表示を避けた位置です。実画面でも確認してください。');
  }
  /** @param {number} x @param {number} y @param {boolean} commit */
  function move(x,y,commit=true) {
    const box=panelBox();if(!box) return;
    const mascot=get('overlay').contentDocument.getElementById('mascot');
    const top=mascot&&!mascot.hidden?mascot.offsetHeight:0;
    const maxX=WIDTH-box.w,maxY=HEIGHT-box.h;
    x=Math.max(0,Math.min(maxX,x));y=Math.max(top,Math.min(maxY,y));
    if(x<SNAP)x=0;if(maxX-x<SNAP)x=maxX;if(y-top<SNAP)y=top;if(maxY-y<SNAP)y=maxY;
    form.elements.x.value=Math.round(x);form.elements.y.value=Math.round(y);
    const child=get('overlay').contentWindow,panel=child.document.getElementById('panel');
    panel.style.left=`${Math.round(x)}px`;panel.style.top=`${Math.round(y)}px`;
    child.PuyoMascot?.place();refresh();if(commit)updatePreview();
  }
  handle.onpointerdown=event=>{
    const box=panelBox();if(!box)return;event.preventDefault();handle.setPointerCapture(event.pointerId);
    dragging={...box,startX:event.clientX,startY:event.clientY};
  };
  handle.onpointermove=event=>{
    if(!dragging)return;const bounds=preview.getBoundingClientRect();
    move(dragging.x+(event.clientX-dragging.startX)*WIDTH/bounds.width,
      dragging.y+(event.clientY-dragging.startY)*HEIGHT/bounds.height,false);
  };
  handle.onpointerup=handle.onpointercancel=()=>{if(dragging){dragging=null;updatePreview();}};
  handle.onkeydown=event=>{
    const delta={ArrowLeft:[-10,0],ArrowRight:[10,0],ArrowUp:[0,-10],ArrowDown:[0,10]}[event.key];
    if(delta){event.preventDefault();const b=panelBox();move(b.x+delta[0],b.y+delta[1]);}
  };
  for(const button of document.querySelectorAll('[data-position]')) button.onclick=()=>{
    const b=panelBox(),position=button.dataset.position;
    // ドラッグ直後の見本の読み込み中でも、位置ボタンの操作を失わない。
    if(!b){get('overlay').addEventListener('load',()=>button.click(),{once:true});return;}
    if(position==='center'){form.elements.x.value='';form.elements.y.value='';updatePreview();return;}
    move(position.endsWith('right')?WIDTH-b.w-MARGIN:MARGIN,position.startsWith('bottom')?HEIGHT-b.h-MARGIN:0);
  };
  get('overlay').addEventListener('load',()=>{
    refresh();const doc=get('overlay').contentDocument,observer=new ResizeObserver(refresh);
    observer.observe(doc.getElementById('panel'));
    if(['battle-a','esports','normal'].includes(doc.getElementById('stage').dataset.theme)&&doc.getElementById('panel').dataset.layout==='ribbon')
      doc.querySelectorAll('#battle-speech,.battle-state').forEach(node=>observer.observe(node));
  });
})();
