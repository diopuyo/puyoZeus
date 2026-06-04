"""未DLのmaster/S級/チャレンジャー動画をticker filter厳守でDL (2026-06-04)。
plC/plD の未取得master動画を video_124+ として連番DL、phase_e_dl_index.tsv に追記。
plB(A/B/C/D級混在)は上級者基準外のため除外。"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
YT=str(ROOT/'venv'/'bin'/'yt-dlp')
IDX=ROOT/'data'/'phase_e_dl_index.tsv'
FRAMES=ROOT/'data'/'frames'
# 上級者tierキーワード(CLAUDE.md): マスター/S級/チャレンジャー(A級複合は保守的に除外)
TIER_KW=('マスター','S級','チャレンジャー')
PLS={'plC':'PLsjREVssD8bZer2yBUdJ9ZPrvJ0SeLJi8','plD':'PLsjREVssD8bYG_VUIlJvREnco92HB5R3t'}
MIN_DUR=600

def done_ids()->set:
    s=set()
    for l in open(IDX,encoding='utf-8'):
        p=l.rstrip('\n').split('\t')
        if len(p)>=4 and p[3]!='video_id': s.add(p[3])
    return s

def next_video_num()->int:
    # plA DL が video_121-123 を使用中のため最低124から開始(衝突回避)。
    START_FLOOR=124
    mx=0
    for f in FRAMES.glob('video_*.mp4'):
        try: mx=max(mx,int(f.stem.split('_')[1]))
        except: pass
    return max(mx+1, START_FLOOR)

def fetch(pid):
    r=subprocess.run([YT,'--flat-playlist','--print','%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s','https://www.youtube.com/playlist?list='+pid],capture_output=True,text=True,timeout=180)
    out=[]
    for l in r.stdout.strip().split('\n'):
        x=l.split('\t',3)
        if len(x)<4: continue
        try: out.append((int(float(x[0])),x[1],int(float(x[2])),x[3]))
        except: pass
    return out

def main():
    done=done_ids(); nxt=next_video_num()
    print(f'[dl] 既DL={len(done)} 次video番号={nxt}',flush=True)
    targets=[]
    for tag,pid in PLS.items():
        for pidx,vid,dur,title in fetch(pid):
            if vid in done or dur<MIN_DUR: continue
            if not any(k in title for k in TIER_KW): 
                print(f'[skip-tier] {tag} {vid} {title[:30]}',flush=True); continue
            targets.append((tag,pidx,vid,dur,title))
    print(f'[dl] 対象master動画 {len(targets)}本',flush=True)
    lines=[]
    for tag,pidx,vid,dur,title in targets:
        out=FRAMES/f'video_{nxt:02d}.mp4'
        print(f'[dl] video_{nxt:02d} <- {tag}:{pidx} {vid} {dur}s {title[:30]}',flush=True)
        # phase_e と同じ video-only mp4 (merge/ffmpeg不要、既存データセットと整合)。
        rc=subprocess.run([YT,'-f','bestvideo[ext=mp4][vcodec^=avc1][height<=720]/bestvideo[ext=mp4][height<=720]','-o',str(out),'--no-playlist','--quiet','https://www.youtube.com/watch?v='+vid]).returncode
        if rc==0 and out.exists() and out.stat().st_size>10_000_000:
            lines.append(f'{nxt}\t{tag}\t{pidx}\t{vid}\t{dur}\t{title}')
            nxt+=1
        else:
            print(f'[dl] FAIL {vid} rc={rc}',flush=True)
    if lines:
        with open(IDX,'a',encoding='utf-8') as f:
            for l in lines: f.write(l+'\n')
    print(f'[dl] 完了 追加={len(lines)}本',flush=True)

if __name__=='__main__': sys.exit(main())
