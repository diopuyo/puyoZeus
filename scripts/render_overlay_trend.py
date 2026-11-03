"""4テーマの実フレーム見本と、製品CSS/SVGを動かす8秒動画を作る。"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import ROOT, OUTPUT, SCENES, font, new_page
from scripts.render_overlay_b3_contents import render_scene, select_scenes

THEMES = {
    'neon': ('NEON / ネオンガラス', '青と赤の光、透明感のある深いグラデーション。'),
    'arena': ('ARENA / 大会中継', '太い色面と斜めの数字。競技中継の存在感。'),
    'pop': ('POP / ぷよの丸み', '丸い輪郭と柔らかな色。かわいさを大人っぽく。'),
    'noir': ('NOIR / 黒と白金', '黒と白金の細い輪郭。静かで上質な読みやすさ。'),
}
SNAPSHOTS = Path('D:/puyo_analyzer/b3_data/codex_tasks/overlay_b3_contents/snapshots.jsonl')
FPS, SECONDS, QUALITY = 24, 8, 90
VIDEO_SIZE = (1280, 720)
MOBILE_SIZE = (1080, 2660)
EARLY_SECONDS, EXCHANGE_SECONDS = 2, 4
MS = 1000
NOTE = '数値・推移：zenchi s1実再生／背景：配置確認用の別場面'
RESULT_NOTE = '決着：演出サンプル（実再生に決着行なし）／背景：別場面'


def result_sample(exchange: dict[str, Any]) -> dict[str, Any]:
    """決着データが収録されていないため、出典と区別した演出用入力を作る。"""
    data = deepcopy(exchange)
    data['timing']['source_available_ms'] += MS
    data['evaluations']['practical'].update(p1_win_probability=0., p2_win_probability=1.,
                                           source='confirmed_death')
    return data


def background(page: Page, scene: str, theme: str) -> None:
    """実画像をローカルだけで読み込み、ブラウザで重ねる。"""
    raw = base64.b64encode(SCENES[scene][1].read_bytes()).decode('ascii')
    mime = 'image/png' if SCENES[scene][1].suffix == '.png' else 'image/jpeg'
    page.evaluate("""({src,note,title})=>{
      let image=document.getElementById('sample-background');
      if(!image){image=document.createElement('img');image.id='sample-background';
        image.style.cssText='position:absolute;inset:0;width:1920px;height:1080px;z-index:-1';
        document.getElementById('stage').prepend(image);}
      image.src=src;
      let label=document.getElementById('sample-label');
      if(!label){label=document.createElement('div');label.id='sample-label';
        label.style.cssText='position:absolute;left:16px;top:12px;padding:8px 14px;background:#0b1423;color:#f9dfac;font:23px Meiryo';
        document.getElementById('stage').append(label);}
      label.textContent=title+' ｜ '+note;
    }""", dict(src=f'data:{mime};base64,{raw}', note=RESULT_NOTE if scene == 'result' else NOTE,
               title=THEMES[theme][0]))
    page.wait_for_function("document.getElementById('sample-background').complete")


def stills(browser: Browser, theme: str, rows: list[dict], scenes: dict, out: Path) -> dict:
    bounds = {}
    for scene in ('early', 'exchange', 'result'):
        page = new_page(browser, 'card&theme='+theme)
        render_scene(page, rows, scenes['exchange' if scene == 'result' else scene])
        if scene == 'result':
            page.evaluate('renderSnapshot', result_sample(scenes['exchange']))
        background(page, scene, theme)
        page.screenshot(path=str(out/f'{theme}_{scene}.jpg'), type='jpeg', quality=QUALITY)
        bounds[scene] = page.locator('#panel').bounding_box()
        page.close()
    return bounds


def mobile(theme: str, bounds: dict, out: Path) -> None:
    sheet = Image.new('RGB', MOBILE_SIZE, '#0b101c')
    draw = ImageDraw.Draw(sheet)
    title, description = THEMES[theme]
    draw.text((36, 25), title, font=font(44), fill='#f8f5ee')
    draw.text((36, 95), description, font=font(26), fill='#bdcee1')
    draw.text((36, 141), '序盤・撃ち合い＝実再生値／決着＝演出サンプル', font=font(25), fill='#f9dfac')
    for index, scene in enumerate(('early', 'exchange', 'result')):
        y = 200+index*675
        draw.text((36, y), SCENES[scene][0], font=font(28), fill='white')
        image = Image.open(out/f'{theme}_{scene}.jpg')
        image.thumbnail((1080, 608), Image.Resampling.LANCZOS)
        sheet.paste(image, (0, y+45))
    draw.text((36, 2235), '表示の拡大 / 撃ち合い', font=font(27), fill='#bdcee1')
    box = bounds['exchange']
    image = Image.open(out/f'{theme}_exchange.jpg')
    image = image.crop((box['x'], box['y'], box['x']+box['width'], box['y']+box['height']))
    image = image.resize((round(image.width*310/image.height), 310), Image.Resampling.LANCZOS)
    sheet.paste(image, ((1080-image.width)//2, 2285))
    draw.text((36, 2610), f'?theme={theme}  /  配置・項目設定と組み合わせ可能', font=font(24), fill='#bdcee1')
    sheet.save(out/'mobile'/f'{theme}.jpg', quality=QUALITY)


def replay_window(rows: list[dict], selected: dict, duration: int) -> list[dict]:
    end = selected['timing']['source_available_ms']
    key = selected['identity']['match_id']
    return [row for row in rows if row['identity']['match_id'] == key and
            end-duration*MS <= row['timing']['source_available_ms'] <= end]


def seed_history(page: Page, rows: list[dict], selected: dict, duration: int) -> None:
    """カットの開始前30秒を実再生から復元し、短い動画でも推移を読めるようにする。"""
    start = selected['timing']['source_available_ms']-duration*MS
    history = [row for row in rows if row['identity']['match_id'] == selected['identity']['match_id']
               and start-30000 <= row['timing']['source_available_ms'] <= start]
    page.emulate_media(reduced_motion='reduce')
    page.evaluate('rows=>rows.forEach(renderSnapshot)', history)
    page.evaluate('resetThemeMotion()')
    page.emulate_media(reduced_motion='no-preference')


def video_frame(page: Page, index: int, windows: dict, scenes: dict, theme: str) -> None:
    seconds = index/FPS
    scene = 'early' if seconds < EARLY_SECONDS else 'exchange' if seconds < EARLY_SECONDS+EXCHANGE_SECONDS else 'result'
    start = 0 if scene == 'early' else EARLY_SECONDS if scene == 'exchange' else EARLY_SECONDS+EXCHANGE_SECONDS
    if index in (0, EARLY_SECONDS*FPS, (EARLY_SECONDS+EXCHANGE_SECONDS)*FPS):
        background(page, scene, theme)
        if scene != 'result':
            page.evaluate('resetThemeMotion()')
    if scene == 'result':
        if index == (EARLY_SECONDS+EXCHANGE_SECONDS)*FPS:
            page.evaluate('renderSnapshot', result_sample(scenes['exchange']))
    else:
        window = windows[scene]
        duration = EARLY_SECONDS if scene == 'early' else EXCHANGE_SECONDS
        target = scenes[scene]['timing']['source_available_ms']-duration*MS+(seconds-start)*MS
        due = [row for row in window if row['timing']['source_available_ms'] <= target]
        if due:
            page.evaluate('row=>{if(lastSourceMs!==row.timing.source_available_ms) renderSnapshot(row)}', due[-1])
    page.clock.run_for(round(MS/FPS))


def movie(browser: Browser, theme: str, rows: list[dict], scenes: dict, out: Path) -> None:
    page = new_page(browser, 'card&theme='+theme)
    page.set_viewport_size(dict(width=VIDEO_SIZE[0], height=VIDEO_SIZE[1]))
    page.evaluate('resizeStage()')
    page.emulate_media(reduced_motion='no-preference')
    page.clock.install(time=0)
    page.clock.pause_at(100)
    windows = {key: replay_window(rows, scenes[key], duration) for key, duration in
               [('early', EARLY_SECONDS), ('exchange', EXCHANGE_SECONDS)]}
    command = [shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out/f'{theme}.mp4')]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for index in range(FPS*SECONDS):
            if index in (0, EARLY_SECONDS*FPS):
                key = 'early' if index == 0 else 'exchange'
                seed_history(page, rows, scenes[key], EARLY_SECONDS if index == 0 else EXCHANGE_SECONDS)
            video_frame(page, index, windows, scenes, theme)
            process.stdin.write(page.screenshot(type='jpeg', quality=QUALITY))
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('動画エンコードに失敗しました')
    page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/'trend')
    parser.add_argument('--snapshots', type=Path, default=SNAPSHOTS)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('見本はリポジトリ外に保存してください')
    (args.output/'mobile').mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.snapshots.read_text(encoding='utf-8').splitlines()]
    scenes = select_scenes(rows)
    manifest = dict(source=str(args.snapshots), notes=[NOTE, RESULT_NOTE], themes=THEMES,
                    video=dict(size=VIDEO_SIZE, fps=FPS, seconds=SECONDS), bounds={})
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        for theme in THEMES:
            bounds = stills(browser, theme, rows, scenes, args.output)
            mobile(theme, bounds, args.output)
            movie(browser, theme, rows, scenes, args.output)
            manifest['bounds'][theme] = bounds
            print(f'{theme}: 静止画3枚・スマホ1枚・8秒動画を保存', flush=True)
        browser.close()
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
