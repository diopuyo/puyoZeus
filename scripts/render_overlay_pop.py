"""製品HTMLから実フレーム見本・縦長比較・短い動画を再生成する。"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import SCENES, SIZE, font, snapshot
from scripts.render_overlay_design2 import show_scene as show_probabilities
from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState

OUTPUT = Path('D:/puyo_analyzer/local_frames/overlay_pop')
THEMES = {'esports': 'eスポーツ', 'pop': 'ポップ', 'normal': 'ノーマル（既定）'}
LAYOUTS = {'card': '中央カード', 'ribbon': 'リボン', 'minimal': '最小表示', 'l': 'L字型'}
CANDIDATES = {'jelly': 'ツヤ玉（採用）', 'pastel': 'パステル', 'sticker': 'ステッカー', 'arcade': 'ゲームUI'}
SHEET_WIDTH, CELL_HEIGHT = 1080, 760
VIDEO_FPS, VIDEO_SECONDS = 12, 6


def show_scene(page: Page, scene: str) -> None:
    """文字幅も確認できる代表値を、見本専用として表示する。"""
    show_probabilities(page, scene)
    p1 = dict(sl_pending=12, ta_q_med=120, ta_m_sl_med=108,
        chain_display_step=3, chain_pred_l=8, k1_ojama=60)
    p2 = dict(sl_pending=0, ta_q_med=210, ta_m_sl_med=210,
        chain_display_step=5, chain_pred_l=9, k1_ojama=90)
    if scene == 'early':
        for side in (p1, p2):
            side.update(sl_pending=0, chain_display_step=0, chain_pred_l=0)
    page.evaluate('renderContents', dict(p1=p1, p2=p2, model='exchange'))


def open_page(browser: Browser, origin: str, theme: str, layout: str) -> Page:
    """マスコットも含め、配布版と同じHTTP経路から読む。"""
    page = browser.new_page(viewport=dict(width=SIZE[0], height=SIZE[1]), reduced_motion='reduce')
    page.goto(f'{origin}/?preview=sample&theme={theme}&layout={layout}&mascot_expressions=true')
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
    return page


def composite(page: Page, frame: Path) -> Image.Image:
    """実画面だけを縮小し、透明な製品表示を合成する。"""
    g = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    image = Image.new('RGBA', SIZE, '#152332')
    with Image.open(frame) as source:
        image.paste(source.convert('RGBA').resize((g['width'], g['height'])), (g['x'], g['y']))
    overlay = Image.open(io.BytesIO(page.screenshot(omit_background=True))).convert('RGBA')
    result = Image.alpha_composite(image, overlay).convert('RGB')
    ImageDraw.Draw(result).text((12, 8), '表示見本 / 数値はサンプル', font=font(22),
        fill='white', stroke_width=3, stroke_fill='#101c2c')
    return result


def sheet(items: list[tuple[str, Path]], target: str, title: str) -> None:
    """スマホで順に見られる1列構成。12通りを縮小しすぎない。"""
    image = Image.new('RGB', (SHEET_WIDTH, 170+CELL_HEIGHT*len(items)), '#101c2c')
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), title, font=font(36), fill='white')
    draw.text((30, 85), '実画面＋製品表示 / 勝率・推移はサンプル / ゼウスくん', font=font(24), fill='#ffdc93')
    for index, (label, path) in enumerate(items):
        y = 170+index*CELL_HEIGHT
        draw.text((30, y), label, font=font(32), fill='white')
        with Image.open(path) as source:
            image.paste(source.resize((SHEET_WIDTH, 608), Image.Resampling.LANCZOS), (0, y+65))
        draw.text((30, y+692), '1P：実線 / 2P：破線・縞 / 中央目盛り：50%', font=font(23), fill='#bdcede')
    image.save(OUTPUT / target)


def render_stills(browser: Browser, origin: str) -> list[tuple[str, Path]]:
    """3テーマ×4配置×3場面を出力する。"""
    items = []
    for theme, name in THEMES.items():
        for layout, label in LAYOUTS.items():
            page = open_page(browser, origin, theme, layout)
            for scene, (_, frame) in SCENES.items():
                show_scene(page, scene)
                target = OUTPUT / f'{theme}_{layout}_{scene}.png'
                composite(page, frame).save(target)
                if scene == 'exchange':
                    items.append((f'{name} / {label}', target))
            page.close()
    return items


def render_candidates(browser: Browser, origin: str) -> None:
    """未採用案も同じ実画面・数値・ゼウスくんで比較する。"""
    items = []
    for key, label in CANDIDATES.items():
        page = open_page(browser, origin, 'pop', 'card')
        page.evaluate('(v)=>document.getElementById("stage").dataset.popVariant=v', key)
        show_scene(page, 'exchange')
        target = OUTPUT / f'candidate_{key}.png'
        composite(page, SCENES['exchange'][1]).save(target)
        items.append((label, target))
        page.close()
    sheet(items, 'pop_candidates.png', 'ポップ案 / 丸み・ツヤ・輪郭を比較')


def render_settings(browser: Browser, origin: str) -> None:
    """設定ページの実画面とドラッグ位置の見本も残す。"""
    page = browser.new_page(viewport=dict(width=1500, height=1100), reduced_motion='reduce')
    page.goto(origin+'/settings')
    page.locator('#image').set_input_files(SCENES['exchange'][1])
    page.locator('[name=theme]').select_option('pop')
    page.locator('#overlay').content_frame.locator('#p1').wait_for()
    page.wait_for_function("document.getElementById('frame').naturalWidth>0")
    page.screenshot(path=str(OUTPUT / 'settings.png'), full_page=True)
    page.close()


def render_video(browser: Browser, origin: str, layout: str) -> None:
    """動画は静止実フレームに動く表示を重ねた見本と明示する。"""
    import imageio_ffmpeg
    page = open_page(browser, origin, 'pop', layout)
    page.emulate_media(reduced_motion='no-preference')
    writer = imageio_ffmpeg.write_frames(str(OUTPUT / f'pop_{layout}_motion.mp4'), SIZE,
        fps=VIDEO_FPS, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=8)
    writer.send(None)
    for index in range(VIDEO_FPS*VIDEO_SECONDS):
        p = .5+.34*math.sin(index/12)
        data = snapshot(p, index*1000/VIDEO_FPS)
        if index >= VIDEO_FPS*(VIDEO_SECONDS-1):
            data['evaluations']['practical'].update(p1_win_probability=1., source='confirmed_death')
        page.evaluate('renderSnapshot', data)
        page.wait_for_timeout(1000/VIDEO_FPS)
        image = composite(page, SCENES['exchange'][1])
        ImageDraw.Draw(image).text((24, 16), '動きの見本 / 静止実画面・サンプル数値 / ゼウスくん', font=font(27), fill='white', stroke_width=3, stroke_fill='#101c2c')
        writer.send(image.tobytes())
    writer.close()
    page.close()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0)
    server.start()
    try:
        with sync_playwright() as runner:
            browser = runner.chromium.launch(channel='msedge', headless=True)
            origin = f'http://127.0.0.1:{server.address()[1]}'
            sheet(render_stills(browser, origin), 'lineup_3x4.png', '3テーマ × 4レイアウト / 全12通り')
            render_candidates(browser, origin)
            render_settings(browser, origin)
            for layout in ('card', 'l'):
                render_video(browser, origin, layout)
            browser.close()
    finally:
        server.stop()
    metadata: dict[str, Any] = dict(sample_values=True, static_game_frames=True,
        themes=THEMES, layouts=LAYOUTS, candidates=CANDIDATES,
        sources={key: str(value[1]) for key, value in SCENES.items()})
    (OUTPUT / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
