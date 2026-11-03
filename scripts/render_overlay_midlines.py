"""製品描画の50%表示4案を、同一入力・背景で比較する。"""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, sync_playwright

from scripts.render_overlay_design import OUTPUT, font, new_page
from scripts.render_overlay_b3_contents import render_scene, select_scenes
from scripts.render_overlay_trend import SNAPSHOTS, background

OPTIONS = {
    'split': ('A / 上下の淡い塗り分け', '面の境界で50%を示す。線なし。'),
    'line': ('B / 細い実線＋50目盛り【おすすめ】', '静かな実線と目盛り。最も迷わず読める。'),
    'glow': ('C / 柔らかな光帯', '中央の光で、拮抗する位置を示す。'),
    'adaptive': ('D / 勝率線の上下色分け', '50%より上は1P色、下は2P色。'),
}
THEMES = ('arena', 'neon')
OUT = OUTPUT / 'trend'
WIDTH, HEADER, ROW_HEIGHT = 1080, 220, 740
MARGIN, IMAGE_WIDTH, QUALITY = 36, 1008, 94
PARTS = (('gauge', '#bar'), ('recent', '.trend'), ('full', '#whole'))


def capture(browser: Browser, theme: str, option: str, rows: list[dict], scene: dict) -> list[Image.Image]:
    """実装そのものを描画し、ゲージと両グラフを個別に保存する。"""
    page = new_page(browser, f'card&theme={theme}&midline={option}&full=true')
    render_scene(page, rows, scene)
    background(page, 'exchange', theme)
    page.screenshot(path=str(OUT / f'midline_{theme}_{option}_scene.jpg'), quality=QUALITY)
    parts = []
    for name, selector in PARTS:
        image = Image.open(io.BytesIO(page.locator(selector).screenshot())).convert('RGB')
        image = image.resize((IMAGE_WIDTH, round(image.height * IMAGE_WIDTH / image.width)), Image.Resampling.LANCZOS)
        image.save(OUT / f'midline_{theme}_{option}_{name}.jpg', quality=QUALITY)
        parts.append(image)
    page.close()
    return parts


def comparison(images: dict[tuple[str, str], list[Image.Image]]) -> None:
    """スマホで縦にスクロールできる1枚に、8組を同じ順序で配置する。"""
    sheet = Image.new('RGB', (WIDTH, HEADER + ROW_HEIGHT * len(images)), '#0c1422')
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 24), '50%の示し方 / 4案 × 2テーマ', font=font(42), fill='white')
    draw.text((MARGIN, 90), '同じ撃ち合いの実再生値 / ゲージ・直近30秒・試合全体', font=font(25), fill='#c4d3e5')
    draw.text((MARGIN, 136), '背景は配置確認用の別場面。全体画像・各拡大も同梱。', font=font(25), fill='#c4d3e5')
    for index, ((theme, option), parts) in enumerate(images.items()):
        y = HEADER + index * ROW_HEIGHT
        title, note = OPTIONS[option]
        draw.text((MARGIN, y), f'{theme.upper()}  {title}', font=font(31), fill='white')
        draw.text((MARGIN, y + 50), note, font=font(25), fill='#aabed5')
        offset = y + 105
        for part in parts:
            sheet.paste(part, (MARGIN, offset))
            offset += part.height + 24
        draw.line((MARGIN, y + ROW_HEIGHT - 24, WIDTH - MARGIN, y + ROW_HEIGHT - 24), fill='#304157')
    sheet.save(OUT / 'midline_options.jpg', quality=QUALITY, subsampling=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SNAPSHOTS.read_text(encoding='utf-8').splitlines()]
    scene = select_scenes(rows)['exchange']
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        images = {(theme, option): capture(browser, theme, option, rows, scene)
                  for theme in THEMES for option in OPTIONS}
        comparison(images)
        browser.close()
    print(OUT / 'midline_options.jpg')


if __name__ == '__main__':
    main()
