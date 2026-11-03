"""照合済みの再生DTOから、中身入りのdata・L字見本だけを更新する。"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import OUTPUT, SCENES, SIZE, JPEG_QUALITY, font, new_page

WIDTH, GAP, TITLE_HEIGHT = 1080, 32, 190
BOUND_FRAME = 6920
LAYOUTS = {'data': 'data', 'l_right': 'l-right', 'l_left': 'l-left'}


def select_scenes(rows: list[dict]) -> dict[str, dict]:
    """静止・通常の返し・予測不能時の上限を、実際に存在する行から選ぶ。"""
    result = {}
    for row in rows:
        c = row['evaluations']['b3_contents']
        sides = (c['p1'], c['p2'])
        if c['model'] == 'static' and all((s['cells_color'] or 0) >= 24 for s in sides):
            result.setdefault('early', row)
        if (any((s['ta_q_med'] or 0) > 20 and not s['ta_bound'] for s in sides)
                and any((s['chain_pred_l'] or 0) > 3 for s in sides)
                and any((s['sl_pending'] or 0) > 0 for s in sides)):
            result.setdefault('exchange', row)
        if c['frame'] == BOUND_FRAME:
            result['bound'] = row
    assert set(result) == {'early', 'exchange', 'bound'}
    return {key: result[key] for key in ('early', 'exchange', 'bound')}


def render_scene(page: Page, rows: list[dict], selected: dict) -> None:
    """履歴も同じ試合の実再生から描画する。"""
    identity = selected['identity']
    for row in rows:
        if row['identity']['match_id'] == identity['match_id'] and row['identity']['input_generation'] <= identity['input_generation']:
            page.evaluate('renderSnapshot', row)
    for node in page.locator('.content-row span').all():
        if node.is_visible():
            assert node.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'), node.inner_text()


def compose(page: Page, background: Path, target: Path) -> Image.Image:
    """背景は従来見本の配置用画像であることを画像上でも区別する。"""
    game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    base = Image.new('RGBA', SIZE, '#0b1423')
    source = Image.open(background).convert('RGBA').resize((game['width'], game['height']))
    base.paste(source, (game['x'], game['y']))
    overlay = Image.open(io.BytesIO(page.screenshot(omit_background=True))).convert('RGBA')
    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    draw.rectangle((8, 8, 1200, 49), fill='#0b1423')
    draw.text((20, 12), '数値：zenchi s1 実再生 ／ 背景：配置確認用の別場面', font=font(23), fill='#ffda9a')
    result.save(target, quality=JPEG_QUALITY, subsampling=0)
    return result


def detail_images(page: Page) -> list[Image.Image]:
    """帯と横列を別々に拡大し、スマホでも読めるサイズにする。"""
    selectors = ['#details']
    if page.evaluate("overlayOptions.layout.startsWith('l')"):
        selectors = ['#contents-main .content-group:nth-child(1)',
                     '#contents-main .content-group:nth-child(2)', '#contents-extra']
    images = []
    for selector in selectors:
        image = Image.open(io.BytesIO(page.locator(selector).screenshot())).convert('RGB')
        width = WIDTH-2*GAP
        image = image.resize((width, round(image.height*width/image.width)), Image.Resampling.LANCZOS)
        images.append(image)
    return images


def mobile(key: str, pictures: list[tuple[str, Image.Image]], details: list[Image.Image], out: Path) -> None:
    thumbs = []
    for label, picture in pictures:
        picture = picture.copy()
        picture.thumbnail((WIDTH, WIDTH*SIZE[1]//SIZE[0]), Image.Resampling.LANCZOS)
        thumbs.append((label, picture))
    height = TITLE_HEIGHT+sum(image.height+70 for _, image in thumbs)+80+sum(image.height+GAP for image in details)
    sheet = Image.new('RGB', (WIDTH, height), '#0b1423')
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, 20), f'{key} ｜ 判定の中身', font=font(37), fill='#f8fafc')
    draw.text((GAP, 80), '数値・推移は zenchi s1 の再生結果', font=font(27), fill='#c4d4e8')
    draw.text((GAP, 126), '背景は配置確認用の別場面。おじゃまの単位は個。', font=font(25), fill='#ffda9a')
    y = TITLE_HEIGHT
    for label, image in thumbs:
        draw.text((GAP, y), label, font=font(28), fill='#f8fafc')
        sheet.paste(image, (0, y+45))
        y += image.height+70
    draw.text((GAP, y), '判定の中身・拡大（撃ち合い）', font=font(30), fill='#c4d4e8')
    y += 65
    for image in details:
        sheet.paste(image, (GAP, y))
        y += image.height+GAP
    sheet.save(out/'mobile'/f'{key}.jpg', quality=JPEG_QUALITY, subsampling=0)


def render_l_trend(browser: Browser, rows: list[dict], selected: dict, out: Path) -> None:
    """テーマ付きL字の例も、同じ実再生値と製品CSSで更新する。"""
    (out/'trend').mkdir(exist_ok=True)
    page = new_page(browser, 'l-right&theme=neon')
    render_scene(page, rows, selected)
    compose(page, SCENES['exchange'][1], out/'trend'/'l_right.jpg')
    page.close()


def render(rows: list[dict], out: Path) -> None:
    scenes = select_scenes(rows)
    labels = dict(early='通常判定', exchange='撃ち合い', bound='予想できず・(目安)')
    manifest = dict(sample_values=False, source='zenchi s1 replay / eval_rows_v7',
                    background='従来見本の配置確認用の別場面', scenes=scenes, layouts={})
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        for key, query in LAYOUTS.items():
            pictures, details = [], []
            for scene, selected in scenes.items():
                page = new_page(browser, query)
                render_scene(page, rows, selected)
                frame = SCENES['result' if scene == 'bound' else scene][1]
                picture = compose(page, frame, out/f'{key}_{scene}.jpg')
                stamp = selected['evaluations']['b3_contents']['t_sec']
                pictures.append((f'{labels[scene]} · {stamp:.3f}秒', picture))
                if scene == 'exchange':
                    details = detail_images(page)
                    picture.save(out/f'{key}.jpg', quality=JPEG_QUALITY, subsampling=0)
                page.close()
            mobile(key, pictures, details, out)
            manifest['layouts'][key] = query
        render_l_trend(browser, rows, scenes['exchange'], out)
        manifest['trend_example'] = 'trend/l_right.jpg (l-right, neon)'
        browser.close()
    (out/'manifest_b3_contents.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (args.output/'mobile').mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.snapshots.read_text(encoding='utf-8').splitlines()]
    render(rows, args.output)


if __name__ == '__main__':
    main()
