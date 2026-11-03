"""製品HTMLを使い、L字配置とオプション例を実フレームに合成する。"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

from scripts.render_overlay_design import (
    OUTPUT, SIZE, SCENES, JPEG_QUALITY, font, new_page, snapshot,
)

VARIANTS = {
    'l_right': ('L1 ゲーム左上 / データ右＋下', 'l-right'),
    'l_left': ('L2 ゲーム右上 / データ左＋下', 'l-left'),
    'options_bottom': ('組合せ① 左下 / 75% / 大きな文字',
        'l&corner=bottom-left&scale=75&font=large'),
    'options_column': ('組合せ② 右上 / 80% / 横帯なし',
        'l&corner=top-right&scale=80&band=false&full=false&details=false&font=large'),
    'options_minimal': ('組合せ③ 最小表示＋直近と全体の推移',
        'minimal&recent=true&full=true&font=small&status=false'),
}
MOBILE_SIZE = (1080, 2780)


def show_scene(page: Page, scene: str) -> None:
    """全体と直近の時間軸の差が見える公開DTOの見本を送る。"""
    import math
    count, step_ms = 61, 2000
    for index in range(count):
        probability = .5+index/1000 if scene == 'early' else .6-index/200+math.sin(index/4)*.07
        data = snapshot(probability, index*step_ms)
        if scene != 'early':
            data['evaluations']['practical']['source'] = 'exchange'
        if scene == 'result' and index == count-1:
            data['evaluations']['practical'].update(p1_win_probability=0.,
                p2_win_probability=1., source='confirmed_death')
        page.evaluate('renderSnapshot', data)


def compose(page: Page, frame: Path, target: Path) -> dict[str, Any]:
    """ゲームの矩形は製品と共通の座標計算から取得する。"""
    game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    base = Image.new('RGBA', SIZE, '#0b1423')
    source = Image.open(frame).convert('RGBA').resize((game['width'], game['height']), Image.Resampling.LANCZOS)
    base.paste(source, (game['x'], game['y']))
    overlay = Image.open(io.BytesIO(page.screenshot(omit_background=True))).convert('RGBA')
    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    is_l = page.evaluate("overlayOptions.layout.startsWith('l')")
    if is_l:
        x, y = game['columnX']+20, game['y']+game['height']-62
        draw.text((x, y), '表示見本', font=font(19), fill='#ffda9a')
        draw.text((x, y+26), '数値・推移はサンプル', font=font(19), fill='#ffda9a')
    else:
        draw.rectangle((12, 8, 620, 53), fill='#0b1423')
        draw.text((24, 13), '表示見本 · 数値と推移はサンプル', font=font(26), fill='#ffda9a')
    result.save(target, quality=JPEG_QUALITY, subsampling=0)
    return dict(game=game, panel=page.locator('#panel').bounding_box())


def mobile_sheet(key: str, title: str, bounds: dict[str, Any]) -> None:
    """3場面の全体図と側列の拡大をスマホ向けにまとめる。"""
    sheet = Image.new('RGB', MOBILE_SIZE, '#0b1423')
    draw = ImageDraw.Draw(sheet)
    draw.text((32, 24), title, font=font(37), fill='#f8fafc')
    draw.text((32, 84), '実フレーム使用｜数値と推移は表示確認用サンプル', font=font(27), fill='#ffda9a')
    draw.text((32, 130), 'ゲーム上には表示を重ねない（L字型）', font=font(27), fill='#c4d4e8')
    for index, (scene, (label, _)) in enumerate(SCENES.items()):
        y = 200+index*675
        draw.text((32, y), label, font=font(28), fill='#f8fafc')
        image = Image.open(OUTPUT / f'{key}_{scene}.jpg')
        image.thumbnail((1080, 608), Image.Resampling.LANCZOS)
        sheet.paste(image, (0, y+44))
    draw.text((32, 2240), '表示の拡大（撃ち合い）', font=font(28), fill='#c4d4e8')
    box = bounds['exchange']['panel']
    image = Image.open(OUTPUT / f'{key}_exchange.jpg')
    crop = image.crop((box['x'], box['y'], box['x']+box['width'], box['y']+min(box['height'], 460)))
    crop.thumbnail((990, 400), Image.Resampling.LANCZOS)
    sheet.paste(crop, ((1080-crop.width)//2, 2290))
    draw.text((32, 2710), '設定ページで配置・表示項目を自由に組合せ', font=font(27), fill='#c4d4e8')
    sheet.save(OUTPUT / 'mobile' / f'{key}.jpg', quality=JPEG_QUALITY, subsampling=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'mobile').mkdir(exist_ok=True)
    manifest: dict[str, Any] = dict(sample_values=True, variants={})
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        for key, (title, query) in VARIANTS.items():
            bounds = {}
            for scene, (_, frame) in SCENES.items():
                page = new_page(browser, query)
                show_scene(page, scene)
                bounds[scene] = compose(page, frame, OUTPUT / f'{key}_{scene}.jpg')
                page.close()
            manifest['variants'][key] = dict(query=query, bounds=bounds)
            mobile_sheet(key, title, bounds)
        browser.close()
    (OUTPUT / 'manifest2.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / 'SAMPLE_VALUES.txt').write_text('L字・オプション見本の勝率と推移は表示確認用サンプルです。\n', encoding='utf-8')
    print('L字2案・組合せ3案、各3場面とmobile画像を保存しました。')


if __name__ == '__main__':
    main()
