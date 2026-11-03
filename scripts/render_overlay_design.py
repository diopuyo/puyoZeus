"""既存HTMLをブラウザで描画し、ローカル実フレーム上の比較資料を作る。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FRAME_ROOT = Path('D:/puyo_analyzer/local_frames')
OUTPUT = FRAME_ROOT / 'overlay_design'
SIZE = (1920, 1080)
MOBILE_WIDTH, MOBILE_HEIGHT = 1080, 2660
JPEG_QUALITY = 90
FONT = 'C:/Windows/Fonts/meiryob.ttc'
LAYOUTS = {
    'card': ('A 中央カード｜おすすめ', '大きな勝率と30秒の推移。盤面とWINの間に。'),
    'ribbon': ('B 横長リボン', '視線の移動を小さく。低めの位置に情報を集約。'),
    'minimal': ('C 最小表示', '勝率と状態だけ。中央の背景を広く残す。'),
    'data': ('D 判定の詳細付き', '勝率と連鎖後の予想、判定の中身を表示。'),
}
SCENES = {
    'early': ('序盤', FRAME_ROOT / 'sealed_eval_2026-10-03/sealed02/g002_0_t287.000.jpg'),
    'exchange': ('撃ち合い・明るい連鎖エフェクト', FRAME_ROOT / 'b3_prep_2026-10-05/c125_rnd11_f19420.png'),
    'result': ('決着', FRAME_ROOT / 'sealed_eval_2026-10-03/sealed02/g002_1_t346.000.jpg'),
}
MOCK_STREAM = """
window.EventSource=class {
  constructor(url){this.url=url;this.listeners={};window.testStream=this;}
  addEventListener(name,callback){this.listeners[name]=callback;}
  emit(data){this.listeners.analysis({data:JSON.stringify(data)});}
};
"""


def new_page(browser: Browser, layout: str = 'card') -> Page:
    """外部通信なしで製品HTMLそのものを開く。"""
    page = browser.new_page(viewport=dict(width=SIZE[0], height=SIZE[1]),
                            reduced_motion='reduce', locale='ja-JP')
    page.add_init_script(MOCK_STREAM)
    page.goto((ROOT / 'src/phase_j/overlay.html').as_uri() + '?layout=' + layout)
    return page


def snapshot(probability: float = .58, time_ms: int = 30000) -> dict[str, Any]:
    """数値はデザイン確認専用。認識・学習結果を装わない。"""
    path = ROOT / 'docs/schemas/examples/live_practical.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['timing']['source_available_ms'] = time_ms
    data['display'].update(input_status='ready', match_phase='playing')
    data['evaluations']['practical'].update(p1_win_probability=probability,
        p2_win_probability=1-probability, advantage_score=(probability-.5)*200,
        is_even=abs(probability-.5)*200 <= 3, calibration_id='b3-stage2-display', source='static')
    data['evaluations']['display_layers'] = dict(current_p1=min(1., probability+.02),
        predicted_p1=None, displayed_p1=probability, includes_prediction=False,
        prediction_discard_reason=None)
    return data


def show_scene(page: Page, scene: str) -> None:
    """公開契約の値を模した序盤・撃ち合い・死亡確認を送る。"""
    values = [.5, .51, .49, .52, .54, .53, .55, .58]
    if scene != 'early':
        values += [.56, .54, .48, .45, .49, .43, .38, .30, .34, .32]
    for index, value in enumerate(values):
        data = snapshot(value, index * 1000)
        if scene != 'early':
            data['evaluations']['practical']['source'] = 'exchange'
        page.evaluate('data=>window.testStream.emit(data)', data)
    if scene == 'result':
        data = deepcopy(data)
        data['evaluations']['practical'].update(p1_win_probability=0.,
            p2_win_probability=1., source='confirmed_death')
        data['timing']['source_available_ms'] += 1000
        page.evaluate('data=>window.testStream.emit(data)', data)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def compose_frame(page: Page, frame: Path, target: Path) -> dict[str, float]:
    """ゲーム画像はリポジトリ外だけで扱い、製品描画をそのまま合成する。"""
    import io
    base = Image.open(frame).convert('RGBA').resize(SIZE, Image.Resampling.LANCZOS)
    overlay = Image.open(io.BytesIO(page.screenshot(omit_background=True))).convert('RGBA')
    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    draw.rectangle((12, 8, 620, 53), fill='#0b1423')
    draw.text((24, 13), '表示見本 · 数値と推移はサンプル', font=font(26), fill='#ffda9a')
    result.save(target, quality=JPEG_QUALITY, subsampling=0)
    return page.locator('#panel').bounding_box()


def mobile_sheet(layout: str, output: Path, bounds: dict[str, Any]) -> None:
    """全体図3場面と拡大図を1枚にまとめ、スマホで比較できるようにする。"""
    sheet = Image.new('RGB', (MOBILE_WIDTH, MOBILE_HEIGHT), '#0b1423')
    draw = ImageDraw.Draw(sheet)
    title, description = LAYOUTS[layout]
    draw.text((38, 25), title, font=font(42), fill='#f8fafc')
    draw.text((38, 90), description, font=font(27), fill='#d0dceb')
    draw.text((38, 137), '実フレーム使用／勝率・推移は表示確認用のサンプル', font=font(25), fill='#ffda9a')
    y = 200
    for scene, (label, _) in SCENES.items():
        draw.text((38, y), label, font=font(28), fill='#f8fafc')
        image = Image.open(output / f'{layout}_{scene}.jpg')
        image.thumbnail((MOBILE_WIDTH, 608), Image.Resampling.LANCZOS)
        sheet.paste(image, (0, y + 45))
        y += 675
    draw.text((38, 2235), '表示の拡大 · 撃ち合い中', font=font(27), fill='#d0dceb')
    box = bounds['exchange']
    image = Image.open(output / f'{layout}_exchange.jpg')
    crop = image.crop((box['x'], box['y'], box['x']+box['width'], box['y']+box['height']))
    crop.thumbnail((950, 310), Image.Resampling.LANCZOS)
    sheet.paste(crop, ((MOBILE_WIDTH-crop.width)//2, 2285))
    draw.text((38, 2610), f'URL: ?layout={layout}  |  OBS 1920×1080', font=font(24), fill='#d0dceb')
    (output / 'mobile').mkdir(exist_ok=True)
    sheet.save(output / 'mobile' / f'{layout}.jpg', quality=JPEG_QUALITY, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('実ゲーム画像の保存先はリポジトリ外を指定してください')
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = dict(sample_values=True, frames={
        scene: str(path) for scene, (_, path) in SCENES.items()}, layouts={})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        for layout in LAYOUTS:
            bounds = {}
            for scene, (_, frame) in SCENES.items():
                page = new_page(browser, layout)
                show_scene(page, scene)
                bounds[scene] = compose_frame(page, frame, output / f'{layout}_{scene}.jpg')
                page.close()
            manifest['layouts'][layout] = bounds
            mobile_sheet(layout, output, bounds)
        browser.close()
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
        indent=2), encoding='utf-8')
    print(json.dumps(manifest['layouts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
