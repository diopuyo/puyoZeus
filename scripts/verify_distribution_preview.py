"""隔離した配布版の実HTMLから、ゲーム画面を含まない見本PNGを保存する。"""
from __future__ import annotations

import argparse
from pathlib import Path
from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import snapshot

OUTPUT = Path('D:/puyo_analyzer/local_frames/dist_logo')
SIZE = dict(width=1920, height=1080)
MOODS = [('even', .5, False), ('ahead', .78, False), ('behind', .24, False),
         ('win', 1., True), ('lose', 0., True)]


def mascot_samples(browser: Browser, url: str, output: Path) -> None:
    page = browser.new_page(viewport=SIZE, reduced_motion='reduce')
    for mood, probability, terminal in MOODS:
        page.goto(url + '/?preview=sample&language=ja&mascot=true&mascot_position=card&theme=neon')
        data = snapshot(probability)
        if terminal:
            data['evaluations']['practical']['source'] = 'confirmed_death'
        page.evaluate('renderSnapshot', data)
        page.wait_for_function("document.querySelector('#mascot img').complete && document.querySelector('#mascot img').naturalWidth>0")
        boxes = [page.locator(selector).bounding_box() for selector in ('#panel', '#mascot')]
        x, y = min(b['x'] for b in boxes)-20, min(b['y'] for b in boxes)-20
        right, bottom = max(b['x']+b['width'] for b in boxes)+20, max(b['y']+b['height'] for b in boxes)+20
        page.screenshot(path=str(output / f'mascot_{mood}_ja.png'), clip=dict(x=x,y=y,width=right-x,height=bottom-y))
    page.close()


def english_samples(browser: Browser, url: str, output: Path) -> None:
    page = browser.new_page(viewport=SIZE, reduced_motion='reduce')
    page.goto(url + '/?preview=sample&language=en&layout=l&theme=arena&mascot=true&mascot_position=left')
    for index in range(31):
        page.evaluate('renderSnapshot', snapshot(.5 + index / 150, index * 1000))
    page.evaluate("document.body.style.background='#14283a'")
    page.screenshot(path=str(output / 'overlay_english.png'))
    page.goto(url + '/?preview=sample&language=ja&layout=l&theme=arena&mascot=true&mascot_position=left')
    page.evaluate('renderSnapshot', snapshot(.73))
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
    page.evaluate("document.body.style.background='#14283a'")
    page.screenshot(path=str(output / 'overlay_l_ja.png'))
    page.goto(url + '/settings')
    page.locator('[name=language]').select_option('en')
    page.locator('[name=mascot]').check()
    page.wait_for_function("document.getElementById('overlay').contentDocument?.getElementById('state')?.textContent==='Playing'")
    page.wait_for_function("document.getElementById('overlay').contentDocument?.querySelector('#mascot img')?.naturalWidth>0")
    page.screenshot(path=str(output / 'settings_english.png'), full_page=True)
    page.goto(url + '/setup')
    page.locator('#language').select_option('en')
    page.screenshot(path=str(output / 'initial_setup_english.png'), full_page=True)
    page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        mascot_samples(browser, args.url, args.output)
        english_samples(browser, args.url, args.output)
        browser.close()


if __name__ == '__main__':
    main()
