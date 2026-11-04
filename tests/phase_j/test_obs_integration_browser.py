"""設定・URL・アプリ起動時刻の結合を実ブラウザで検証する。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from playwright.sync_api import Browser, expect

from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_browser import select_language, open_appearance
from scripts.render_overlay_design import snapshot

FRAMES: Path


@pytest.fixture(autouse=True)
def frame_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """画像は一時領域か明示された保存先にだけ出力する。"""
    destination = Path(os.environ.get('OBS_TEST_FRAMES', str(tmp_path / 'frames')))
    monkeypatch.setitem(globals(), 'FRAMES', destination)


@pytest.mark.parametrize('language,labels', [
    ('ja', ['普通', '高い', 'エレクション']), ('en', ['Normal', 'High', 'Election'])])
def test_dopamine_labels_preserve_saved_values(browser: Browser, distribution: tuple,
                                              language: str, labels: list[str]) -> None:
    """表示名だけを変更し、既定・URL・保存値と他設定の翻訳を維持する。"""
    runtime, url = distribution
    page = browser.new_page()
    page.goto(url + '/settings')
    open_appearance(page)
    select = page.locator('[name=mascot_dopamine]')
    assert select.input_value() == 'normal'
    select_language(page, language)
    expect(select.locator('option')).to_have_text(labels)
    expect(page.locator('[name=mascot_speed] option')).to_have_text(['0.5×', '1×', '1.5×', '2×'])
    assert page.evaluate("['低','普通','高'].map(PuyoI18n.t)") == (
        ['低', '普通', '高'] if language == 'ja' else ['Low', 'Normal', 'High'])
    for value, label in zip(['low', 'normal', 'high'], labels):
        open_appearance(page)
        with page.expect_response(lambda r: r.url.endswith('/api/settings') and f'\"mascot_dopamine\":\"{value}\"' in (r.request.post_data or '')):
            select.select_option(label=label)
        assert page.locator('#url').input_value() == url + '/'
        expect(page.locator('#save-status')).to_have_text('保存しました' if language == 'ja' else 'Saved')
        assert runtime.settings['overlay']['mascot_dopamine'] == value
        page.reload()
        assert select.input_value() == value
        expect(select.locator('option:checked')).to_have_text(label)
    page.close()


@pytest.mark.parametrize('theme', ['normal', 'esports', 'pop'])
@pytest.mark.parametrize('layout', ['card', 'ribbon', 'minimal', 'l'])
def test_rendered_marker_detects_actual_overlay(browser: Browser, distribution: tuple, theme: str, layout: str) -> None:
    from src.phase_j.distribution_input import has_feedback
    _, url = distribution
    page = browser.new_page(viewport=dict(width=1920,height=1080))
    page.goto(url+f'/?preview=sample&theme={theme}&layout={layout}&x=0&y=300')
    page.evaluate('renderSnapshot',snapshot(.65))
    pixels = cv2.imdecode(np.frombuffer(page.screenshot(),dtype=np.uint8),cv2.IMREAD_COLOR)
    assert has_feedback(pixels)
    page.close()


@pytest.mark.parametrize('minutes,stage', [(0,0),(29,0),(30,1),(45,2),(60,3)])
def test_bloodshot_uses_app_start_and_survives_refresh(browser: Browser, distribution: tuple,
                                                     minutes: int, stage: int) -> None:
    runtime, url = distribution
    runtime.started_at = time.time() - minutes * 60
    page = browser.new_page(viewport=dict(width=1920,height=1080))
    suffix = f'_b{stage}' if stage else ''
    page.goto(url+'/?preview=sample&mascot=true&mascot_bloodshot=true&mascot_speed=2')
    expect(page.locator('#mascot img')).to_have_attribute('src',f'/mascot/50{suffix}_s2.webp')
    page.reload()
    expect(page.locator('#mascot img')).to_have_attribute('src',f'/mascot/50{suffix}_s2.webp')
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth===320")
    page.close()


@pytest.mark.parametrize('dopamine', ['low','normal','high'])
def test_dopamine_only_changes_upper_faces(browser: Browser, distribution: tuple, dopamine: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url+f'/?preview=sample&mascot=true&mascot_expressions=true&mascot_dopamine={dopamine}')
    for probability, band in ((.2,'30'),(.8,'70'),(.95,'90')):
        page.evaluate('renderSnapshot',snapshot(probability))
        suffix = '_'+dopamine if band in ('70','90') and dopamine != 'normal' else ''
        expect(page.locator('#mascot img')).to_have_attribute('src',f'/mascot/{band}{suffix}.webp')
        page.wait_for_function("document.querySelector('#mascot img').naturalWidth===320")
    page.close()


def test_options_saved_and_preview_screens(browser: Browser, distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_runtime.shortcuts',lambda *args:None)
    FRAMES.mkdir(parents=True,exist_ok=True)
    page = browser.new_page(viewport=dict(width=1440,height=1000))
    page.goto(url+'/settings')
    open_appearance(page)
    for key in ('mascot','mascot_dopamine_mode','mascot_bloodshot'):
        page.locator(f'[name={key}]').check()
    page.locator('[name=mascot_dopamine]').select_option('high')
    page.locator('[name=mascot_speed]').select_option('1.5')
    expect(page.locator('#save-status')).to_have_text('保存しました')
    assert runtime.settings['overlay']['mascot_speed'] == '1.5'
    assert runtime.settings['overlay']['mascot_dopamine'] == 'high'
    assert page.locator('#url').input_value() == url + '/' and runtime.settings['overlay']['mascot_bloodshot']
    page.screenshot(path=str(FRAMES/'settings-ja.png'),full_page=True)
    select_language(page, 'en')
    page.screenshot(path=str(FRAMES/'settings-en.png'),full_page=True)
    page.locator('#tab-stream').click()
    page.locator('#obs-check').scroll_into_view_if_needed()
    page.screenshot(path=str(FRAMES/'obs-guide-en.png'))
    runtime.started_at = time.time()-60*60
    page.goto(url+'/?preview=sample&mascot=true&mascot_expressions=true&mascot_dopamine=high&mascot_bloodshot=true&mascot_speed=1.5')
    page.evaluate('renderSnapshot',snapshot(.95))
    expect(page.locator('#mascot img')).to_have_attribute('src','/mascot/90_high_b3_s1.5.webp')
    page.screenshot(path=str(FRAMES/'mascot-high-bloodshot.png'))
    page.close()


def test_feedback_label_fits_card(browser: Browser, distribution: tuple) -> None:
    _, url = distribution
    page = browser.new_page(viewport=dict(width=1920,height=1080))
    page.goto(url+'/?preview=sample')
    page.evaluate("renderSnapshot({display:{input_status:'overlay_feedback'}})")
    expect(page.locator('#state')).to_have_text('表示の映り込み')
    assert page.locator('#panel').evaluate('(node)=>node.scrollWidth<=node.clientWidth+1')
    page.screenshot(path=str(FRAMES/'feedback-warning.png'))
    page.close()
