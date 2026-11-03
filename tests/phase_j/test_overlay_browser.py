"""実ブラウザで配置・公開DTO表示・接続状態・推移の退行を検査する。"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Iterator

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import LAYOUTS, ROOT, new_page, snapshot

SAFE_LEFT, SAFE_RIGHT, SAFE_TOP, SAFE_BOTTOM = 704, 1216, 480, 932


@pytest.fixture(scope='module')
def browser() -> Iterator[Browser]:
    """製品やOBSを起動せず、隔離されたヘッドレスブラウザを使う。"""
    with sync_playwright() as runner:
        channel = os.environ.get('OVERLAY_BROWSER_CHANNEL', 'msedge' if os.name == 'nt' else 'chromium')
        instance = runner.chromium.launch(channel=channel, headless=True)
        yield instance
        instance.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    instance = new_page(browser)
    yield instance
    instance.close()


def test_score_roi_alignment_status(page: Page) -> None:
    """任意フラグの状態だけを表示し、OFFの旧DTOでは欄を隠す。"""
    data = snapshot(.5)
    data['display']['score_roi_alignment'] = {
        '1P': dict(ready=False, samples=3, offset=[0, 0]),
        '2P': dict(ready=True, samples=12, offset=[0, -2]),
    }
    page.evaluate('renderSnapshot', data)
    label = page.locator('#score-roi-status')
    assert not label.is_visible()  # 採用表示では状態の説明を隠す。値の受け取りは維持する。
    assert label.inner_text() == '式の枠: 1P 合わせ中 / 2P 0,-2px'
    del data['display']['score_roi_alignment']
    page.evaluate('renderSnapshot', data)
    assert not label.is_visible()


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('probability', [0., .5, 1.])
def test_layout_stays_in_gap(browser: Browser, layout: str, probability: float) -> None:
    page = new_page(browser, layout)
    page.evaluate('renderSnapshot', snapshot(probability))
    box = page.locator('#panel').bounding_box()
    assert SAFE_LEFT <= box['x'] and box['x']+box['width'] <= SAFE_RIGHT
    # 中身を全項目表示するdata案は、中央の縦長欄を使う。
    top, bottom = (200, 1060) if layout == 'data' else (SAFE_TOP, SAFE_BOTTOM)
    if layout == 'ribbon':  # 旧 ribbon は NEXT の間に収まる中央の柱。
        top, bottom = 175, 935
        assert 830 <= box['x'] and box['x']+box['width'] <= 1090
    assert top <= box['y'] and box['y']+box['height'] <= bottom
    for selector in ('.number', '.meta', '#details'):
        assert page.locator(selector).first.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
    for number in page.locator('.number').all():
        assert number.evaluate('e=>{const r=e.getBoundingClientRect(),p=e.parentElement.getBoundingClientRect();return r.left>=p.left&&r.right<=p.right}')
    page.close()


@pytest.mark.parametrize('width,height', [(1920, 1080), (1280, 720), (480, 270), (390, 844)])
def test_proportional_viewport(page: Page, width: int, height: int) -> None:
    page.set_viewport_size(dict(width=width, height=height))
    page.evaluate('resizeStage()')
    box = page.locator('#panel').bounding_box()
    scale = min(width/1920, height/1080)
    assert box['width'] == pytest.approx(500*scale, abs=1)
    assert box['y'] >= 0 and box['y']+box['height'] <= height


@pytest.mark.parametrize('phase,expected', [
    ('no_puyo_screen', '試合待ち'), ('calibrating', '画面調整中 42%'), ('verifying', '入力確認中')])
def test_input_states_clear_previous_value(page: Page, phase: str, expected: str) -> None:
    data = snapshot()
    page.evaluate('renderSnapshot', data)
    data['display'].update(input_status=phase, calibration_progress=42)
    page.evaluate('renderSnapshot', data)
    assert page.locator('#state').inner_text() == expected
    assert page.locator('#p1').inner_text() == '—'
    assert page.locator('#trace').get_attribute('d') == ''


@pytest.mark.parametrize('filename,expected', [
    ('initial_hidden.json', '試合待ち'), ('hold_visible.json', '判定中'),
    ('terminal_p1.json', '決着 · 1P'), ('result_p2.json', '決着 · 2P'),
    ('integrity_fault.json', '入力確認中')])
def test_existing_contract_examples(page: Page, filename: str, expected: str) -> None:
    data = json.loads((ROOT / 'docs/schemas/examples' / filename).read_text(encoding='utf-8'))
    page.evaluate('renderSnapshot', data)
    assert page.locator('#state').inner_text() == expected


@pytest.mark.parametrize('probability', [None, -1, 2, '0.5'])
def test_invalid_probability_never_becomes_a_number(page: Page, probability: object) -> None:
    data = snapshot()
    data['evaluations']['practical']['p1_win_probability'] = probability
    page.evaluate('renderSnapshot', data)
    assert page.locator('#p1').inner_text() == '—'


def test_b3_source_distinguishes_death_from_prediction(page: Page) -> None:
    data = snapshot(1.)
    data['evaluations']['practical']['source'] = 'unavoidable_death'
    page.evaluate('renderSnapshot', data)
    assert '決着' not in page.locator('#state').inner_text()
    data['evaluations']['practical']['source'] = 'confirmed_death'
    page.evaluate('renderSnapshot', data)
    assert page.locator('#state').inner_text() == '決着 · 1P'
    assert page.locator('#p1').inner_text() == '100'


def test_between_matches_clears_terminal(page: Page) -> None:
    data = snapshot(1.)
    data['evaluations']['practical']['source'] = 'confirmed_death'
    page.evaluate('renderSnapshot', data)
    data['display']['match_phase'] = 'between_matches'
    page.evaluate('renderSnapshot', data)
    assert page.locator('#state').inner_text() == '試合待ち'
    assert page.locator('#p1').inner_text() == '—'


def test_counter_pending_keeps_published_value(page: Page) -> None:
    data = snapshot(.64)
    data['evaluations']['counter_search'] = dict(pending=True)
    page.evaluate('renderSnapshot', data)
    assert page.locator('#state').inner_text() == '判定中'
    assert page.locator('#p1').inner_text() == '64'


def test_history_uses_source_time_and_resets_between_matches(page: Page) -> None:
    data = snapshot()
    for time_ms in (0, 1000, 30000, 31000, 31000):
        data['timing']['source_available_ms'] = time_ms
        page.evaluate('renderSnapshot', data)
    assert page.evaluate('history.map(p=>p.time)') == [1000, 30000, 31000]
    data['identity']['match_id'] = 'new-match'
    page.evaluate('renderSnapshot', data)
    assert page.evaluate('history.length') == 1
    data['timing']['source_available_ms'] = 0
    page.evaluate('renderSnapshot', data)
    assert page.evaluate('history.length') == 1


def test_data_panel_uses_optional_public_fields(page: Page) -> None:
    data = snapshot(.6)
    original = deepcopy(data)
    page.evaluate('renderSnapshot', data)
    assert data == original
    assert page.locator('#basis').inner_text() == '1P 62%'
    assert page.locator('#forecast').inner_text() == '—'
    data['evaluations']['display_layers'].update(predicted_p1=.75, includes_prediction=True)
    page.evaluate('renderSnapshot', data)
    assert page.locator('#forecast').inner_text() == '1P 75%'
    assert page.locator('#prediction').is_hidden()
    del data['evaluations']['display_layers']
    page.evaluate('renderSnapshot', data)
    assert page.locator('#basis').inner_text() == '—'
    assert page.locator('#prediction').is_hidden()


def test_stale_updates_are_not_connection_loss(page: Page) -> None:
    page.clock.install()
    page.evaluate('data=>testStream.emit(data)', snapshot())
    page.clock.run_for(3600)
    assert page.locator('#state').inner_text() == '更新待ち'
    page.evaluate('testStream.onerror()')
    assert page.locator('#state').inner_text() == '接続待ち'
    page.evaluate('testStream.onopen()')
    page.evaluate('data=>testStream.emit(data)', snapshot(.72))
    assert page.locator('#p1').inner_text() == '72'


def test_waiting_resends_remain_waiting(page: Page) -> None:
    page.clock.install()
    data = snapshot()
    data['display']['input_status'] = 'no_puyo_screen'
    for _ in range(10):
        page.evaluate('data=>testStream.emit(data)', data)
        page.clock.run_for(500)
    page.clock.run_for(4000)
    assert page.locator('#state').inner_text() == '試合待ち'


def test_motion_finishes_without_delaying_terminal(page: Page) -> None:
    page.emulate_media(reduced_motion='no-preference')
    page.clock.install()
    page.evaluate('renderSnapshot', snapshot(.2))
    page.evaluate('renderSnapshot', snapshot(.8, 31000))
    assert page.locator('#p1').inner_text() == '80'
    page.clock.run_for(200)
    assert page.locator('#p1').inner_text() == '80'
    data = snapshot(1., 32000)
    data['evaluations']['practical']['source'] = 'confirmed_death'
    page.evaluate('renderSnapshot', data)
    assert page.locator('#p1').inner_text() == '100'


def test_unknown_layout_falls_back_and_has_no_page_errors(browser: Browser) -> None:
    page = new_page(browser, 'unknown')
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.evaluate('data=>testStream.emit(data)', snapshot())
    page.evaluate("testStream.listeners.analysis({data:'invalid json'})")
    assert page.locator('#panel').get_attribute('data-layout') == 'center'
    assert page.locator('#state').inner_text() == '入力確認中'
    assert not errors
    page.close()
