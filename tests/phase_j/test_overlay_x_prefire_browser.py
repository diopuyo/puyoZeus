"""撃つ直前の暫定 X の 1 行 (overlay_x_prefire.js) を実ブラウザで検査する。

- 実場面 3 つ (確定・取り下げ 2 種) の公開欄を、実際の間隔で流し込んで表示と消えるまでを確かめる。
  公開欄の元は段3 の 21c 再生結果 (fixtures/x_prefire_public_scenes.json に出典 SHA を記録)。
- 公開内容に x_provisional が無い (= フラグ OFF) とき、画面の DOM が 1 つも変わらないことを確かめる。
- HTTP 配信は、表示フラグ OFF なら script を足さず、ON のときだけ足すことを確かめる。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import Browser, Page

from scripts.render_overlay_design import LAYOUTS, ROOT, new_page, snapshot
from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState
from tests.phase_j.test_overlay_browser import browser  # noqa: F401  (module fixture)

SCRIPT = ROOT / 'src/phase_j/overlay_x_prefire.js'
FIXTURE = Path(__file__).parent / 'fixtures/x_prefire_public_scenes.json'
SCENES = {s['name']: s for s in json.loads(FIXTURE.read_text(encoding='utf-8'))['scenes']}
LINE = '#x-prefire-line'
SIDE_NAMES = ('1P', '2P')
PENDING_TEXT, CONFIRMED_TEXT, RETRACTED_TEXT = '確定待ち', '確定', '取り下げ'
HOLD_SHOWN_MAX_SEC, HOLD_GONE_MIN_SEC = 1.4, 1.6   # 「確定」「取り下げ」は 1〜2 秒出して消す
STALE_SEC = 3.5
CLOCK_START_MS = 1000
TAG = '\n<script src="overlay_x_prefire.js"></script>\n'
ENV_DISPLAY = 'PUYO_LIVE_X_PREFIRE'


def entry(status: str = 'pending', score: int = 1240, chain: int = 3, landing_t: float = 10.) -> dict[str, Any]:
    """公開欄の片側 (live_v49f_prefire.public() と同じ鍵)。"""
    return dict(status=status, score=score, chain=chain, generated=None, landing_t=landing_t,
                source='image', col=2, rotation=0, age_sec=0.)


def field(first: dict | None = None, second: dict | None = None) -> dict[str, Any]:
    return dict(label='撃つ直前の火力 X (暫定)', sides=[first, second])


def with_x(x_provisional: Any = ..., time_ms: int = 30000) -> dict[str, Any]:
    data = snapshot(.58, time_ms)
    if x_provisional is not ...:
        data['evaluations']['x_provisional'] = x_provisional
    return data


def line_text(page: Page) -> str | None:
    node = page.locator(LINE)
    return node.inner_text() if node.count() else None


def expected(side: int, phase: str, value: dict) -> str:
    return f'暫定 X ({SIDE_NAMES[side]}): {value["chain"]}連鎖 {value["score"]:,}点 {phase}'


def loaded_page(browser: Browser, layout: str = 'card') -> Page:
    """製品 HTML に 1 行の script を足し、時計を止めて手で進められるようにする。"""
    page = new_page(browser, layout)
    page.clock.install(time=0)
    page.clock.pause_at(CLOCK_START_MS)
    page.add_script_tag(path=str(SCRIPT))
    return page


def play(page: Page, scene: dict) -> list[tuple[float, dict | None, str | None]]:
    """標本の間隔どおりに時計を進めながら、公開欄を 1 つずつ流し込む。(時刻, 側の欄, 表示の文字) の列。"""
    played, previous = [], scene['rows'][0]['t']
    for row in scene['rows']:
        page.clock.run_for(round((row['t']-previous)*1000))
        previous = row['t']
        page.evaluate('renderSnapshot', with_x(row['x_provisional']))
        played.append((row['t'], row['x_provisional']['sides'][scene['side']], line_text(page)))
    return played


def test_confirmed_scene_shows_pending_then_confirmed_then_disappears(browser: Browser) -> None:
    scene = SCENES['confirmed']
    page = loaded_page(browser)
    played = play(page, scene)
    side = scene['side']
    pending = [(v, text) for _, v, text in played if v and v['status'] == 'pending']
    assert pending and all(text == expected(side, PENDING_TEXT, v) for v, text in pending)
    first = next(i for i, (_, v, _) in enumerate(played) if v and v['status'] == 'confirmed')
    start = played[first][0]
    assert '15,360点' in played[first][2]
    for t, value, text in played[first:]:
        if t-start <= HOLD_SHOWN_MAX_SEC:
            assert text == expected(side, CONFIRMED_TEXT, value), t-start
        elif t-start >= HOLD_GONE_MIN_SEC:
            # 公開内容にはまだ confirmed が残っているが、再表示しない。
            assert value['status'] == 'confirmed' and text is None, t-start
    assert played[-1][2] is None
    page.clock.run_for(10000)
    assert line_text(page) is None
    page.close()


@pytest.mark.parametrize('name', ['retract_next_moved', 'retract_timeout'])
def test_retract_scenes_show_retracted_then_disappear(browser: Browser, name: str) -> None:
    scene = SCENES[name]
    page = loaded_page(browser)
    played = play(page, scene)
    side = scene['side']
    pending = [(v, text) for _, v, text in played if v]
    assert pending and all(text == expected(side, PENDING_TEXT, v) for v, text in pending)
    last_pending = pending[-1][0]
    first = next(i for i, (_, v, _) in enumerate(played) if v is None and i and played[i-1][1])
    start = played[first][0]
    for t, value, text in played[first:]:
        assert value is None
        if t-start <= HOLD_SHOWN_MAX_SEC:
            assert text == expected(side, RETRACTED_TEXT, last_pending), t-start
        elif t-start >= HOLD_GONE_MIN_SEC:
            assert text is None, t-start
    assert played[-1][2] is None
    page.close()


def test_pending_without_updates_is_dropped_silently(browser: Browser) -> None:
    page = loaded_page(browser)
    page.evaluate('renderSnapshot', with_x(field(entry())))
    assert line_text(page) == expected(0, PENDING_TEXT, entry())
    page.clock.run_for(round((STALE_SEC-.1)*1000))
    assert line_text(page) == expected(0, PENDING_TEXT, entry())
    page.clock.run_for(200)
    assert line_text(page) is None, '更新が途絶えた暫定は取り下げと出さず黙って消す'
    page.close()


def test_new_pending_replaces_old_without_retract_flash(browser: Browser) -> None:
    page = loaded_page(browser)
    page.evaluate('renderSnapshot', with_x(field(entry(score=1240, chain=3, landing_t=10.))))
    newer = entry(score=2620, chain=4, landing_t=10.4)
    page.evaluate('renderSnapshot', with_x(field(newer)))
    assert line_text(page) == expected(0, PENDING_TEXT, newer)
    page.close()


def test_both_sides_and_second_side(browser: Browser) -> None:
    page = loaded_page(browser)
    right = entry(score=880, chain=1, landing_t=11.)
    page.evaluate('renderSnapshot', with_x(field(None, right)))
    assert line_text(page) == expected(1, PENDING_TEXT, right)
    page.evaluate('renderSnapshot', with_x(field(entry(), right)))
    assert line_text(page) == expected(0, PENDING_TEXT, entry()) + ' / ' + expected(1, PENDING_TEXT, right)
    page.close()


@pytest.mark.parametrize('broken', [dict(score=None), dict(chain=None), dict(landing_t=None),
                                    dict(score='abc'), dict(status='unknown')])
def test_unusable_side_is_not_drawn(browser: Browser, broken: dict) -> None:
    page = loaded_page(browser)
    page.evaluate('renderSnapshot', with_x(field(dict(entry(), **broken))))
    assert line_text(page) is None
    page.close()


def test_line_removed_when_field_disappears(browser: Browser) -> None:
    page = loaded_page(browser)
    page.evaluate('renderSnapshot', with_x(field(entry())))
    assert line_text(page) is not None
    page.evaluate('renderSnapshot', with_x())   # x_provisional が公開内容から消えた
    assert line_text(page) is None
    page.clock.run_for(10000)
    assert line_text(page) is None
    page.close()


@pytest.mark.parametrize('absent', [..., None, dict(label='x', sides=None), dict(label='x'),
                                    dict(label='x', sides=[None, None]), dict(label='x', sides=[])])
def test_off_or_empty_field_leaves_dom_unchanged(browser: Browser, absent: Any) -> None:
    """x_provisional が無い・空のとき、script を足しても DOM・style は 1 つも変わらない。"""
    page = new_page(browser)
    page.clock.install(time=0)
    page.clock.pause_at(CLOCK_START_MS)
    page.evaluate('renderSnapshot', with_x())
    snapshot_dom = ("[document.getElementById('stage').outerHTML, document.querySelectorAll('style').length,"
                    "document.body.children.length]")
    before = page.evaluate(snapshot_dom)
    page.add_script_tag(path=str(SCRIPT))
    assert page.evaluate(snapshot_dom) == before, 'script を読んだだけで DOM が変わった'
    page.evaluate('renderSnapshot', with_x(absent))
    page.clock.run_for(10000)
    page.evaluate('renderSnapshot', with_x(absent))
    assert page.evaluate(snapshot_dom) == before
    assert page.locator(LINE).count() == 0
    page.close()


EXISTING_ITEMS = ('#p1', '#p2', '#bar', '.trend')
FLEX_FILL_ITEMS = ('#details',)   # L 字の縦パネルでは、残りの高さを埋める欄。中身の位置は変わらず、箱だけ縮む。
# 縦に大きい data 配置は既存の「画面下端に収める」補正で、行が増えた分だけパネル全体が上へ動く。
BOTTOM_CLAMPED_LAYOUTS = ('data',)
CLAMP_SHIFT_MAX = 70   # 2 行 (両側同時) + 余白の分


def item_boxes(page: Page) -> dict[str, tuple]:
    """見えている既存の欄の、パネルの左上を原点にした位置と大きさ。"""
    panel = page.locator('#panel').bounding_box()
    boxes = {s: page.locator(s).first.bounding_box() for s in EXISTING_ITEMS}
    return {s: (b['x']-panel['x'], b['y']-panel['y'], b['width'], None if s in FLEX_FILL_ITEMS else b['height'])
            for s, b in boxes.items() if b is not None}


@pytest.mark.parametrize('layout', [*LAYOUTS, 'l-right', 'l-left', 'l&band=false', 'l&corner=bottom-left'])
def test_line_does_not_move_existing_items_and_fits(browser: Browser, layout: str) -> None:
    """既存の欄の位置・大きさを変えず、横にはみ出さず、パネルと画面の中に収まる。"""
    page = loaded_page(browser, layout)
    page.evaluate('renderSnapshot', with_x())
    panel_before, items_before = page.locator('#panel').bounding_box(), item_boxes(page)
    page.evaluate('renderSnapshot', with_x(field(entry(score=15360, chain=12), entry(score=980, chain=2, landing_t=11.))))
    assert page.locator(LINE).is_hidden()  # 火力Xの数字は配信に出さない。
    assert len(items_before) >= 3 and item_boxes(page) == items_before
    panel = page.locator('#panel').bounding_box()
    if layout in BOTTOM_CLAMPED_LAYOUTS:
        assert 0 <= panel_before['y']-panel['y'] <= CLAMP_SHIFT_MAX
    else:
        assert (panel['x'], panel['y']) == (panel_before['x'], panel_before['y'])
    assert page.locator(LINE).bounding_box() is None
    assert 0 <= panel['x'] and panel['x']+panel['width'] <= 1920 and 0 <= panel['y'] and panel['y']+panel['height'] <= 1080
    page.close()


def test_one_line_in_card_layout_stays_in_safe_gap(browser: Browser) -> None:
    page = loaded_page(browser, 'card')
    page.evaluate('renderSnapshot', with_x(field(entry(score=15360, chain=12))))
    assert page.locator(LINE).is_hidden()
    box = page.locator('#panel').bounding_box()
    assert 704 <= box['x'] and box['x']+box['width'] <= 1216 and 480 <= box['y'] and box['y']+box['height'] <= 932
    page.close()


@pytest.fixture
def server() -> Iterator[LiveOverlayServer]:
    instance = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0)
    instance.start()
    yield instance
    instance.stop()


def fetch(server: LiveOverlayServer, path: str) -> tuple[int, str, str]:
    with urlopen(f'http://127.0.0.1:{server.address()[1]}{path}') as response:
        return response.status, response.headers['Content-Type'], response.read().decode('utf-8')


def test_http_flag_off_serves_same_page_and_on_adds_only_the_script_tag(
        server: LiveOverlayServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_DISPLAY, raising=False)
    _, _, unset = fetch(server, '/')
    monkeypatch.setenv(ENV_DISPLAY, '0')
    _, _, zero = fetch(server, '/')
    monkeypatch.setenv(ENV_DISPLAY, '1')
    _, _, on = fetch(server, '/')
    assert 'x_prefire' not in unset and unset == zero
    assert on == unset + TAG
    status, mime, body = fetch(server, '/overlay_x_prefire.js')
    assert status == 200 and mime.startswith('text/javascript') and body == SCRIPT.read_text(encoding='utf-8')


def test_http_page_draws_only_when_flag_on(browser: Browser, server: LiveOverlayServer,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    origin = f'http://127.0.0.1:{server.address()[1]}/?preview=sample'
    for flag, drawn in (('0', False), ('1', True)):
        monkeypatch.setenv(ENV_DISPLAY, flag)
        page = browser.new_page(reduced_motion='reduce')
        errors: list[str] = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(origin)
        page.evaluate('renderSnapshot', with_x(field(entry())))
        assert (line_text(page) == expected(0, PENDING_TEXT, entry())) is drawn
        assert page.locator(LINE).count() == int(drawn)
        assert not errors
        page.close()
