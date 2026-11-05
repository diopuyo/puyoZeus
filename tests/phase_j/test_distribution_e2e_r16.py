"""映像の実表示、設定の余白、詳細の順番と映り込み検出の目印を測る。"""
import pytest
import threading
from playwright.sync_api import Browser, Page, expect
from websockets.sync.server import serve

from src.phase_j.distribution_config import load_settings
from src.phase_j.distribution_obs import SOURCE_PREFIX
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_e2e_r14 import SIZES, frame, publish
from tests.phase_j.test_distribution_e2e_r13 import LAYOUTS
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_obs_integration import MockObs

MIN_VIDEO_RATIO = .9
MIN_CONTROL_GAP = 4
TOLERANCE = 1
MARKER_WIDTH = 24
MARKER_HEIGHT = 18
MARKER_CELLS = 12
VIDEO_ASPECT = 16 / 9
ASPECT_TOLERANCE = .01
MAX_PAGE_HEIGHT = 830
VIDEO = '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#385575"/></svg>'


def check_video(page: Page) -> None:
    """画像の読み込み後に枠と16:9の比率を測る。"""
    page.wait_for_function("document.querySelector('#input-preview').naturalWidth>0")
    expect(page.locator('#input-preview')).to_be_visible()
    video = page.locator('#input-preview').bounding_box()
    panel = page.locator('.ux-video-frame').bounding_box()
    assert video['width'] / panel['width'] >= MIN_VIDEO_RATIO
    assert abs(video['width'] / video['height'] - VIDEO_ASPECT) < ASPECT_TOLERANCE


def check_gaps(page: Page) -> None:
    """表示中のラベル文字と隣接する入力部品を実座標で測る。"""
    gaps = page.locator('label:visible').evaluate_all('''labels => labels.flatMap(label => {
      const nodes=[...label.childNodes].filter(n=>n.nodeType===1 || n.textContent.trim());
      const boxes=nodes.map(n=>{const r=document.createRange();r.selectNode(n);return r.getBoundingClientRect();});
      return boxes.slice(1).map((b,i)=>b.top>=boxes[i].bottom ? b.top-boxes[i].bottom : b.left-boxes[i].right);
    })''')
    assert gaps and min(gaps) >= MIN_CONTROL_GAP
    for left, right in [('stream-url', 'copy-obs-url'), ('obs-password-clear', 'obs-check')]:
        if not page.locator('#' + left).is_visible():
            continue
        a, b = [page.locator('#' + key).bounding_box() for key in (left, right)]
        gap = b['y'] - a['y'] - a['height'] if b['y'] >= a['y'] + a['height'] else b['x'] - a['x'] - a['width']
        assert gap >= MIN_CONTROL_GAP


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('language', ['ja', 'en'])
def test_settings_spacing(browser: Browser, distribution: tuple, size: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'language': language})
    page = browser.new_page(viewport=dict(width=size[0], height=size[1]))
    page.route('**/api/input-preview.jpg', lambda route: route.fulfill(body=VIDEO, content_type='image/svg+xml'))
    page.goto(url + '/settings'); page.wait_for_function('window.distributionReady')
    check_video(page); check_gaps(page)
    prefix = f'r16_{language}_{size[0]}'
    frame(page, prefix + '_initial')
    page.route('**/api/state', lambda route: route.fulfill(json=page.evaluate('window.lastDistributionState')))
    publish(page, 'ready')
    check_video(page); frame(page, prefix + '_playing')
    page.locator('#stream-details>summary').click()
    for selector in ['#app-notice', '#obs-connection']:
        a, b = [page.locator(s).bounding_box() for s in (selector, '#live-diagnostics')]
        assert a['y'] + a['height'] <= b['y']
    page.locator('#stream-steps').evaluate('(node)=>node.open=true')
    page.locator('#obs-manual').evaluate('(node)=>node.open=true')
    check_gaps(page); frame(page, prefix + '_all_open')
    publish(page, 'overlay_feedback'); frame(page, prefix + '_attention')
    for tab in ['appearance', 'other']:
        page.locator('#tab-' + tab).click()
        check_gaps(page); frame(page, prefix + '_' + tab)
    page.close()


@pytest.mark.parametrize('layout', LAYOUTS)
def test_feedback_marker_geometry(browser: Browser, distribution: tuple, layout: str) -> None:
    """カードを含む全配置で目印が上辺中央の24×18pxである。"""
    runtime, url = distribution
    runtime.configure({'overlay': {'layout': layout, 'theme': 'pop'}})
    page = browser.new_page(viewport=dict(width=1920, height=1080))
    page.goto(url + '/?preview=live')
    geometry = page.evaluate('''() => {
      const marker=feedbackMarker, panel=el('panel'), a=marker.getBoundingClientRect(), b=panel.getBoundingClientRect();
      return {width:marker.offsetWidth,height:marker.offsetHeight,cells:marker.children.length,
        center:Math.abs(a.x+a.width/2-b.x-panel.clientLeft-panel.clientWidth/2),top:Math.abs(a.y-b.y-panel.clientTop)};
    }''')
    assert geometry['width'] == MARKER_WIDTH and geometry['height'] == MARKER_HEIGHT
    assert geometry['cells'] == MARKER_CELLS
    assert geometry['center'] <= TOLERANCE and geometry['top'] <= TOLERANCE
    frame(page, 'r16_marker_' + layout)
    page.close()


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('language', ['ja', 'en'])
def test_collapsed_video(browser: Browser, distribution: tuple, size: tuple, language: str) -> None:
    """準備済みの次回表示と解析中にも実画像が枠いっぱいに表示される。"""
    runtime, url = distribution
    runtime.configure({'language': language, 'stream_obs_done': True,
                       'stream_camera_done': True, 'stream_analysis_done': True})
    page = browser.new_page(viewport=dict(width=size[0], height=size[1]))
    page.route('**/api/input-preview.jpg', lambda route: route.fulfill(body=VIDEO, content_type='image/svg+xml'))
    page.goto(url + '/settings'); page.wait_for_function('window.distributionReady')
    expect(page.locator('#stream-steps')).not_to_have_attribute('open', '')
    check_video(page)
    prefix = f'r16_{language}_{size[0]}'
    frame(page, prefix + '_returning')
    page.route('**/api/state', lambda route: route.fulfill(json=page.evaluate('window.lastDistributionState')))
    publish(page, 'ready')
    expect(page.locator('#stream-steps')).not_to_have_attribute('open', '')
    check_video(page); frame(page, prefix + '_collapsed_playing')
    assert page.evaluate('document.documentElement.scrollHeight') <= MAX_PAGE_HEIGHT
    page.locator('#tab-appearance').click()
    page.locator('select[name=layout]').select_option('card')
    expect(page.locator('select[name=layout]')).to_have_value('card')
    expect(page.frame_locator('#overlay').locator('#stage')).to_have_attribute('data-layout', 'ribbon')
    frame(page, prefix + '_card')
    page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('source_exists', [True, False])
def test_upgrade_obs_step(browser: Browser, distribution: tuple, language: str, source_exists: bool) -> None:
    """上書き利用者は既存ソースを確認した時だけ①が済みになり、②は保留される。"""
    runtime, url = distribution
    obs = MockObs()
    if source_exists:
        obs.items = [dict(sourceName=SOURCE_PREFIX + ' abcdef12', inputKind='browser_source')]
    with serve(obs.handle, '127.0.0.1', 0, subprotocols=['obswebsocket.json']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        runtime.configure(dict(completed=True, obs_enabled=True, obs_password='secret',
                               obs_port=server.socket.getsockname()[1], language=language))
        assert not runtime.settings['stream_obs_done']
        page = browser.new_page(viewport=dict(width=1100, height=800))
        page.goto(url + '/settings'); page.wait_for_function('window.distributionReady')
        expect(page.locator('#obs-connection')).to_contain_text('✓ Connected' if language == 'en' else '✓ 連携済み')
        expect(page.locator('#step-obs-mark')).to_have_text('✓' if source_exists else '①')
        saved = load_settings(runtime.root)
        assert saved['stream_obs_done'] is source_exists
        assert not saved['stream_camera_done'] and not saved['stream_analysis_done']
        expect(page.locator('#step-camera-mark')).to_have_text('②')
        page.reload(); page.wait_for_function('window.distributionReady')
        expect(page.locator('#step-obs-mark')).to_have_text('✓' if source_exists else '①')
        assert not any(call.startswith(('Create', 'Start', 'Set')) for call in obs.calls)
        frame(page, f'r16_upgrade_{language}_{source_exists}')
        page.close()
