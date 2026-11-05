"""準備をたたんだ配置とOBSの小さな図の距離を確かめる。"""
import pytest
from playwright.sync_api import Browser, Page, expect

from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_e2e_r14 import SIZES, frame, publish
from tests.phase_j.test_overlay_browser import browser

MAX_CONTROL_GAP = 12
MAX_VIDEO_GAP = 24
POSITION_TOLERANCE = 2


def check_video(page: Page) -> None:
    """左端をそろえ、準備の見出し直下に映像を置く。"""
    content = page.locator('.stream-content').bounding_box()
    video = page.locator('.ux-video').bounding_box()
    steps = page.locator('#stream-steps').bounding_box()
    assert abs(video['x'] - content['x']) <= POSITION_TOLERANCE
    assert 0 <= video['y'] - steps['y'] - steps['height'] <= MAX_VIDEO_GAP
    assert video['x'] + video['width'] <= content['x'] + content['width'] + POSITION_TOLERANCE
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


def check_diagram(page: Page) -> None:
    """開始と歯車、各ラベルと選択欄が同じ行で隣り合う。"""
    rows = page.locator('.camera-controls, .camera-output-window>div')
    assert rows.count() == 3
    for row in rows.all():
        left, right = [node.bounding_box() for node in row.locator(':scope>*').all()]
        assert 0 <= right['x'] - left['x'] - left['width'] <= MAX_CONTROL_GAP
        assert abs((left['y'] + left['height'] / 2) -
                   (right['y'] + right['height'] / 2)) <= POSITION_TOLERANCE


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('language', ['ja', 'en'])
def test_stream_layout(browser: Browser, distribution: tuple, size: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'language': language})
    page = browser.new_page(viewport={'width': size[0], 'height': size[1]})
    page.goto(url + '/settings');page.wait_for_function('window.distributionReady')
    check_diagram(page)
    frame(page, f'r15_{language}_{size[0]}_initial')
    runtime.configure({'stream_obs_done': True, 'stream_camera_done': True,
                       'stream_analysis_done': True})
    page.reload();page.wait_for_function('window.distributionReady')
    expect(page.locator('#stream-steps')).not_to_have_attribute('open', '')
    check_video(page)
    frame(page, f'r15_{language}_{size[0]}_returning')
    page.route('**/api/state', lambda route: route.fulfill(json=page.evaluate('window.lastDistributionState')))
    publish(page, 'ready')
    check_video(page)
    frame(page, f'r15_{language}_{size[0]}_playing')
    page.locator('#stream-steps summary').first.click()
    check_diagram(page)
    frame(page, f'r15_{language}_{size[0]}_reopened')
    page.locator('#stream-steps summary').first.click()
    check_video(page)
    publish(page, 'overlay_feedback')
    expect(page.locator('#stream-steps')).to_have_attribute('open', '')
    check_diagram(page)
    frame(page, f'r15_{language}_{size[0]}_feedback')
    page.close()
