"""試験用パネル(debug_panel)の実ブラウザ試験: 既定OFFで画面が変わらない・ONで盤面の色が対応する・列から出ない。"""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest
from playwright.sync_api import Browser, Page

from scripts.render_debug_panel_screens import frame_from_result, payload, sample_contents
from scripts.render_overlay_design import new_page
from src.phase_j.live_debug_panel import BOARD_COLS, BOARD_ROWS, DEBUG_PANEL_KEY, build_debug_panel
from tests.phase_j.test_overlay_browser import browser  # noqa: F401  (fixture)

ON = 'l&debug_panel=true'
# overlay_debug.js の FILL と同じ対応 (赤 青 緑 黄 紫 おじゃま 空 不明)
EXPECTED_RGB = {'1': (229, 72, 77), '2': (59, 130, 246), '3': (47, 165, 106), '4': (242, 207, 29),
                '5': (162, 89, 230), 'o': (154, 163, 178), '.': (23, 33, 47), '?': (255, 61, 245)}
CELL_CORNER = 3  # 文字が乗らないマスの角の画素で色を読む


def blank(rows: int = BOARD_ROWS) -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(rows)]


def board_with_every_kind() -> list[list[int]]:
    cells = blank()
    for column, value in enumerate((1, 2, 3, 4, 5, 9)):
        cells[12][column] = value
    cells[11][0] = 10
    return cells


def frame_for(confirmed, reading=None) -> dict:
    side = dict(confirmed_board=confirmed, raw_cnn_board=reading, state='STABLE', board_provenance='observed',
                score=100, tsumo_count=1, formula=None)
    return dict(frame=5, t_sec=1.0, sides=[side, dict(side, confirmed_board=None, raw_cnn_board=None)])


def emit(page: Page, data: dict) -> None:
    page.evaluate('data=>window.testStream.emit(data)', data)
    page.wait_for_timeout(100)


def pixel(page: Page, index: int, row: int, column: int, cell: int) -> tuple[int, int, int]:
    return tuple(page.evaluate("""([i,x,y])=>{
        const c=document.querySelectorAll('#debug canvas')[i];
        return Array.from(c.getContext('2d').getImageData(x,y,1,1).data).slice(0,3);}""",
                               [index, column * cell + CELL_CORNER, row * cell + CELL_CORNER]))


def test_board_colors_map_to_cells(browser: Browser) -> None:  # noqa: F811
    page = new_page(browser, ON)
    emit(page, payload(frame_for(board_with_every_kind()), sample_contents(None)))
    cell = page.evaluate('document.querySelector("#debug canvas").width/6')
    for column, code in enumerate('12345o'):
        assert pixel(page, 0, 12, column, cell) == EXPECTED_RGB[code], code
    assert pixel(page, 0, 11, 0, cell) == EXPECTED_RGB['?']
    assert pixel(page, 0, 10, 3, cell) == EXPECTED_RGB['.']
    assert '1P 確定盤' in page.locator('#debug .dbg-title').first.inner_text()
    page.close()


def test_unconfirmed_reading_is_labeled_and_mismatch_is_outlined(browser: Browser) -> None:  # noqa: F811
    page = new_page(browser, ON)
    confirmed, reading = board_with_every_kind(), board_with_every_kind()
    reading[12][0] = 2
    emit(page, payload(frame_for(confirmed, reading), sample_contents(None)))
    cell = page.evaluate('document.querySelector("#debug canvas").width/6')
    # 違うマスは読取盤 (2枚目) の側だけ白い枠 (角から2画素内側)。確定盤 (1枚目) には出ない
    outline = page.evaluate("""([cell])=>{const g=document.querySelectorAll('#debug canvas')[1].getContext('2d');
        return Array.from(g.getImageData(2,12*cell+3,1,1).data).slice(0,3);}""", [cell])
    assert outline == [255, 255, 255]
    assert pixel(page, 0, 12, 0, cell) != (255, 255, 255)
    emit(page, payload(frame_for(None, reading), sample_contents(None)))
    titles = page.locator('#debug .dbg-title').all_inner_texts()
    assert titles[0] == '1P 確定盤なし' and titles[1] == '1P 読取盤'
    assert titles[2] == '2P 確定盤なし' and titles[3] == '2P 読取なし'
    page.close()


def test_chain_and_hands_rows(browser: Browser) -> None:  # noqa: F811
    page = new_page(browser, ON)
    emit(page, payload(frame_for(blank()), sample_contents(None)))
    labels = page.locator('#debug .dbg-rows span:nth-child(3n+1)').all_inner_texts()
    assert labels == ['予想連鎖', '置ける手数', '予告おじゃま', '得点']
    emit(page, payload(frame_for(blank()), sample_contents(4)))
    text = ' '.join(page.locator('#debug .dbg-rows span').all_inner_texts())
    assert '7連鎖' in text and '3.4秒' in text and '4手' in text
    assert '連鎖中 4連鎖目' in page.locator('#debug .dbg-state').first.inner_text()
    page.close()


def frame_states(p1_state: str, p2_state: str, match_active: bool = True, confirmed: bool = True) -> dict:
    """状態だけを変えた frame (盤面は空盤)。"""
    board = blank()
    frame = frame_for(board if confirmed else None, board)
    frame['match_active'] = match_active
    frame['sides'][0]['state'], frame['sides'][1]['state'] = p1_state, p2_state
    frame['sides'][1]['confirmed_board'] = board if confirmed else None
    frame['sides'][1]['raw_cnn_board'] = board
    return frame


@pytest.mark.parametrize('p1,p2,match,confirmed,expected', [
    ('CHAIN', 'STABLE', True, True, ('連鎖中', '安定 (確定盤)')),
    ('OJAMA_FALL', 'TSUMO_FALL', True, True, ('おじゃま落下中', 'ツモ落下中')),
    ('STABLE', 'STABLE', True, False, ('認識不安定', '認識不安定')),
    ('GRAVITY_SETTLE', 'EFFECT', True, True, ('連鎖後の着地待ち', '演出中(全消し等)')),
    ('MENU', 'MENU', False, True, ('試合外', '試合外')),
    ('STABLE', 'STABLE', False, True, ('試合外', '試合外'))])
def test_state_is_shown_in_japanese(browser: Browser, p1, p2, match, confirmed, expected) -> None:  # noqa: F811
    page = new_page(browser, ON)
    emit(page, payload(frame_states(p1, p2, match, confirmed), sample_contents(None)))
    texts = page.locator('#debug .dbg-state').all_inner_texts()
    assert texts[0].startswith(expected[0]) and texts[1].startswith(expected[1]), texts
    page.close()


def test_chain_number_fallbacks(browser: Browser) -> None:  # noqa: F811
    """連鎖数: 画面の式の段 → 観測した式の段数 → 段不明、の順。値を新しく作らない。"""
    page = new_page(browser, ON)
    emit(page, payload(frame_states('CHAIN', 'STABLE'), sample_contents(3)))
    assert page.locator('#debug .dbg-state').first.inner_text() == '連鎖中 3連鎖目'
    emit(page, payload(frame_states('CHAIN', 'STABLE'), sample_contents(3, display_step_missing=True)))
    assert page.locator('#debug .dbg-state').first.inner_text() == '連鎖中 式3段観測'
    assert page.evaluate("PuyoDebug.statusOf({state:'CHAIN'},{},true)") == {'text': '連鎖中 段不明', 'kind': 'chain'}
    assert page.evaluate("PuyoDebug.statusOf({state:'CHAIN'},{chain_active:1,chain_n_steps:0},true).text") == '連鎖中 段不明'
    page.close()


def test_confirmed_and_reading_boards_are_both_shown(browser: Browser) -> None:  # noqa: F811
    """1P/2Pそれぞれ確定盤と読取盤の2枚、計4枚。マスの文字は読める大きさ (一辺18px以上)。"""
    page = new_page(browser, ON)
    emit(page, payload(frame_states('STABLE', 'STABLE'), sample_contents(None)))
    sizes = page.evaluate("[...document.querySelectorAll('#debug canvas')].map(c=>[c.width/6,c.getBoundingClientRect().width/6])")
    assert len(sizes) == 4 and all(cell >= 18 for cell, _ in sizes), sizes
    assert page.locator('#debug .dbg-title').all_inner_texts() == ['1P 確定盤', '1P 読取盤', '2P 確定盤', '2P 読取盤']
    page.close()


@pytest.mark.parametrize('layout,scale,band,font', list(product(['l', 'l-left'], [70, 80, 85], ['true', 'false'],
                                                                 ['small', 'large'])))
def test_debug_panel_stays_inside_column(browser: Browser, layout: str, scale: int, band: str, font: str) -> None:  # noqa: F811
    page = new_page(browser, f'{layout}&debug_panel=true&scale={scale}&band={band}&font={font}')
    emit(page, payload(frame_for(board_with_every_kind(), board_with_every_kind()), sample_contents(3)))
    box = page.locator('#debug').bounding_box()
    panel = page.locator('#panel').bounding_box()
    assert box['y'] + box['height'] <= panel['y'] + panel['height'] + 1, (layout, scale, band, font)
    assert box['x'] >= panel['x'] and box['x'] + box['width'] <= panel['x'] + panel['width'] + 1
    assert page.evaluate('()=>{const e=document.getElementById("debug");return e.scrollWidth<=e.clientWidth+1}')
    page.close()


def test_off_by_default_and_not_in_other_layouts(browser: Browser) -> None:  # noqa: F811
    for query in ('l', 'card', 'ribbon', 'minimal', 'data', 'card&debug_panel=true', 'minimal&debug_panel=true'):
        page = new_page(browser, query)
        assert page.locator('#debug').count() == 0, query
        assert page.evaluate('window.PuyoDebug.enabled') is False
        page.close()


def test_win_rate_screen_identical_with_and_without_added_field(browser: Browser) -> None:  # noqa: F811
    """追加欄の有無で、勝率・状態・主要表示(#panel の debug 以外)が完全に同じ。"""
    data = payload(frame_for(board_with_every_kind()), sample_contents(3))
    without = deepcopy(data)
    del without[DEBUG_PANEL_KEY]
    texts = []
    for body, query in ((without, 'l'), (data, 'l'), (data, ON)):
        page = new_page(browser, query)
        emit(page, body)
        page.evaluate("document.getElementById('debug')?.remove()")
        texts.append(page.evaluate("()=>[document.getElementById('p1').textContent,document.getElementById('p2').textContent,"
                                   "document.getElementById('state').textContent,document.getElementById('fill').style.width,"
                                   "document.getElementById('contents-main').innerText,document.getElementById('basis').textContent]"))
        page.close()
    assert texts[0] == texts[1]
    # ON のとき勝率欄の数値は同じ (連鎖の中身欄は debug 表示で隠れるが値は変えない)
    assert texts[0][:4] == texts[2][:4] and texts[0][5] == texts[2][5]


def test_missing_added_field_does_not_break(browser: Browser) -> None:  # noqa: F811
    page = new_page(browser, ON)
    data = payload(frame_for(board_with_every_kind()), None, with_panel=False)
    emit(page, data)
    assert '盤面の追加欄なし' in page.locator('#debug .dbg-note').inner_text()
    assert page.locator('#p1').inner_text() == '46'
    page.close()


def test_builder_output_renders(browser: Browser) -> None:  # noqa: F811
    """本番の組み立て関数が出す形をそのまま描画できる。"""
    page = new_page(browser, ON)
    data = payload(frame_for(board_with_every_kind()), sample_contents(2))
    assert data[DEBUG_PANEL_KEY] == build_debug_panel(frame_for(board_with_every_kind()))
    emit(page, data)
    assert page.locator('#debug canvas').count() == 4
    page.close()
