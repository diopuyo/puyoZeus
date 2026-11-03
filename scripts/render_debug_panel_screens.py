"""試験用パネル(debug_panel)の見た目確認: 記録済みの認識結果の盤面を使い、L字配置を描画して PNG に保存する。

盤面は --journal (evaluation-journal-*.pickle) の実認識結果。連鎖数・置ける手数などの B3 値は
表示確認用の値 (tests/test_b3_contents.py の feature_row を基にした見本) で、実測ではない。
ゲーム画像が無いため、背景は暗い灰色の見本枠 (L字配置のゲーム領域の位置) にする。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import pickle
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Page, sync_playwright

from scripts.render_overlay_design import ROOT, new_page, snapshot
from src.b3_contents import contents_from_row
from src.phase_j.live_debug_panel import DEBUG_PANEL_KEY, build_debug_panel
from src.record_redesign_observer import grid

OUTPUT = Path(_public_dev_text('<work>/debugpanel2'))
LAYOUT_QUERY = 'l&debug_panel=true'
GAME_PLACEHOLDER = ('body{background:#2b2f36}#game-frame{position:absolute;left:0;top:0;width:1536px;height:864px;'
                    'background:#3a4048;color:#7c8794;font:700 40px sans-serif;display:flex;'
                    'align-items:center;justify-content:center;border:2px dashed #7c8794;box-sizing:border-box}')


def load_journal(path: Path) -> list[Any]:
    """journal は pickle(bytes) の連なり。update 行の PipelineResult だけを返す。"""
    results: list[Any] = []
    with path.open('rb') as stream:
        while True:
            try:
                record = pickle.loads(pickle.load(stream))
            except EOFError:
                return results
            if record[0] == 'update' and record[1]:
                results.append(record[1][0])


CHAIN_MIN_PUYOS = 30
EARLY_RANGE = (8, 14)


def count(side: Any) -> int:
    """確定盤のぷよ数。確定盤が無ければ 0。"""
    return 0 if side.confirmed_board is None else int((side.confirmed_board._grid != 0).sum())


def find(results: list[Any], predicate: Any) -> Any:
    return next(result for result in results if predicate(result))


def index_of(results: list[Any], result: Any) -> int:
    return next(i for i, item in enumerate(results) if item is result)


def frame_from_result(result: Any, confirmed: bool = True, reading: bool = True, index: int = 0,
                      tweak: tuple[int, int, int] | None = None,
                      states: tuple[str | None, str | None] = (None, None),
                      match_active: bool = True) -> dict[str, Any]:
    """B3 観測 frame と同じ形 (sides[].confirmed_board / raw_cnn_board) を認識結果から作る。"""
    sides = []
    for key, forced in zip(('p1', 'p2'), states):
        side = getattr(result, key)
        cells = grid(side.confirmed_board)
        shown = [row[:] for row in cells] if cells else None
        if shown and tweak and key == 'p1':
            shown[tweak[0]][tweak[1]] = tweak[2]  # 読取盤だけ1マス違う見本
        sides.append(dict(confirmed_board=cells if confirmed else None, raw_cnn_board=shown if reading else None,
                          state=forced or side.state.name, board_provenance=side.board_provenance, score=side.score,
                          tsumo_count=index, formula=None))
    return dict(frame=index, t_sec=index / 30, match_active=match_active, sides=sides)


def payload(frame: dict[str, Any], contents: dict[str, Any] | None, time_ms: int = 30000,
            with_panel: bool = True) -> dict[str, Any]:
    """公開 payload の見本: 既存の見本 snapshot + b3_contents + 試験用の追加欄。"""
    data = snapshot(.46, time_ms)
    data['evaluations']['practical'].update(source='exchange', calibration_id='b3-stage2-display')
    data['input'].update(recognition_quality=dict(status='trusted', reason_codes=[]))
    if contents is not None:
        data['evaluations']['b3_contents'] = contents
    if with_panel:
        data[DEBUG_PANEL_KEY] = build_debug_panel(frame)
    return data


def sample_contents(chain_step: int | None, display_step_missing: bool = False) -> dict[str, Any]:
    """見本: 1P が連鎖中なら、受け側の 2P が置ける手数を持つ (B3 の向きの規則どおり)。"""
    from tests.test_b3_contents import feature_row
    row = dict(feature_row(), ta_H_med_p1=None, ta_H_med_p2=None)
    if chain_step is not None:
        row.update(chain_active_p1=1., chain_n_steps_p1=float(chain_step), chain_rem_sec_p1=3.4,
                   chain_pred_l_p1=7., ta_H_med_p1=4., ta_H_med_p2=None)
    else:
        row.update(chain_active_p1=0., chain_pred_l_p1=None, chain_rem_sec_p1=None)
    contents = contents_from_row(row)
    contents['p1']['chain_display_step'] = None if display_step_missing else chain_step
    contents['p2']['chain_display_step'] = None
    return contents


def shoot(browser: Browser, name: str, data: dict[str, Any], output: Path, query: str = LAYOUT_QUERY) -> Path:
    page: Page = new_page(browser, query)
    page.add_style_tag(content=GAME_PLACEHOLDER)
    page.evaluate("document.body.insertAdjacentHTML('afterbegin','<div id=game-frame>ゲーム画面の位置</div>')")
    page.evaluate('data=>window.testStream.emit(data)', data)
    page.wait_for_timeout(300)
    target = output / f'{name}.png'
    page.screenshot(path=str(target))
    page.close()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_journal(args.journal)
    chain = find(results, lambda r: r.p1.state.name == 'CHAIN' and count(r.p1) >= CHAIN_MIN_PUYOS and count(r.p2) > 0)
    full = max((r for r in results if r.p1.state.name == r.p2.state.name == 'STABLE' and count(r.p2) > 0),
               key=lambda r: count(r.p1))
    early = find(results, lambda r: EARLY_RANGE[0] <= count(r.p1) <= EARLY_RANGE[1] and count(r.p2) > 0)
    ojama = find(results, lambda r: 'OJAMA_FALL' in (r.p1.state.name, r.p2.state.name)
                 and count(r.p1) > 0 and count(r.p2) > 0)
    tsumo = find(results, lambda r: r.p1.state.name == 'TSUMO_FALL' and count(r.p1) >= CHAIN_MIN_PUYOS and count(r.p2) > 0)
    menu = find(results, lambda r: r.p1.state.name == 'MENU' and r.p2.state.name == 'MENU')
    scenes = {'01_chain_midgame': (chain, sample_contents(3), dict(tweak=(12, 5, 2))),
              '02_chain_step_missing': (chain, sample_contents(3, display_step_missing=True), {}),
              '03_ojama_fall': (ojama, sample_contents(None), {}),
              '04_tsumo_fall': (tsumo, sample_contents(None), {}),
              '05_full_board_stable': (full, sample_contents(None), {}),
              '06_unconfirmed_reading_only': (chain, sample_contents(2), dict(confirmed=False)),
              '07_out_of_match': (menu, sample_contents(None), dict(match_active=False)),
              '08_early_game': (early, sample_contents(None), {})}
    with sync_playwright() as runner:
        browser = runner.chromium.launch(headless=True)
        for name, (result, contents, options) in scenes.items():
            print(shoot(browser, name, payload(frame_from_result(result, index=index_of(results, result), **options),
                                               contents), args.output))
        hold = payload(frame_from_result(early), None, with_panel=False)
        hold['display'].update(input_status='verifying', visibility='visible', status='hold', message='入力確認中')
        print(shoot(browser, '09_input_verifying_no_panel_field', hold, args.output))
        print(shoot(browser, '10_panel_off_same_payload', payload(frame_from_result(chain), sample_contents(3)),
                    args.output, query='l'))
        browser.close()
    print(shoot_over_http(args.output, chain))


def shoot_over_http(output: Path, result: Any) -> Path:
    """本物の LivePublisher → SSE → ブラウザの経路で、overlay 無指定の既定だけで描く (ポート自動割当・映像入力なし)。"""
    import time
    from src.phase_j.live_publish import LivePublisher
    frame = frame_from_result(result, tweak=(12, 5, 2))
    row = dict(frame=1, t_sec=1.0, game=0, captured_at=time.perf_counter(), recognized_at=time.perf_counter(),
               evaluated_at=time.perf_counter(), queue_depth=0, raw_probability=.46, probability=.46,
               advantage=-8.0, source='exchange', capture_gap=False, digest='sha256:test', hold=False,
               calibration_id='b3-stage2-display', b3_contents=sample_contents(3),
               **{DEBUG_PANEL_KEY: build_debug_panel(frame)})
    assets = {name: 'test' for name in ('app_build_id', 'recognition_model_hash', 'recognition_config_hash',
                                        'prediction_model_hash', 'calibration_hash')}
    publisher = LivePublisher(assets, '127.0.0.1', 0)  # overlay 無指定 = 既定 (L字 + デバッグ表示)
    publisher.start()
    try:
        publisher.offer(row)
        port = publisher.server.address()[1]
        with sync_playwright() as runner:
            browser = runner.chromium.launch(headless=True)
            page = browser.new_page(viewport=dict(width=1920, height=1080), locale='ja-JP')
            page.goto(f'http://127.0.0.1:{port}/')
            page.add_style_tag(content=GAME_PLACEHOLDER)
            page.wait_for_function("document.querySelectorAll('#debug canvas').length===4 && "
                                   "document.querySelector('#debug .dbg-note').textContent.includes('フレーム')",
                                   timeout=15000)
            page.wait_for_timeout(1500)  # 2Hz公開の続報を受けても崩れないことを見る
            target = output / '11_e2e_http_sse.png'
            page.screenshot(path=str(target))
            browser.close()
        return target
    finally:
        publisher.close()


if __name__ == '__main__':
    main()
