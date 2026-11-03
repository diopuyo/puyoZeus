"""video_30 の配信出力 × レビュー動画の突き合わせ台: 時刻の対応・数値・欠けの数え方・台自体の健全性。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts import compare_live_video30_review as c
from scripts import live_x_video30_spec as spec

FPS = spec.VIDEO_FPS


def steps(count: int, seed: int, every: int = 40) -> np.ndarray:
    """every コマごとにレベルが変わる階段状の確率列 (連鎖などの事象で値が跳ぶ様子の代用)。"""
    levels = np.random.RandomState(seed).uniform(.05, .95, count//every+1)
    return np.repeat(levels, every)[:count]


def review_rows(first: int = spec.REVIEW_FIRST_FRAME, count: int = spec.REVIEW_FRAME_COUNT) -> c.ReviewRows:
    frames = np.arange(first, first+count, dtype=np.int64)
    return c.ReviewRows(frames, {name: steps(count, seed) for seed, name in enumerate(c.REVIEW_COLUMNS)})


def write_csv(path: Path, review: c.ReviewRows, bom: bool = True) -> None:
    with path.open('w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', *c.REVIEW_COLUMNS, 'base_source'])
        for i, frame in enumerate(review.frames):
            writer.writerow([frame, *(review.values[name][i] for name in c.REVIEW_COLUMNS), 'exchange'])


def live_rows(condition: str, frames: np.ndarray, stamps: np.ndarray, b3: np.ndarray, g: np.ndarray | None = None) -> c.LiveRows:
    return c.LiveRows(condition, frames, stamps, dict(b3=b3, g=b3 if g is None else g), np.zeros(len(frames), bool))


# --- 読み込みと物差しの健全性 ---------------------------------------------------------------

def test_to_probability_treats_none_and_nonfinite_as_missing() -> None:
    assert c.to_probability(.25) == .25 and c.to_probability(1) == 1.
    assert all(np.isnan(c.to_probability(v)) for v in (None, float('nan'), float('inf'), 'x'))


def test_load_review_reads_bom_csv_and_checks_sha(tmp_path: Path) -> None:
    review = review_rows(count=10)
    path = tmp_path/'r.csv'
    write_csv(path, review)
    loaded = c.load_review(path, c.file_sha256(path))
    assert list(loaded.frames) == list(review.frames)
    assert np.allclose(loaded.values['B3_M3_p1'], review.values['B3_M3_p1'])
    with pytest.raises(ValueError, match='SHA256'):
        c.load_review(path, '0'*64)


def test_load_review_rejects_gaps_and_out_of_range(tmp_path: Path) -> None:
    review = review_rows(count=10)
    gap = c.ReviewRows(np.delete(review.frames, 4), {k: np.delete(v, 4) for k, v in review.values.items()})
    write_csv(tmp_path/'gap.csv', gap)
    with pytest.raises(ValueError, match='連続'):
        c.load_review(tmp_path/'gap.csv', None)
    loud = c.ReviewRows(review.frames, dict(review.values, B3_M3_p1=review.values['B3_M3_p1']+1.))
    write_csv(tmp_path/'loud.csv', loud)
    with pytest.raises(ValueError, match='確率'):
        c.load_review(tmp_path/'loud.csv', None)


def test_groups_cover_two_matches_with_verification_counts() -> None:
    groups = c.groups_for(review_rows())
    assert [(g.name, g.last-g.first+1) for g in groups] == [('match1', 2021), ('match2', 2761), ('range', 6000)]
    with pytest.raises(ValueError, match='範囲'):
        c.groups_for(review_rows(count=100))


# --- 時刻の対応 ---------------------------------------------------------------------------------

def test_zoh_lookup_returns_value_on_screen_and_its_age() -> None:
    shown, age = c.zoh_lookup(np.array([1., 2., 3.]), np.array([.1, .2, .3]), np.array([.5, 1., 1.5, 2.999, 3., 10.]))
    assert np.isnan(shown[0]) and np.isnan(age[0])
    assert list(shown[1:]) == [.1, .1, .2, .3, .3] and age[3] == pytest.approx(.999) and age[5] == 7.


def test_zoh_tolerates_float_noise_in_frame_times() -> None:
    noisy = [f for f in range(2000) if f/FPS+1/FPS != (f+1)/FPS]
    assert noisy, '浮動小数の丸めが起きる例が無い (許容の根拠が消えた)'
    f = noisy[0]
    shown, _ = c.zoh_lookup(np.array([f/FPS, (f+1)/FPS]), np.array([.1, .9]), np.array([f/FPS+1/FPS]))
    assert shown[0] == .9    # 許容が無いと 1 コマ前の .1 を拾う


def test_frame_lookup_marks_hits_only_on_exact_frames() -> None:
    shown, hit = c.frame_lookup(np.array([10, 12, 14]), np.array([.1, .2, .3]), np.array([9, 10, 11, 14, 15]))
    assert list(hit) == [False, True, False, True, False]
    assert shown[1] == .1 and shown[3] == .3 and np.isnan(shown[[0, 2, 4]]).all()


def test_pick_rows_last_or_first_per_frame() -> None:
    rows = [dict(frame=1, probability=.1), dict(frame=1, probability=.2), dict(frame=2, probability=.3)]
    assert [r['probability'] for r in c.pick_rows(rows, 'last').values()] == [.2, .3]
    assert [r['probability'] for r in c.pick_rows(rows, 'first').values()] == [.1, .3]


def test_frame_condition_rows_put_rows_at_input_time_and_none_is_missing() -> None:
    rows = [dict(frame=120, probability=.4, v49f=dict(p1=.6), hold=False),
            dict(frame=122, probability=None, v49f=None, hold=True)]
    live = c.frame_condition_rows(rows, 'last')
    assert live.condition == 'F' and list(live.stamps) == [2., 122/FPS]
    assert live.values['b3'][0] == .4 and np.isnan(live.values['b3'][1]) and np.isnan(live.values['g'][1])
    assert list(live.hold) == [False, True]


def wall_fixture(origin: float = 1000.) -> tuple[list[dict], list[dict]]:
    evaluations = [dict(frame=f, t_sec=f/FPS, captured_at=origin+f/FPS, evaluated_at=origin+f/FPS+.05,
                        v49f=dict(p1=.1*(i+1)), probability=.5) for i, f in enumerate((600, 602, 604))]
    publications = [dict(frame=604, published_at=origin+604/FPS+.3, evaluated_at=origin+604/FPS+.05,
                         probability=.7, hold=False),
                    dict(frame=600, published_at=origin+600/FPS+.2, evaluated_at=origin+600/FPS+.05,
                         probability=.6, hold=True),
                    dict(frame=999, published_at=origin+20., evaluated_at=1., probability=None, hold=False)]
    return evaluations, publications


def test_wall_condition_maps_publication_clock_to_video_seconds() -> None:
    evaluations, publications = wall_fixture()
    live = c.wall_condition_rows(evaluations, publications)
    # 公開の壁時計 − (captured_at − t_sec) = 映像秒。公開順 (時刻昇順) に並べ直す。
    assert list(live.frames) == [600, 604, 999]
    assert live.stamps[0] == pytest.approx(600/FPS+.2) and live.stamps[1] == pytest.approx(604/FPS+.3)
    assert live.stamps[2] == pytest.approx(20.)
    assert live.values['b3'][0] == .6 and live.values['g'][0] == pytest.approx(.1) and live.values['g'][1] == pytest.approx(.3)
    assert np.isnan(live.values['b3'][2]) and np.isnan(live.values['g'][2])     # 評価行に結合できない公開は G 欠測
    assert list(live.hold) == [True, False, False]


def test_wall_condition_refuses_non_realtime_origin() -> None:
    evaluations, publications = wall_fixture()
    evaluations[1]['captured_at'] += 1.
    with pytest.raises(ValueError, match='一定でない'):
        c.wall_condition_rows(evaluations, publications)


# --- 数値 -------------------------------------------------------------------------------------

def test_abs_stats_with_denominators() -> None:
    stats = c.abs_stats(np.array([0., 1., 5., 6., 12., 30.]))
    assert stats['n'] == 6 and stats['median_pt'] == 5.5 and stats['max_pt'] == 30.
    assert stats['over_5pt'] == dict(count=3, n=6, ratio=.5) and stats['over_10pt'] == dict(count=2, n=6, ratio=2/6)
    empty = c.abs_stats(np.array([]))
    assert empty['n'] == 0 and empty['median_pt'] is None and empty['over_5pt']['ratio'] is None


def f_fixture() -> tuple[c.ReviewRows, c.LiveRows, c.FrameGrid]:
    """レビュー 100〜139 コマ (130 以降は .75)。配信は偶数コマだけ: 110 は行なし・120 は値なし・130 以降は 1.0。"""
    frames = np.arange(100, 140, dtype=np.int64)
    review = np.where(frames >= 130, .75, .5)
    sampled = np.array([f for f in range(100, 140, 2) if f != 110], dtype=np.int64)
    live_b3 = np.where(sampled >= 130, 1.0, .5)
    live_b3[list(sampled).index(120)] = np.nan
    return (c.ReviewRows(frames, {name: review for name in c.REVIEW_COLUMNS}),
            live_rows('F', sampled, sampled/FPS, live_b3), c.FrameGrid(100, 2))


def test_compare_group_f_separates_not_sampled_missing_and_none() -> None:
    review, live, grid = f_fixture()
    result = c.compare_group(review, live, c.VALUES[0], c.Group('t', 100, 139), grid)
    assert result['counts'] == dict(review_frames=40, sampled_frames=20, not_sampled=20, missing_no_row=1,
                                    missing_value_none=1, compared=18)
    stats = result['stats']
    assert stats['n'] == 18 and stats['max_pt'] == 25. and stats['over_10pt'] == dict(count=5, n=18, ratio=5/18)
    assert stats['median_pt'] == 0.
    assert result['end']['frame'] == 139 and result['end']['live'] == 1.0 and result['end']['diff_pt'] == 25.
    assert result['end']['live_age_sec'] == pytest.approx(1/FPS)       # 139 は見ないコマ。最後に見た 138 の値
    assert [(round(s['start_sec']*FPS), round(s['end_sec']*FPS), s['frames']) for s in result['worst_segments']] == [
        (130, 138, 5)]


def test_compare_group_l_counts_missing_and_stale() -> None:
    frames = np.arange(0, 600, dtype=np.int64)
    review = c.ReviewRows(frames, {name: np.full(600, .5) for name in c.REVIEW_COLUMNS})
    stamps = np.array([1., 2., 6.])          # 6 秒の公開まで 4 秒間は更新なし → 古い値
    live = live_rows('L', np.array([60, 120, 360]), stamps, np.array([.5, np.nan, .5]))
    result = c.compare_group(review, live, c.VALUES[0], c.Group('t', 0, 599), c.FrameGrid(0, 2))
    counts = result['counts']
    assert counts['missing_no_row'] == 60           # 最初の公開 (1 秒) より前
    assert counts['missing_value_none'] == 240      # 2〜6 秒は値なしの公開が出ている
    assert counts['compared'] == 600-60-240 == result['stats']['n']
    assert counts['stale_frames'] == 358            # 2 秒の公開は 3 秒超 (179 コマ)、6 秒の公開は 7 秒超 (179 コマ) が古い
    assert result['stats']['max_pt'] == 0.


def test_estimate_lag_recovers_known_delay_despite_float_noise() -> None:
    frames = np.arange(0, 900, dtype=np.int64)
    truth = steps(900, 7, every=25)
    delay = 15      # 0.25 秒
    live = live_rows('L', frames, (frames+delay)/FPS, truth)   # query の f/FPS+k/FPS とは丸めが違う
    lag = c.estimate_lag(frames/FPS, truth, live, 'b3')
    assert lag['best_shift_sec'] == pytest.approx(delay/FPS) and lag['mean_abs_at_best_pt'] == pytest.approx(0.)
    assert lag['mean_abs_at_zero_pt'] > lag['mean_abs_at_best_pt']


def test_estimate_lag_without_overlap_reports_none() -> None:
    frames = np.arange(0, 100, dtype=np.int64)
    live = live_rows('L', np.array([0]), np.array([500.]), np.array([.5]))
    lag = c.estimate_lag(frames/FPS, np.full(100, .5), live, 'b3')
    assert lag['best_shift_sec'] is None and lag['n_at_best'] == 0


def test_top_segments_merge_within_gap_and_rank_by_area() -> None:
    seconds = np.arange(600)/FPS
    diff = np.zeros(600)
    diff[10:15] = 20.        # A
    diff[20:23] = 10.        # A に併合 (間 5 コマ < 0.5 秒)
    diff[100:141] = 12.      # B (長いが小さい)
    diff[300:302] = 50.      # C (短いが大きい)
    diff[400:410] = 9.9      # 線未満
    found = c.top_segments(seconds, diff)
    assert [round(s['start_sec']*FPS) for s in found] == [100, 10, 300]
    assert found[1]['frames'] == 8 and found[1]['peak_pt'] == 20. and found[1]['area_pt_sec'] == pytest.approx((100+30)/FPS)
    assert found[0]['area_pt_sec'] == pytest.approx(41*12/FPS)


def test_top_segments_limits_to_ten_and_ignores_nan() -> None:
    seconds = np.arange(3000)/FPS
    diff = np.full(3000, np.nan)
    for i in range(12):
        diff[i*200:i*200+3] = 11.+i
    assert len(c.top_segments(seconds, diff)) == c.TOP_SEGMENTS == 10
    assert c.top_segments(seconds, np.full(3000, np.nan)) == []


# --- 報告 -------------------------------------------------------------------------------------

def aligned_live(review: c.ReviewRows, grid: c.FrameGrid, condition: str = 'F') -> c.LiveRows:
    on_grid = grid.contains(review.frames)
    frames = review.frames[on_grid]
    return live_rows(condition, frames, frames/FPS, review.values['B3_M3_p1'][on_grid], review.values['G_p1'][on_grid])


def test_build_report_labels_g_as_other_ruler_and_is_json() -> None:
    review, grid = review_rows(), c.FrameGrid(10500, 2)
    report = c.build_report(review, aligned_live(review, grid), grid)
    assert report['values']['b3']['same_ruler'] is True and report['values']['g']['same_ruler'] is False
    assert set(report['values']['b3']['groups']) == {'match1', 'match2', 'range'}
    assert report['live']['rows_off_grid'] == 0 and report['grid'] == dict(first_frame=10500, stride=2)
    assert report['ruler']['g']['same_ruler'] is False and report['ruler']['b3']['differences']
    json.dumps(report)
    assert 'match1' in c.format_summary(dict(report, lag_baseline=c.lag_baseline(review, grid)))


def test_build_report_g_column_choice_changes_the_reference() -> None:
    review, grid = review_rows(), c.FrameGrid(10500, 2)
    live = aligned_live(review, grid)
    default = c.build_report(review, live, grid)['values']['g']['groups']['match1']['stats']['max_pt']
    other = c.build_report(review, live, grid, 'v416_p1')['values']['g']
    assert default == 0. and other['review_column'] == 'v416_p1' and other['groups']['match1']['stats']['max_pt'] > 0.


def test_live_off_grid_rows_are_counted() -> None:
    review, grid = review_rows(), c.FrameGrid(10501, 2)       # 格子の取り違え (本当は偶数コマ)
    summary = c.live_summary(aligned_live(review, c.FrameGrid(10500, 2)), grid)
    assert summary['rows_off_grid'] == summary['rows'] > 0


def test_selftest_passes_and_can_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    review = review_rows()
    result = c.selftest(review)
    assert result['passed'] and all(result['checks'].values()) and len(result['lag_baseline']) == len(c.BASELINE_LAGS_SEC)
    # 遅れの推定が壊れた台は、既知のずらしを検出できず落ちる (台は既知の失敗を再現できること)
    monkeypatch.setattr(c, 'estimate_lag', lambda *a, **k: dict(best_shift_sec=0., mean_abs_at_best_pt=0.,
                                                                  mean_abs_at_zero_pt=0., n_at_best=1))
    assert not c.selftest(review)['passed']


def test_lag_baseline_grows_with_lag() -> None:
    review, grid = review_rows(), c.FrameGrid(10500, 2)
    table = c.lag_baseline(review, grid)
    p95 = [row['groups']['match1']['p95_pt'] for row in table]
    assert [row['lag_sec'] for row in table] == list(c.BASELINE_LAGS_SEC)
    assert p95[-1] > p95[0] and table[0]['groups']['match1']['n'] == 2021     # 階段の位置で単調とは限らない


# --- 入出力 -----------------------------------------------------------------------------------

def write_spool(root: Path, review: c.ReviewRows, delay_sec: float = 0.) -> None:
    """配信ソフトの出力 (spool) を模す。レビューと同じ値を偶数コマに出し、公開は delay_sec 遅らせる。"""
    spool = root/'output'/'spool'
    spool.mkdir(parents=True)
    origin = 5000.
    evaluations = []
    for f in range(10500, 17040, 2):
        known = f >= spec.REVIEW_FIRST_FRAME
        evaluations.append(dict(
            frame=f, t_sec=f/FPS, captured_at=origin+f/FPS, evaluated_at=origin+f/FPS+.01, hold=False,
            probability=float(review.values['B3_M3_p1'][f-spec.REVIEW_FIRST_FRAME]) if known else .5,
            v49f=dict(p1=float(review.values['G_p1'][f-spec.REVIEW_FIRST_FRAME]) if known else .5)))
    with (spool/'evaluations.pickle').open('wb') as stream:
        for row in evaluations:
            pickle.dump(row, stream)
    with (spool/'publications.pickle').open('wb') as stream:
        for row in evaluations:
            pickle.dump(dict(frame=row['frame'], published_at=row['captured_at']+delay_sec,
                             evaluated_at=row['evaluated_at'], probability=row['probability'], hold=False), stream)


@pytest.mark.parametrize('condition,delay', [('F', 0.), ('L', .25)])
def test_main_compare_end_to_end_on_synthetic_spool(condition: str, delay: float, tmp_path: Path,
                                                     capsys: pytest.CaptureFixture) -> None:
    review = review_rows()
    write_csv(tmp_path/'review.csv', review)
    write_spool(tmp_path/'run', review, delay)
    out = tmp_path/'report.json'
    code = c.main(['compare', '--run', str(tmp_path/'run'), '--condition', condition,
                   '--review-csv', str(tmp_path/'review.csv'), '--out', str(out)])
    report = json.loads(out.read_text(encoding='utf-8'))
    assert code == 0 and 'match2' in capsys.readouterr().out
    b3 = report['values']['b3']['groups']
    assert report['live']['unique_frames'] == 3270 and len(report['lag_baseline']) == len(c.BASELINE_LAGS_SEC)
    if condition == 'F':
        assert b3['match1']['stats']['max_pt'] == 0. and b3['match1']['counts']['not_sampled'] > 0
    else:
        assert all(abs(b3[m]['lag']['best_shift_sec']-delay) <= 2/FPS for m in ('match1', 'match2'))


def test_grid_for_prefers_arguments_then_receipt_then_spec(tmp_path: Path) -> None:
    assert c.grid_for(tmp_path, None, None) == c.FrameGrid(10500, 2)
    (tmp_path/'receipt.json').write_text(json.dumps(dict(workload=dict(first_frame=10502, stride=4))), encoding='utf-8')
    assert c.grid_for(tmp_path, None, None) == c.FrameGrid(10502, 4)
    assert c.grid_for(tmp_path, 10500, None) == c.FrameGrid(10500, 4)


def test_resolve_output_accepts_arm_folder_or_output(tmp_path: Path) -> None:
    (tmp_path/'output'/'spool').mkdir(parents=True)
    (tmp_path/'output'/c.EVALUATIONS).write_bytes(b'')
    assert c.resolve_output(tmp_path) == tmp_path/'output' and c.resolve_output(tmp_path/'output') == tmp_path/'output'


@pytest.mark.skipif(not spec.REVIEW_CSV.exists(), reason='レビュー成果物が無い環境')
def test_real_review_csv_is_pinned_and_selftest_passes() -> None:
    review = c.load_review()          # SHA256・連続・確率の範囲を確かめる
    assert len(review.frames) == spec.REVIEW_FRAME_COUNT and int(review.frames[0]) == spec.REVIEW_FIRST_FRAME
    assert c.selftest(review)['passed']
    assert [g.last-g.first+1 for g in c.groups_for(review)[:2]] == [2021, 2761]
