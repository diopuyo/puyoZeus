"""video_30 の配信ソフト出力 (F/L 条件) を、レビュー動画の値と 1 コマずつ突き合わせる (読み取り専用)。

対応のさせ方:
- F (全コマ処理): コマ番号で対応させる。配信ソフトは 60fps を stride 2 で認識するので、レビューの奇数側のコマは
  「配信ソフトが見ないコマ」(not_sampled) として母数から分けて数える。1 コマに複数の評価行があれば --pick の行を使う。
- L (配信と同じ時間制約): 公開の壁時計を映像秒へ直し、各レビューコマの時刻に「その時刻に画面に出ている値」
  (直前の公開の値) を対応させる。壁時計→映像秒は captured_at−t_sec (実時間入力で一定) を原点にする。
出す数値は試合ごと・値ごと: 差の絶対値の中央値・P95・最大、5pt 超 / 10pt 超の割合 (母数つき)、遅れの推定、
差が大きい区間の上位 10、試合の終わりの値、配信側の欠け。比べる物差しの違い (RULER) も報告へ入れる。
selftest は実データのレビュー CSV 自身を「配信の出力」に見立て、差 0・遅れ 0 と、0.5 秒ずらした入力の遅れ 0.5 秒を
検出できること (台自体の健全性) を確かめる。流し直しの結果が無くても回せる。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import live_x_video30_spec as spec  # noqa: E402
from scripts.measure_live_x_timing import pickle_stream, run_origin  # noqa: E402

FPS = spec.VIDEO_FPS
POINTS = 100.                           # 確率 → pt
THRESHOLDS_PT = (5., 10.)               # 「差が大きい」コマの割合を出す線 (依頼の指定)
PERCENTILE = 95.
LAG_RANGE_FRAMES = (-60, 300)           # 遅れの探索範囲 (映像コマ、負 = 配信が先)。60fps で -1〜+5 秒
LAG_STEP_SEC = 1./FPS                   # 探索の刻み = 映像 1 コマ
TIME_EPS_SEC = 1e-9                     # 映像秒の比較の許容 (コマ/fps の浮動小数の丸め。1 コマ = 約 0.0167 秒よりずっと小さい)
LAG_MIN_COVERAGE = .5                   # 探索で比べられたコマが群の半分未満の遅れ候補は採らない
SEGMENT_THRESHOLD_PT = 10.              # 差が大きい区間の線
SEGMENT_GAP_SEC = .5                    # この秒数以内に続く超過コマは同じ区間にまとめる
TOP_SEGMENTS = 10
STALE_SEC = 1.                          # L: 直前の公開がこれより古ければ「古い値」として数える
PICKS = ('last', 'first')
CONDITIONS = ('F', 'L')
REVIEW_COLUMNS = ('B3_M3_p1', 'G_p1', 'v416_p1', 'B3_v6_base')
EVALUATIONS, PUBLICATIONS = 'spool/evaluations.pickle', 'spool/publications.pickle'
SELFTEST_SHIFT_SEC = .5
BASELINE_LAGS_SEC = (1./30, .1, .25, .5, 1.)   # 遅れの目安表の遅れ (配信 30Hz の 1 コマ・0.1・0.25・0.5・1 秒)
SELFTEST_LAG_TOLERANCE_SEC = LAG_STEP_SEC
READ_BLOCK = 1 << 20


@dataclass(frozen=True)
class ValueSpec:
    """配信の値とレビューの列の対応。same_ruler が False の対応は「ずれ」と呼ばず参考値として別扱いにする。"""

    key: str
    label: str
    review_column: str
    same_ruler: bool


VALUES = (ValueSpec('b3', '実戦勝率 B3 (M3)', 'B3_M3_p1', True),
          ValueSpec('g', '撃ち合い勝率 v4.9f (配信) vs G (レビュー)', 'G_p1', False))

# 比べる物差しの違い。調べた根拠は DECISIONS / 本台の報告に書く。値のずれと混同しないよう報告へ必ず載せる。
RULER = dict(
    b3=dict(same_ruler=True, same=[
        'モデル: models/b3/model.joblib SHA256 11f7b32a…・timing_table SHA256 1521db5c… (21c 配布 app = レビューの b3_config.json)',
        'M3 (295 列)・fast_ema=True・台帳フラグ・死亡根拠 production_R_F10・認識設定 (21c の recognition_config.json)'],
        differences=[
            '評価の同期: 配信=非同期 (async_enabled=True・completion_driven=True・結果は数コマ遅れて届く) / レビュー=同期',
            '時間解像度: 配信=stride 2 (30Hz) / レビュー=stride 1 (60Hz)。表示 EMA は通知ごとに進み、状態機械のコマ数定数は'
            ' 30fps 前提なので、実時間の効きが 2 倍違う',
            '公開: 配信 (L)=最新 1 件・完了駆動の公開 / レビュー=全コマの値']),
    g=dict(same_ruler=False, same=[
        '混ぜ元の B3: models/v49f/b3_v6.joblib SHA256 6113e42b…・表示関数 prefire_v49d.display・係数 c=0.4608258505895786'],
        differences=[
            'エンジン: 配信=LiveV49FEngine の PostV49F (v4.9f) / レビューの G=PostV416 (→PostV415→PostV412→PostV49G の受け連鎖'
            '完走後の証明) の値を同じ表示関数へ渡したもの。同じでないので差は「ずれ」でなく別物の差の可能性がある',
            'v416_p1 (レビューの 3 枠目) は配信に無い。比べない',
            'B3 側と同じ非同期・30Hz/60Hz の差を、さらに受ける']))


@dataclass(frozen=True)
class ReviewRows:
    """レビュー CSV。frames は連続、values は列名 → 確率 (0〜1、欠測 NaN)。"""

    frames: np.ndarray
    values: dict[str, np.ndarray]

    def seconds(self) -> np.ndarray:
        return self.frames/FPS


@dataclass(frozen=True)
class LiveRows:
    """配信側の行。stamps は映像秒で昇順、values は 'b3' / 'g' → 確率 (欠測 NaN)。"""

    condition: str
    frames: np.ndarray
    stamps: np.ndarray
    values: dict[str, np.ndarray]
    hold: np.ndarray


@dataclass(frozen=True)
class FrameGrid:
    """配信ソフトが認識するコマ (first から stride おき)。F の「見ないコマ」の判定に使う。"""

    first: int
    stride: int

    def contains(self, frames: np.ndarray) -> np.ndarray:
        return (frames >= self.first) & ((frames-self.first) % self.stride == 0)


@dataclass(frozen=True)
class Group:
    """突き合わせる範囲 (映像コマ、両端を含む)。"""

    name: str
    first: int
    last: int


def file_sha256(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def to_probability(value: Any) -> float:
    """None・非数は NaN (欠測)。"""
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else float('nan')


def load_review(path: Path = spec.REVIEW_CSV, expected_sha256: str | None = spec.REVIEW_CSV_SHA256) -> ReviewRows:
    """レビュー CSV を読む。SHA256・コマの連続・確率の範囲を先に確かめる (物差しの健全性)。"""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f'レビュー CSV の SHA256 が固定値と違う: {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    frames = np.array([int(row['frame']) for row in rows], dtype=np.int64)
    if not len(frames) or not np.array_equal(frames, np.arange(frames[0], frames[0]+len(frames))):
        raise ValueError('レビュー CSV のコマが連続でない')
    values = {name: np.array([to_probability(float(row[name]) if row[name] != '' else None) for row in rows])
              for name in REVIEW_COLUMNS}
    for name, column in values.items():
        finite = column[np.isfinite(column)]
        if not len(finite) or finite.min() < 0. or finite.max() > 1.:
            raise ValueError(f'レビュー CSV の {name} が 0〜1 の確率でない')
    return ReviewRows(frames, values)


def groups_for(review: ReviewRows) -> list[Group]:
    """試合 1・試合 2・レビュー区間全体。試合のコマ数が verification.json と違えば止める。"""
    found = []
    for match in spec.MATCHES:
        first, last = match.frames()
        if last-first+1 != spec.EXPECTED_MATCH_FRAMES[match.index]:
            raise ValueError(f'試合 {match.index} のコマ数が verification.json と違う')
        found.append(Group(f'match{match.index}', first, last))
    if found[0].first < review.frames[0] or found[-1].last > review.frames[-1]:
        raise ValueError('試合がレビュー CSV の範囲に入っていない')
    return found+[Group('range', int(review.frames[0]), int(review.frames[-1]))]


def pick_rows(evaluations: Iterable[dict], pick: str) -> dict[int, dict]:
    """1 コマに複数の評価行 (完了通知で出し直した行) があれば pick の行を使う。"""
    chosen: dict[int, dict] = {}
    for row in evaluations:
        if pick == 'last' or row['frame'] not in chosen:
            chosen[row['frame']] = row
    return chosen


def frame_condition_rows(evaluations: Iterable[dict], pick: str) -> LiveRows:
    """F: 評価行を入力時刻 (コマ/fps) へ置く。値は B3 の表示確率と v4.9f の p1。"""
    chosen = pick_rows(evaluations, pick)
    frames = np.array(sorted(chosen), dtype=np.int64)
    rows = [chosen[f] for f in frames]
    return LiveRows('F', frames, frames/FPS,
                    dict(b3=np.array([to_probability(r.get('probability')) for r in rows]),
                         g=np.array([to_probability((r.get('v49f') or {}).get('p1')) for r in rows])),
                    np.array([bool(r.get('hold')) for r in rows], dtype=bool))


def wall_condition_rows(evaluations: list[dict], publications: Iterable[dict]) -> LiveRows:
    """L: 公開の壁時計を映像秒へ直す。G は公開票を評価行 (frame, evaluated_at) へ結合して得る。"""
    origin = run_origin(evaluations)
    index = {(r['frame'], r['evaluated_at']): r for r in evaluations}
    shown = sorted(publications, key=lambda p: p['published_at'])
    joined = [index.get((p['frame'], p['evaluated_at'])) or {} for p in shown]
    return LiveRows('L', np.array([p['frame'] for p in shown], dtype=np.int64),
                    np.array([p['published_at']-origin for p in shown]),
                    dict(b3=np.array([to_probability(p.get('probability')) for p in shown]),
                         g=np.array([to_probability((j.get('v49f') or {}).get('p1')) for j in joined])),
                    np.array([bool(p.get('hold')) for p in shown], dtype=bool))


def load_live(output: Path, condition: str, pick: str) -> LiveRows:
    """配信ソフトの出力 (spool/evaluations.pickle・publications.pickle) から行を作る。"""
    evaluations = list(pickle_stream(output/EVALUATIONS))
    if not evaluations:
        raise ValueError(f'評価行が無い: {output/EVALUATIONS}')
    if condition == 'F':
        return frame_condition_rows(evaluations, pick)
    return wall_condition_rows(evaluations, pickle_stream(output/PUBLICATIONS))


def zoh_lookup(stamps: np.ndarray, values: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各時刻に表示されている値 = その時刻以前で最後の行の値。先頭より前は NaN。age は経過秒 (無ければ NaN)。"""
    index = np.searchsorted(stamps, queries+TIME_EPS_SEC, side='right')-1
    found = index >= 0
    shown, age = np.full(queries.shape, np.nan), np.full(queries.shape, np.nan)
    shown[found] = values[index[found]]
    age[found] = queries[found]-stamps[index[found]]
    return shown, age


def frame_lookup(frames: np.ndarray, values: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """コマ番号が一致する行の値と、一致した印。frames は昇順・重複なし。"""
    index = np.minimum(np.searchsorted(frames, queries), len(frames)-1)
    hit = frames[index] == queries
    return np.where(hit, values[index], np.nan), hit


def abs_stats(diff_pt: np.ndarray) -> dict[str, Any]:
    """差の絶対値 (pt) の中央値・P95・最大と、各線を超えたコマの割合 (件数と母数つき)。"""
    count = int(len(diff_pt))
    if not count:
        return dict(n=0, median_pt=None, p95_pt=None, max_pt=None,
                    **{f'over_{int(t)}pt': dict(count=0, n=0, ratio=None) for t in THRESHOLDS_PT})
    return dict(n=count, median_pt=float(np.median(diff_pt)), p95_pt=float(np.percentile(diff_pt, PERCENTILE)),
                max_pt=float(diff_pt.max()),
                **{f'over_{int(t)}pt': dict(count=int((diff_pt > t).sum()), n=count,
                                            ratio=float((diff_pt > t).sum()/count)) for t in THRESHOLDS_PT})


def estimate_lag(seconds: np.ndarray, review: np.ndarray, live: LiveRows, key: str) -> dict[str, Any]:
    """配信の値を shift 秒だけ先へ読んだとき (配信が shift 秒遅れているとみなしたとき) 差が最小になる shift。"""
    shifts = np.arange(LAG_RANGE_FRAMES[0], LAG_RANGE_FRAMES[1]+1)/FPS   # 整数コマ/fps (0 が厳密に 0 になる)
    scores = []
    for shift in shifts:
        shown, _ = zoh_lookup(live.stamps, live.values[key], seconds+shift)
        shown[seconds+shift > live.stamps[-1]+TIME_EPS_SEC] = np.nan   # 配信の出力が尽きた先は比べない
        ok = np.isfinite(shown) & np.isfinite(review)
        scores.append((float(np.abs(shown[ok]-review[ok]).mean()*POINTS) if ok.sum() >= LAG_MIN_COVERAGE*len(seconds)
                       else float('inf'), int(ok.sum())))
    candidates = [(s[0], abs(shift), shift, s[1]) for s, shift in zip(scores, shifts) if np.isfinite(s[0])]
    zero = next((s[0] for s, shift in zip(scores, shifts) if shift == 0.), float('inf'))
    if not candidates:
        return dict(best_shift_sec=None, mean_abs_at_best_pt=None, mean_abs_at_zero_pt=None, n_at_best=0)
    best = min(candidates)
    return dict(best_shift_sec=float(best[2]), mean_abs_at_best_pt=best[0], n_at_best=best[3],
                mean_abs_at_zero_pt=zero if np.isfinite(zero) else None)


def top_segments(seconds: np.ndarray, diff_pt: np.ndarray) -> list[dict[str, Any]]:
    """差が SEGMENT_THRESHOLD_PT 以上のコマを、SEGMENT_GAP_SEC 以内の連なりでまとめ、面積 (pt・秒) の大きい順に上位。"""
    over = np.flatnonzero(np.nan_to_num(diff_pt, nan=0.) >= SEGMENT_THRESHOLD_PT)
    if not len(over):
        return []
    runs = np.split(over, np.flatnonzero(np.diff(seconds[over]) > SEGMENT_GAP_SEC)+1)
    found = [dict(start_sec=float(seconds[r[0]]), end_sec=float(seconds[r[-1]]), frames=int(len(r)),
                  peak_pt=float(diff_pt[r].max()), mean_pt=float(diff_pt[r].mean()),
                  area_pt_sec=float(diff_pt[r].sum()/FPS)) for r in runs]
    return sorted(found, key=lambda s: -s['area_pt_sec'])[:TOP_SEGMENTS]


def end_value(review: ReviewRows, live: LiveRows, key: str, column: str, group: Group) -> dict[str, Any]:
    """試合の最後のコマ時点の値。配信は「その時刻以前で最後の行」(F は最後に見たコマ、L は画面に出ている値)。"""
    at = np.array([group.last/FPS])
    shown, age = zoh_lookup(live.stamps, live.values[key], at)
    reference = float(review.values[column][group.last-review.frames[0]])
    return dict(frame=group.last, sec=float(at[0]), review=reference, live=to_probability(shown[0]),
                diff_pt=None if np.isnan(shown[0]) else float(abs(shown[0]-reference)*POINTS),
                live_age_sec=None if np.isnan(age[0]) else float(age[0]))


def pair_group(review: ReviewRows, live: LiveRows, key: str, column: str, group: Group,
               grid: FrameGrid) -> dict[str, Any]:
    """1 つの群の対応づけ。review/live (NaN は比べられない)・秒・内訳を返す。"""
    lo, hi = group.first-int(review.frames[0]), group.last-int(review.frames[0])+1
    frames, seconds, reference = review.frames[lo:hi], review.seconds()[lo:hi], review.values[column][lo:hi]
    if live.condition == 'F':
        shown, hit = frame_lookup(live.frames, live.values[key], frames)
        sampled = grid.contains(frames)
        counts = dict(review_frames=len(frames), sampled_frames=int(sampled.sum()),
                      not_sampled=int((~sampled).sum()), missing_no_row=int((sampled & ~hit).sum()),
                      missing_value_none=int((hit & np.isnan(shown)).sum()))
    else:
        shown, age = zoh_lookup(live.stamps, live.values[key], seconds)
        counts = dict(review_frames=len(frames), sampled_frames=len(frames), not_sampled=0,
                      missing_no_row=int(np.isnan(age).sum()),
                      missing_value_none=int((np.isfinite(age) & np.isnan(shown)).sum()),
                      stale_over_sec=STALE_SEC, stale_frames=int((age > STALE_SEC).sum()))
    both = np.isfinite(shown) & np.isfinite(reference)
    counts['compared'] = int(both.sum())
    return dict(seconds=seconds, review=reference, live=shown, both=both, counts=counts)


def compare_group(review: ReviewRows, live: LiveRows, value: ValueSpec, group: Group,
                  grid: FrameGrid) -> dict[str, Any]:
    pair = pair_group(review, live, value.key, value.review_column, group, grid)
    diff = np.where(pair['both'], np.abs(pair['live']-pair['review'])*POINTS, np.nan)
    return dict(group=group.name, frames=[group.first, group.last], counts=pair['counts'],
                stats=abs_stats(diff[pair['both']]),
                lag=estimate_lag(pair['seconds'], pair['review'], live, value.key),
                worst_segments=top_segments(pair['seconds'], diff),
                end=end_value(review, live, value.key, value.review_column, group))


def live_summary(live: LiveRows, grid: FrameGrid) -> dict[str, Any]:
    """配信側の行の母数。F で grid に乗らない行があれば、first_frame / stride の指定違いを疑う。"""
    on_grid = grid.contains(live.frames)
    return dict(condition=live.condition, rows=int(len(live.frames)), unique_frames=int(len(np.unique(live.frames))),
                first_sec=float(live.stamps[0]), last_sec=float(live.stamps[-1]),
                rows_off_grid=int((~on_grid).sum()) if live.condition == 'F' else None,
                hold_rows=int(live.hold.sum()),
                value_rows={key: int(np.isfinite(column).sum()) for key, column in live.values.items()})


def build_report(review: ReviewRows, live: LiveRows, grid: FrameGrid, g_column: str = 'G_p1') -> dict[str, Any]:
    """試合ごと・値ごとの突き合わせ報告。G は物差しが違うので same_ruler=False を付けて別枠にする。"""
    values = [ValueSpec(v.key, v.label, g_column if v.key == 'g' else v.review_column, v.same_ruler) for v in VALUES]
    return dict(condition=live.condition, grid=dict(first_frame=grid.first, stride=grid.stride),
                review=dict(frames=[int(review.frames[0]), int(review.frames[-1])], csv=str(spec.REVIEW_CSV)),
                live=live_summary(live, grid), ruler=RULER,
                values={v.key: dict(label=v.label, review_column=v.review_column, same_ruler=v.same_ruler,
                                    groups={g.name: compare_group(review, live, v, g, grid)
                                            for g in groups_for(review)}) for v in values})


def line(group: dict[str, Any]) -> str:
    stats, counts, lag, end = group['stats'], group['counts'], group['lag'], group['end']
    if not stats['n']:
        return f"  {group['group']}: 比べられたコマ 0 (欠け {counts['missing_no_row']+counts['missing_value_none']})"
    return (f"  {group['group']}: n={stats['n']}/{counts['review_frames']} 中央値 {stats['median_pt']:.2f} "
            f"P95 {stats['p95_pt']:.2f} 最大 {stats['max_pt']:.2f}pt / >5pt {stats['over_5pt']['ratio']:.1%}"
            f" >10pt {stats['over_10pt']['ratio']:.1%} / 遅れ {lag['best_shift_sec']}s"
            f" / 欠け {counts['missing_no_row']}+{counts['missing_value_none']} 見ない {counts['not_sampled']}"
            f" / 終わり 配信 {end['live']:.3f} レビュー {end['review']:.3f}")


def baseline_lines(table: list[dict]) -> list[str]:
    """遅れだけで出る差の目安 (B3・試合ごと)。実測の P95・最大がこれ以下なら、遅れだけで説明がつく。"""
    if not table:
        return []
    out = ['[目安] レビュー自身を遅らせたときの B3 の差 (pt)']
    for row in table:
        cells = ' / '.join(f"{name} P95 {g['p95_pt']:.2f} 最大 {g['max_pt']:.2f}"
                           for name, g in row['groups'].items() if name != 'range')
        out.append(f"  遅れ {row['lag_sec']:.3f}s: {cells}")
    return out


def format_summary(report: dict[str, Any]) -> str:
    """人が読む要約。詳細 (上位区間・内訳) は JSON に残す。"""
    out = [f"条件 {report['condition']} 配信の行 {report['live']['rows']} (コマ {report['live']['unique_frames']})"]
    for key, value in report['values'].items():
        out.append(f"[{key}] {value['label']} ({'同じ物差し' if value['same_ruler'] else '別の物差し・参考'})")
        out += [line(g) for g in value['groups'].values()]
    return '\n'.join(out)


def shifted_live(review: ReviewRows, shift_sec: float) -> LiveRows:
    """レビュー自身を shift_sec 遅らせた L 条件の出力に見立てる (全コマ・欠測なし)。"""
    return LiveRows('L', review.frames, review.seconds()+shift_sec,
                    dict(b3=review.values['B3_M3_p1'], g=review.values['G_p1']), np.zeros(len(review.frames), bool))


def lag_baseline(review: ReviewRows, grid: FrameGrid, lags: tuple[float, ...] = BASELINE_LAGS_SEC) -> list[dict]:
    """遅れだけで出る差の目安。レビュー自身を lag 秒遅らせて B3 を突き合わせる。実測の差が遅れだけで説明できるかの物差し。"""
    table = []
    for lag in lags:
        groups = build_report(review, shifted_live(review, lag), grid)['values']['b3']['groups']
        table.append(dict(lag_sec=lag, groups={name: dict(
            n=g['stats']['n'], median_pt=g['stats']['median_pt'], p95_pt=g['stats']['p95_pt'], max_pt=g['stats']['max_pt'],
            over_5pt=g['stats']['over_5pt']['ratio'], over_10pt=g['stats']['over_10pt']['ratio'])
            for name, g in groups.items()}))
    return table


def selftest(review: ReviewRows) -> dict[str, Any]:
    """レビュー自身を配信の出力に見立てて、差 0・遅れ 0 と、既知のずらしの検出を確かめる。"""
    grid = FrameGrid(spec.first_frame(), spec.stride_for())
    on_grid = grid.contains(review.frames)
    frames = review.frames[on_grid]
    same = LiveRows('F', frames, frames/FPS, dict(b3=review.values['B3_M3_p1'][on_grid],
                                                  g=review.values['G_p1'][on_grid]), np.zeros(len(frames), bool))
    exact, moved = build_report(review, same, grid), build_report(review, shifted_live(review, SELFTEST_SHIFT_SEC), grid)
    first, second = exact['values']['b3']['groups'], moved['values']['b3']['groups']
    checks = dict(
        identical_max_zero=all(g['stats']['max_pt'] == 0. for g in first.values()),
        identical_lag_zero=all(g['lag']['best_shift_sec'] == 0. for g in first.values()),
        stride_split=all(g['counts']['not_sampled'] > 0 and g['counts']['missing_no_row'] == 0 for g in first.values()),
        shifted_lag_detected=all(abs(g['lag']['best_shift_sec']-SELFTEST_SHIFT_SEC) <= SELFTEST_LAG_TOLERANCE_SEC
                                 for g in second.values() if g['group'] != 'range'),
        shifted_has_difference=all(g['stats']['max_pt'] > 0. for g in second.values()))
    return dict(passed=all(checks.values()), checks=checks, lag_baseline=lag_baseline(review, grid))


def resolve_output(run: Path) -> Path:
    """腕のフォルダ (…/F1_OFF) でも出力フォルダ (…/F1_OFF/output) でも受け取る。"""
    return run/'output' if (run/'output'/EVALUATIONS).exists() else run


def grid_for(run: Path, first_frame: int | None, stride: int | None) -> FrameGrid:
    """認識するコマの格子。指定が無ければ受領票 (receipt.json) の workload、無ければ spec の既定。"""
    receipt = run/'receipt.json'
    work = json.loads(receipt.read_text(encoding='utf-8')).get('workload', {}) if receipt.exists() else {}
    return FrameGrid(first_frame if first_frame is not None else work.get('first_frame', spec.first_frame()),
                     stride if stride is not None else work.get('stride', spec.stride_for()))


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selftest', help='レビュー自身を入力にして台の健全性を確かめる')
    compare = sub.add_parser('compare', help='配信ソフトの出力をレビューの値と突き合わせる')
    compare.add_argument('--run', type=Path, required=True, help='腕のフォルダ (F1_OFF など) か、その output')
    compare.add_argument('--condition', choices=CONDITIONS, required=True)
    compare.add_argument('--pick', choices=PICKS, default='last', help='F: 1 コマに複数の評価行があるときの選び方')
    compare.add_argument('--g-column', choices=REVIEW_COLUMNS, default='G_p1')
    compare.add_argument('--first-frame', type=int)
    compare.add_argument('--stride', type=int)
    compare.add_argument('--review-csv', type=Path, default=spec.REVIEW_CSV)
    compare.add_argument('--out', type=Path, help='報告 JSON の出力先')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == 'selftest':
        result = selftest(load_review())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['passed'] else 1
    sha = spec.REVIEW_CSV_SHA256 if args.review_csv == spec.REVIEW_CSV else None
    review, grid = load_review(args.review_csv, sha), grid_for(args.run, args.first_frame, args.stride)
    report = build_report(review, load_live(resolve_output(args.run), args.condition, args.pick), grid, args.g_column)
    report['lag_baseline'] = lag_baseline(review, grid)
    print(format_summary(report))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
