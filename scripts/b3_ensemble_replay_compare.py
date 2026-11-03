"""保存済みの B3 入力の行 (再生なし) で、large 1 本と ref 3 本の平均を比べる (読み取りのみ・原票は書き換えない)。

入力 (24d の採点 obs24d_score_20261025 と同じ原票):
  L  = 24d の配信の記録 (c2_on/output/...)          b3_time_gate.jsonl の result 行 (B3 の入力の行 501 キー)・spool の評価票と公開票
  F1 = 同じ録画を 30Hz 全コマで流した再生 (replay_F1)  同上 (録画 130〜730 秒だけ)
手順 (どの段も、飛ばさず数字で確かめてから次へ進む):
  1. 行の取り出し: result 行を (epoch, frame) で取り、評価票の judgment_id[:2] で結ぶ。結べない行 0 を確かめる。
  2. 測定器の健全性: 今の large で行を予測し直した値が、保存された raw_probability と 1 ビットも違わない (全行)。
  3. 表示の再構成: 評価票の raw_probability を新しい予測へ置き換え (由来が static / exchange の行だけ。死亡の由来は不変)、
     表示の平滑 (SwitchAwareDisplayEMA、同じ認識時刻の再評価は前の状態へ戻す) を同じ順で再生する。保存された表示値で検証する。
  4. 段 B と同じ時刻合わせ (stage_b_compare24d.py): 30Hz 基準コマ τ に「τ 以前で最後の公開」を比べる。
  5. 勝者と逆の向きにいた秒数: 勝者は終了の組 (end_events_C2.json) から。p = 1P の勝率で、勝者 1P なら p < 0.5、2P なら p > 0.5 が逆。
注意: 平滑の再生は physical_count (物理イベントの記録数) を保存物から復元できないので、由来切替のブレンド判定だけが近似。
保存された表示値との一致率を出し、モデルを替えた効果は「同じ再生での差」で見る。生の予測 (平滑なし) での比較も併記する。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import gzip
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np

HOME = 'C:/' if os.name == 'nt' else '/mnt/c/'
L_DIR = Path(HOME+'PuyoLive_OBS_20261025d_USER_ONLY/c2_on/output/20261025T115259_794377')
SCORE_DIR = Path(_public_dev_text('<work>/obs24d_score_20261025'))
F_DIR = SCORE_DIR/'replay_F1/output'
WORK_DIR = Path(_public_dev_text('<work>/b3_ensemble_20261025'))
LARGE_MODEL = Path(_public_dev_text('<work>/b3_c2_model_20261025/large_seed1003.joblib'))
REF_MODELS = tuple(WORK_DIR/'models'/f'ref_seed{seed}.joblib' for seed in (1003, 1004, 1005))
WINDOW = (130.0, 730.0)           # stage_b_compare24d.WINDOW (録画秒)
END_MATCH_MARGIN_SEC = 8.0        # 終了の組を試合へ結びつける範囲 [開始, 終了+8秒] (analysis_misc24d.py と同じ)
FPS = 30.0
POINTS = 100.0
MIN_PAIRS = 30
THRESHOLDS_PT = (5.0, 10.0)
MODEL_SOURCES = ('static', 'exchange')   # B3 の予測から来る表示の由来 (死亡の由来は予測に依らない)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 5000, 1
VARIANTS = ('large', 'ref_s1003', 'ref3_logit', 'ref3_mean_p')    # 比べる構成


def read_pickles(path: Path) -> list[Any]:
    out = []
    with path.open('rb') as stream:
        while True:
            try:
                out.append(pickle.load(stream))
            except EOFError:
                return out


def iter_results(gate: Path) -> Iterator[dict]:
    """b3_time_gate.jsonl の result 行 (B3 の入力の行を含む)。"""
    with gate.open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('{"monotonic"') and '"kind": "result"' in line[:80]:
                yield json.loads(line)


def load_run(directory: Path) -> SimpleNamespace:
    results = {(r['epoch'], r['frame']): r for r in iter_results(directory/'b3_time_gate.jsonl') if r['error'] is None}
    return SimpleNamespace(directory=directory, results=results,
                           evals=read_pickles(directory/'spool/evaluations.pickle'),
                           pubs=read_pickles(directory/'spool/publications.pickle') if (directory/'spool/publications.pickle').exists() else [])


def link_check(run: SimpleNamespace) -> dict[str, int]:
    """モデル由来の評価行が、judgment_id[:2] = (epoch, frame) の result 行へ全て結べ、生の確率が一致するか。"""
    linked = missing = mismatch = 0
    for row in run.evals:
        if row['source'] in MODEL_SOURCES:
            result = run.results.get(tuple(row['judgment_id'][:2]))
            missing += result is None
            if result is not None:
                linked += 1
                mismatch += result['raw_probability'] != row['raw_probability']
    return dict(model_source_evals=linked+missing, linked=linked, missing=missing, raw_mismatch=mismatch)


def member_predictions(run: SimpleNamespace, members: dict[str, Any]) -> dict[str, dict[tuple, float]]:
    """行ごと・メンバーごとの 1 行予測 (配信と同じ threadpool_limits(1)・predict_row)。"""
    from threadpoolctl import threadpool_limits
    out: dict[str, dict[tuple, float]] = {name: {} for name in members}
    with threadpool_limits(limits=1):
        for key, result in run.results.items():
            for name, model in members.items():
                out[name][key] = model.predict_row(result['row'])
    return out


def variant_predictions(singles: dict[str, dict[tuple, float]]) -> dict[str, dict[tuple, float]]:
    """メンバーの予測から比べる構成を作る (ref 3 本は src.b3_ensemble の平均関数。predict_row と同じ関数)。"""
    from src.b3_ensemble import AGG_MEAN_LOGIT, AGG_MEAN_P, aggregate_probabilities
    refs = [f'ref_s{seed}' for seed in (1003, 1004, 1005)]
    keys = list(singles['large'])
    stack = np.array([[singles[name][key] for key in keys] for name in refs])
    out = dict(large=singles['large'], ref_s1003=singles['ref_s1003'])
    for name, aggregate in (('ref3_logit', AGG_MEAN_LOGIT), ('ref3_mean_p', AGG_MEAN_P)):
        out[name] = dict(zip(keys, aggregate_probabilities(stack, aggregate).tolist()))
    return out


def reproduction(run: SimpleNamespace, large: dict[tuple, float]) -> dict[str, float]:
    """今の large で行を予測し直した値と、保存された raw_probability の差 (健全性。1 ビット違いは OS / libm の差)。"""
    diff = np.array([abs(large[key]-result['raw_probability']) for key, result in run.results.items()])
    return dict(rows=len(diff), bit_identical=int((diff == 0).sum()), over_1e12=int((diff > 1e-12).sum()), max_abs=float(diff.max()))


def row_difference_table(run: SimpleNamespace, predictions: dict[str, dict[tuple, float]]) -> dict[str, Any]:
    """同じ行での予測の差 (pt)。static / exchange 別・平均 / 中央値 / P95 / 最大・符号つきの平均 (新 − 旧)。"""
    modes = {key: int(result['mode']) for key, result in run.results.items()}
    table: dict[str, Any] = {}
    for new in ('ref_s1003', 'ref3_logit', 'ref3_mean_p'):
        for label, wanted in (('all', (0, 1)), ('static', (0,)), ('exchange', (1,))):
            keys = [k for k in modes if modes[k] in wanted]
            diff = np.array([predictions[new][k]-predictions['large'][k] for k in keys])*POINTS
            table[f'{new}_vs_large/{label}'] = describe_abs(diff)
    return table


def describe_abs(signed: np.ndarray) -> dict[str, float]:
    absolute = np.abs(signed)
    return dict(n=int(len(signed)), mean=float(absolute.mean()), median=float(np.median(absolute)),
                p95=float(np.percentile(absolute, 95)), max=float(absolute.max()), signed_mean=float(signed.mean()))


def raw_series(run: SimpleNamespace, predictions: dict[tuple, float] | None) -> list[float | None]:
    """評価行ごとの生の確率。predictions を渡すとモデル由来の行だけを置き換える (None = 保存どおり)。"""
    out = []
    for row in run.evals:
        raw = row['raw_probability']
        if predictions is not None and row['source'] in MODEL_SOURCES:
            raw = predictions[tuple(row['judgment_id'][:2])]
        out.append(raw)
    return out


def replay_shown(run: SimpleNamespace, raws: Sequence[float | None], fast_ema: bool = True) -> np.ndarray:
    """評価行ごとの表示確率 (平滑後)。B3Display.update と LiveB3._restore_display_tick と同じ手順の再生。"""
    from src.exchange_display_smoothing import SwitchAwareDisplayEMA
    smoother, game, tick, before = SwitchAwareDisplayEMA(fast_ema=fast_ema), None, None, None
    shown = np.empty(len(run.evals))
    for index, (row, raw) in enumerate(zip(run.evals, raws)):
        if row['game'] != game:
            smoother, game, tick = SwitchAwareDisplayEMA(fast_ema=fast_ema), row['game'], None
        if tick == row['t_sec']:
            smoother = deepcopy(before)
        else:
            tick, before = row['t_sec'], deepcopy(smoother)
        if raw is None:
            smoother, shown[index] = SwitchAwareDisplayEMA(fast_ema=fast_ema), 0.5
        else:
            tracker = SimpleNamespace(source=row['source'], _game_idx=row['game'], current=None, records=range(0))
            shown[index] = smoother.apply(tracker, (0.0, raw), row['t_sec'])[1]
    return shown


def replay_agreement(run: SimpleNamespace, shown: np.ndarray) -> dict[str, float]:
    """保存された表示値 (評価票の probability) との一致 (再生の精度)。"""
    logged = np.array([row['probability'] for row in run.evals])
    diff = np.abs(shown-logged)
    return dict(n=int(len(diff)), exact=int((diff == 0).sum()), over_1e9=int((diff > 1e-9).sum()), over_0p001=int((diff > 1e-3).sum()),
                mean_pt=float(diff.mean()*POINTS), max_pt=float(diff.max()*POINTS))


def frame_reference(run: SimpleNamespace, values: Sequence[float], window: tuple[float, float]) -> dict[str, np.ndarray]:
    """30Hz 基準コマ: 1 コマ複数行は最後の行 (stage_b_compare.load_reference と同じ)。保留 (hold) は NaN。"""
    chosen: dict[int, int] = {}
    for index, row in enumerate(run.evals):
        chosen[row['frame']] = index
    indices = [chosen[f] for f in sorted(chosen)]
    tau = np.array([run.evals[i]['t_sec'] for i in indices])
    p = np.array([np.nan if (run.evals[i]['probability'] is None or run.evals[i].get('hold')) else values[i] for i in indices])
    active = np.array([bool(run.evals[i].get('match_active')) for i in indices])
    keep = (tau >= window[0]) & (tau <= window[1])
    return dict(tau=tau[keep], p=p[keep], active=active[keep])


def live_series(run: SimpleNamespace, values: Sequence[float], meta: dict, offset_sec: float) -> dict[str, np.ndarray]:
    """配信の公開列 (録画秒)。公開の値 = その公開のコマの最後の評価行の表示値。一致しない公開は保存どおり (件数を出す)。"""
    last = {row['frame']: index for index, row in enumerate(run.evals)}
    times, vals, kept = [], [], 0
    for pub in run.pubs:
        index = last.get(pub['frame'])
        logged = np.nan if pub['probability'] is None else pub['probability']
        same = index is not None and pub['probability'] is not None and run.evals[index]['probability'] == pub['probability']
        kept += not same
        vals.append(values[index] if same else logged)
        times.append(pub['published_at']-meta['origin_mono']-offset_sec)
    return dict(times=np.array(times), values=np.array(vals), kept_as_logged=kept, n=len(run.pubs))


def value_at(times: np.ndarray, values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """τ にその時刻に出ている値 = τ 以前で最後の公開 (stage_b_compare.value_at の lag=0)。無ければ NaN。"""
    index = np.searchsorted(times, tau, side='right')-1
    out = np.full(len(tau), np.nan)
    ok = index >= 0
    out[ok] = values[index[ok]]
    return out


def diff_stats(ref: np.ndarray, live: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ok = mask & np.isfinite(ref) & np.isfinite(live)
    d = np.abs(ref[ok]-live[ok])*POINTS
    if len(d) < MIN_PAIRS:
        return dict(n=int(len(d)))
    result = dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))
    for t in THRESHOLDS_PT:
        result[f'over_{int(t)}pt'] = float((d > t).mean())
    return result


def bootstrap_mean(values: list[float]) -> dict[str, float]:
    """試合を単位にした平均の 95% 区間 (標本が小さいので目安)。"""
    if len(values) < 3:
        return dict(n=len(values))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = [float(np.mean(rng.choice(values, len(values)))) for _ in range(BOOTSTRAP_DRAWS)]
    return dict(n=len(values), mean=float(np.mean(values)), lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)))


def match_table(origin_unix: float) -> tuple[list[dict], dict[int, str]]:
    """試合表 (記録秒) と、終了の組の勝者 (試合番号 → '1P' / '2P')。

    終了の組は analysis_misc24d.end_events_vs_matches と同じ規則で試合へ結ぶ ([開始, 終了+8秒]、複数なら開始が最後の試合)。
    end_events_C2.json の match_ids は m8・m30 が空 (検出が試合の終了の 3.9 / 5.7 秒後) なので使わない。"""
    matches = json.loads(gzip.decompress((SCORE_DIR/'tables_C2/matches.json.gz').read_bytes()))
    events = json.loads((SCORE_DIR/'end_events_C2.json').read_text(encoding='utf-8'))
    winners: dict[int, str] = {}
    for event in events:
        t = event['first']-origin_unix
        candidates = [m for m in matches if m['start'] <= t <= m['end']+END_MATCH_MARGIN_SEC]
        if candidates:
            winners[max(candidates, key=lambda m: m['start'])['id']] = event['winner']
    return matches, winners


def wrong_direction(p: np.ndarray, winner: str) -> np.ndarray:
    """勝者と逆の向き (1P 勝者なら p < 0.5、2P 勝者なら p > 0.5)。p = 0.5 ちょうどは向きなし。"""
    return (p < 0.5) if winner == '1P' else (p > 0.5)


def wrong_seconds(grids: list[tuple[np.ndarray, str, int]]) -> dict[str, Any]:
    """試合ごとの (値のある秒, 逆向きの秒, 値のない秒) と合計。grids = [(p の列, 勝者, 試合番号)]、1 コマ = 1/FPS 秒。"""
    per_match, valid, wrong, missing = [], 0, 0, 0
    for p, winner, match_id in grids:
        finite = np.isfinite(p)
        n_valid, n_wrong = int(finite.sum()), int((wrong_direction(p, winner) & finite).sum())
        per_match.append(dict(id=match_id, winner=winner, valid_sec=n_valid/FPS, wrong_sec=n_wrong/FPS, missing_sec=int((~finite).sum())/FPS))
        valid, wrong, missing = valid+n_valid, wrong+n_wrong, missing+int((~finite).sum())
    return dict(matches=len(grids), valid_sec=valid/FPS, wrong_sec=wrong/FPS, missing_sec=missing/FPS,
                wrong_share=wrong/valid if valid else None, per_match=per_match)


def direction_tables(series: dict[str, Any], matches: list[dict], winners: dict[int, str], offset: float) -> dict[str, Any]:
    """(iii) 3 つの母集団: L 全試合 (L 自身の時計で 30Hz に標本化)・F の窓内・L を F と同じコマ (窓内) で見たもの。"""
    shift = -offset
    ref, live = series['reference'], series['live']
    live_record = dict(times=live['times']+offset, values=live['values'])      # 録画秒 → 記録秒
    all_l, in_f, l_on_f = [], [], []
    for match in matches:
        winner = winners.get(match['id'])
        if winner is None:
            continue
        grid = np.arange(match['start'], match['end'], 1.0/FPS)
        all_l.append((value_at(live_record['times'], live_record['values'], grid), winner, match['id']))
        low, high = match['start']+shift, match['end']+shift
        mask = ref['active'] & (ref['tau'] >= low) & (ref['tau'] <= high)
        if mask.any():
            in_f.append((ref['p'][mask], winner, match['id']))
            l_on_f.append((value_at(live['times'], live['values'], ref['tau'][mask]), winner, match['id']))
    return dict(L_all_matches=wrong_seconds(all_l), F_window=wrong_seconds(in_f), L_on_F_frames=wrong_seconds(l_on_f))


def build_series(L: SimpleNamespace, F: SimpleNamespace, values: dict[str, Sequence[float]], meta: dict, offset: float) -> dict[str, Any]:
    return dict(reference=frame_reference(F, values['F'], WINDOW), live=live_series(L, values['L'], meta, offset))


def stage_b(series: dict[str, Any], matches: list[dict], shift: float) -> dict[str, Any]:
    """段 B と同じ「見える差」(lag 0): 全体と、試合ごとの平均の試合単位の区間。"""
    ref, live = series['reference'], series['live']
    live_p = value_at(live['times'], live['values'], ref['tau'])
    result: dict[str, Any] = dict(as_viewed=diff_stats(ref['p'], live_p, ref['active']))
    per_match = {}
    for match in matches:
        mask = ref['active'] & (ref['tau'] >= match['start']+shift) & (ref['tau'] <= match['end']+shift)
        stats = diff_stats(ref['p'], live_p, mask)
        if 'mean' in stats:
            per_match[match['id']] = stats['mean']
    result['per_match_mean_pt'] = bootstrap_mean(list(per_match.values()))
    result['per_match'] = per_match
    return result


def evaluate_variant(name: str, runs: dict[str, SimpleNamespace], predictions: dict[str, dict[tuple, float]] | None, kinds: dict) -> dict[str, Any]:
    """1 つの構成 (None = 保存された値そのまま) について、生の予測と表示 (平滑後) の両方の段 B と向きの秒数を出す。"""
    meta, offset, matches, winners, rec_shift = kinds['meta'], kinds['offset'], kinds['matches'], kinds['winners'], -kinds['offset']
    raws = {k: raw_series(runs[k], None if predictions is None else predictions[k]) for k in ('L', 'F')}
    raw_values = {k: [np.nan if v is None else v for v in raws[k]] for k in raws}
    if predictions is None:     # 保存された表示値そのまま (採点 stage_b.json と同じ数字が出るかの基準)
        shown_values = {k: np.array([row['probability'] for row in runs[k].evals]) for k in raws}
    else:
        shown_values = {k: replay_shown(runs[k], raws[k]) for k in raws}
    out: dict[str, Any] = dict(name=name)
    for label, values in (('raw', raw_values), ('shown', shown_values)):
        series = build_series(runs['L'], runs['F'], values, meta, offset)
        out[label] = dict(stage_b=stage_b(series, matches, rec_shift), direction=direction_tables(series, matches, winners, offset),
                          pubs_kept_as_logged=series['live']['kept_as_logged'], pubs=series['live']['n'])
    if predictions is not None:
        out['shown_replay_agreement'] = {k: replay_agreement(runs[k], shown_values[k]) for k in raws}
    return out


def paired_summary(report: dict[str, Any]) -> dict[str, Any]:
    """各構成 − large を試合ごとの対で見る (段 B の平均差と、勝者と逆の向きの秒数)。標本が小さいので区間は目安。"""
    out: dict[str, Any] = {}
    base = report['variants']['large']
    for name, variant in report['variants'].items():
        if name == 'large':
            continue
        entry: dict[str, Any] = {}
        for kind in ('shown', 'raw'):
            a, b = base[kind]['stage_b']['per_match'], variant[kind]['stage_b']['per_match']
            entry[f'{kind}/stage_b_per_match_mean_delta_pt'] = bootstrap_mean([b[i]-a[i] for i in a if i in b])
            entry[f'{kind}/stage_b_all_delta_pt'] = variant[kind]['stage_b']['as_viewed']['mean']-base[kind]['stage_b']['as_viewed']['mean']
            for pop in ('L_all_matches', 'F_window', 'L_on_F_frames'):
                x = {m['id']: m['wrong_sec'] for m in base[kind]['direction'][pop]['per_match']}
                y = {m['id']: m['wrong_sec'] for m in variant[kind]['direction'][pop]['per_match']}
                delta = [y[i]-x[i] for i in x if i in y]
                entry[f'{kind}/{pop}/wrong_sec'] = dict(delta_total=float(np.sum(delta)), better=int(sum(d < -1e-9 for d in delta)),
                                                        worse=int(sum(d > 1e-9 for d in delta)), same=int(sum(abs(d) <= 1e-9 for d in delta)),
                                                        per_match_mean_delta=bootstrap_mean(delta))
        out[name] = entry
    return out


def logged_stage_b() -> dict[str, Any]:
    """採点の保存値 (stage_b.json)。この台が同じ数字を再現するかの基準。"""
    return json.loads((SCORE_DIR/'stage_b.json').read_text(encoding='utf-8'))['as_viewed']


def load_models() -> dict[str, Any]:
    from src.b3_train_model import load_model
    members = {'large': load_model(LARGE_MODEL)}
    members.update({f'ref_s{seed}': load_model(path) for seed, path in zip((1003, 1004, 1005), REF_MODELS)})
    return members


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=WORK_DIR/'replay_compare.json')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    started = perf_counter()
    meta = json.loads((SCORE_DIR/'stage_a.json').read_text(encoding='utf-8'))['C2']
    offset = float(json.loads((SCORE_DIR/'cal_c2.json').read_text(encoding='utf-8'))['offset_sec'])
    runs = dict(L=load_run(L_DIR), F=load_run(F_DIR))
    matches, winners = match_table(meta['origin_unix'])
    report: dict[str, Any] = dict(sources=dict(L=str(L_DIR), F=str(F_DIR)), logged_stage_b_as_viewed=logged_stage_b(),
                                  links={k: link_check(runs[k]) for k in runs}, results={k: len(runs[k].results) for k in runs})
    singles = {k: member_predictions(runs[k], load_models()) for k in runs}
    predictions = {k: variant_predictions(singles[k]) for k in runs}
    report['large_reproduces_logged_raw'] = {k: reproduction(runs[k], predictions[k]['large']) for k in runs}
    report['row_differences'] = {k: row_difference_table(runs[k], predictions[k]) for k in runs}
    kinds = dict(meta=meta, offset=offset, matches=matches, winners=winners)
    report['logged'] = evaluate_variant('logged (saved values)', runs, None, kinds)
    report['variants'] = {name: evaluate_variant(name, runs, {k: predictions[k][name] for k in runs}, kinds) for name in VARIANTS}
    report['paired_vs_large'] = paired_summary(report)
    report['winners'] = dict(matches=len(matches), with_winner=len(winners), p1=sum(w == '1P' for w in winners.values()),
                             p2=sum(w == '2P' for w in winners.values()))
    report['elapsed_sec'] = perf_counter()-started
    options.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(options.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
