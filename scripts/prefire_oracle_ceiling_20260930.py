"""発火前予測の価値上限 (オラクル) と、発火前の表示鮮度を測る (Phase 1a、2026-09-30)。

保存済みの本番構成の表示列 (logs/pending_expiry/e36b_on) を入力に取り、
「撃ち合いの結果が発火の N 秒前に分かっていたら」を、採点時だけ表示勝率へ注入して測る。
再生・認識は一切やり直さない (検証の梯子 段2)。実行時の評価器は変えない。

注入の定義:
  - 対象区間: 同じ試合で [max(発火 - N, 直前の撃ち合い終了, 試合先頭), 発火)
  - 目標値: 撃ち合い終了時点 (closed_sec、次の発火が先ならその直前) の表示勝率 (EMA 後)
            = 撃ち合いの帰結を表示が消化した後の値
  - 混合: logit(p) = (1-w)·logit(元の表示) + w·logit(目標値)。w=1 が完全な先読み、w=0.5 は半分だけ当たる予測
N=0 (注入なし) で本番記録の値 (q .507567 / zenchi 7,671/8,333) を再現することを先に確認する (原則3)。
使い方: PYTHONPATH=. python -m scripts.prefire_oracle_ceiling_20260930
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from scripts import aggregate_e3_exchange_eval_20260926 as e3

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
RECORD_ROOT = EXEV / 'logs/pending_expiry/e36b_on/on'
ZENCHI_GAMES = EXEV / 'logs/review_zenchi_part3/official_games.json'
OUT = Path('logs/prefire_prediction')
RECORD_DIRS = {'q_7gc4TgFig': 'renders/q_7gc4TgFig/on', 'fcXG83vInDY': 'renders/fcXG83vInDY/on',
               'mia8KCjr52g': 'renders/mia8KCjr52g/on', 'review': 'review', 'zenchi': 'zenchi'}
LABELED = ('q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g')   # 勝敗区間を持つ記録
LEADS_SEC = (1.0, 3.0, 5.0, 10.0)
SUBSET_LEADS_SEC = (3.0, 10.0)   # 発火前区間に限った採点の部分集合 (全条件で同じ行)
BLENDS = (1.0, 0.5)
PLACEBO_LEAD_SEC, PLACEBO_SEED = 3.0, 20260930   # 対照 (目標値の入れ替え) の条件
FPS = e3.FPS
EPS = e3.PROBABILITY_EPSILON
MOVE_EPS = 1e-3            # 表示勝率がこれ以上動いたフレームを「動いた」と数える (0.1%pt)
MIN_TOTAL_MOVE = 0.05      # 先読み率の分母に使う撃ち合い全体の変化の下限 (5%pt)
BASELINE_Q_LL, BASELINE_Z_HITS, BASELINE_Z_FRAMES = .5075669993215693, 7671, 8333
ZENCHI_LAST_FRACTION = 2 / 3   # 終盤1/3 (report_e10b.agreement と同定義)


def load(source: str) -> tuple[dict, list[dict]]:
    """表示列とイベント原票を読む。"""
    root = RECORD_ROOT / RECORD_DIRS[source]
    with np.load(root / 'display.npz') as data:
        display = {k: data[k].copy() for k in data.files}
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    return display, events


def value_at(display: dict, t: float) -> tuple[int, float]:
    """時刻 t より前の最後のフレーム番号と表示勝率を返す。"""
    idx = max(0, int(np.searchsorted(display['t_sec'], t, side='left')) - 1)
    return idx, float(display['display_p1'][idx])


def exchange_rows(display: dict, events: list[dict]) -> list[dict]:
    """撃ち合いごとに発火・終了・直前の終了・目標値を作る (同じ試合内で時系列順)。"""
    rows, times, games = [], display['t_sec'], display['game_idx']
    ordered = sorted(events, key=lambda e: e['trigger_sec'])
    for pos, event in enumerate(ordered):
        game, trigger = event['game_idx'], event['trigger_sec']
        same = [e for e in ordered if e['game_idx'] == game]
        before = [e for e in same if e['trigger_sec'] < trigger]
        after = [e['trigger_sec'] for e in same if e['trigger_sec'] > trigger]
        game_times = times[games == game]
        if not len(game_times):
            continue
        close = event['closed_sec'] if event['closed_sec'] is not None else float(game_times[-1])
        close = min([close] + after)
        prev_close = max([(e['closed_sec'] or e['trigger_sec']) for e in before] + [float(game_times[0])])
        _, target = value_at(display, close)
        _, fire_value = value_at(display, trigger)
        rows.append(dict(exchange_id=event['exchange_id'], game=game, trigger=trigger, close=close,
                         prev_close=prev_close, target=target, at_fire=fire_value))
    return rows


def logit(p: np.ndarray) -> np.ndarray:
    """数値安定な logit。"""
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def inject(display: dict, rows: list[dict], lead: float, blend: float) -> tuple[np.ndarray, np.ndarray]:
    """発火前 lead 秒の区間へ目標値を混合した表示勝率と、書換マスクを返す。"""
    p1 = display['display_p1'].copy()
    mask = np.zeros(len(p1), dtype=bool)
    times, games = display['t_sec'], display['game_idx']
    for row in rows:
        start = max(row['trigger'] - lead, row['prev_close'])
        sel = (games == row['game']) & (times >= start) & (times < row['trigger'])
        mixed = (1 - blend) * logit(p1[sel]) + blend * logit(np.array([row['target']]))
        p1[sel] = 1 / (1 + np.exp(-mixed))
        mask |= sel
    return p1, mask


def q_scores(display: dict, p1: np.ndarray, windows: list[dict], only: np.ndarray | None) -> dict:
    """e3.m3_scores と同定義 (区間内全フレーム) の LL/AUC。only を与えればその部分集合だけ。"""
    frames = np.rint(display['t_sec'] * FPS).astype(int)
    labels = np.full(len(frames), np.nan)
    for window in windows:
        sel = (frames >= window['start']) & (frames < window['end'])
        labels[sel] = int(window['winner'] == '1P')
    valid = np.isfinite(labels) if only is None else np.isfinite(labels) & only
    y, p = labels[valid], np.clip(p1[valid], EPS, 1 - EPS)
    if not len(y):
        return dict(frames=0, log_loss=None, auc=None)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return dict(frames=int(valid.sum()), log_loss=float(log_loss(y, p, labels=[0, 1])), auc=auc)


def zenchi_hits(display: dict, p1: np.ndarray, mask: np.ndarray, only_mask: bool) -> dict:
    """終盤1/3の勝者一致 (report_e10b.agreement 同定義)。only_mask なら書換フレームだけ数える。"""
    subset = mask if only_mask else np.ones(len(p1), dtype=bool)
    return zenchi_subset(display, p1, mask, subset)


def window_freshness(display: dict, row: dict, lead: float) -> dict | None:
    """発火前 lead 秒の表示の動き (動いたフレーム率・先読み率) を1撃ち合い分求める。"""
    times, games, p1 = display['t_sec'], display['game_idx'], display['display_p1']
    start = max(row['trigger'] - lead, row['prev_close'])
    sel = np.flatnonzero((games == row['game']) & (times >= start) & (times < row['trigger']))
    if len(sel) < 2:
        return None
    values = p1[sel]
    moved = float(np.mean(np.abs(np.diff(values)) > MOVE_EPS))
    total = row['target'] - values[0]
    before = values[-1] - values[0]
    ratio = float(before / total) if abs(total) >= MIN_TOTAL_MOVE else None
    return dict(moved_frac=moved, before=float(before), total=float(total), anticipation=ratio,
                jump_after=float(row['target'] - values[-1]), frames=len(sel))


def freshness_summary(items: list[dict]) -> dict:
    """撃ち合い単位の鮮度を母数つきで集約する。"""
    ratios = [i['anticipation'] for i in items if i['anticipation'] is not None]
    moved = [i['moved_frac'] for i in items]
    jumps = [abs(i['jump_after']) for i in items]
    before = [abs(i['before']) for i in items]
    pct = lambda xs, q: float(np.percentile(xs, q)) if xs else None
    return dict(exchanges=len(items), moved_frac_p50=pct(moved, 50), moved_any=sum(m > 0 for m in moved),
                abs_move_before_p50=pct(before, 50), abs_move_before_p90=pct(before, 90),
                abs_jump_after_p50=pct(jumps, 50), abs_jump_after_p90=pct(jumps, 90),
                anticipation_n=len(ratios), anticipation_p50=pct(ratios, 50),
                anticipation_ge_half=sum(r >= 0.5 for r in ratios))


def global_moved_frac(display: dict) -> float:
    """試合中の全フレームで表示勝率が動いたフレームの割合 (参照値)。"""
    games, p1 = display['game_idx'], display['display_p1']
    same = games[1:] == games[:-1]
    return float(np.mean(np.abs(np.diff(p1))[same] > MOVE_EPS))


def source_ceiling(source: str) -> dict[str, Any]:
    """1記録について注入なし・各 N・各混合の採点と鮮度を返す。"""
    display, events = load(source)
    rows = exchange_rows(display, events)
    windows = e3.outcomes(source)[0] if source in LABELED else None
    result: dict[str, Any] = dict(exchanges=len(rows), global_moved_frac=global_moved_frac(display),
                                  freshness={}, variants={})
    for lead in LEADS_SEC:
        items = [f for f in (window_freshness(display, r, lead) for r in rows) if f is not None]
        result['freshness'][str(lead)] = freshness_summary(items)
        result.setdefault('freshness_items', {})[str(lead)] = items
    subsets = {f'prefire{n:g}s': inject(display, rows, n, 0.0)[1] for n in SUBSET_LEADS_SEC}
    base_mask = np.zeros(len(display['t_sec']), dtype=bool)
    variants = [(0.0, 0.0)] + [(n, w) for n in LEADS_SEC for w in BLENDS]
    for lead, blend in variants:
        p1, mask = inject(display, rows, lead, blend) if lead else (display['display_p1'], base_mask)
        result['variants'][f'N{lead:g}_w{blend:g}'] = score_variant(source, display, p1, mask,
                                                                     windows, subsets)
    p1, mask = inject(display, placebo_rows(rows), PLACEBO_LEAD_SEC, 1.0)
    result['variants'][f'placebo_shuffled_N{PLACEBO_LEAD_SEC:g}_w1'] = score_variant(
        source, display, p1, mask, windows, subsets)
    return result


def placebo_rows(rows: list[dict]) -> list[dict]:
    """対照: 目標値を同じ記録内の別の撃ち合いのものへ入れ替える (情報なしの書換で改善しないことの確認)。"""
    order = np.random.default_rng(PLACEBO_SEED).permutation(len(rows))
    return [dict(row, target=rows[j]['target']) for row, j in zip(rows, order)]


def score_variant(source: str, display: dict, p1: np.ndarray, mask: np.ndarray,
                  windows: list[dict] | None, subsets: dict[str, np.ndarray]) -> dict[str, Any]:
    """1条件の採点。部分集合 (発火前 3s/10s の和集合) は条件によらず同じ行で比べる。"""
    entry: dict[str, Any] = dict(modified_frames=int(mask.sum()))
    if windows is not None:
        entry['q_all'] = q_scores(display, p1, windows, None)
        for name, subset in subsets.items():
            entry[f'q_{name}'] = q_scores(display, p1, windows, subset)
    if source == 'zenchi':
        entry['zenchi_all'] = zenchi_hits(display, p1, mask, False)
        for name, subset in subsets.items():
            entry[f'zenchi_{name}'] = zenchi_subset(display, p1, mask, subset)
    return entry


def zenchi_subset(display: dict, p1: np.ndarray, mask: np.ndarray, subset: np.ndarray) -> dict:
    """発火前部分集合に限った終盤1/3一致。書換フレームは p1 の符号、他は表示 adv の符号。"""
    games = json.loads(ZENCHI_GAMES.read_text(encoding='utf-8'))
    times = display['t_sec']
    sign_display = np.where(mask, np.sign(p1 - 0.5), np.sign(display['display_adv']))
    hits = frames = 0
    for game in games:
        begin = game['start'] + (game['end'] - game['start']) * ZENCHI_LAST_FRACTION
        sel = (times >= begin) & (times < game['end']) & subset
        sign = 1 if game['winner'] == '1P' else -1
        hits += int(np.count_nonzero(sign_display[sel] * sign > 0))
        frames += int(sel.sum())
    return dict(hits=hits, frames=frames, agreement=hits / frames if frames else None)


def main() -> None:
    """全記録を採点し、注入なしが本番記録値と一致することを確かめてから保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    result = {source: source_ceiling(source) for source in RECORD_DIRS}
    base_q = result['q_7gc4TgFig']['variants']['N0_w0']['q_all']
    base_z = result['zenchi']['variants']['N0_w0']['zenchi_all']
    reproduced = dict(q=abs(base_q['log_loss'] - BASELINE_Q_LL) < 1e-9 and base_q['frames'] == 6526,
                      zenchi=base_z['hits'] == BASELINE_Z_HITS and base_z['frames'] == BASELINE_Z_FRAMES)
    result['_pooled_freshness'] = {str(n): freshness_summary(
        [i for s in RECORD_DIRS for i in result[s]['freshness_items'][str(n)]]) for n in LEADS_SEC}
    result['_meta'] = dict(record_root=str(RECORD_ROOT), leads=LEADS_SEC, blends=BLENDS,
                           move_eps=MOVE_EPS, min_total_move=MIN_TOTAL_MOVE, reproduced=reproduced)
    (OUT / 'oracle_ceiling.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(reproduced), flush=True)
    if not all(reproduced.values()):
        raise SystemExit('注入なしの採点が本番記録値と一致しない (測定器の不一致)')


if __name__ == '__main__':
    main()
