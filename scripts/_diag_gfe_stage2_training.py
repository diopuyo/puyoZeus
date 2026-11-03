"""学習元の欠測と同一保存場面の提供変換を照合する。"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scripts.gfe_phase_stage1_20261002 import memory_guard, datasets, REFERENCE, SOURCE, SHARED, save_json
from scripts._diag_gfe_stage2_analyze import OUT, stats
from src.exchange_event_features import D_COLUMNS

KEYS = ['video_id', 'game_idx', 'frame', 'side']
SAMPLE_COUNT = 30
TOLERANCE = 2e-6


def raw_training() -> tuple[pd.DataFrame, pd.DataFrame]:
    """0補完前CSVを学習行と同じキー順にする。"""
    rows, design, _ = datasets(REFERENCE)
    raw = pd.read_csv(SOURCE, usecols=KEYS+list(D_COLUMNS)).set_index(KEYS)
    raw.xs('video_29', level='video_id').reset_index().to_csv(OUT/'video29_source_rows.csv', index=False)
    key = rows[['video_id','frame','source_side']].rename(columns={'source_side':'side'})
    key['game_idx'] = rows.game_id.str.rsplit(':', n=1).str[-1].astype(int)
    aligned = raw.reindex(pd.MultiIndex.from_frame(key[KEYS]))
    np.testing.assert_array_equal(design, aligned[list(D_COLUMNS)].fillna(0).to_numpy(dtype=np.float32))
    pd.DataFrame(stats(aligned[list(D_COLUMNS)].to_numpy(), D_COLUMNS, 'train_raw')).to_csv(OUT/'training_raw_columns.csv', index=False)
    return rows, aligned


def inspect_npz() -> None:
    """学習元候補の保存形式を確認する。"""
    root = SHARED/'data/indicators_v2/boards_lean_phase_l_2026-08-11'
    with np.load(root/'29.npz', allow_pickle=False) as raw:
        print({k: (raw[k].shape, str(raw[k].flatten()[:3])) for k in raw.files}, flush=True)


def match_scenes(rows: pd.DataFrame, aligned: pd.DataFrame) -> None:
    """評価セット外video_29で同時刻・保存盤面から提供Dを計算する。"""
    path = SHARED/'data/indicators_v2/boards_lean_phase_l_2026-08-11/29.npz'
    with np.load(path, allow_pickle=False) as raw:
        arrays = {k: raw[k] for k in raw.files}
    print('npz fields', list(arrays), flush=True)
    save_json(OUT/'same_scene_source.json', dict(path=str(path), fields=list(arrays),
        note='保存NPZから同じ時刻・盤面を復元。元動画の再認識ではない。'))
    subset = rows[rows.video_id == 'video_29']
    chosen = subset.iloc[np.linspace(0, len(subset)-1, SAMPLE_COUNT).astype(int)]
    results = []
    for idx, row in chosen.iterrows():
        game = int(row.game_id.rsplit(':', 1)[1])
        own = np.flatnonzero((arrays['frame_idx'] == row.frame) & (arrays['side'] == row.source_side)
                            & (arrays['game_idx'] == game))
        if len(own) != 1:
            continue
        own_idx = own[0]
        candidates = np.flatnonzero((arrays['side'] != row.source_side) & (arrays['game_idx'] == game)
                                    & (arrays['t_sec'] <= arrays['t_sec'][own_idx]))
        if not len(candidates):
            continue
        opp_idx = candidates[np.argmax(arrays['t_sec'][candidates])]
        assert abs(float(arrays['t_sec'][own_idx])-row.t_sec) < TOLERANCE
        result = compare_scene(arrays, own_idx, opp_idx, aligned.iloc[idx], int(row.row_id))
        results.extend(result)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT/'same_scene_values.csv', index=False)
    summary = frame.groupby('column').agg(scenes=('row_id','size'),
        mismatches=('mismatch','sum'), maximum_error=('error','max')).reset_index()
    summary.to_csv(OUT/'same_scene_columns.csv', index=False)


def compare_scene(data: dict, own: int, opp: int, expected: pd.Series, row_id: int) -> list[dict]:
    """同じ保存盤面ペアでDを生成。保存されていない会計入力は照合対象外。"""
    from scripts.visualize_advantage_overlay import _side_feats_full_base, _side_feats_full
    from src.board import Board
    own_base, opp_base = (_side_feats_full_base(Board.from_list(data['grids'][i].tolist())) for i in (own, opp))
    actual = _side_feats_full(own_base, opp_base, 0, 0)
    dynamic = {'ojama_net_balance','ojama_forecast','ojama_net_balance_synced','ojama_margin',
               'all_clear_bonus_pending','opp_all_clear_bonus_pending'}
    results = []
    for name in D_COLUMNS:
        if name in dynamic:
            continue
        left, right = float(expected[name]), float(actual[name])
        error = abs(left-right) if np.isfinite(left) and np.isfinite(right) else 0.0
        mismatch = not (np.isnan(left) and np.isnan(right) or np.isclose(left,right,atol=TOLERANCE,rtol=0))
        results.append(dict(row_id=row_id, column=name, train=left, serving=right,
            error=error, mismatch=mismatch, own_frame=int(data['frame_idx'][own]),
            opponent_frame=int(data['frame_idx'][opp]), t_sec=float(data['t_sec'][own])))
    return results


def main() -> None:
    """学習元だけを読み、診断の表を作る。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    rows, raw = raw_training()
    inspect_npz()
    match_scenes(rows, raw)


if __name__ == '__main__':
    main()
