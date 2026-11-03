"""実際に保持表示された静止入力の経過秒でCVを再重み付けする。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT, OUTPUT, REFERENCE, datasets, display_frames, loss
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from src.exchange_event_user_phase import USER_COLUMNS

PHASE_COUNT = 3
TOLERANCE = 1e-12


def held_inputs() -> pd.DataFrame:
    """直近の通常静止呼出しを表示へ対応付け、予測値で対応を検証する。"""
    directory = Path('models/exchange_event_v1')
    manifest = json.loads((directory/'manifest.json').read_text())
    model = joblib.load(directory/manifest['models']['G_fe']['file'])
    display = display_frames()
    result = []
    columns = [USER_COLUMNS.index('U_'+phase) for phase in ('early','middle','late')]
    for source, d in display[display.route == 'G_fe'].groupby('source'):
        root = OUT/('capture_V3' if source == 'c1' else 'capture')/source
        with np.load(root/'inputs.npz') as raw:
            keep = raw['caller'] == '_evaluate'
            values = pd.DataFrame(dict(input_t=raw['context'][keep,0], input_elapsed=raw['elapsed'][keep],
                input_p=model.predict_proba(raw['x59'][keep])[:,1],
                U=raw['x54'][keep][:,columns].argmax(axis=1)))
        with np.load(root/'display.npz') as timeline:
            maximum = timeline['t_sec'].max()
        values = values.sort_values('input_t',kind='stable').drop_duplicates('input_t',keep='last')
        d = d[d.t <= maximum].sort_values('t')
        merged = pd.merge_asof(d,values,left_on='t',right_on='input_t',direction='backward')
        assert merged.input_p.notna().all(), source
        np.testing.assert_allclose(merged.input_p,merged.p0,atol=TOLERANCE,rtol=0,err_msg=source)
        result.append(merged)
    return pd.concat(result,ignore_index=True)


def main() -> None:
    """表示中の保持も重みへ含め、1秒刻みとUの同時分布を確認する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    target = held_inputs()
    rows, _, phase = datasets(REFERENCE)
    cv = pd.read_csv(OUTPUT/'predictions.csv',usecols=['row_id','label','V0','V3'])
    ids = cv.row_id.to_numpy()
    with np.load(OUTPUT/'user_phase.npz') as saved:
        u = saved['phase']
    elapsed = np.floor(phase['elapsed']).astype(int)
    observed = np.floor(target.input_elapsed).astype(int).to_numpy()
    l0,l3 = loss(cv.V0,cv.label),loss(cv.V3,cv.label)
    sides = rows.source_side.to_numpy()[ids]
    pd.DataFrame([dict(side=side, predictions=int((sides == side).sum()),
        baseline=float(l0[sides == side].mean()), V3=float(l3[sides == side].mean()),
        difference=float((l3-l0)[sides == side].mean())) for side in ('1P','2P')]).to_csv(OUT/'cv_side.csv',index=False)
    results = []
    joint, target_joint = elapsed*PHASE_COUNT+u, observed*PHASE_COUNT+target.U.to_numpy()
    schemes = [('保持入力の経過秒',elapsed,observed,False),('保持入力の経過秒×U',joint,target_joint,False),
               ('保持入力の経過秒×U・1P行限定',joint,target_joint,True)]
    for name, train_bins, target_bins, only_1p in schemes:
        size = max(train_bins.max(),target_bins.max())+1
        selected = sides == '1P' if only_1p else np.ones(len(ids),dtype=bool)
        counts = np.bincount(train_bins[ids][selected],minlength=size)
        mass = np.bincount(target_bins,weights=target.weight,minlength=size)
        mass /= mass.sum()
        w = np.divide(mass,counts,out=np.zeros(size),where=counts > 0)[train_bins[ids]]*selected
        results.append(dict(weighting=name, display_frames=len(target), unsupported_mass=mass[counts == 0].sum(),
            baseline=np.average(l0,weights=w),V3=np.average(l3,weights=w),difference=np.average(l3-l0,weights=w)))
    pd.DataFrame(results).to_csv(OUT/'cv_input_time_reweighted.csv',index=False)
    save_json(OUT/'HELD_INPUT_VERIFICATION.json',dict(display_frames=len(target),
        baseline_probability_match=True, max_error=float(np.max(np.abs(target.input_p-target.p0))),
        note='s2部分再生以外は全区間。G_feの実入力を保持時間分だけ重み付け。'))


if __name__ == '__main__':
    main()
