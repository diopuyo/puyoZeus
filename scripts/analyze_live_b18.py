"""B18: 1本の実時間run(RUN/candidate + RUN/probe)から遅延分布と分単位の取得待ち分解を出す。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.analyze_live_b9 import available, lines, samples

MS, HZ_TICKS, MINUTE, MIN_FRAMES = 1000.0, 100.0, 60.0, 200
SLOT_MS = 1000.0/30


def pct(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()), max=float(a.max()))


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or x[ok].std() == 0 or y[ok].std() == 0:
        return None
    rx, ry = (np.argsort(np.argsort(v[ok])).astype(float) for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def load_probe(probe: Path) -> tuple[dict, dict, dict]:
    names = {}
    for p in probe.glob('proc_*.json'):
        d = json.loads(p.read_text())
        names[d['pid']] = d['name']

    def read(pattern: str) -> dict:
        return {int(p.stem.split('_')[1]): [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                for p in probe.glob(pattern)}
    return names, read('source_*.jsonl'), read('gc_*.jsonl')


def overall(metrics: dict, data: dict, sent: list) -> dict:
    subscriber = [(r['received_at']-r['payload']['timing']['capture_monotonic_sec'])*MS for r in sent]
    ev_sse = [(r['received_at']-r['payload']['timing']['evaluated_monotonic_sec'])*MS for r in sent]
    return dict(
        expected_frames=metrics['expected_frames'], recognized=int(len(data['t_sec'])),
        capture_dropped=metrics['capture_dropped_in_measured_window'],
        capture_drop_fraction=metrics['capture_drop_fraction'],
        dropped_frames_in_window=metrics['dropped_frames_in_measured_window'],
        capture_to_sse_subscriber=pct(subscriber), evaluation_to_sse_subscriber=pct(ev_sse),
        server_side_stages=metrics['latency_ms'],
        recognition_ms=pct(data['recognition_ms']), recognition_cpu_ms=pct(data['cpu_ms']),
        acquisition_wait_ms=pct((data['acquired_at']-data['captured_at'])*MS),
        state_update_ms=pct([r['milliseconds'] for r in metrics['state_updates']]),
        probability_calculation_ms=pct([r['milliseconds'] for r in metrics['probability_calculations']]),
        queue=metrics['evaluation_queue'] | dict(observed=None),
        published_sse_available=len(sent), batch_sizes=pct(metrics['batch_sizes']))


def proc_window(proc: list, roles: dict, lo: float, hi: float) -> dict:
    rows = [r for r in proc if lo <= r['at'] <= hi]
    if len(rows) < 2:
        return dict(cpu={}, rss={}, write_mb=0.0, sizes={}, busy=None, steal=None)
    first, last = rows[0], rows[-1]
    dt = last['at']-first['at']
    before = {p['pid']: p for p in first['procs']}
    cpu, rss, write = {}, {}, 0.0
    for p in last['procs']:
        name = roles.get(p['pid'], 'other')
        rss[name] = max(rss.get(name, 0), p['rss']/1e6)
        q = before.get(p['pid'])
        if q:
            cpu[name] = cpu.get(name, 0.0)+((p['utime']+p['stime'])-(q['utime']+q['stime']))/HZ_TICKS/dt*100
            write += (p['write_bytes']-q['write_bytes'])/1e6
    ticks = np.array(last['cpu_ticks'], float)-np.array(first['cpu_ticks'], float)
    busy = float(100*(1-(ticks[3]+ticks[4])/ticks.sum()))
    steal = float(100*ticks[7]/ticks.sum()) if len(ticks) > 7 else None
    return dict(cpu=cpu, rss=rss, write_mb=write, sizes=last['sizes'], busy=busy, steal=steal)


def per_minute(run: Path, metrics: dict, data: dict, sent: list) -> tuple[list, dict]:
    cand = run/'candidate'
    names, sources, gcs = load_probe(run/'probe')
    rec_pid = next(p for p, n in names.items() if n == 'recognition-process')
    src = {int(r['frame']): r for r in sources[rec_pid]}
    rows_src = [src.get(int(f)) for f in data['frame']]
    col = lambda key: np.array([r.get(key, np.nan) if r else np.nan for r in rows_src], float)
    t, rec = data['t_sec'], data['recognition_ms']
    acq = (data['acquired_at']-data['captured_at'])*MS
    source_ms, read_ms, grab_ms = col('source_ms'), col('read_ms'), col('grab_ms')
    overshoot = col('sleep_act')-col('sleep_req')
    consumer_shift = np.r_[col('prev_consumer_ms')[1:], np.nan]  # フレームnの消費側時間は行n+1に付く
    gc_src, gc_cons = col('gc_source_ms'), np.r_[col('prev_consumer_gc_ms')[1:], np.nan]
    lag_resume = acq-source_ms
    nonstable = np.array([any(s != 'STABLE' for s in st) for st in data['states']])
    load, runtime, proc = (lines(cand/n) for n in ('cpu_load.jsonl', 'runtime.jsonl', 'b18_proc.jsonl'))
    origin = float(np.median(data['captured_at']-t))
    minute = ((t-t[0])//MINUTE).astype(int)
    out = []
    for m in np.unique(minute):
        k = minute == m
        if k.sum() < MIN_FRAMES:
            continue
        lo, hi = float(t[k].min()), float(t[k].max())
        sel = [r for r in sent if lo <= r['payload']['timing']['source_available_ms']/MS <= hi]
        lat = [(r['received_at']-r['payload']['timing']['capture_monotonic_sec'])*MS for r in sel]
        rt = [r for r in runtime if lo <= r['progress'].get('t_sec', -1) <= hi]
        ld = [r for r in load if origin+lo <= r['start'] <= origin+hi]
        pw = proc_window(proc, names, float(data['captured_at'][k].min()), float(data['captured_at'][k].max()))
        gcrec = [g for g in gcs.get(rec_pid, []) if data['captured_at'][k].min() <= g['second'] <= data['captured_at'][k].max()]
        st, ns = k & ~nonstable, k & nonstable
        out.append(dict(minute=int(m), n=int(k.sum()), nonstable_frac=float(nonstable[k].mean()),
            acq_mean=float(acq[k].mean()), acq_p50=float(np.percentile(acq[k], 50)), acq_p95=float(np.percentile(acq[k], 95)),
            lag_resume_p95=float(np.nanpercentile(lag_resume[k], 95)), frac_late_gt10ms=float((lag_resume[k] > 10).mean()),
            rec_p50=float(np.percentile(rec[k], 50)), rec_p95=float(np.percentile(rec[k], 95)),
            rec_stable_p50=float(np.percentile(rec[st], 50)) if st.sum() > 30 else None,
            rec_nonstable_p50=float(np.percentile(rec[ns], 50)) if ns.sum() > 30 else None,
            consumer_mean=float(np.nanmean(consumer_shift[k])), overhead_mean=float(np.nanmean(consumer_shift[k]-rec[k])),
            frac_consumer_gt_slot=float(np.nanmean(consumer_shift[k] > SLOT_MS)),
            source_mean=float(np.nanmean(source_ms[k])), read_mean=float(np.nanmean(read_ms[k])),
            grab_mean=float(np.nanmean(grab_ms[k])), sleep_overshoot_mean=float(np.nanmean(overshoot[k])),
            gc_src_ms_per_frame=float(np.nanmean(gc_src[k])), gc_consumer_ms_per_frame=float(np.nanmean(gc_cons[k])),
            gc_gen2_events=int(sum(g.get('n2', 0) for g in gcrec)),
            gc_max_ms=float(max([g.get('max_ms', 0) for g in gcrec], default=0)),
            dropped=int(data['dropped_before'][k].sum()),
            lat_p50=float(np.percentile(lat, 50)) if lat else None, lat_p95=float(np.percentile(lat, 95)) if lat else None,
            lat_p99=float(np.percentile(lat, 99)) if lat else None, lat_n=len(lat),
            queue_p95=float(np.percentile([r['queue_depth'] for r in rt], 95)) if rt else None,
            queue_max=max([r['queue_depth'] for r in rt], default=None),
            external_cores_p50=float(np.percentile([r['external_cores'] for r in ld], 50)) if ld else None,
            cpu_pct_by_role=pw['cpu'], rss_mb_by_role=pw['rss'], system_busy_pct=pw['busy'], steal_pct=pw['steal'],
            write_mb_all=pw['write_mb'], journal_mb=pw['sizes'].get('journal', 0)/1e6,
            spool_mb=pw['sizes'].get('spool', 0)/1e6, out_total_mb=pw['sizes'].get('total', 0)/1e6))
    return out, names


def correlations(rows: list) -> dict:
    keys = ['nonstable_frac', 'rec_p50', 'rec_p95', 'rec_stable_p50', 'consumer_mean', 'overhead_mean',
            'frac_consumer_gt_slot', 'read_mean', 'gc_src_ms_per_frame', 'gc_consumer_ms_per_frame',
            'gc_gen2_events', 'queue_p95', 'external_cores_p50', 'journal_mb', 'spool_mb', 'out_total_mb',
            'system_busy_pct', 'minute']
    out = dict(n_minutes=len(rows))
    for target in ('acq_mean', 'acq_p95', 'lat_p95'):
        y = [r[target] for r in rows]
        out[target] = {k: spearman([np.nan if r[k] is None else r[k] for r in rows], y) for k in keys}
    return out


def frame_level(run: Path, data: dict) -> dict:
    names, sources, _ = load_probe(run/'probe')
    rec_pid = next(p for p, n in names.items() if n == 'recognition-process')
    src = sources[rec_pid]
    acq = np.array([(r['acq']-r['cap'])*MS for r in src])
    cons = np.array([r.get('prev_consumer_ms', np.nan) for r in src])
    out = {}
    for lo, hi in ((0, 25), (25, 30), (30, SLOT_MS), (SLOT_MS, 40), (40, 1e9)):
        k = (cons >= lo) & (cons < hi)
        out[f'prev_consumer_{lo:.1f}_{hi:.1f}'] = dict(n=int(k.sum()),
            acq_mean=float(acq[k].mean()) if k.any() else None,
            acq_p95=float(np.percentile(acq[k], 95)) if k.any() else None)
    nonstable = np.array([any(s != 'STABLE' for s in st) for st in data['states']])
    tt = (data['t_sec']-data['t_sec'][0])/MINUTE
    for name, k in (('stable', ~nonstable), ('nonstable', nonstable)):
        out['rec_ms_slope_per_min_'+name] = float(np.polyfit(tt[k], data['recognition_ms'][k], 1)[0])
    out['acq_ms_slope_per_min_all'] = float(np.polyfit(tt, (data['acquired_at']-data['captured_at'])*MS, 1)[0])
    return out


def main() -> None:
    run = Path(sys.argv[1])
    cand = run/'candidate'
    metrics = json.loads((cand/'metrics.json').read_text())
    sent = [r for r in samples(cand) if available(r)]
    with np.load(cand/'recognition.npz') as z:
        data = {k: z[k] for k in z.files}
    rows, names = per_minute(run, metrics, data, sent)
    result = dict(run=str(run), overall=overall(metrics, data, sent), per_minute=rows,
                  roles={str(k): v for k, v in names.items()}, correlations=correlations(rows),
                  frame_level=frame_level(run, data))
    (run/'analysis.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(result['overall'], indent=1, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
