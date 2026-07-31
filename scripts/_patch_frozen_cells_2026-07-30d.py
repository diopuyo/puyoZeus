import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

old_process = """    for rec in _load_npz(npz_path):
        for gidx in sorted(set(rec.game_idx.tolist())):
            mask = rec.game_idx == gidx
            g = _subset(rec, mask)
            order = np.argsort(g.t_sec)
            g = NpzRecord(
                video_id=g.video_id, side=g.side,
                t_sec=g.t_sec[order], game_idx=g.game_idx[order],
                grids=g.grids[order], score=g.score[order],
            )
            col_ev = _detect_column_collapses(g, stem, rec.side, int(gidx))
            col_events.extend(col_ev)
            cell_events.extend(_detect_cell_freezes(g, stem, rec.side, int(gidx), col_ev))
    return col_events, cell_events"""

new_process = """    for rec in _load_npz(npz_path):
        for gidx in sorted(set(rec.game_idx.tolist())):
            mask = rec.game_idx == gidx
            g = _subset(rec, mask)
            order = np.argsort(g.t_sec)
            g = NpzRecord(
                video_id=g.video_id, side=g.side,
                t_sec=g.t_sec[order], game_idx=g.game_idx[order],
                grids=g.grids[order], score=g.score[order],
            )
            # game_idx desyncバグ対策: scoreの減少箇所で再分割してから検出する
            for seg_idx, seg in enumerate(_split_into_match_segments(g)):
                col_ev = _detect_column_collapses(seg, stem, rec.side, int(gidx) * 1000 + seg_idx)
                col_events.extend(col_ev)
                cell_events.extend(
                    _detect_cell_freezes(seg, stem, rec.side, int(gidx) * 1000 + seg_idx, col_ev)
                )
    return col_events, cell_events"""

assert old_process in text
text = text.replace(old_process, new_process, 1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("patched process_video ok")
