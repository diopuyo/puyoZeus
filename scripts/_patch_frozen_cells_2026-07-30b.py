import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

old_detect = """            prev_colored = int(np.isin(prev_col, list(COLORED_VALUES)).sum())
            prev_ojama = int((prev_col == COLOR_OJAMA).sum())
            score_delta = int(g.score[i + 1]) - int(g.score[i])
            expected_min = SCORE_FLOOR_PER_COLORED_PUYO * prev_colored
            recovery_idx = _scan_column_recovery(g.grids, i + 1, col)
            events.append(ColumnCollapseEvent(
                video_stem=video_stem, side=side, game_idx=game_idx, col=col,
                t_before=float(g.t_sec[i]), t_collapse=float(g.t_sec[i + 1]),
                prev_colored_count=prev_colored, prev_ojama_count=prev_ojama,
                score_before=int(g.score[i]), score_after=int(g.score[i + 1]),
                score_delta=score_delta, expected_min_score=expected_min,
                score_suspicious=score_delta < expected_min,
                t_recovery=None if recovery_idx is None else float(g.t_sec[recovery_idx]),"""
new_detect = """            prev_colored = int(np.isin(prev_col, list(COLORED_VALUES)).sum())
            prev_ojama = int((prev_col == COLOR_OJAMA).sum())
            score_before_v, score_after_v = int(g.score[i]), int(g.score[i + 1])
            score_delta = score_after_v - score_before_v
            expected_min = SCORE_FLOOR_PER_COLORED_PUYO * prev_colored
            score_available = score_before_v != SCORE_UNKNOWN and score_after_v != SCORE_UNKNOWN
            recovery_idx = _scan_column_recovery(g.grids, i + 1, col)
            events.append(ColumnCollapseEvent(
                video_stem=video_stem, side=side, game_idx=game_idx, col=col,
                t_before=float(g.t_sec[i]), t_collapse=float(g.t_sec[i + 1]),
                prev_colored_count=prev_colored, prev_ojama_count=prev_ojama,
                score_before=score_before_v, score_after=score_after_v,
                score_delta=score_delta, expected_min_score=expected_min,
                score_suspicious=score_available and score_delta < expected_min,
                score_available=score_available,
                t_recovery=None if recovery_idx is None else float(g.t_sec[recovery_idx]),"""
assert old_detect in text, "detect marker not found"
text = text.replace(old_detect, new_detect, 1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("patched detect ok")
