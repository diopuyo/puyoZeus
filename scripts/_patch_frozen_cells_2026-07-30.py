import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

marker_old_dataclass = """    score_suspicious: bool   # score_delta < expected_min_score
    t_recovery: float | None
    freeze_duration_sec: float | None
    right_censored: bool
    phase_bucket: str
    row_band_summary: str"""
marker_new_dataclass = """    score_suspicious: bool   # score_delta < expected_min_score (score_available時のみ有効)
    score_available: bool    # score_before/afterともにOCR成功していたか (-1センチネル除外)
    t_recovery: float | None
    freeze_duration_sec: float | None
    right_censored: bool
    phase_bucket: str
    row_band_summary: str"""
assert marker_old_dataclass in text, "dataclass marker not found"
text = text.replace(marker_old_dataclass, marker_new_dataclass, 1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("patched dataclass ok")
