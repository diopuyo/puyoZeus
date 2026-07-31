import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1) SCORE_UNKNOWN 定数追加 (collect_boards_lean.py の -1センチネルに対応)
old_const = "# 列崩壊トリガー: 崩壊前の非空セル数がこれ以上 (色+おじゃま、空/UNKNOWN除く)。\nMIN_STACK_BEFORE_COLLAPSE: int = 2"
new_const = (
    "# score OCR失敗センチネル (collect_boards_lean.py の変換規則、-1=None)。\n"
    "SCORE_UNKNOWN: int = -1\n\n"
    "# 列崩壊トリガー: 崩壊前の非空セル数がこれ以上 (色+おじゃま、空/UNKNOWN除く)。\n"
    "MIN_STACK_BEFORE_COLLAPSE: int = 2"
)
assert old_const in text
text = text.replace(old_const, new_const, 1)

# 2) セグメント分割関数を追加 (game_idx desync対策、project_game_idx_desync_bug_2026-07-29)
anchor = "# ============================\n# per-video 処理\n# ============================"
segment_fn = '''# ============================
# 試合セグメント分割 (game_idx desync対策)
# ============================
#
# memory project_game_idx_desync_bug_2026-07-29 の通り、npzのgame_idx列は
# 1P/2P独立カウンタのズレで信頼できない場合がある。scoreが減少する遷移は
# 物理的にありえない(1試合中は単調非減少)ため、試合境界の実測シグナルとして
# 採用し、同一game_idxラベル内でも試合を再分割する。これを怠ると試合終了時の
# 勝敗演出(盤面がUIで覆われる等)や次試合の空盤面が「列崩壊」に誤分類される
# (2026-07-30実測: c11 1P g1でscoreが238から26へ減少する境界を確認、これを
# セグメント分割せずに検出した結果は誤検出だった)。


def _split_into_match_segments(g: NpzRecord) -> list[NpzRecord]:
    """scoreが減少する遷移を境界として、1つの試合区間に再分割する。

    score_before/afterのいずれかがSCORE_UNKNOWN(-1)の場合はOCR欠測であり
    真の減少と断定できないため分割トリガーにしない (境界の見誤りを避ける)。
    """
    n = g.grids.shape[0]
    if n < 2:
        return [g]
    boundaries: list[int] = [0]
    for i in range(n - 1):
        prev_s, next_s = int(g.score[i]), int(g.score[i + 1])
        if prev_s == SCORE_UNKNOWN or next_s == SCORE_UNKNOWN:
            continue
        if next_s < prev_s:
            boundaries.append(i + 1)
    boundaries.append(n)
    segments: list[NpzRecord] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        segments.append(NpzRecord(
            video_id=g.video_id, side=g.side,
            t_sec=g.t_sec[start:end], game_idx=g.game_idx[start:end],
            grids=g.grids[start:end], score=g.score[start:end],
        ))
    return segments


''' + anchor
assert anchor in text
text = text.replace(anchor, segment_fn, 1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("patched const+segment_fn ok")
