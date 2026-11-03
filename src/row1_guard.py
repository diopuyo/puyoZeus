"""3列目12段目 (row1, col2) の幻ぷよ抑止 — v2.1 (既定OFF、2026-10-14)。

背景 (<work>/X_MARK_MISREAD.md 8・9節): 確定盤の row1 col2 に、実際には無いぷよ
(空きマスの赤い×印 7件、操作中・回転中ツモの取り込み 6件) が数秒〜十数秒残る。
15秒超の 4件は自分の大連鎖の凍結区間で延命されていた。

方針: 確定盤そのもの (state machine の内部) には触れず、**公開する確定盤** (SideResult.
confirmed_board) の row1 col2 だけを、毎フレーム次の順で検査して外す。stateless 判定は
`floating_r1c2`、時間を跨ぐ判定だけを `Row1Guard` が持つ (外部 wrapper 方針)。

1. 浮き (物理): row1 col2 に色があり row2 col2 が空 → 空にする。浮いたぷよは物理的にありえない。
2. 画面が空: 生の読取 (×マスク後) の row1 col2 が EMPTY_EVIDENCE_SEC 続けて空 → 空にする。
   外した後は、置き終わりの合図と読取の一致がそろうまで戻さない。
   ×が見えている間は読取が空になる (×の検出は既存の UiMaskMatcher に依存する補助)。
   連鎖凍結中も毎フレーム走るので、凍結による延命は止まる。
3. スポーン形: 前フレームで空だった row1 col2 が、隣接セル (row0 col2 / row1 col1 / row1 col3 /
   row2 col2) の少なくとも 1つと同時に新規出現したら保留する。
   同側の NEXT 移動・おじゃま落下・掛け算式の立上りと読取一致で、そのコマに公開する。
   先行NEXTは着地投票窓内の最初の着地診断に一度だけ対応付ける。
   合図前に読取から消えた組は幻として外す。時間経過だけでは公開しない。

v2.1 は全内部更新後の公開境界で row1 col2 だけを変更する。評価には元盤を渡す。
合図からの追加待ちは0コマ。着地から8コマ以内かは画像経路で別途実測が必要。
"""
from __future__ import annotations

from src.board import Board, COLOR_EMPTY

# 対象セル (raw row: 0=隠し段, 1=画面内最上段, 2=その下)。col は 3列目 = index 2。
GUARD_COL: int = 2
ROW_HIDDEN: int = 0
ROW_TOP: int = 1
ROW_BELOW: int = 2
# スポーン形の同時出現を調べる隣接セル (row, col)。
SPAWN_COMPANIONS: tuple[tuple[int, int], ...] = (
    (ROW_HIDDEN, GUARD_COL), (ROW_TOP, GUARD_COL - 1),
    (ROW_TOP, GUARD_COL + 1), (ROW_BELOW, GUARD_COL),
)
# 旧API/import互換のため残す。v2 の判定には使わない。
SPAWN_HOLD_SEC: float = 0.5
# 読取が空を示し続けたら幻とみなす秒数 (×マスクの瞬間的な欠けを拾わない長さ)。
EMPTY_EVIDENCE_SEC: float = 0.5
# 既存の着地投票窓 (0.4秒) と同じ範囲で、先行NEXTを最初の着地にだけ対応付ける。
NEXT_LANDING_WINDOW_SEC: float = 0.4
# 秒の比較に使う浮動小数の許容 (0.1 秒刻みの積み上げ誤差で 0.5 秒ちょうどを取りこぼさない)。
TIME_EPSILON: float = 1e-6
# 監査カウンタのキー。
COUNT_KEYS: tuple[str, ...] = (
    "frames", "floating", "screen_empty", "spawn_held", "spawn_released",
    "spawn_released_chain", "screen_restored",
    "spawn_released_next", "spawn_released_ojama", "spawn_removed",
)


def floating_r1c2(board: Board) -> bool:
    """row1 col2 に何かあり、真下の row2 col2 が空なら True (物理的にありえない)。"""
    return (
        int(board.get(ROW_TOP, GUARD_COL)) != COLOR_EMPTY
        and int(board.get(ROW_BELOW, GUARD_COL)) == COLOR_EMPTY
    )


def _blank(board: Board, cells: list[tuple[int, int]]) -> Board:
    """指定セルを空にした複製を返す (元の盤面は変更しない)。"""
    out = board.copy()
    for row, col in cells:
        out.set(row, col, COLOR_EMPTY)
    return out


class Row1Guard:
    """片側 (1P か 2P) の row1 col2 保護。1 フレームに 1 回 `step` を呼ぶ。"""

    def __init__(
        self,
        spawn_hold_sec: float = SPAWN_HOLD_SEC,
        empty_evidence_sec: float = EMPTY_EVIDENCE_SEC,
    ) -> None:
        # spawn_hold_sec は既存呼出しとの互換引数。時間では保留解除しない。
        self._empty_evidence_sec = empty_evidence_sec
        self.counts: dict[str, int] = {key: 0 for key in COUNT_KEYS}
        self.reset()

    def reset(self) -> None:
        """時間を跨ぐ状態を捨てる (確定盤が無い・試合境界)。カウンタは残す。"""
        self._prev_cells: dict[tuple[int, int], int] | None = None
        self._held_since: float | None = None
        self._held_cells: list[tuple[int, int]] = []
        self._empty_since: float | None = None
        self._suppressed: bool = False
        self._agree_since: float | None = None
        self._previous_signals: tuple[bool, bool, bool] = (False, False, False)
        self.release_reason: str | None = None
        self._next_signal_sec: float | None = None

    def _snapshot(self, board: Board) -> dict[tuple[int, int], int]:
        cells = [(ROW_TOP, GUARD_COL), *SPAWN_COMPANIONS]
        return {cell: int(board.get(*cell)) for cell in cells}

    def _new_spawn_cells(
        self, prev: dict[tuple[int, int], int], now: dict[tuple[int, int], int],
    ) -> list[tuple[int, int]] | None:
        """スポーン形なら保留すべきセルを返す。そうでなければ None。"""
        top = (ROW_TOP, GUARD_COL)
        if prev[top] != COLOR_EMPTY or now[top] == COLOR_EMPTY:
            return None
        new = [c for c in SPAWN_COMPANIONS
               if prev[c] == COLOR_EMPTY and now[c] != COLOR_EMPTY]
        if not new:
            return None
        # v2 は対象外セルへの直接変更も避ける。隠し段は観測にだけ使う。
        return [top]

    def step(
        self, board: Board, raw_cnn: Board | None, time_sec: float,
        in_chain: bool = False,
        next_moved: bool = False, ojama_falling: bool = False,
        settled: bool = False,
    ) -> Board:
        """公開用の確定盤を返す。変更が無ければ引数のオブジェクトをそのまま返す。

        in_chain: 互換引数名。v2 の呼出元は同側の掛け算式の実読結果を渡す。
        各合図は立上りのみ使い、継続中の古い合図で新しい組を確定しない。
        """
        self.counts["frames"] += 1
        signal = self._placement_signal(in_chain, next_moved, ojama_falling)
        signal = self._align_next_landing(signal, time_sec, settled)
        self.release_reason = None
        top = (ROW_TOP, GUARD_COL)
        now = self._snapshot(board)
        prev, self._prev_cells = self._prev_cells, now
        if now[top] == COLOR_EMPTY:
            self._held_since, self._held_cells, self._empty_since = None, [], None
            self._suppressed, self._agree_since = False, None
            return board
        if floating_r1c2(board):
            self.counts["floating"] += 1
            return _blank(board, [top])
        seen = int(raw_cnn.get(*top)) if raw_cnn is not None else None
        if self._held_since is not None and seen == COLOR_EMPTY:
            self._held_since, self._held_cells = None, []
            self._suppressed = True
            self.counts["spawn_removed"] += 1
            self.release_reason = "raw_disappeared"
            return _blank(board, [top])
        if self._screen_says_empty(seen, time_sec):
            self.counts["screen_empty"] += 1
            self._suppressed, self._agree_since = True, None
            return _blank(board, [top])
        if self._suppressed:
            return self._suppressed_view(board, seen, now[top], signal)
        return self._spawn_hold(board, prev, now, seen, time_sec, signal)

    def _align_next_landing(self, signal: str | None, time_sec: float,
                            settled: bool) -> str | None:
        """先行したNEXTを着地時に一度だけ使う。時間だけでは解除しない。"""
        if signal == 'next_moved':
            self._next_signal_sec = time_sec
        pending = self._next_signal_sec
        if pending is not None and time_sec - pending > NEXT_LANDING_WINDOW_SEC + TIME_EPSILON:
            self._next_signal_sec = None
        if settled:
            aligned = self._next_signal_sec is not None
            self._next_signal_sec = None
            if signal is None and aligned:
                return 'next_moved'
        elif signal in ('chain_started', 'ojama_falling'):
            self._next_signal_sec = None
        return signal

    def _placement_signal(self, chain: bool, moved: bool, ojama: bool) -> str | None:
        """同じ側の合図の立上りを取り出す。時間による解除はしない。"""
        current = (chain, moved, ojama)
        previous, self._previous_signals = self._previous_signals, current
        names = ("chain_started", "next_moved", "ojama_falling")
        return next((name for name, now, old in zip(names, current, previous)
                     if now and not old), None)

    def _screen_says_empty(self, seen: int | None, time_sec: float) -> bool:
        if seen != COLOR_EMPTY:
            self._empty_since = None
            return False
        if self._empty_since is None:
            self._empty_since = time_sec
        return time_sec - self._empty_since >= self._empty_evidence_sec - TIME_EPSILON

    def _suppressed_view(
        self, board: Board, seen: int | None, value: int, signal: str | None,
    ) -> Board:
        """読みぶれでは戻さず、合図と確定値の一致がそろったコマに戻す。"""
        if signal is not None and seen == value:
            self._suppressed, self._agree_since = False, None
            self._held_since, self._held_cells = None, []
            self.release_reason = signal
            self.counts["screen_restored"] += 1
            return board
        return _blank(board, [(ROW_TOP, GUARD_COL)])

    def _spawn_hold(
        self, board: Board, prev: dict[tuple[int, int], int] | None,
        now: dict[tuple[int, int], int], seen: int | None, time_sec: float,
        signal: str | None,
    ) -> Board:
        if self._held_since is None and prev is not None:
            cells = self._new_spawn_cells(prev, now)
            if cells is not None:
                self._held_since, self._held_cells = time_sec, cells
        if self._held_since is None:
            return board
        top = (ROW_TOP, GUARD_COL)
        if signal is not None and seen == now[top]:
            self._held_since, self._held_cells = None, []
            counter = {"chain_started": "spawn_released_chain",
                       "next_moved": "spawn_released_next",
                       "ojama_falling": "spawn_released_ojama"}[signal]
            self.counts[counter] += 1
            self.counts["spawn_released"] += 1
            self.release_reason = signal
            return board
        self.counts["spawn_held"] += 1
        return _blank(board, self._held_cells)
