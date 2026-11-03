"""ama評価v5の窒息・着弾規則。明示呼出専用で本番既定OFF。

v4まで「着弾溢れ = 窒息」としていたが、これはuser未承認の仮定だった。
v5の窒息は「連鎖の後に、3列目の画面内最上段 (DEATH_ROW=1) にぷよがある」だけ。
入りきらないおじゃま (14段目に置かれる分) の扱いは引数で切り替える。
- discard (暫定既定): 14段目に置いたぷよが直後に消える規則と同じ扱いで、予告から外れて消える。
- carry: 予告に残り、次の着弾で降る。
"""
from __future__ import annotations
from src.prefire_build_projection import drop_one

COLS = 6
DEATH_ROW, DEATH_COL = 1, 2
DEATH_CELL = DEATH_ROW*COLS+DEATH_COL
DROP_MAX = 30
OJAMA_COLOR = 9
OVERFLOW_DISCARD, OVERFLOW_CARRY = 'discard', 'carry'
OVERFLOW_MODES = (OVERFLOW_DISCARD, OVERFLOW_CARRY)
DEFAULT_OVERFLOW_MODE = OVERFLOW_DISCARD
REASON_DEATH_CELL = '死亡セル'


def landing_proof(board: list, pending: int, mode: str = DEFAULT_OVERFLOW_MODE) -> dict:
    """1回の着弾 (最大30個) を再現する。窒息は着弾後の死亡セルだけで判定する。

    landed は予告から減る個数。carry では溢れた分が予告に残るので、その分を引く。
    """
    assert mode in OVERFLOW_MODES, mode
    board, overflow = list(board), 0
    count = min(DROP_MAX, max(0, pending))
    for i in range(count):
        overflow += int(not drop_one(board, i % COLS, OJAMA_COLOR))
    dead = bool(board[DEATH_CELL])
    return dict(board=board, landed=count-(overflow if mode == OVERFLOW_CARRY else 0),
                overflow=overflow, dead=dead, reason=REASON_DEATH_CELL if dead else None,
                overflow_mode=mode)
