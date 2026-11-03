"""配信の認識ループの Python 側の出力不変の高速化 (live_recog_speed_py、既定 OFF)。

背景 (2026-10-25): 24d の単独試験 (C2) は認識 1 コマ平均 46.97ms・周期 48.17ms で、30 コマ/秒には
-14.8ms が要る。洗い出し (<work>/python_hotspots_20261025/RESULT.json) のうち、
「結果を変えずに削れる」項目だけをここに集める。read_board の中身・hybrid_classifier のループは
Rust 移植の担当が触るので、このモジュールは触らず「呼ぶ側」で重複を除く。

全項目の約束 (崩すと出力が変わる):
- 同じ入力 (同じコマの画像・同じ分類器の状態) に対する、同じ関数の結果だけを使い回す。
- 使い回すのは「画像オブジェクトそのもの (is)」が同じときだけ。id() は使わない
  (解放後に別の画像が同じ id を取りうる)。強参照を持つので、色補正 buffer の再利用
  (参照数が 2 のときだけ上書き) も、保持中の画像を上書きしない。
- 分類器の状態 (色範囲・彩度スケール・モード) が変わったら使い回さない。
- 想定外 (属性が無い・型が違う) は黙って別の結果を返さず、従来の呼出へ戻す。

旗: 環境変数 PUYO_LIVE_RECOG_SPEED_PY=1 (puyo_live.json の live_recog_speed_py。既定 OFF)。
  壁時計に依存する項目 (終了の組の crop の worker 側複製) は、この旗では働かず、専用の旗
  PUYO_LIVE_RECOG_SPEED_PY_PAIR_LAZY=1 (puyo_live.json の live_recog_speed_py_pair_lazy。既定 OFF) でだけ働く。
  終了の組の照合は元から 0.5 秒ごとの壁時計で動き、評価に使うコマが 1 コマずれうるので、全行一致の確認
  (live_recog_speed_py) とは別に承認する。
  PUYO_LIVE_RECOG_SPEED_PY_ITEMS=a,b を併せて渡すと、その項目だけが働く (項目ごとの削減量の測定・
  壁時計依存の項目を外した全行一致の確認用。旗が OFF なら無視)。
  PUYO_LIVE_RECOG_SPEED_PY_AUDIT=1 は、使い回した結果を毎回やり直した結果と突き合わせる
  (検証用。違えば STATS['audit_mismatch'] を数える)。
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

# ---- 旗と項目 ----
ENV_FLAG = "PUYO_LIVE_RECOG_SPEED_PY"
ENV_ITEMS = "PUYO_LIVE_RECOG_SPEED_PY_ITEMS"
ENV_AUDIT = "PUYO_LIVE_RECOG_SPEED_PY_AUDIT"
ENV_PAIR_LAZY = "PUYO_LIVE_RECOG_SPEED_PY_PAIR_LAZY"
FLAG_ON = "1"
ITEM_SEPARATOR = ","

ITEM_HSV_MEMO = "hsv_memo"                # A2: read_board_hsv_only の同コマ重複呼び
ITEM_SLIDE_GRAY = "slide_gray"            # A7: NextSlideDetector の前コマのグレー化
ITEM_IMAGE_CROPS = "image_crops"          # A3a: 発火前画像リングを盤面 2 領域だけ保持
ITEM_IMAGE_BATCH = "image_batch"          # A3b: 発火前画像の分類を一括 (その分類器が既に native のときだけ Rust 1 回)
ITEM_DISPLAY_SCORES = "display_scores"    # A6: 表示得点を同コマの生得点から (b3_share_score_read が前提)
ITEM_PAIR_LAZY = "pair_lazy_crop"         # A1: 終了の組の crop を worker 側で (壁時計依存)
# 出力が全行一致で確かめられる項目 (旗 live_recog_speed_py が働かせる)。
STRICT_ITEMS: tuple[str, ...] = (
    ITEM_HSV_MEMO, ITEM_SLIDE_GRAY, ITEM_IMAGE_CROPS, ITEM_IMAGE_BATCH, ITEM_DISPLAY_SCORES,
)
# 壁時計に依存して全行一致を厳密には取れない項目 → 専用の旗の環境変数 (元から終了の組の照合は 0.5 秒ごとの壁時計)。
WALL_CLOCK_ENV: dict[str, str] = {ITEM_PAIR_LAZY: ENV_PAIR_LAZY}
ALL_ITEMS: tuple[str, ...] = STRICT_ITEMS + tuple(WALL_CLOCK_ENV)

# 診断用の通過件数 (出力には一切影響しない)。検証台が「実際に使われたか」を母数つきで示すために読む。
STATS: Counter = Counter()


def reset_stats() -> None:
    STATS.clear()


def enabled(item: str) -> bool:
    """項目 item が働くか。旗が OFF なら常に False (従来の経路と完全に同じ)。"""
    if item in WALL_CLOCK_ENV:   # 壁時計に依存する項目は、専用の旗だけで決まる (live_recog_speed_py では働かない)
        return os.environ.get(WALL_CLOCK_ENV[item]) == FLAG_ON
    if os.environ.get(ENV_FLAG) != FLAG_ON:
        return False
    only = os.environ.get(ENV_ITEMS)
    if not only:
        return True
    return item in {name.strip() for name in only.split(ITEM_SEPARATOR)}


def runtime_report() -> dict | None:
    """実効値の記録。どの項目が働く設定か (環境変数から) と、実際に使い回した件数。全項目 OFF なら None。"""
    items = {name: enabled(name) for name in ALL_ITEMS}
    if not any(items.values()):
        return None
    return dict(items=items, stats=dict(STATS))


def audit_enabled() -> bool:
    """使い回しを毎回やり直しと突き合わせるか (検証用。既定 OFF)。"""
    return os.environ.get(ENV_AUDIT) == FLAG_ON


# ----------------------------------------------------------------------------
# A2: read_board_hsv_only の同コマ重複呼びを 1 回にする
# ----------------------------------------------------------------------------

REGION_KEY_FIELDS: tuple[str, ...] = ("x", "y", "width", "height")


def _classifier_state(reader: Any) -> tuple | None:
    """HSV-only 盤面の結果を決める分類器の状態。取れなければ None (= 使い回さない)。

    read_board_hsv_only の結果は、画像・領域・HSV 分類器の状態 (色範囲・彩度スケール・
    モード) だけで決まる。色範囲は set_color_ranges_from_simple / 較正の巻き戻しで
    「新しい dict」へ置き換わるので、同一性 (is) で変化を見分ける。
    """
    try:
        clf = reader._get_hsv_classifier()
        if clf is None:
            return None
        return (clf, clf._ranges, float(clf._s_min_scale), bool(clf._vote_mode),
                bool(clf._enable_red_hue_wrap_fix), bool(clf._enable_specular_robust_saturation),
                bool(clf._enable_native_hsv), bool(reader._live_board_vectorized))
    except AttributeError:
        return None


def _same_state(left: tuple, right: tuple) -> bool:
    """分類器と色範囲は同一オブジェクト、残りは値が等しいときだけ同じ状態とみなす。"""
    return left[0] is right[0] and left[1] is right[1] and left[2:] == right[2:]


class HsvOnlyMemo:
    """1 コマ分の HSV-only 盤面。コマ (画像オブジェクト) か分類器の状態が変われば作り直す。"""

    __slots__ = ("frame", "state", "boards")

    def __init__(self, frame: Any, state: tuple) -> None:
        self.frame = frame          # 強参照 (is で比べる。保持中は色補正 buffer も上書きしない)
        self.state = state
        self.boards: dict[tuple, Any] = {}


def _region_key(region: Any) -> tuple:
    return tuple(int(getattr(region, name)) for name in REGION_KEY_FIELDS)


def read_hsv_only(reader: Any, frame: Any, region: Any) -> Any:
    """`reader.read_board_hsv_only(frame, region)` と同じ盤面を返す。同コマの 2 回目以降は使い回す。

    呼ぶ側 (recognition_pipeline._step_side と collect_boards_lean の記録時補正) が使う。
    返す盤面は常に呼出ごとに独立 (呼出元が書き換えても次の呼出に影響しない)。
    旗が OFF・状態が取れない場合は従来どおり reader を直接呼ぶ。
    """
    if not enabled(ITEM_HSV_MEMO):
        return reader.read_board_hsv_only(frame, region)
    state = _classifier_state(reader)
    if state is None:
        return reader.read_board_hsv_only(frame, region)
    memo = getattr(reader, "_hsv_only_memo", None)
    if memo is None or memo.frame is not frame or not _same_state(memo.state, state):
        memo = HsvOnlyMemo(frame, state)
        reader._hsv_only_memo = memo
    key = _region_key(region)
    cached = memo.boards.get(key)
    if cached is None:
        board = reader.read_board_hsv_only(frame, region)
        memo.boards[key] = board.copy()
        STATS["hsv_only_miss"] += 1
        return board
    STATS["hsv_only_hit"] += 1
    if audit_enabled():
        fresh = reader.read_board_hsv_only(frame, region)
        STATS["audit_checked"] += 1
        if fresh._grid.tobytes() != cached._grid.tobytes():
            STATS["audit_mismatch"] += 1
    result = cached.copy()
    # 従来の呼出は盤面をライブ窓の観測 (_live_hsv_boards) へも残す。窓はコマごとに作り直されるので、
    # 同じ画像オブジェクトが次のコマにも渡された場合に備え、使い回しでも同じ記録を残す。
    remember = getattr(reader, "_remember_live_hsv", None)
    if remember is not None:
        remember(result, region)
    return result
