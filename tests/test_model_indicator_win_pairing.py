"""
scripts/model_indicator_win.py の pair_sides_for_win 回帰テスト。

## 背景 (2026-07-28 監査)
study CSV の game_idx は「ジョブ窓内の相対インデックス」であり、動画を
本体(0-300s) / _gap(300-900s) / _mid(1200-1560s) の3区間に分割して並列収集して
いるため、**同一動画・同一 game_idx が複数の別試合を指す**ことがある
(実データ例: video_c10 の game_idx=0 が t_sec=120.7〜1240.7 に散在)。

pair_sides_for_win() は docstring 上「game_idx でグループ化せず、t_sec の
近傍マッチで同時刻のペアを構成する」と明記されている (model_indicator_win.py:150-152)。
この約束を検証する pytest が存在しなかったため、本ファイルで固定化する。

もし将来 「game_idx でグループ化した方が速い」等の善意のリファクタで
groupby(["video_id", "game_idx"]) 的な実装に変わると、同一 game_idx を
共有する別試合の 1P/2P 行が誤ってペア化されうる。本テストの本命ケースは
それを検出する。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.model_indicator_win import DEFAULT_MAX_TDIFF, pair_sides_for_win


def _row(video_id: str, game_idx: int, t_sec: float, side: str, won: int) -> dict:
    """study CSV の1行を模したダミーレコードを作る。"""
    return {
        "video_id": video_id,
        "game_idx": game_idx,
        "t_sec": t_sec,
        "side": side,
        "won": won,
    }


# =============================================================================
# 本命ケース: 同一 game_idx・別試合(t_sec 大きく離れる)が混同されないこと
# =============================================================================


def test_同一game_idxでも時刻が離れた別試合は誤ってペア化されない() -> None:
    """
    実データ相当の状況を再現する:
    - 試合A (例: 本体窓由来) : game_idx=0, t_sec=100.0 (1P) / 100.4 (2P)
    - 試合B (例: _mid窓由来) : game_idx=0 (Aと同じ相対インデックス!), t_sec=1300.0 (1P) / 1300.6 (2P)
    - 試合A/B の間には約1200秒の空白があり、実データの876.9秒空白と同種の状況。

    game_idx を使わず t_sec 近傍マッチのみで組む実装であれば、
    1Pの t=100.0 は 2Pの t=100.4 とだけペアになり、
    1Pの t=1300.0 は 2Pの t=1300.6 とだけペアになるはずで、
    A と B が混ざったペア (例: t_sec_1p=100.0 と t_sec_2p=1300.6) は絶対に出てはならない。
    """
    df = pd.DataFrame([
        _row("video_x", 0, 100.0, "1P", won=1),
        _row("video_x", 0, 100.4, "2P", won=0),
        _row("video_x", 0, 1300.0, "1P", won=0),
        _row("video_x", 0, 1300.6, "2P", won=1),
    ])

    paired = pair_sides_for_win(df, DEFAULT_MAX_TDIFF)

    # 2試合ぶん、正しく2ペアだけ成立すること (4通りの組み合わせ全部が
    # ペア化される cartesian join バグ、あるいは0ペアになる過剰フィルタ
    # バグのどちらでもないことを確認)
    assert len(paired) == 2

    paired_sorted = paired.sort_values("t_sec_1p").reset_index(drop=True)

    # 試合A: 1P(t=100.0) は 2P(t=100.4) とペアになる (2P(t=1300.6)ではない)
    row_a = paired_sorted.iloc[0]
    assert row_a["t_sec_1p"] == pytest.approx(100.0)
    assert row_a["t_sec_2p"] == pytest.approx(100.4)
    assert row_a["t_diff"] == pytest.approx(0.4, abs=1e-6)

    # 試合B: 1P(t=1300.0) は 2P(t=1300.6) とペアになる (2P(t=100.4)ではない)
    row_b = paired_sorted.iloc[1]
    assert row_b["t_sec_1p"] == pytest.approx(1300.0)
    assert row_b["t_sec_2p"] == pytest.approx(1300.6)
    assert row_b["t_diff"] == pytest.approx(0.6, abs=1e-6)

    # 両ペアとも game_idx は同じ値(0)だが、それにもかかわらず
    # 時刻に基づき正しく別試合として扱われたことを明示的に確認する
    assert row_a["game_idx_1p"] == 0 and row_a["game_idx_2p"] == 0
    assert row_b["game_idx_1p"] == 0 and row_b["game_idx_2p"] == 0

    # t_diff が max_tdiff (既定1.0秒) を大きく下回ることも確認
    # (=1200秒級の空白をまたいだ誤ペアはそもそも許容範囲外)
    assert (paired["t_diff"] <= DEFAULT_MAX_TDIFF).all()


# =============================================================================
# 正常系: 同時刻の1P/2P行が正しくペアになる
# =============================================================================


def test_同時刻の1P_2P行は正しくペアになる() -> None:
    """t_diff <= 1.0秒 の1P/2P行が正常にペア化されることを確認する基本ケース。"""
    df = pd.DataFrame([
        _row("video_y", 3, 50.0, "1P", won=1),
        _row("video_y", 3, 50.2, "2P", won=0),
    ])

    paired = pair_sides_for_win(df, DEFAULT_MAX_TDIFF)

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["video_id_1p"] == "video_y"
    assert row["t_sec_1p"] == pytest.approx(50.0)
    assert row["t_sec_2p"] == pytest.approx(50.2)
    assert row["t_diff"] == pytest.approx(0.2, abs=1e-6)
    assert row["won_1p"] == 1
    assert row["won_2p"] == 0


def test_許容時刻差を超える行はペア化されない() -> None:
    """
    |t_sec_1p - t_sec_2p| > max_tdiff の場合はペアが成立しないこと。

    注意: 全行が不成立になると別途固定化した既知のKeyErrorバグ
    (末尾の進捗ログが空DataFrameの"t_diff"列を参照する) を踏むため、
    ここでは正常に成立する別ペアを1組混在させてクラッシュを避けつつ、
    許容時刻差を超えた行だけが除外されることを検証する。
    """
    df = pd.DataFrame([
        # 正常に成立するペア (クラッシュ回避用、かつ「他のペアには影響しない」ことの確認も兼ねる)
        _row("video_y", 5, 200.0, "1P", won=1),
        _row("video_y", 5, 200.1, "2P", won=0),
        # 許容差(1.0秒)を超えるペア候補: 成立してはならない
        _row("video_y", 0, 10.0, "1P", won=1),
        _row("video_y", 0, 12.0, "2P", won=0),  # 差2.0秒 > DEFAULT_MAX_TDIFF(1.0)
    ])

    paired = pair_sides_for_win(df, DEFAULT_MAX_TDIFF)

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["t_sec_1p"] == pytest.approx(200.0)
    # t_sec=10.0 の行がどの2P行とも(誤って)ペア化されていないこと
    assert 10.0 not in paired["t_sec_1p"].values


# =============================================================================
# won 整合チェック: won_1p + won_2p == 1 でないペアは除外される
# =============================================================================


def test_won不整合ペアは除外され整合ペアのみ残る() -> None:
    """
    won_1p + won_2p != 1 (例: 両者とも won=1、データ不整合/誤対応の兆候) の
    ペアは除外され、整合する方のペアだけが結果に残ることを確認する。

    2組を混在させ、全滅(空DataFrame化)によるKeyErrorを避けつつ
    フィルタ挙動そのものを検証する (空DataFrame化時の挙動は別テストで扱う)。
    """
    df = pd.DataFrame([
        # 整合ペア (t=50台): won合計=1 -> 採用される
        _row("video_z", 0, 50.0, "1P", won=1),
        _row("video_z", 0, 50.2, "2P", won=0),
        # 不整合ペア (t=60台、50台とは時刻が離れており誤混同しない): won合計=2 -> 除外される
        _row("video_z", 1, 60.0, "1P", won=1),
        _row("video_z", 1, 60.3, "2P", won=1),
    ])

    paired = pair_sides_for_win(df, DEFAULT_MAX_TDIFF)

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["t_sec_1p"] == pytest.approx(50.0)
    assert row["won_1p"] == 1
    assert row["won_2p"] == 0


# =============================================================================
# 既知バグの固定化 (修正はしない・報告専用)
# =============================================================================


def test_全ペアがwon不整合で除外されると既知のKeyErrorが発生する_バグ固定化() -> None:
    """
    [既知バグ] pair_sides_for_win は won 整合チェックで全行が除外されると
    rows=[] のまま pd.DataFrame([]) を作るため列 "t_diff" が存在せず、
    末尾の進捗ログ print(f"...{paired['t_diff'].median():.2f}...") で
    KeyError: 't_diff' が送出される (model_indicator_win.py:196 付近)。

    本テストは実装を修正せず、現状の(望ましくない)挙動をそのまま固定化する
    ためのものである。将来この関数を触る際に、この挙動が意図的に残されて
    いるものではなく既知の未修正バグであることを気づかせる目的で追加している。
    """
    df = pd.DataFrame([
        _row("video_w", 0, 20.0, "1P", won=1),
        _row("video_w", 0, 20.1, "2P", won=1),  # won合計=2 -> 唯一の候補ペアが除外される
    ])

    with pytest.raises(KeyError, match="t_diff"):
        pair_sides_for_win(df, DEFAULT_MAX_TDIFF)
