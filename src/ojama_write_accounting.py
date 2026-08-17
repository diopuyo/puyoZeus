"""W25根治 第3弾・最終 (2026-08-18): CNN観測入力段の会計整合フィルタ。

docs/KNOWN_WEAKNESSES.md W25、
data/verify/diag_c13c22_recheck_2026-08-17/w25_guard_gap.md 参照。

## 背景 (根治に至った経緯)

おじゃま落下時の白雲パーティクル (HSV S≈14-20/V≈250) が下段セル群を覆い、
CNN が既存の色ぷよセルを誤っておじゃま(9)と観測する。この誤り値は
以下 3 つの独立した下流機構それぞれに焼き付く経路を持つことが実測で
確定した:

    (1) cycle 71n の STABLE 長期不一致 override (案4、2026-08-17)
    (2) DriftDetector needs_resync による sm.reset() (第2弾、2026-08-17)
    (3) 設計C 事後復旧ゲート (`_apply_stable_recovery_gate`) の色→色訂正
        (第3弾で新規確認、enable_recovery_counter_carryover 併用時に
        OJAMA_FALL 振動をまたいで CNN 不一致カウンタが持ち越される)

個別ガードで (1)(2) を塞いでも (3) が残り、「ガード2つ超で根治自動昇格」
(feedback_kill_known_weaknesses) のルールに基づき、下流の各機構を
個別に塞ぐアプローチを終了し、**入力段 (cnn_board が state machine に
渡る直前) で誤り値そのものを訂正する** アプローチに根治した。
下流の (1)(2)(3) はいずれも無改修 (最初から正しい観測を見るため、
訂正の必要自体が生じない)。

## 設計

- `filter_ojama_write_by_accounting`: 1 セル分の純関数 (stateless)。
- `apply_ojama_write_accounting_filter`: 盤面全体への適用ラッパー (純関数)。
- 呼び出し側 (`RecognitionPipeline._step_side`) が「セル単位の直近安定色
  メモリ」(state machine reset の影響を受けない外部 wrapper 保持) と
  「pending おじゃま予告量から算出した列別クレジット」を用意し、本モジュールの
  純関数に渡す。本モジュール自身は一切 state を持たない。

## 実測に基づく設計修正 (2026-08-18、アーキ確認要)

当初のアーキ設計では「非空色セルへの9書込みかつ column_pending_ojama_
credit<=0 の場合のみ」棄却し、credit>0 なら素通しする仕様だった。
c13 実機検証 (`scripts/_verify_w25_3rd_fix_2026-08-17.py`) で、対象9セル
(2P) が **credit>0 の素通し分岐によって解消0/9のまま** であることが判明した。
根因追跡の結果:

  - c13 2P の pending 予告量が試合中盤で `OjamaAccountingTracker` 自身の
    絶対サニティ上限 (`PENDING_ABS_CAP`=216) に達しており
    (score OCR 異常・chain 境界追跡の既存バグに起因、本 W25 修正の対象外)、
    floor(216/6)=36 という現実的にあり得ない大きさのクレジットを生成する。
  - **ぷよぷよのルール上、おじゃまぷよは空セルにのみ着弾する
    (`reference_ojama_landing_pattern`)。非空色セルへ直接 9 が「着弾」する
    という物理事象自体が存在しない** ため、credit の大小に関わらず
    「非空色セル→9」を許容する理由は本来存在しない。
  - 実験的に credit>0 の素通し分岐を無効化 (常時棄却) したところ、対象9セル
    は 9/9 全て解消した (新規悪化0件)。

以上の実測・ルール確認に基づき、`column_pending_ojama_credit` は
**引数として残す** (将来の telemetry / 別用途拡張に備える、シグネチャ
後方互換) が、「非空色セルへの9書込み」の判定には使わず常時棄却する
よう修正する。空セルへの9書込み (正規の着弾経路) は本修正の対象外で
従来どおり無条件で通過する (design要件(c)は不変)。
"""
from __future__ import annotations

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, HIDDEN_ROWS, Board


def filter_ojama_write_by_accounting(
    prev_stable_color: int,
    new_cnn_value: int,
    column_pending_ojama_credit: int,
) -> int:
    """1 セル分の会計整合フィルタ (純関数、stateless)。

    「非空色セルへの 9 書込み」は常に直近安定色 (`prev_stable_color`) へ
    差し替える (= 棄却)。ぷよぷよのルール上おじゃまは空セルにのみ着弾する
    (reference_ojama_landing_pattern) ため、非空色セルへの 9 は
    column_pending_ojama_credit の大小に関わらず物理的に説明不可能であり、
    実測 (c13、本モジュール docstring 冒頭「実測に基づく設計修正」参照) でも
    credit ベースの素通し例外が実害を防げないことが確定している。
    空セルへの 9 書込み (= 正規のおじゃま着弾経路) はフィルタ対象外で
    無条件に素通しする。

    Args:
        prev_stable_color: このセルの直近安定色 (state machine reset の
            影響を受けない外部メモリから取得、未観測セルは COLOR_EMPTY 扱い)。
        new_cnn_value: 今フレームの CNN 観測値。
        column_pending_ojama_credit: このセルが属する列の pending おじゃま
            着弾クレジット (floor(予告個数/6) の6列均等下限保証分のみ、
            呼び出し側で算出済みの値をそのまま渡す)。現在の判定ロジックでは
            未使用 (シグネチャ後方互換・将来拡張用に保持)。

    Returns:
        フィルタ後の値 (書換えが起きなければ `new_cnn_value` そのもの)。
    """
    del column_pending_ojama_credit  # 実測に基づき判定には使わない (docstring参照)
    if prev_stable_color != COLOR_EMPTY and new_cnn_value == COLOR_OJAMA:
        return prev_stable_color
    return new_cnn_value


def apply_ojama_write_accounting_filter(
    cnn_board: Board,
    stable_color_memory: "dict[tuple[int, int], int]",
    column_pending_ojama_credit: int,
) -> Board:
    """盤面全体に会計整合フィルタを適用する (純関数、stateless)。

    `column_pending_ojama_credit` は全列共通の値を渡す想定
    (floor(予告個数/6) は6列均等の下限保証分であり、端数がどの列に
    偏って着弾するかのランダム性は使わない安全側設計のため、列ごとに
    異なる値を計算しない)。

    Args:
        cnn_board: フィルタ対象の CNN 観測盤面 (変更しない、コピーを返す)。
        stable_color_memory: (row, col) -> 直近安定色 の辞書。未登録セルは
            COLOR_EMPTY として扱う (= フィルタ対象外、まだ安定観測が無い
            セルへの書込みは無条件で通す)。
        column_pending_ojama_credit: 全列共通の pending クレジット。

    Returns:
        フィルタ適用後の新規 Board (`cnn_board` 自体は変更しない)。
    """
    filtered = cnn_board.copy()
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            prev = stable_color_memory.get((r, c), COLOR_EMPTY)
            cur = int(cnn_board.get(r, c))
            out = filter_ojama_write_by_accounting(
                prev, cur, column_pending_ojama_credit,
            )
            if out != cur:
                filtered.set(r, c, out)
    return filtered
