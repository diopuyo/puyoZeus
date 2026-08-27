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

## 固着対策 (2026-08-18、data/verify/diag_w25_regression2_2026-08-18/)

「非空色セル→9 を常時棄却」への修正 (上記) は c13 の実害を解消したが、
28チャンクA/B で新規悪化2件 (**永久固着**) が判明した:

  - c10 (r8,1)(r10,2): 着地推論の一過性誤り (フィルタ無しなら 0.8秒で
    自己修復) を、本フィルタが「正しい9観測」まで毎フレーム棄却して
    自己修復機構の入力そのものを汚染し、永久固着させていた。
  - c109 (r3,2): 紫ぷよが CHAIN 中に消去され (凍結中は「空」が STABLE
    確定されないため直近安定色メモリは紫のまま) → 直後に正規のおじゃま
    着弾が発生 → メモリが紫のままの棄却ロジックが正規着弾を永久に拒否。

いずれも「直近安定色メモリが陳腐化 (stale) しているのに、生の CNN 観測が
9 を一貫して示し続けている」という共通構造。**陳腐化したメモリはいつか
実観測に屈服すべき** という別の (棄却側とは独立した) 正当化に基づき、
セル単位の「生CNN観測が連続して9を示している持続時間」
(`OJAMA_REJECT_TIMEOUT_SEC` 秒、SEC基準・呼び出し側で外部wrapperとして
保持) がタイムアウトに達したら棄却を解除し9を受理する。

### 論拠の分離 (重要、2つの正当化は独立)

- **棄却側の論拠**: 白雲パーティクルの持続時間は実測 0.85〜1.0秒
  (n=2波、data/verify/diag_c13c22_recheck_2026-08-17/) であり、雲由来の
  誤観測が連続 `OJAMA_REJECT_TIMEOUT_SEC` (=1.5秒、実測上限の1.5倍
  マージン) に届くことは無い。
- **受理側の論拠**: 上記の雲の実測とは無関係に、「陳腐化したメモリは
  いつか実観測に屈服すべき」という構造的な永久固着防止原則。 これは
  雲の持続時間の長さとは論理的に独立しており、雲がどれだけ短くても
  長くても、メモリが誤って stale なままなら実観測が最終的に勝つべき、
  という別の要請。

この2つを同じ定数 `OJAMA_REJECT_TIMEOUT_SEC` で表現しているのは偶然の
一致ではなく、「雲の実測上限に安全マージンを載せた値」を「stale メモリの
最大許容陳腐化期間」としても採用する、という設計判断であることに注意
(単一の値が2つの異なる論拠を同時に満たすように選定されている)。

## 色→別色棄却 (拡張、2026-08-18、user発見・原因特定済み)

user がレビューで発見した別現象への対処: 連鎖発火時の白〜青の閃光
エフェクトが画面全体を覆い、既に置かれている色ぷよが別の色へ誤読される
(例: 青→緑、赤→黄)。 W25 本体 (上記) は `new_cnn_value == COLOR_OJAMA`
の場合しか介入しないため、この「色→別色」(new_cnn_value が 1〜5 のまま)
は素通ししていた (スコープの取りこぼし、実装不具合ではない)。

`filter_ojama_write_by_accounting` に `reject_color_swap` (既定 False) を
追加し、True の場合のみ「`prev_stable_color` と `new_cnn_value` が共に
色ぷよ (1〜5) かつ不一致」を追加で棄却する。`_stable_color_memory_1p/2p`
をそのまま流用し (別メモリを新設しない)、固着対策も W25 と同型
(`OJAMA_REJECT_TIMEOUT_SEC` 秒連続で同一の誤色を観測したら陳腐化メモリの
屈服として受理、呼出し側 `_update_ojama_color_swap_streak` 参照)。

**スコープを OJAMA_FALL 中に限定** (呼出し側 `RecognitionPipeline.
_step_side` が `sm.context.state == BoardState.OJAMA_FALL` の場合のみ
`reject_color_swap=True` を渡す): おじゃま落下中は物理的に色ぷよが
別の色へ変わることはあり得ないため、副作用が理論上ゼロ。CHAIN 中も同種の
違反が大量に観測されているが、多段消去・重力補充の高速遷移とのエイリア
シングが未精査のため、今回は対象外 (フラグ
`enable_ojama_fall_color_swap_guard` は OJAMA_FALL 限定、CHAIN では
一切発火しない設計)。

既存の `reject_ojama_write` (既定 True、W25 本体の色→9棄却の有効/無効
切替) と独立した引数のため、`enable_ojama_write_accounting_guard` が
False でも `enable_ojama_fall_color_swap_guard` 単独で機能する
(呼出し側でどちらか一方が True なら本モジュールが呼ばれ、それぞれの
判定は個別の flag で独立に有効化される)。

### フリッカ許容なし (重要)

ストリーム判定は 1 フレームでも生CNN観測が9以外を示したら即座にリセット
する (許容フレームを設けない)。 許容を入れると「雲が断続的に (フリッカ
しながら) 累積 `OJAMA_REJECT_TIMEOUT_SEC` 秒分観測される」というシナリオ
が発生した場合、棄却側の論拠 (雲は連続 0.85〜1.0秒で晴れる) が崩れて
しまう。 呼び出し側 (`RecognitionPipeline._update_ojama_raw9_streak`) が
この厳格なリセット規則を実装する。

### 新規許容の明文化 (暗黙ガード禁止、user恒久指示対応)

本タイムアウトの導入により、「正規のおじゃま着弾の反映が最大
`OJAMA_REJECT_TIMEOUT_SEC` (=1.5) 秒遅れる」ケースが新たに生じ得る
(直近安定色メモリが誤って stale な色を保持していた場合、実際の着弾から
最大1.5秒間は棄却され続け、タイムアウト到達で初めて反映される)。
既存の「設置→盤面反映は8フレーム以内」(feedback_placement_reflection_
8frames) はツモ設置 (TSUMO_FALL) の反映遅延基準であり、本モジュールが
扱うおじゃま着弾の反映経路には元々適用されない (対象が異なる)。
既存の着弾遅延基準 (reference_ojama_landing_gated_by_placement 等) も
サブ秒精度を要求しない構造 (連鎖アニメ時間・ターン単位の粒度) のため、
本タイムアウトによる新規許容は既存基準と抵触しない。

### 色→別色棄却の固着対策 (2026-08-18)

「色→別色棄却」も W25 同様に永久固着リスクを持つ (直近安定色メモリが
陳腐化した状態で、閃光後の本当の色変化 [連鎖消去に伴う次の色ぷよ設置等]
を誤って棄却し続ける可能性)。 呼出し側 (`RecognitionPipeline.
_update_ojama_color_swap_streak`) が「セル単位の連続同一誤色観測時間」を
追跡し、`OJAMA_REJECT_TIMEOUT_SEC` 到達で受理する、W25 と同型のタイムアウト
機構を持つ。 フリッカ許容なし (観測値が変わったら新しいストリークとして
再起動、W25 と同じ理由)。
"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    HIDDEN_ROWS,
    Board,
)

# W25固着対策 (2026-08-18): 直近安定色メモリが陳腐化 (stale) している場合の
# 永久固着防止タイムアウト。 生CNN観測が連続してこの秒数だけ9を示したら、
# 直近安定色メモリを信用せず9を受理する (SEC基準、フレーム基準禁止・W4教訓)。
# 根拠は2つの独立した正当化から成る (詳細はモジュール docstring
# 「固着対策」節参照、棄却側と受理側で論拠が異なることに注意):
#   - 棄却側: 白雲パーティクル持続時間実測 0.85〜1.0秒 (n=2波) の
#     1.5倍相当の安全マージン → 雲由来の誤観測はこの秒数に届かない。
#   - 受理側: 陳腐化したメモリはいつか実観測に屈服すべき、という構造的な
#     永久固着防止原則 (雲の実測とは無関係な別の要請)。
# 色→別色棄却 (拡張、2026-08-18) のタイムアウトにも同じ定数を流用する
# (モジュール docstring「色→別色棄却の固着対策」節参照、閃光の持続時間は
# 雲パーティクルと同系の演出エフェクトのため同一マージンで安全)。
OJAMA_REJECT_TIMEOUT_SEC: float = 1.5

# 色→別色棄却 (2026-08-18) が対象とする「色ぷよ」値の範囲 (おじゃま(9)・
# 空(0)・不明(10) を含まない、board.py の COLOR_RED〜COLOR_PURPLE が連番で
# あることに依存、マジックナンバー回避)。
_MIN_COLOR_PUYO: int = COLOR_RED
_MAX_COLOR_PUYO: int = COLOR_PURPLE


def filter_ojama_write_by_accounting(
    prev_stable_color: int,
    new_cnn_value: int,
    column_pending_ojama_credit: int,
    consecutive_raw9_duration_sec: float = 0.0,
    *,
    reject_ojama_write: bool = True,
    reject_color_swap: bool = False,
    consecutive_color_swap_duration_sec: float = 0.0,
) -> int:
    """1 セル分の会計整合フィルタ (純関数、stateless)。

    「非空色セルへの 9 書込み」は (`reject_ojama_write=True`、既定 True =
    従来挙動) 常に直近安定色 (`prev_stable_color`) へ差し替える (= 棄却)。
    ぷよぷよのルール上おじゃまは空セルにのみ着弾する
    (reference_ojama_landing_pattern) ため、非空色セルへの 9 は
    column_pending_ojama_credit の大小に関わらず物理的に説明不可能であり、
    実測 (c13、本モジュール docstring 冒頭「実測に基づく設計修正」参照) でも
    credit ベースの素通し例外が実害を防げないことが確定している。
    空セルへの 9 書込み (= 正規のおじゃま着弾経路) はフィルタ対象外で
    無条件に素通しする。

    固着対策 (2026-08-18、モジュール docstring「固着対策」節参照):
    `consecutive_raw9_duration_sec` (呼び出し側が算出する、生CNN観測が
    連続してこの秒数だけ9を示している持続時間) が `OJAMA_REJECT_TIMEOUT_
    SEC` 以上ならタイムアウト受理し、`prev_stable_color` に関わらず
    `new_cnn_value` (=9) をそのまま返す。 直近安定色メモリが陳腐化
    (stale) している場合の永久固着を防ぐ (棄却側の論拠とは独立、詳細は
    モジュール docstring 参照)。

    色→別色棄却 (拡張、2026-08-18、モジュール docstring「色→別色棄却」節
    参照): `reject_color_swap=True` の場合のみ、`prev_stable_color` と
    `new_cnn_value` が共に色ぷよ (1〜5) かつ不一致のケースも追加で棄却する
    (連鎖発火の閃光エフェクトによる色→別色誤読への対処、既定 False =
    従来挙動と bit-identical)。 固着対策は W25 本体と同型
    (`consecutive_color_swap_duration_sec` が `OJAMA_REJECT_TIMEOUT_SEC`
    以上ならタイムアウト受理)。

    Args:
        prev_stable_color: このセルの直近安定色 (state machine reset の
            影響を受けない外部メモリから取得、未観測セルは COLOR_EMPTY 扱い)。
        new_cnn_value: 今フレームの CNN 観測値。
        column_pending_ojama_credit: このセルが属する列の pending おじゃま
            着弾クレジット (floor(予告個数/6) の6列均等下限保証分のみ、
            呼び出し側で算出済みの値をそのまま渡す)。現在の判定ロジックでは
            未使用 (シグネチャ後方互換・将来拡張用に保持)。
        consecutive_raw9_duration_sec: 生CNN観測が連続して9を示している
            持続時間 (秒、呼び出し側が算出。フリッカ [1frameでも9以外を
            観測] があれば即0にリセットされた値を渡すこと)。既定 0.0
            (backwards compat、渡さなければタイムアウト機能は無効)。
        reject_ojama_write: W25 本体 (色→9棄却) の有効/無効切替。既定 True
            (backwards compat、既存呼出しは全て従来挙動のまま)。
        reject_color_swap: 色→別色棄却 (拡張) の有効/無効切替。既定 False
            (backwards compat、既定 OFF = 従来挙動 bit-identical)。
        consecutive_color_swap_duration_sec: 生CNN観測が連続して同一の
            (直近安定色と異なる) 色を示している持続時間 (秒)。既定 0.0。

    Returns:
        フィルタ後の値 (書換えが起きなければ `new_cnn_value` そのもの)。
    """
    del column_pending_ojama_credit  # 実測に基づき判定には使わない (docstring参照)
    if prev_stable_color == COLOR_EMPTY:
        return new_cnn_value
    if reject_ojama_write and new_cnn_value == COLOR_OJAMA:
        if consecutive_raw9_duration_sec >= OJAMA_REJECT_TIMEOUT_SEC:
            return new_cnn_value  # 固着対策: タイムアウト到達→実観測を受理
        return prev_stable_color
    if (
        reject_color_swap
        and _MIN_COLOR_PUYO <= prev_stable_color <= _MAX_COLOR_PUYO
        and _MIN_COLOR_PUYO <= new_cnn_value <= _MAX_COLOR_PUYO
        and new_cnn_value != prev_stable_color
    ):
        if consecutive_color_swap_duration_sec >= OJAMA_REJECT_TIMEOUT_SEC:
            return new_cnn_value  # 固着対策: タイムアウト到達→実観測を受理
        return prev_stable_color
    return new_cnn_value


def apply_ojama_write_accounting_filter(
    cnn_board: Board,
    stable_color_memory: "dict[tuple[int, int], int]",
    column_pending_ojama_credit: int,
    consecutive_raw9_duration_by_cell: "dict[tuple[int, int], float] | None" = None,
    *,
    reject_ojama_write: bool = True,
    reject_color_swap: bool = False,
    consecutive_color_swap_duration_by_cell: "dict[tuple[int, int], float] | None" = None,
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
        consecutive_raw9_duration_by_cell: (row, col) -> 生CNN観測が連続
            して9を示している持続時間 (秒)。未登録セルは 0.0 扱い
            (= タイムアウト未到達、従来の棄却ロジックのみ適用)。
            既定 None (backwards compat、渡さなければタイムアウト機能は
            全セルで無効)。
        reject_ojama_write: W25 本体 (色→9棄却) の有効/無効切替。既定 True
            (backwards compat)。
        reject_color_swap: 色→別色棄却 (拡張、2026-08-18) の有効/無効切替。
            既定 False (backwards compat、既定 OFF = 従来挙動 bit-identical)。
        consecutive_color_swap_duration_by_cell: (row, col) -> 生CNN観測が
            連続して同一の (直近安定色と異なる) 色を示している持続時間
            (秒)。未登録セルは 0.0 扱い。既定 None。

    Returns:
        フィルタ適用後の新規 Board (`cnn_board` 自体は変更しない)。
    """
    consecutive_raw9_duration_by_cell = consecutive_raw9_duration_by_cell or {}
    consecutive_color_swap_duration_by_cell = (
        consecutive_color_swap_duration_by_cell or {}
    )
    filtered = cnn_board.copy()
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            prev = stable_color_memory.get((r, c), COLOR_EMPTY)
            cur = int(cnn_board.get(r, c))
            duration9 = consecutive_raw9_duration_by_cell.get((r, c), 0.0)
            duration_swap = consecutive_color_swap_duration_by_cell.get(
                (r, c), 0.0,
            )
            out = filter_ojama_write_by_accounting(
                prev, cur, column_pending_ojama_credit, duration9,
                reject_ojama_write=reject_ojama_write,
                reject_color_swap=reject_color_swap,
                consecutive_color_swap_duration_sec=duration_swap,
            )
            if out != cur:
                filtered.set(r, c, out)
    return filtered
