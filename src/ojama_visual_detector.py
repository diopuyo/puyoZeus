"""OjamaVisualDetector — 視覚ベースのお邪魔降下検知 (フェーズ A 精緻化).

連鎖終了直後のお邪魔降下を「可視最上段付近の ROI にお邪魔色が出現」する
視覚情報で確実に捉え、 infer_placement 誤配置 (幽霊) の根絶を目指す。

設計方針:
  - 一次判定: cnn_board の対象 ROI (HIDDEN_ROWS 〜 HIDDEN_ROWS+OJAMA_ROI_HEIGHT 行)
    に COLOR_OJAMA(9) が存在、 または hsv_board が非 None ならそこにも存在。
  - 発火条件: 一次 positive かつ (前フレームより増加 or 前が 0) かつ
    OJAMA_CONSEC_THRESH フレーム連続観測 → BoardState.OJAMA_FALL へ遷移。
  - 完了判定: OJAMA_FALL 中に ROI お邪魔 count==0 で即 STABLE 復帰。
    enable_ojama_settle_detection=True 時はさらに OJAMA_SETTLE_CONSEC フレーム
    不変でも STABLE 復帰 (お邪魔が静止した = 落下完了) 判定を行う。
  - stateless Protocol 準拠: 内部 state は dataclass field で管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, HIDDEN_ROWS, Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    GRAVITY_SETTLE_MAX_SEC,
    GRAVITY_SETTLE_MIN_FRAMES,
    GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
    GRAVITY_SETTLE_PHYSICS_CLEAR_MIN_SEC,
    GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD,
    StateContext,
)

# ============================
# 定数
# ============================

# お邪魔検知 ROI: 可視最上段 (HIDDEN_ROWS 行目) から何行を対象とするか。
# ぷよぷよ eスポーツではお邪魔は最上段から降ってくるため 2 行で十分。
OJAMA_ROI_HEIGHT: int = 2

# 発火に必要な連続観測フレーム数。1 フレームの CNN ぶれを除去する。
OJAMA_CONSEC_THRESH: int = 2

# settle 判定: OJAMA_FALL 中に ROI お邪魔 count が N フレーム不変なら
# お邪魔が静止完了と判断して STABLE 復帰する。
OJAMA_SETTLE_CONSEC: int = 3

# 案B (第2の根本原因対処, 2026-07-24):
# OJAMA_FALL 退出が早すぎ (滞在中央値 0.0167s = 1 frame) で着弾前に状態が抜け、
# empty_to_color 3 票ゲートが成立せず本物おじゃまが却下される問題への対処。
# GravitySettleDetector (src/state_detectors.py) と同じ「全盤面ぷよ数の静止」
# 判定方式を OJAMA_FALL 退出条件にも適用する。
# 定数は GravitySettle 側の値を初期値として流用するが、 今後 ojama 固有の
# チューニングが chain (GRAVITY_SETTLE) 側に波及しないよう独立命名する。
OJAMA_FALL_SETTLE_MIN_FRAMES: int = GRAVITY_SETTLE_PHYSICS_CLEAR_MIN
# フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正として使う。
# OJAMA_FALL_SETTLE_MIN_FRAMES (frame 定数) は既存 import 互換のため残置する。
# 60fps 動画では (frame_idx 差分)/60 == time_sec 差分 が恒等式のため
# _detect_ojama_fall_exit_board_settle の判定は bit-identical。
OJAMA_FALL_SETTLE_MIN_SEC: float = GRAVITY_SETTLE_PHYSICS_CLEAR_MIN_SEC
OJAMA_FALL_SETTLE_STABLE_FRAMES: int = GRAVITY_SETTLE_MIN_FRAMES
OJAMA_FALL_SETTLE_DIFF_THRESHOLD: int = GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD
OJAMA_FALL_MAX_SEC: float = GRAVITY_SETTLE_MAX_SEC

# 案4-lite (enable_ojama_fall_entry_hardening, 2026-08-13 根因調査追補):
# OJAMA_CONSEC_THRESH (=2 frame 連続) の実時間版。 テストコード
# (tests/test_ojama_visual_detector.py の `_make_signals`) 及び
# --normalize-fps-30 収集経路は time_sec = frame_idx/30 (30fps 前提) で
# 一貫しているため、 GRAVITY_SETTLE_PHYSICS_CLEAR_MIN_SEC (/60 換算、
# 60fps 前提) とは分母を変える。 診断 (--stride 2) で実測した通り、
# frame 数ベースの「2 フレーム連続」は間引き幅 (stride) に応じて実時間の
# 意味が伸縮する (docs/DEMO_REVIEW_2026-08-13.md 場面2)。 実時間基準に
# 切り替えることでこの非対称を解消する。
OJAMA_ENTRY_CONSEC_SEC: float = OJAMA_CONSEC_THRESH / 30.0

# 案4-lite: ctx.state == CHAIN からの OJAMA_FALL entry (= CHAIN 割り込み) に
# 要求する持続時間の倍率。 ChainPhaseDetector が chain_event の瞬間欠落
# (stride 間引き下で発生しやすい) 時に ojama_top_positive 保留で None を
# 返すと、 本 detector が同一 frame 内で ctx.state==CHAIN のまま entry 判定に
# 呼ばれ、 通常閾値のまま CHAIN を横取りしてしまう (場面2 の実測根因)。
# CHAIN からの割り込みのみ必要時間を延長し、 瞬間的な検出空白への割り込みを
# 抑止する。
OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER: int = 3

# 案4-lite 拡張 (coordinator追加指示, 2026-08-13、場面2実測 52→69 の悪化対処):
# 「ctx.state == CHAIN」の瞬間条件だけでは実際の割り込み経路を捉えられない
# (実測: chain_to_ojama_interrupts は baseline でも常に 0 = 直接の
# CHAIN→OJAMA_FALL 隣接フレーム遷移は存在しない)。実際は
# 「chain_event 検出が数秒の空白を作る → state が既に CHAIN を離れた後
# (GRAVITY_SETTLE 等の中間 state 経由) に OJAMA_FALL entry が判定される」
# という state 非依存の経路であるため、「直近 T 秒以内に自 side の
# chain_event がアクティブだった」を実時間で判定する方式に拡張する。
# T は連鎖1リンクの実測 (~0.4秒) の 2〜3 倍 (= 空白がここに収まる想定)。
OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC: float = 0.4 * 2.5

# 案4-lite 拡張 (coordinator追加指示, 2026-08-13、悪化原因の切り分け対象):
# 案2 (enable_ojama_fall_placement_override) の早期 exit 直後は ROI に
# visual artifact が残っている可能性があり、 即座に再度 entry 判定される
# と「exit→即re-entry」の振動を誘発しうる。 exit 直後 本秒数以内の entry は
# CHAIN 割り込みと同じ倍率で厳格化する (enable_ojama_fall_entry_hardening
# 配下、 案2単体 (entry_hardening OFF) には一切影響しない)。
OJAMA_REENTRY_SUPPRESS_SEC: float = 0.4

# 案1 (enable_ojama_fall_scoped_exit, 2026-08-13、OJAMA_FALL出口の根治、
# docs/BURST_GUARD_DESIGN_2026-08-05.md §12.5 原典): 案B (全盤面 settle) は
# 盤面全体のぷよ数 (色ぷよ込み) の静止を待つため、 自分のツモ設置で数値が
# 変化し続け出口判定を塞ぐ (根因調査 2026-08-13 で実測: 設置ブロック16/16件、
# 高速振動)。 根治は「おじゃまセルの個数」のみを監視対象にし、 色ぷよの
# 増減を無視する。 定数は案B系 (Board Settle) の値を初期値として流用しつつ、
# 今後 scoped 系固有のチューニングが他系に波及しないよう独立命名する。
OJAMA_FALL_SCOPED_EXIT_MIN_SEC: float = OJAMA_FALL_SETTLE_MIN_SEC
OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES: int = OJAMA_FALL_SETTLE_STABLE_FRAMES
OJAMA_FALL_SCOPED_EXIT_DIFF_THRESHOLD: int = OJAMA_FALL_SETTLE_DIFF_THRESHOLD

# 案1 会計連動: 未着弾予告量 (会計トラッカー由来、 getattr 安全参照) が
# 0 以下 (= もう降ってくる予告が無い) のとき、 静止確認に必要な連続フレーム数
# をこの倍率で短縮し、 滞在上限を短くする。 最低 1 フレームは静止確認する
# 安全弁として `max(1, ...)` を必ず適用する。
OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR: int = 2

# 案2修正 (2026-08-15、実害46セル根治、data/verify/yardstick_v2_2026-08-14/
# scoring_ablation/_diag_c1_new_regressions_2026-08-15.json 診断+実データ
# 直接確認で確定): own_score_delta>0 の単一フレーム即決 evidence 判定が、
# (a) score OCR の孤立 1 フレームぶれ (実測: c17/c96/c13/c23/c10 の5シート
# 全てで own_score_delta=1 のような小さい孤立増分が繰り返し観測され、
# 直前直後の own_chain_hold_until_sec は time_sec から数秒以上前=自chain無関係
# だった)、 (b) 自 side の小連鎖・全消しボーナスの roll-up アニメ中の score
# 上昇 (実測: own_score_delta=320/2100、 own_chain_hold_until_sec が
# time_sec 近傍=自chain継続中) の両方を「実設置」と誤認していた。
#
# 修正は (b) の chain-active 除外のみを適用する ((a) 用の実時間ヒステリシス
# 案は 55盤面物差しで実測した上で不採用に確定、 理由は下記
# `_confirm_placement_evidence` の docstring 参照)。

# 案2修正 (b) 用: 「直近何秒以内に自 chain がアクティブだったか」の判定窓。
# OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC (連鎖1リンク実測 ~0.4秒の2〜3倍) と
# 同じ物理的根拠を再利用する (独立命名)。
OJAMA_FALL_PLACEMENT_SCORE_CHAIN_EXCLUDE_SEC: float = (
    OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC
)

# ROI の開始行 (= HIDDEN_ROWS = 1 = 可視最上段)
_ROI_ROW_START: int = HIDDEN_ROWS
# ROI の終了行 (exclusive)
_ROI_ROW_END: int = HIDDEN_ROWS + OJAMA_ROI_HEIGHT


# ============================
# ヘルパー (static)
# ============================


def _count_top_ojama(
    cnn_board: Board,
    hsv_board: "Board | None" = None,
) -> int:
    """対象 ROI 内の COLOR_OJAMA セル数を返す (一次判定).

    cnn_board に OJAMA が存在するか、 または hsv_board 非 None かつそちらにも
    OJAMA が存在する場合に count を返す。
    両方 0 なら 0 を返し「お邪魔なし」 と判断する。

    Args:
        cnn_board: CNN 認識結果の盤面。
        hsv_board: HSV-only 認識盤面 (None なら cnn_board のみで判定)。

    Returns:
        ROI 内の OJAMA セル数 (0 以上の整数)。
    """
    count_cnn: int = 0
    for r in range(_ROI_ROW_START, _ROI_ROW_END):
        for c in range(BOARD_COLS):
            if int(cnn_board.get(r, c)) == COLOR_OJAMA:
                count_cnn += 1

    if hsv_board is None:
        return count_cnn

    count_hsv: int = 0
    for r in range(_ROI_ROW_START, _ROI_ROW_END):
        for c in range(BOARD_COLS):
            if int(hsv_board.get(r, c)) == COLOR_OJAMA:
                count_hsv += 1

    # CNN または HSV 一方でも検知した場合はその最大値を採用 (感度優先)
    return max(count_cnn, count_hsv)


def _count_board_ojama(board: Board) -> int:
    """盤面可視領域全体 (HIDDEN_ROWS〜最下段) の COLOR_OJAMA セル数を返す.

    案1 (enable_ojama_fall_scoped_exit) の静止判定専用ヘルパー。
    `_count_top_ojama` の ROI (可視最上段 2 行のみ) はおじゃまが ROI を
    通過して下段に着地した後は捕捉できないため、 「まだ盤面のどこかで
    おじゃまが動いているか」 を見るにはスコープを盤面全体に広げる必要が
    ある。 色ぷよは対象外とし、 COLOR_OJAMA セルのみを数える (= 自分の
    ツモ設置による色ぷよの増減はこの値に一切影響しない)。

    Args:
        board: 対象盤面 (通常は signals.cnn_board)。

    Returns:
        可視領域内の COLOR_OJAMA セル数 (0 以上の整数)。
    """
    count = 0
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            if int(board.get(r, c)) == COLOR_OJAMA:
                count += 1
    return count


def _has_ojama_fall_placement_evidence(signals: DetectorSignals) -> bool:
    """案2: OJAMA_FALL 滞在中の「生」実設置証拠を判定する (stateless ヘルパー).

    ⚠️ この関数の True は「即座に STABLE へ復帰してよい」ことを意味しない
    (2026-08-15 修正)。 own_score_delta 経路は自 chain roll-up を拾って
    しまうため、 実際の確定判定には `OjamaVisualDetector.
    _confirm_placement_evidence` (chain-active 除外つき) を使う。 本関数は
    「今フレーム単体で見た生の signal 有無」のみを返す stateless ヘルパー
    として、 従来の呼び出し互換 (直接呼ぶテスト等) のために残す。

    実測パターン (docs/DEMO_REVIEW_2026-08-13.md 場面1): 全盤面ぷよ数の
    静止判定 (`_detect_ojama_fall_exit_board_settle`) は「盤面全体」を
    見るため自分のツモ設置でもぷよ数が変化し続け、 OJAMA_FALL⇔STABLE が
    0.15-0.3 秒周期で振動する (出口判定のスコープ過大)。 以下いずれかを
    検知したら実設置の可能性ありとみなす (確定は呼び出し側の責務)。

    - signals.slide_motion: NEXT ROI のスライド検知
      (= 次ツモへ繰り上がった = 設置完了の物理的証拠、 TsumoPhaseDetector の
      R-7 と同じ signal を流用)。
    - signals.own_score_delta > 0: 自 side の score 増分 (= 落下ボーナス等)。
      おじゃま受け側は連鎖しないため自 score は通常動かない。 増分があれば
      設置イベントの証拠 (reference_ojama_landing_gated_by_placement:
      おじゃまは受け側ツモ着地時に降る、 の裏付けとも整合する) …という
      2026-08-13 導入時の前提だったが、 診断で「小連鎖・全消しの roll-up」
      と「単発 OCR ぶれ」の両方を誤検知することが確定した (定数コメント参照)。

    Args:
        signals: 現 frame の DetectorSignals。

    Returns:
        生の実設置 signal (slide_motion or own_score_delta>0) が今フレーム
        観測できたか。
    """
    if signals.slide_motion:
        return True
    return signals.own_score_delta > 0


def _is_own_chain_recently_active_for_score_exclusion(
    signals: DetectorSignals,
) -> bool:
    """案2修正: 「直近まで自 side の chain がアクティブだったか」を判定する.

    own_score_delta による実設置誤検知のうち、 自 chain の roll-up アニメ
    (score が連鎖アニメの間フレームごとに増え続ける) が原因のケースを除外
    するための判定。 `_is_hardened_entry_context` (entry hardening 機能側)
    の条件4と同じ計算式だが、 本関数は「案2修正 (score_delta 除外)」専用の
    独立実装として切り離す (今後の調整が entry hardening 側に波及しない
    ようにするため、 既存コードベースの独立命名規約と同じ方針)。

    signals.own_chain_hold_until_sec <= 0.0 は「まだ一度も自 chain が発生
    していない」sentinel (RecognitionPipeline.reset() の初期値と同じ規約)
    のため除外条件から外す (序盤の time_sec が小さい間の誤判定防止、
    `_is_hardened_entry_context` と同じ既知の落とし穴)。

    Args:
        signals: 現 frame の DetectorSignals。

    Returns:
        直近 `OJAMA_FALL_PLACEMENT_SCORE_CHAIN_EXCLUDE_SEC` 秒以内に自 chain
        がアクティブだったか。
    """
    return (
        signals.own_chain_hold_until_sec > 0.0
        and signals.time_sec - signals.own_chain_hold_until_sec
        < OJAMA_FALL_PLACEMENT_SCORE_CHAIN_EXCLUDE_SEC
    )


# ============================
# OjamaVisualDetector
# ============================


@dataclass
class OjamaVisualDetector:
    """視覚ベースのお邪魔降下検知 detector.

    OjamaPhaseDetector (score 差分ベース) より前に登録し、
    連鎖終了直後のお邪魔降下を視覚で即座に捉える。

    フラグ (RecognitionPipeline 側で注入):
        enable_ojama_visual_chain_exit: True で CHAIN 中でも発火を許可。
        enable_ojama_settle_detection: True で settle (お邪魔静止) 判定を有効化。
        enable_ojama_fall_board_settle: 案B (2026-07-24)。True で OJAMA_FALL
            退出条件を「全盤面ぷよ数が静止するまで待つ」方式 (GravitySettle と
            同型) に切替える。default False = 既存挙動と完全 bit-identical。
        enable_ojama_entry_gravity_settle_guard: 修正B (2026-08-08、バグB)。
            True で GRAVITY_SETTLE 中の新規 OJAMA_FALL 発火を禁止する。
            連鎖の段間重力待ち (GRAVITY_SETTLE) 中に本 detector が横取りして
            OJAMA_FALL に遷移すると、 GravitySettleDetector の内部カウンタが
            残留し、 次回 GRAVITY_SETTLE 再進入時に誤って 1 frame で STABLE
            化するバグ (バグC、 src/state_detectors.py 側の
            enable_gravity_settle_reset_on_exit と対) を誘発する。
            default False = 既存挙動と完全 bit-identical。
        enable_ojama_fall_placement_override: 案2 (2026-08-13、OJAMA_FALL
            誤分類根因調査、2026-08-15 evidence 一発判定欠陥修正)。True で
            OJAMA_FALL 滞在中に実設置の証拠 (NEXT スライド、 または自 side
            score の非連鎖増分) を検知したら、 settle 判定を待たず即座に
            STABLE へ復帰する。全盤面ぷよ数の静止を待つ既存の退出判定
            (`_detect_ojama_fall_exit_board_settle`) は自分のツモ設置でも
            盤面全体のぷよ数が変化し続けるため OJAMA_FALL⇔STABLE が
            0.15-0.3 秒周期で振動する実害が確認された
            (docs/DEMO_REVIEW_2026-08-13.md 場面1)。
            2026-08-15 追加修正 (data/verify/yardstick_v2_2026-08-14/
            scoring_ablation/_diag_c1_new_regressions_2026-08-15.json 診断):
            導入時の own_score_delta>0 単一フレーム即決は (a) score OCR の
            孤立ぶれ、 (b) 自 chain roll-up アニメ中の score 上昇、 の両方を
            実設置と誤認し実害46セルを持ち込んでいた。 `_confirm_placement_
            evidence` で直近自 chain アクティブなら own_score_delta を評価
            対象から除外するよう変更した ((b) の根治)。 (a) 用の実時間
            ヒステリシス (2フレーム連続要求) も試作したが、 55盤面物差しの
            再実測で **有意に悪化** (新規劣化 46→57セル、 全体一致率
            95.18%→93.95%) したため不採用に確定した — 真因は
            `collect_boards_lean._should_emit` が STABLE 復帰時のみ snapshot
            を記録する仕様で、 ヒステリシスが早期 exit の頻度を落とすと
            OJAMA_FALL 中の snapshot 空白期間が伸び、 物差し query フレームが
            古い (もっと前の) snapshot に nearest match してしまう二次被害が
            (a) 自体を残す一次効果を上回った。 chain除外のみ (ヒステリシス
            無し) では 46→25セル (46%減) かつ全体一致率 95.18%→95.63%
            (改善) を実測、 この構成を最終採用する。
            default False = 既存挙動と完全 bit-identical
            (フラグ OFF 時は `_confirm_placement_evidence` 自体が呼ばれない)。
        enable_ojama_fall_entry_hardening: 案4-lite (2026-08-13、根因調査追補、
            coordinator追加指示で拡張)。True で OJAMA_FALL entry 判定を
            frame 数連続でなく実時間連続 (`OJAMA_ENTRY_CONSEC_SEC`) に
            切り替え、 `_is_hardened_entry_context` が True の間
            (a) ctx.state==CHAIN、 (b) 直近 `OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_
            SEC` 秒以内に自 side の chain_event がアクティブだった
            (state 非依存、 chain_event 検出の空白で state が既に CHAIN を
            離れた後の割り込みも捉える)、 (c) 直近
            `OJAMA_REENTRY_SUPPRESS_SEC` 秒以内に案2 (placement override) の
            早期 exit があった (exit→即re-entry 振動対策) — のいずれかで
            持続時間を `OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER` 倍に厳格化する。
            stride 間引き下で chain_event が瞬間的に欠落した隙に低発火閾値の
            OJAMA_FALL entry が CHAIN を奪う実害への対策 (場面2)。
            default False = 既存挙動と完全 bit-identical。
        enable_ojama_fall_scoped_exit: 案1 (2026-08-13、OJAMA_FALL出口の根治、
            docs/BURST_GUARD_DESIGN_2026-08-05.md §12.5 原典)。True で
            OJAMA_FALL 退出条件を「おじゃまセル限定の個数安定」
            (`_count_board_ojama`、 色ぷよの増減は無視) に切り替える。
            案B (`enable_ojama_fall_board_settle`) は盤面全体のぷよ数を見る
            ため自分のツモ設置でも数値が変化し続け出口判定を塞ぐ
            (根因調査で実測: 設置ブロック16/16件、 高速振動)。 さらに
            `signals.own_pending_ojama_forecast` (お邪魔会計トラッカー由来、
            getattr 安全参照) が 0 以下なら静止確認に必要な連続フレーム数を
            `OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR` で短縮する。
            本フラグが True の間は `enable_ojama_fall_board_settle` より
            優先され、 案2 (`enable_ojama_fall_placement_override`) とは
            併存可能 (案2 の即時 exit 判定が本フラグより先にチェックされる
            ため矛盾しない)。 default False = 既存挙動と完全 bit-identical。

    いずれも False のままなら既存挙動に近い動作となる (= STABLE/TSUMO_FALL 時の
    新規お邪魔出現のみ OJAMA_FALL に遷移)。
    """

    # 外部フラグ: RecognitionPipeline 側から __init__ 後に代入する。
    enable_ojama_visual_chain_exit: bool = False
    enable_ojama_settle_detection: bool = False
    # 案B (2026-07-24): OJAMA_FALL 退出を全盤面 settle 判定に置き換えるフラグ。
    # default False で `_detect_ojama_fall_exit` (既存ロジック) を完全維持する。
    enable_ojama_fall_board_settle: bool = False
    # 修正B (2026-08-08、バグB): GRAVITY_SETTLE 中の OJAMA_FALL 新規発火を
    # 禁止するガード。default False = 既存挙動と完全 bit-identical。
    enable_ojama_entry_gravity_settle_guard: bool = False
    # 案2 (2026-08-13、OJAMA_FALL誤分類根因調査): 実設置検知による早期 exit。
    # default False = 既存挙動と完全 bit-identical。
    enable_ojama_fall_placement_override: bool = False
    # 案4-lite (2026-08-13、根因調査追補): entry 判定の実時間ハードニング。
    # default False = 既存挙動と完全 bit-identical。
    enable_ojama_fall_entry_hardening: bool = False
    # 案1 (2026-08-13、OJAMA_FALL出口の根治): おじゃまセル限定の静止判定。
    # default False = 既存挙動と完全 bit-identical。
    enable_ojama_fall_scoped_exit: bool = False

    # 内部 state (dataclass field、 init=False)
    _consec_count: int = field(default=0, init=False, repr=False)
    _prev_top_ojama_count: int = field(default=0, init=False, repr=False)
    _last_frame_idx: int = field(default=-1, init=False, repr=False)
    _settle_count: int = field(default=0, init=False, repr=False)
    _prev_settle_count: int = field(default=0, init=False, repr=False)
    # 案B: 全盤面 settle 判定用の内部 state (GravitySettleDetector と同型)。
    _settle_start_frame: int = field(default=-1, init=False, repr=False)
    _settle_start_time: float = field(default=0.0, init=False, repr=False)
    _board_stable_consec: int = field(default=0, init=False, repr=False)
    _prev_board_puyo_count: int = field(default=-1, init=False, repr=False)
    # 案4-lite: 実時間ベース entry 判定用の開始時刻 (-1.0 = 未開始)。
    _entry_trigger_start_time: float = field(
        default=-1.0, init=False, repr=False,
    )
    # 案4-lite 拡張 (coordinator追加指示, 2026-08-13): 案2 (placement
    # override) による直近の早期 exit 時刻 (-1.0 = 未発生)。
    # `_reset_internal_state()` では意図的にクリアしない
    # (re-entry 抑制窓の判定に「exit してから」の経過時間が必要なため)。
    _placement_override_exit_time: float = field(
        default=-1.0, init=False, repr=False,
    )
    # 案4-lite 拡張・第2ラウンド (coordinator追加指示, 2026-08-13、実データ
    # 直接確認で判明): ChainPhaseDetector が最優先のため、 本 detector の
    # entry 判定は「真に CHAIN 中」はほぼ呼ばれない (chain_event がアクティブ
    # な間 ChainPhaseDetector が毎 frame 勝つため)。 実際に呼ばれるのは
    # CHAIN 終了直後の GRAVITY_SETTLE 中、 またはそこから即 STABLE に
    # 抜けた直後の 1-2 frame。 ctx.state==GRAVITY_SETTLE の「瞬間」条件では
    # GRAVITY_SETTLE→STABLE→OJAMA_FALL という第2の割り込み経路 (実測:
    # t=198.267 gravity_settle → t=198.400 stable → t=198.500 ojama_fall、
    # 経過 0.233秒) を捉えられない。 本 detector 自身が「直近に
    # CHAIN/GRAVITY_SETTLE を観測した時刻」 を内部で追跡し (新規の
    # pipeline レベル状態追跡ではなく、 本 detector 内で完結する自己観測)、
    # state 遷移の経路によらず一貫して厳格化できるようにする。
    # (-1.0 = 未観測)。
    _last_chain_or_settle_observed_time: float = field(
        default=-1.0, init=False, repr=False,
    )
    # 案1 (2026-08-13): おじゃまセル限定 settle 判定用の内部 state。
    # 案B (`_settle_start_frame` 等) と独立命名し、 両フラグが同時に True の
    # 構成 (通常は想定しないが安全側に倒す) でも状態が混線しないようにする。
    _scoped_exit_start_frame: int = field(default=-1, init=False, repr=False)
    _scoped_exit_start_time: float = field(default=0.0, init=False, repr=False)
    _scoped_stable_consec: int = field(default=0, init=False, repr=False)
    _prev_board_ojama_count: int = field(default=-1, init=False, repr=False)

    def detect(
        self,
        ctx: StateContext,
        signals: DetectorSignals,
    ) -> BoardState | None:
        """state 遷移を返す (None = 遷移なし).

        OJAMA_FALL 中の完了判定と、 新規お邪魔出現による OJAMA_FALL 発火を担う。
        """
        cur_count = _count_top_ojama(
            signals.cnn_board, signals.hsv_board,
        )

        if ctx.state == BoardState.OJAMA_FALL:
            # 案2 (2026-08-13、2026-08-15 chain除外修正): 実設置の証拠が
            # 確定できたら settle 判定を待たず即 STABLE 復帰する。
            # (`_confirm_placement_evidence` が roll-up 誤検知を除外する)
            if (
                self.enable_ojama_fall_placement_override
                and self._confirm_placement_evidence(signals)
            ):
                # 案B と同じ理由 (2026-07-24): keep_top_ojama_count で
                # ROI に残る実カウントを保持し、 直後の再突入振動を防ぐ。
                self._reset_internal_state(keep_top_ojama_count=cur_count)
                # 案4-lite 拡張 (coordinator追加指示, 2026-08-13): exit 時刻を
                # 記録する (`_reset_internal_state()` はクリアしないフィールド)。
                # enable_ojama_fall_entry_hardening=True 時の re-entry 抑制窓
                # 判定に使う (entry_hardening=False なら参照されず無害)。
                self._placement_override_exit_time = signals.time_sec
                return BoardState.STABLE
            # 案1 (2026-08-13): おじゃまセル限定の静止判定 (根治)。
            # 案B より優先する (根治が exit を先に決める設計、両立可能)。
            if self.enable_ojama_fall_scoped_exit:
                return self._detect_ojama_fall_exit_scoped(ctx, signals)
            # 案B (2026-07-24): 全盤面 settle 判定に委譲 (default False で既存不変)。
            if self.enable_ojama_fall_board_settle:
                return self._detect_ojama_fall_exit_board_settle(ctx, signals)
            return self._detect_ojama_fall_exit(cur_count)

        return self._detect_ojama_fall_entry(ctx, signals, cur_count)

    def _confirm_placement_evidence(self, signals: DetectorSignals) -> bool:
        """案2修正 (2026-08-15): 実設置証拠を「確定」判定する.

        `_has_ojama_fall_placement_evidence` の生判定をそのまま採用すると
        46セルの新規劣化を持ち込む (診断確定、 モジュール定数コメント参照)。
        経路ごとに以下の扱いに分ける:

        - slide_motion: 従来通り即採用 (診断で単一フレーム誤爆の対象になって
          いない)。
        - own_score_delta: 直近自 chain アクティブなら除外する (roll-up
          誤検知対策)、 それ以外は従来通り単一フレームで即採用する。

        ⚠️ 実時間ヒステリシス (own_score_delta が N秒間途切れず観測される
        ことを要求する案) も試作したが、 55盤面物差しで **有意に悪化**
        すると実測で確定し不採用にした (46→57セル・全体一致率 95.18%→
        93.95%)。 真因: `collect_boards_lean._should_emit` は STABLE 復帰
        時のみ snapshot を記録する仕様のため、 ヒステリシスが早期 exit の
        頻度を落とすと OJAMA_FALL 中の snapshot 空白期間が伸びる。 物差し
        query フレームがその空白の中に落ちると、 nearest match が
        「OJAMA_FALL に入る前の古い snapshot」まで遡ってしまい、 単発ノイズ
        1件を防ぐ一次効果より snapshot 空白の二次被害が大きい
        (chain除外のみなら 46→25セル・全体一致率 95.18%→95.63% と改善する
        ため、 hysteresis 抜きのこの実装を最終採用する)。

        Args:
            signals: 現 frame の DetectorSignals。

        Returns:
            実設置と確定できたか (True なら即 STABLE 復帰してよい)。
        """
        if signals.slide_motion:
            return True
        if signals.own_score_delta <= 0:
            return False
        return not _is_own_chain_recently_active_for_score_exclusion(signals)

    def _detect_ojama_fall_exit(
        self, cur_count: int,
    ) -> BoardState | None:
        """OJAMA_FALL 中の完了判定 (STABLE 復帰).

        ROI が空になれば即 STABLE 復帰。
        settle 判定 ON 時は cur_count が OJAMA_SETTLE_CONSEC フレーム不変でも復帰。
        """
        # お邪魔消滅: 即 STABLE 復帰
        if cur_count == 0:
            self._reset_internal_state()
            return BoardState.STABLE

        # settle 判定: count が N フレーム不変 → 落下完了
        if self.enable_ojama_settle_detection:
            if cur_count == self._prev_settle_count:
                self._settle_count += 1
            else:
                self._settle_count = 1
            self._prev_settle_count = cur_count
            if self._settle_count >= OJAMA_SETTLE_CONSEC:
                self._reset_internal_state()
                return BoardState.STABLE

        # OJAMA_FALL 継続中: 遷移なし
        return None

    def _detect_ojama_fall_exit_board_settle(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        """案B: 全盤面ぷよ数の静止を待つ OJAMA_FALL 退出判定.

        GravitySettleDetector (src/state_detectors.py) と同型のロジック。
        ROI (可視最上段 2 行) でなく盤面全体のぷよ数 (お邪魔含む) を見ることで、
        おじゃまが着弾しきる前に退出してしまう問題 (真因: 滞在中央値 1 frame) を防ぐ。

        条件 (いずれか):
          - タイムアウト: OJAMA_FALL_MAX_SEC 秒経過で安全弁として強制 STABLE。
          - OJAMA_FALL_SETTLE_MIN_FRAMES フレーム以上経過 かつ
            ぷよ数差分 < OJAMA_FALL_SETTLE_DIFF_THRESHOLD の状態が
            OJAMA_FALL_SETTLE_STABLE_FRAMES フレーム連続。
        """
        cur_board_count = signals.cnn_board.count_puyos()

        # OJAMA_FALL に初めて入ったフレームを記録
        if self._settle_start_frame < 0:
            self._settle_start_time = signals.time_sec
            self._settle_start_frame = ctx.frame_idx
            self._board_stable_consec = 0
            self._prev_board_puyo_count = cur_board_count
            return None  # 最初のフレームは必ず継続

        # タイムアウト: 安全弁として強制 STABLE 復帰
        elapsed = signals.time_sec - self._settle_start_time
        if elapsed >= OJAMA_FALL_MAX_SEC:
            # バグ修正 (2026-07-24): reset で _prev_top_ojama_count を無条件 0
            # にすると、 ROI にまだ残っているお邪魔 (着弾済み・不動) を次
            # フレームの _detect_ojama_fall_entry が「新規出現」と誤認し
            # 即座に OJAMA_FALL へ再突入する (振動ループ、 TSUMO_FALL 検出を
            # 最大 +7秒 遅延させる真因)。 退出時点の実カウントを保持する。
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE

        # 最低待機時間未達: 継続 (カウンタのみ更新)。
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `ctx.frame_idx -
        # self._settle_start_frame` (frame 差分) を、同関数内で既に算出済の
        # `elapsed` (time_sec 差分, タイムアウト判定と同一変数) に置換。
        # 60fps 動画では (frame_idx 差分)/60 == time_sec 差分 が恒等式のため
        # 判定は bit-identical。30fps 動画では実秒基準になる。
        if elapsed < OJAMA_FALL_SETTLE_MIN_SEC:
            self._prev_board_puyo_count = cur_board_count
            self._board_stable_consec = 0
            return None

        # ぷよ数変化の安定性チェック
        diff = abs(cur_board_count - self._prev_board_puyo_count)
        self._prev_board_puyo_count = cur_board_count
        if diff < OJAMA_FALL_SETTLE_DIFF_THRESHOLD:
            self._board_stable_consec += 1
        else:
            self._board_stable_consec = 0

        if self._board_stable_consec >= OJAMA_FALL_SETTLE_STABLE_FRAMES:
            # バグ修正 (2026-07-24): タイムアウト分岐と同様、 settle 退出時も
            # ROI 実カウントを保持して次フレームの誤再突入を防ぐ。
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE

        return None  # 継続

    def _detect_ojama_fall_exit_scoped(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        """案1: おじゃまセル限定の静止判定による OJAMA_FALL 退出判定 (根治).

        `_detect_ojama_fall_exit_board_settle` (案B) は盤面全体のぷよ数
        (色ぷよ込み) を見るため自分のツモ設置でも数値が変化し続け、 退出
        判定が延々と塞がれる (根因調査 2026-08-13 で実測: 設置ブロック
        16/16件)。 本メソッドは `_count_board_ojama` (おじゃまセルのみ) の
        静止だけを見て、 色ぷよの増減を無視する。 さらに
        `signals.own_pending_ojama_forecast` が 0 以下なら
        `_scoped_required_stable_frames` により静止確認フレーム数を短縮する。
        """
        cur_ojama_count = _count_board_ojama(signals.cnn_board)

        if self._scoped_exit_start_frame < 0:
            self._scoped_exit_start_time = signals.time_sec
            self._scoped_exit_start_frame = ctx.frame_idx
            self._scoped_stable_consec = 0
            self._prev_board_ojama_count = cur_ojama_count
            return None  # 最初のフレームは必ず継続

        elapsed = signals.time_sec - self._scoped_exit_start_time
        if elapsed >= OJAMA_FALL_MAX_SEC:
            # 安全弁: タイムアウトで強制 STABLE (案B と同じ安全弁を流用)。
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE

        if elapsed < OJAMA_FALL_SCOPED_EXIT_MIN_SEC:
            self._prev_board_ojama_count = cur_ojama_count
            self._scoped_stable_consec = 0
            return None

        diff = abs(cur_ojama_count - self._prev_board_ojama_count)
        self._prev_board_ojama_count = cur_ojama_count
        if diff < OJAMA_FALL_SCOPED_EXIT_DIFF_THRESHOLD:
            self._scoped_stable_consec += 1
        else:
            self._scoped_stable_consec = 0

        required = self._scoped_required_stable_frames(signals)
        if self._scoped_stable_consec >= required:
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE
        return None  # 継続

    def _scoped_required_stable_frames(self, signals: DetectorSignals) -> int:
        """会計連動: 未着弾予告が尽きていれば必要静止フレーム数を短縮する.

        `signals.own_pending_ojama_forecast` は getattr 安全参照 (トラッカー
        無効構成では属性自体が既定 None のため、 従来通りの必要フレーム数を
        返す)。 0 以下 (= もう降ってくる予告が無い) の場合のみ短縮する。
        """
        pending = getattr(signals, "own_pending_ojama_forecast", None)
        base = OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES
        if pending is not None and pending <= 0:
            return max(1, base // OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR)
        return base

    def _detect_ojama_fall_entry(
        self, ctx: StateContext, signals: DetectorSignals, cur_count: int,
    ) -> BoardState | None:
        """OJAMA_FALL 発火判定 (STABLE/TSUMO_FALL/CHAIN → OJAMA_FALL).

        CHAIN 中からの発火は enable_ojama_visual_chain_exit が True の時のみ。

        発火ロジック:
          - 「開始トリガー」: cur_count > 0 かつ (prev_count == 0 or cur_count > prev_count)
            = お邪魔が新規出現または増加したフレームでカウント開始。
          - 「継続」: 開始後 cur_count > 0 の間はカウントを継続する。
          - enable_ojama_fall_entry_hardening=False (既定) は
            OJAMA_CONSEC_THRESH フレーム到達で発火 (frame 数連続、従来挙動)。
            True なら実時間連続判定に切り替える (`_detect_entry_time_based`)。

        防御コード (案B, 2026-07-24): CHAIN 等に途中横取りされて OJAMA_FALL から
        state が抜けると、 このメソッドが (exit メソッドではなく) 毎 frame
        呼ばれ続ける。 その間に settle 用内部 state をここで都度リセットしておかないと、
        次回 OJAMA_FALL 再突入時に前回の残骸 (_settle_start_frame 等) が残り、
        即 timeout 誤判定を起こす。
        """
        self._settle_start_frame = -1
        self._settle_start_time = 0.0
        self._board_stable_consec = 0
        self._prev_board_puyo_count = -1
        # 案1 (2026-08-13): おじゃまセル限定 settle 判定用の内部 state も同様に
        # 都度リセットする (案B と同じ理由、 再突入時の残骸汚染を防ぐ)。
        self._scoped_exit_start_frame = -1
        self._scoped_exit_start_time = 0.0
        self._scoped_stable_consec = 0
        self._prev_board_ojama_count = -1

        # 案4-lite 拡張・第2ラウンド (coordinator追加指示, 2026-08-13):
        # detect() が呼ばれた時点 (= 他の高優先度 detector が勝たなかった
        # frame) の ctx.state を無条件で観測記録する。 早期 return 分岐の
        # 前に置くことで、 CHAIN/GRAVITY_SETTLE 中に発火しない分岐でも
        # 観測時刻を確実に更新する。
        if ctx.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):
            self._last_chain_or_settle_observed_time = signals.time_sec

        # CHAIN 中かつフラグ OFF → 発火しない
        if (
            ctx.state == BoardState.CHAIN
            and not self.enable_ojama_visual_chain_exit
        ):
            self._reset_internal_state()
            return None

        # MENU/EFFECT: 発火しない
        if ctx.state in (BoardState.MENU, BoardState.EFFECT):
            self._reset_internal_state()
            return None

        # 修正B (2026-08-08、バグB): GRAVITY_SETTLE 中は発火しない
        # (enable_ojama_entry_gravity_settle_guard=True 時のみ)。
        # 連鎖の段間重力待ち中に本 detector が横取りすると、
        # GravitySettleDetector 内部カウンタが残留し次回誤 STABLE 化を招く
        # (バグC)。 default False = 既存挙動と完全 bit-identical。
        if (
            self.enable_ojama_entry_gravity_settle_guard
            and ctx.state == BoardState.GRAVITY_SETTLE
        ):
            self._reset_internal_state()
            return None

        prev_count = self._prev_top_ojama_count
        self._prev_top_ojama_count = cur_count

        if cur_count == 0:
            # お邪魔なし: カウントリセット
            self._consec_count = 0
            self._entry_trigger_start_time = -1.0
            return None

        # 開始トリガー: 前フレームが 0 またはお邪魔が増加した時にカウントを (再) 開始
        is_new_trigger = prev_count == 0 or cur_count > prev_count

        if self.enable_ojama_fall_entry_hardening:
            return self._detect_entry_time_based(ctx, signals, is_new_trigger)
        return self._detect_entry_frame_based(ctx, is_new_trigger)

    def _detect_entry_frame_based(
        self, ctx: StateContext, is_new_trigger: bool,
    ) -> BoardState | None:
        """従来 (frame 数連続) の OJAMA_FALL entry 判定.

        enable_ojama_fall_entry_hardening=False (既定) で使われる、
        従来ロジックと bit-identical な経路。
        """
        if self._consec_count == 0 and not is_new_trigger:
            # まだ開始トリガーが来ていない: カウントしない
            return None

        # 連続観測カウント (同 frame 二重カウント防止)
        if ctx.frame_idx != self._last_frame_idx:
            if is_new_trigger or self._consec_count > 0:
                self._consec_count += 1
            self._last_frame_idx = ctx.frame_idx

        if self._consec_count >= OJAMA_CONSEC_THRESH:
            self._consec_count = 0
            return BoardState.OJAMA_FALL

        return None

    def _detect_entry_time_based(
        self, ctx: StateContext, signals: DetectorSignals,
        is_new_trigger: bool,
    ) -> BoardState | None:
        """案4-lite: 実時間ベースの OJAMA_FALL entry 判定.

        frame 数でなく time_sec 差分で「連続観測」を判定するため、 フレーム
        間引き (stride) 下でも実時間の意味が変わらない。 さらに
        `_is_hardened_entry_context` が True の間 (CHAIN 中/直近 CHAIN
        アクティブ/直近 案2 早期 exit 直後) は
        OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER 倍の持続時間を要求し、
        chain_event 関連の瞬間的な検出空白への割り込みを抑止する
        (docs/DEMO_REVIEW_2026-08-13.md 場面2 の根因対策、
        coordinator追加指示 2026-08-13 で state 非依存の実時間判定に拡張)。
        """
        if is_new_trigger:
            self._entry_trigger_start_time = signals.time_sec
        if self._entry_trigger_start_time < 0:
            # 開始トリガーが一度も来ていない: カウントしない
            return None

        required_sec = OJAMA_ENTRY_CONSEC_SEC
        if self._is_hardened_entry_context(ctx, signals):
            required_sec *= OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER

        elapsed = signals.time_sec - self._entry_trigger_start_time
        if elapsed >= required_sec:
            self._entry_trigger_start_time = -1.0
            return BoardState.OJAMA_FALL

        return None

    def _is_hardened_entry_context(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> bool:
        """entry 厳格化 (持続時間倍率) を適用すべき文脈か判定する.

        coordinator追加指示 (2026-08-13、場面2実測 52→69 の悪化対処、実データ
        直接確認で2ラウンドの調査を要した)。当初「ctx.state == CHAIN」の
        瞬間条件だけでは実際の割り込み経路を捉えられないと判明した。
        実データ (frame_records 直接確認、
        logs/verify_ojama_fall_fix_scene2_2026-08-13_v2.json) で少なくとも
        2 つの割り込み経路を確認した:
          (a) CHAIN → GRAVITY_SETTLE → OJAMA_FALL → CHAIN
              (~1.2〜1.3秒周期、連鎖1リンクの実測とほぼ一致)。
          (b) GRAVITY_SETTLE → STABLE (一瞬) → OJAMA_FALL
              (実測: t=198.267 gravity_settle → t=198.400 stable →
              t=198.500 ojama_fall、経過 0.233秒。 GravitySettleDetector が
              先に STABLE へ正常解決した直後、 まだ ROI trigger が残った
              状態で本 detector の entry がすり抜ける)。
        (b) は ctx.state の瞬間条件では原理的に捉えられない (STABLE は
        「厳格化すべきでない通常 state」でもあるため)。 本 detector 自身が
        「直近に CHAIN/GRAVITY_SETTLE を観測した時刻」 を内部で追跡する
        ことで、 経由する中間 state の種類によらず一貫して捉える
        (`_last_chain_or_settle_observed_time`、 detect() 呼び出し時の
        自己観測、 新規のpipelineレベル状態追跡ではない)。 以下いずれかで
        True (OR):
            1. ctx.state == CHAIN (直接 CHAIN 中からの entry、安全弁として維持。
               ただし ChainPhaseDetector が最優先のため実運用では chain_event
               空白時の 1 frame 程度でしか成立しない)。
            2. ctx.state == GRAVITY_SETTLE (経路(a)の直接判定)。
            3. 直近 OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC 秒以内に本 detector が
               CHAIN/GRAVITY_SETTLE を観測していた (経路(b)を含む、 中間
               state 非依存の判定)。
            4. 直近 OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC 秒以内に自 side の
               chain_event がアクティブだった
               (signals.own_chain_hold_until_sec との差分。 既存の追跡値
               `_chain_until_1p`/`_chain_until_2p` を再利用する保険的な
               二重チェック、 条件3 と独立に評価する)。
            5. 直近 OJAMA_REENTRY_SUPPRESS_SEC 秒以内に案2 (placement
               override) の早期 exit があった (exit→即re-entry 振動対策)。
        """
        if ctx.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):
            return True
        if (
            self._last_chain_or_settle_observed_time >= 0
            and signals.time_sec - self._last_chain_or_settle_observed_time
            < OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC
        ):
            return True
        # own_chain_hold_until_sec <= 0.0 は「まだ一度も自 chain が発生してい
        # ない」sentinel (RecognitionPipeline.reset() の初期値と同じ規約)。
        # これを除外しないと、 試合序盤の time_sec がまだ小さい間、 default
        # 値 0.0 との差分が偶然小さく見えて誤って「直近アクティブ」と判定
        # してしまう (2026-08-13 実装時にユニットテストで検出、time_sec=0
        # 起点のテスト規約でも再現)。
        if (
            signals.own_chain_hold_until_sec > 0.0
            and signals.time_sec - signals.own_chain_hold_until_sec
            < OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC
        ):
            return True
        if (
            self._placement_override_exit_time >= 0
            and signals.time_sec - self._placement_override_exit_time
            < OJAMA_REENTRY_SUPPRESS_SEC
        ):
            return True
        return False

    def _reset_internal_state(
        self, keep_top_ojama_count: int | None = None,
    ) -> None:
        """内部カウンタを全リセットする.

        Args:
            keep_top_ojama_count: None (既定) なら従来通り
                `_prev_top_ojama_count` を 0 にリセットする (bit-identical)。
                int を渡すとその値を `_prev_top_ojama_count` に保持する。
                案B (`_detect_ojama_fall_exit_board_settle`) の退出時のみ、
                退出時点の ROI 実カウントを渡すことで、 まだ ROI に残る
                (着弾済み・不動の) お邪魔を次フレームが新規出現と誤認して
                OJAMA_FALL へ即再突入する振動バグを防ぐ。
        """
        self._consec_count = 0
        self._prev_top_ojama_count = (
            0 if keep_top_ojama_count is None else keep_top_ojama_count
        )
        self._last_frame_idx = -1
        self._settle_count = 0
        self._prev_settle_count = 0
        # 案B: 全盤面 settle 判定用の内部 state もリセット。
        self._settle_start_frame = -1
        self._settle_start_time = 0.0
        self._board_stable_consec = 0
        self._prev_board_puyo_count = -1
        # 案1: おじゃまセル限定 settle 判定用の内部 state もリセット。
        self._scoped_exit_start_frame = -1
        self._scoped_exit_start_time = 0.0
        self._scoped_stable_consec = 0
        self._prev_board_ojama_count = -1
        # 案4-lite: 実時間ベース entry 判定の開始時刻もリセット。
        self._entry_trigger_start_time = -1.0
