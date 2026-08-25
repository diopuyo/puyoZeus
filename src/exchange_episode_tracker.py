"""交換エピソード会計の入力アダプタ (Gate 3-2a、2026-08-24)。

`src/chain_id_resolver.py` (chain_id 復元) と `src/exchange_ledger.py`
(会計コア) を合成し、動画 1 本ぶんの連鎖イベント列を流し込んで診断値
(`docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` §13 の D1〜D7) を出す。

## この段階でやらないこと (最重要)

**表示も判定も一切変えない。数値を観測して記録するだけ。**
`ExchangeLedger.allows_hard_override` 等の判定結果は「出す」が「使わない」。

## 既定 OFF・未配線

`src/recognition_pipeline.py` / `src/chain_detector.py` / `src/score_ocr.py` /
`src/ojama_accounting.py` / `scripts/visualize_advantage_overlay.py` /
`src/production_config.py` のいずれからも呼ばれない。呼び出し配線は別タスク。
`enabled=False` (既定) では `observe` は即 return し、何も計算しない。

## CHAIN_END_SIGNAL に依存しない設計

連鎖終了の絶対律検出器 (`CHAIN_END_SIGNAL`) は既知の事故源
(memory `project_slide_false_positive_root_cause_2026-08-22`、連鎖中に
1.37 秒おきに誤検知した前科) であるため、本モジュールは一切使わない。
`ChainIdResolver` は `CHAIN_END_SIGNAL` が無くても
`GROWING -> BASELINE で FINALIZE` の経路で正しく動く
(`tests/test_chain_id_resolver.py` の遷移表で確認済み)。

## D5 を出さない理由

D5 (直近の物理イベント種別+時刻) はフレームごとの時系列列であり、
tracker (動画 1 本の集計値を返す部品) の責務ではない。
Gate 3-2b で dump 行へ直接足す。

## 単位換算 (2026-08-24 コーディネーター指摘を受けて追加)

**`ExchangeLedger.amount` はおじゃまの個数であり、スコア点ではない。**
最初の実装は `ChainEventObservation.total_score` (スコア点) をそのまま
`amount` に渡していたため、`FINALIZE_DOWNWARD_TOLERANCE=4.0` (おじゃま
4 個) が実質機能しない欠陥があった (720 点を 4 個の許容帯と比較していた)。

`src/scoring.py:score_to_ojama` (マージンタイム逓減込み) で暫定・確定の
両方を**同一の規約**で換算してから台帳へ渡す (§4.1.2)。
`prev_leftover=0` 固定: 繰越はサイド別の状態であり、この tracker は
chain_id をまたぐ繰越を保持しない (責務外)。レート 70 なら最大 69 点
(= 1 個未満) の切り捨て差にしかならないため、この段階では許容する。

換算の正しさを機械検査するため、確定側については
`ChainEventObservation.ojama_sent` (= `chain_detector` が繰越込みで
算出した権威ある値) との差を自己検算として記録する (D3)。
**この自己検算は診断のみに使い、会計 (台帳への入力) には使わない。**
"""
from __future__ import annotations

from dataclasses import dataclass

from src.chain_detector import (
    CHAIN_MECHANISM_BASELINE,
    CHAIN_MECHANISM_FORMULA,
    CHAIN_MECHANISM_FORMULA_READ,
)
from src.chain_id_resolver import (
    FINALIZED_SOURCE_SCORE_OCR_DIFF,
    FINALIZED_SOURCE_SIMULATE_FALLBACK,
    ChainIdResolver,
    ChainObservation,
    ObservationKind,
    ResolvedChain,
    ResolverStats,
)
from src.exchange_ledger import (
    OJAMA_MAX_DROP_PER_TURN,
    EpisodeStatus,
    EventKind,
    ExchangeEvent,
    ExchangeLedger,
    FINALIZE_DOWNWARD_TOLERANCE,
    LedgerSnapshot,
    PhysicalContext,
    Side,
)
from src.scoring import OJAMA_RATE_STANDARD, score_to_ojama

# ============================
# 定数
# ============================

# ChainEvent.mechanism -> ObservationKind の対応表 (タスク仕様の変換表)。
# それ以外の mechanism は FORMULA_STEP として扱わず、カウンタに記録して無視する。
#
# 【2026-08-25 是正】 CHAIN_MECHANISM_BASELINE は `ChainSimulator` による
# 盤面からの推定であり、score OCR 由来の確定値ではなかった
# (`src/chain_detector.py:277-317`)。値の権威を持たない
# `ObservationKind.CHAIN_SETTLED` (連鎖が終わった合図のみ) にマップする。
# 値の権威 (`ObservationKind.SCORE_FINALIZE`) は `observe_generation()`
# 経由の別チャネルから供給する (下記 `GenerationObservation` 参照)。
_MECHANISM_TO_KIND: dict[str, ObservationKind] = {
    CHAIN_MECHANISM_FORMULA: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_FORMULA_READ: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_BASELINE: ObservationKind.CHAIN_SETTLED,
}

# `observe_generation()` が resolver へ渡す診断用 mechanism 文字列
# (`ChainObservation.mechanism` は判定に使わない診断専用フィールド)。
_SCORE_FINALIZE_MECHANISM_LABEL: str = "score_ocr_generated_delta"

# ChainEventObservation.side ("1P" / "2P") -> ExchangeLedger.Side の対応表。
_SIDE_TO_LEDGER_SIDE: dict[str, Side] = {"1P": Side.P1, "2P": Side.P2}

# 浮動小数点の比較誤差許容値。src/exchange_ledger.py 内の他の許容誤差
# (`_provisional_residual` 等の `1e-9`) と揃える。シーンからの逆算ではない。
_FLOAT_EPS: float = 1e-9

# 換算に使う繰越初期値。tracker は chain_id をまたぐ繰越 (leftover) を
# 保持しないため常に 0 で呼ぶ (docstring 冒頭「単位換算」節を参照)。
_PREV_LEFTOVER_FOR_TRACKER: int = 0

# `_build_timeline`/`_replay_timeline` の同一 t_sec 内での処理優先順位
# (Fix【2】、2026-08-25 追加)。小さい値ほど先に処理する。
_TIMELINE_KIND_CHAIN: int = 0
_TIMELINE_KIND_WIPE: int = 1
_TIMELINE_KIND_SETTLEMENT: int = 2


# ============================
# 入力
# ============================

@dataclass(frozen=True)
class ChainEventObservation:
    """ChainEvent から抜き出した、tracker が必要とする最小の情報。

    ChainEvent そのものを受け取らないのは、盤面 (before_board) を保持すると
    メモリを食い、かつ tracker の責務 (会計) に不要なため。

    `elapsed_sec` (2026-08-24 追加): 試合開始からの経過秒。
    `score_to_ojama` のマージンタイム逓減計算に必須 (`src/scoring.py:271`)。

    **既定値を持たせない (必須フィールド)。** 既定値 0.0 を許すと、
    呼び出し側が渡し忘れてもエラーにならず「マージンタイムが効かず終盤の
    火力が黙って過小になる」という**気づけない誤り**を静かに許してしまう
    (memory `feedback_wiring_gap_vs_wiring_error_2026-08-22`: 「漏れ」型の
    配線事故はどれも「動いてしまう」ことが原因だった)。
    呼び出し元がまだ 0 件 (未配線) の段階なので、後方互換を優先する理由も無い。
    渡し忘れは `TypeError` で止まるのが正しい挙動である。

    `ojama_sent` についての警告 (2026-08-25 W38 実データ検証で発覚):
    **この値は W38 の影響下にあり、会計にも検算にも使わない。**
    `src/chain_detector.py:186` の `match_start_sec` が `0.0` 固定のため、
    `ChainEvent` 側の経過秒が「動画の絶対時刻」になってしまい、320秒を
    超える動画位置では `compute_effective_rate` が下限 1 まで落ちて
    「点数がそのままおじゃま個数」という壊れた値になる (実測 16/16 で
    レート1)。診断で参照する可能性があるため残すが、値そのものは信用
    しないこと (会計にも検算にも使わない)。
    【2026-08-25 W38 根治後の追記】pipeline の score リセット境界 reset()
    が `match_start_sec` を渡すようになり、境界通過後の `ojama_sent` は
    試合相対時刻で計算される (tests/test_w38_match_start_wiring_2026-08-25.py)。
    ただし処理窓の先頭〜最初の境界までの区間は依然 0.0 起点のため、
    「会計にも検算にも使わない」方針は据え置く (会計の権威は従来通り
    score OCR 確定差分)。

    **【2026-08-25 Fix【5】で `authoritative_ojama` を削除】** 新設計では
    `SCORE_FINALIZE` (score OCR 確定差分) の値がそのまま会計の確定値に
    なるため、それを検算する**独立な第二の権威値が存在しない**。
    `authoritative_ojama` の供給元は `finalized_score` と**同一の
    accumulator** (`OjamaAccountingTracker.total_generated_by_pX`) であり、
    フレーム一致を直しても常に差 0 になるだけ (tautological、自明に
    成立する無意味な検算)。2026-08-25 の実測で
    `n_authoritative_ojama_present = 0/20, 0/7` (供給自体がほぼ機能して
    いなかった) も確認済み。意味のない「0 が合格」を残すと誤読される
    ため、フィールドごと削除した (前任コーダと診断役が独立に同じ結論)。
    """

    side: str
    t_sec: float
    mechanism: str
    chain_count: int
    total_score: int
    ojama_sent: int
    game_idx: int
    elapsed_sec: float


@dataclass(frozen=True)
class SettlementObservation:
    """1 フレーム分の相殺・着弾の観測 (累積カウンタの差分から作る)。

    `src/ojama_accounting.py` は変更しない。`OjamaAccountSnapshot`
    (`src/ojama_accounting.py:169-223`) が既に公開している累積値の
    **前フレームとの差分**を呼び出し側が計算して渡す。

    **報告 (実物確認結果)**: タスク指定の `total_offset` という単一
    フィールドは実在しない。実際は side 別に分かれた
    `total_offset_by_p1` / `total_offset_by_p2` である。
    `total_dropped_to_p1` / `total_dropped_to_p2` は指定通り実在する。
    このフィールド名 (`canceled_by_1p` 等) は `SettlementObservation`
    独自の命名であり、`OjamaAccountSnapshot` のフィールド名をそのまま
    使う必要はない (呼び出し側が差分を計算して渡す設計のため)。

    - `canceled_by_1p` = `total_offset_by_p1` の増分
      (1P が発火することで、1P に向かっていた予告=2P の生成量を相殺した量)
    - `landed_on_1p` = `total_dropped_to_p1` の増分
      (1P に実際に降った量=2P の生成量が着弾した量)
    - `canceled_by_2p` / `landed_on_2p` は対称に 2P 側。
    """

    t_sec: float
    game_idx: int
    canceled_by_1p: float
    canceled_by_2p: float
    landed_on_1p: float
    landed_on_2p: float


@dataclass(frozen=True)
class GenerationObservation:
    """score OCR 確定差分による生成観測 (2026-08-25 追加、fable アーキ裁定)。

    `OjamaAccountSnapshot.total_generated_by_p1` / `total_generated_by_p2`
    (`src/ojama_accounting.py:187-188`。実装は `:720` `s.total_generated += gen`
    で、`gen` は score OCR の確定差分 `chain_total = score_after - score_start`
    を `score_to_ojama` で換算した**個数**) が side ごとに増えた瞬間の増分。

    **単位はおじゃま個数。換算不要** (コーディネーター指定)。
    `ChainEventObservation.total_score` (スコア点) とは単位が違うことに注意
    (`_provisional_ojama_for`/`_finalized_ojama_for` 参照)。

    **`SettlementObservation` と同じ経路では処理しない。** `SettlementObservation`
    は chain_id 解決が完了した後 (`finish()` の `_replay_timeline`) に台帳へ
    供給するのに対し、この観測は resolver 自身の入力
    (`ObservationKind.SCORE_FINALIZE`) であり、chain_id 解決の**途中**で
    `observe()` と同じタイミングで即座に反映しなければならない
    (resolver の `flush()` より前に必要な入力のため)。そのため
    `SettlementObservation` へフィールドを足す設計は採らず、専用の型と
    `observe_generation()` を新設した (判断根拠は報告参照)。

    `game_idx` (2026-08-25 追加): `observe()` と独立したチャネルでも
    試合境界をまたいだ汚染を防ぐため必須にした。**「呼び出し順序を呼び出し側
    の注意力に頼らない」** (コーディネーター指摘): `observe_generation()`
    単独でも `_maybe_inject_match_boundary` を呼び、`observe()` を挟まずに
    試合境界をまたいでもチャネル単独で境界を検出できるようにする。

    **【2026-08-25 Fix【5】】** 自己検算 (D3) は削除した。`generated_delta`
    自体が会計の確定値そのものであり、それを検算する独立な第二の権威値が
    存在しないため (`ChainEventObservation` docstring 参照)。
    """

    side: str          # "1P" / "2P"
    t_sec: float
    game_idx: int
    generated_delta: int  # おじゃま個数の増分 (score OCR 確定差分、換算不要)


# ============================
# Fix【1】: 上限なしの pending レベル差分からの観測経路再構成
# (2026-08-25 追加、gate3_episode_v3 実装タスク)
# ============================

@dataclass(frozen=True)
class PendingUncappedFrame:
    """1 フレーム分の cap 前 pending 観測 (Fix【1】、2026-08-25 追加)。

    `OjamaAccountSnapshot.pending_p1_uncapped`/`pending_p2_uncapped`
    (`src/ojama_accounting.py:222-223`。上限 216 の切り捨てを受けない
    並行帳簿。実在確認済み、報告参照) の値そのもの。前フレームとの
    差分から相殺・着弾・ワイプを判別的に再構成する入力になる
    (`classify_pending_uncapped_delta` 参照)。

    `p1_tsumo_placed`/`p2_tsumo_placed`: このフレームでその side が
    ツモを 1 手設置した直後か (着弾は設置のタイミングで起きるため。
    `src/ojama_accounting.py` の `total_dropped_to_pX` docstring
    「tsumo_settled drain の累積」と同じ根拠)。
    `p1_chain_finalized`/`p2_chain_finalized`: このフレームでその side
    自身の連鎖が score OCR 確定差分で確定したか (相殺は
    `cancel_own_pending_then_send_surplus` が連鎖確定の瞬間に計算する
    ため。`observe_generation()`/`GenerationObservation` と同じ経路の
    情報を呼び出し側が渡す)。
    """

    t_sec: float
    game_idx: int
    p1_uncapped: float
    p2_uncapped: float
    p1_tsumo_placed: bool
    p2_tsumo_placed: bool
    p1_chain_finalized: bool
    p2_chain_finalized: bool


@dataclass(frozen=True)
class PendingDeltaClassification:
    """1 フレーム分の pending_uncapped 差分の判別結果 (Fix【1】、2026-08-25 追加)。

    `settlement`: CANCEL/LAND として分類できた分 (`SettlementObservation`
    と同形。相殺・着弾のどちらも無ければ `None`)。
    `wiped_sides`: このフレームでワイプ (一括で 0 になった、かつ
    相殺・着弾のどちらの条件にも一致しない) が起きた side の一覧。
    `unclassified_drop_p1`/`unclassified_drop_p2`: 減少はしたが
    相殺・着弾・ワイプのどの条件にも一致しなかった量。**0 が期待値。**
    0 でなければ判別規則が不十分な証拠であり、黙って捨てない
    (呼び出し側がカウンタへ積む)。
    """

    settlement: SettlementObservation | None
    wiped_sides: tuple[Side, ...]
    unclassified_drop_p1: float
    unclassified_drop_p2: float


@dataclass(frozen=True)
class _SideDeltaResult:
    """side 1 つ分の pending_uncapped 差分判別結果 (内部専用)。"""

    canceled: float = 0.0
    landed: float = 0.0
    wiped: bool = False
    unclassified: float = 0.0


def _classify_side_delta(
    prev_val: float, curr_val: float, tsumo_placed: bool, chain_finalized: bool,
) -> _SideDeltaResult:
    """side 1 つ分の pending_uncapped 変化を判別する (増加・変化なしは無視)。

    優先順位 (減少時): 自分の連鎖確定と同時 (CANCEL) ->
    ツモ設置直後かつ `OJAMA_MAX_DROP_PER_TURN` 以下 (LAND) ->
    一括で 0 になった (WIPE) -> それ以外 (unclassified)。

    CANCEL (`cancel_own_pending_then_send_surplus`、連鎖確定の瞬間) と
    LAND (`tsumo_settled` drain、ツモ設置の瞬間) は物理的に別のタイミング
    で起きる独立事象であり、通常は同一フレームで両方が真になることは
    ない。理論上の衝突に備え、より直接的な観測 (score OCR 確定差分)
    である CANCEL を優先する。
    """
    delta = curr_val - prev_val
    if delta >= 0.0:
        return _SideDeltaResult()
    drop = -delta
    if chain_finalized:
        return _SideDeltaResult(canceled=drop)
    if tsumo_placed and drop <= OJAMA_MAX_DROP_PER_TURN:
        return _SideDeltaResult(landed=drop)
    if curr_val <= _FLOAT_EPS:
        return _SideDeltaResult(wiped=True)
    return _SideDeltaResult(unclassified=drop)


def classify_pending_uncapped_delta(
    prev: PendingUncappedFrame, curr: PendingUncappedFrame,
) -> PendingDeltaClassification:
    """1 フレーム分の pending_uncapped 差分を相殺・着弾・ワイプに判別する
    純関数 (Fix【1】、2026-08-25 追加)。

    上限 216 で切り捨てる前の値 (`pending_pX_uncapped`) を使うため、
    真の生成が 216 を超えても超過分が観測から漏れない
    (根因 (A) への対処。報告参照)。I/O なし、状態を持たない。
    """
    p1 = _classify_side_delta(
        prev.p1_uncapped, curr.p1_uncapped, curr.p1_tsumo_placed, curr.p1_chain_finalized,
    )
    p2 = _classify_side_delta(
        prev.p2_uncapped, curr.p2_uncapped, curr.p2_tsumo_placed, curr.p2_chain_finalized,
    )
    settlement = None
    if p1.canceled or p1.landed or p2.canceled or p2.landed:
        settlement = SettlementObservation(
            t_sec=curr.t_sec, game_idx=curr.game_idx,
            canceled_by_1p=p1.canceled, canceled_by_2p=p2.canceled,
            landed_on_1p=p1.landed, landed_on_2p=p2.landed,
        )
    wiped = tuple(s for s, r in ((Side.P1, p1), (Side.P2, p2)) if r.wiped)
    return PendingDeltaClassification(
        settlement=settlement, wiped_sides=wiped,
        unclassified_drop_p1=p1.unclassified, unclassified_drop_p2=p2.unclassified,
    )


@dataclass(frozen=True)
class _ObservationContext:
    """t_sec に対応する tracker 内部の補助情報 (診断・換算専用)。

    `authoritative_ojama` は Fix【5】(2026-08-25) で削除した (自己検算
    そのものの廃止。`ChainEventObservation` docstring 参照)。
    """

    game_idx: int
    elapsed_sec: float
    ojama_sent: int


@dataclass(frozen=True)
class _ChainOjama:
    """1 本の chain_id を `score_to_ojama` で換算した結果 (個数、cap 前)。"""

    provisional_ojama: int
    finalized_ojama: int | None


# ============================
# 出力 (§13 D1〜D7。D5 は出さない)
# ============================

@dataclass(frozen=True)
class D1EpisodeTotals:
    """episode 単位の生成・相殺・着弾・unreconciled の累計と保存則違反件数。

    `ExchangeLedger.closed_episodes()` (2026-08-24 追加、承認済み) から
    CLOSED / CLOSED_FORCED した episode の要約を取り、正常 CLOSED の
    episode についてのみ保存則 (I1) を検査する。

    **CANCEL / LAND の供給は Gate 3-2b で `src/ojama_accounting.py` から
    配線予定。それまで `conservation_violation_count` は 0 が正常であり、
    `episodes_without_settlement_input` が全 episode 数と一致するのが
    期待値である。** `ChainEventObservation` (この tracker の入力) には
    相殺・着弾に対応する概念が無いため、CANCEL/LAND が 1 件も無い
    episode は「生成だけが積まれ、相殺/着弾が 0 のまま残る」構造上
    避けられない不一致になる。これを「会計のバグ」として誤検出しない
    ため、`has_settlement_input=False` の episode は保存則の検査対象から
    除外し、件数だけを `episodes_without_settlement_input` に出す
    (黙って捨てない)。

    **【2026-08-24 追記: E1 削除後の期待値】**
    `ExchangeLedger._should_close` から終了条件 E1 (`net_raw()==0`) を
    削除した (fable アーキ裁定。相殺の記録なしに決着扱いする欠陥だった)。
    その結果、**配線前 (CANCEL/LAND が供給されない間) は、両側が撃ち合った
    episode が 1 本も正常 CLOSED されなくなる。** すべて `EPISODE_MAX_SEC`
    の安全弁で `CLOSED_FORCED` になる (残量は unreconciled として明示記録
    されるため保存則 I1 自体は成立する)。**これは正常である。**
    したがって Gate 3-2b の配線完了前は、`closed_episodes()` に積まれる
    要約のほとんどが `status=CLOSED_FORCED` かつ `has_settlement_input=False`
    になるのが期待値であり、**配線完了後はこの件数がほぼゼロになる**
    ことを受け入れ条件とする。

    **【2026-08-24 追記: 超過決済 (over-settlement) の可視化】**
    `保存則違反件数が 0 でも、`oversettled_total > 0` なら供給側が
    壊れている**。`ChainRecord.outstanding` は下限 0 でクリップされる
    ため、相殺・着弾が生成量を超えて供給されると (例: `src/ojama_accounting.py`
    から断片化イベント経由で同じ相殺を二度供給する配線ミス)、
    `outstanding` は 0 のまま E2 (`_all_settled()`) を満たし、
    `conservation_violation_count` の検査 (`total_generated` と
    `total_canceled+total_landed+unreconciled` の一致) を素通りして
    「きれいに帳簿が合った」ように見えてしまう。これは配線事故のうち
    「間違い (別の値が届く)」型であり、突合では検出できない
    (`feedback_wiring_gap_vs_wiring_error_2026-08-22`)。
    `oversettled_total` はこの見落としを構造的に潰すための独立した指標。

    **【2026-08-24 追記: 相殺・着弾の帰属 (Gate 3-2b 純粋部分)】**
    `unattributed_settlement_total`: `SettlementObservation` を FIFO で
    帰属させたとき、帰属先の未決着 chain が尽きて捨てられずに残った量の
    合計 (`_attribute` 参照)。0 が正常。非 0 は「観測された相殺・着弾に
    対応する chain が台帳に無い」ことを意味し、chain_id 復元の欠落や
    観測の時刻ズレを疑う信号になる。

    **【実装1、2026-08-25 追加】測定器の是正 — `total_unreconciled` は
    「閉じた episode のぶんだけ」である。** 窓の終わりにまだ OPEN な
    episode が抱える残量はこの値に一切含まれない。2026-08-25 の実測:
    v51 の窓を t1=545 で切ると `total_unreconciled` は 1,163 になるが、
    同じ動画を t1=533 で切ると `total_unreconciled` は **0** になる。
    しかし `ExchangeLedger._chains` の生値 (`ledger_residual_all`) は
    どちらの窓でも 1,163 のまま変化しない。**窓を早く切ると
    `total_unreconciled` が 0 になり「合格」に見えるが、それは
    「決着した」ではなく「まだ数えていない」だけである。**
    `total_unreconciled` / `open_episode_outstanding` / `ledger_residual_all`
    の 3 つを揃えて見なければ解決率は判断できない。

    - `open_episode_outstanding`: 窓終了時点でまだ OPEN な episode が
      抱える残量 (`ExchangeLedger.open_episode_outstanding()`)。
      OPEN な episode が無ければ 0.0。
    - `ledger_residual_all`: 台帳に残っている全 chain の outstanding 合計
      (episode に属さないものも含む、`ExchangeLedger.
      total_outstanding_all_chains()`)。窓の切り方に関わらず不変な
      台帳の生値そのもの。

    **【実装2、2026-08-25 追加】退役 (`retire_side_chains`/
    `_retire_all_chains_at_match_boundary`) で `self._chains` から削除
    される chain は、`_chains_for_episode` の `if cid in self._chains`
    ガードに弾かれ、その chain の相殺・着弾・生成量ごと episode の集計
    から丸ごと消えていた (2026-08-25 実測: v51 chain6 の相殺 18 個が
    `total_canceled` から 327 -> 309 へ落ちていた)。以下は退役前に
    転記された値であり、`total_canceled`/`total_landed`/`total_generated`
    に**含まれていない**分を別枠で示す (`ExchangeLedger.snapshot()` の
    `retired_canceled`/`retired_landed`/`retired_generated` を転記)。
    真の合計を見るときは、これらを対応する `total_*` へ足すこと。
    `retired_unreconciled` は既存の `ExchangeLedger` 側フィールドの転記
    (D1 に単独で出ていなかったため、ここで初めて D1 に露出する)。

    **【コーディネーター指摘、2026-08-25 追加】**
    `post_close_settlement_dropped_count`/`_amount`: close 済み chain へ
    close 後に届いた相殺・着弾の件数・量 (`ExchangeLedger.snapshot()` の
    同名フィールドの転記)。**根治は未着手。この値が大きければ close の
    判定が早すぎる証拠**である (`ExchangeLedger._maybe_count_post_close_
    settlement` docstring参照)。黙って落とさないための可視化のみ。
    """

    total_generated: float
    total_canceled: float
    total_landed: float
    total_unreconciled: float
    conservation_violation_count: int
    episodes_without_settlement_input: int
    oversettled_total: float
    oversettled_chain_count: int
    unattributed_settlement_total: float
    open_episode_outstanding: float = 0.0
    ledger_residual_all: float = 0.0
    retired_unreconciled: float = 0.0
    retired_canceled: float = 0.0
    retired_landed: float = 0.0
    retired_generated: float = 0.0
    post_close_settlement_dropped_count: int = 0
    post_close_settlement_dropped_amount: float = 0.0


@dataclass(frozen=True)
class D2FinalizeDivergence:
    """finalize 乖離 (確定 - 暫定) の一覧と下げ置換ゲートの発動件数。

    **単位はおじゃま個数** (2026-08-24 単位修正)。`score_to_ojama` で
    換算した後の値どうしの差であり、スコア点の差ではない。

    **【2026-08-25 是正】** `divergences` は**台帳に受け入れられた確定値
    (`finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF`。I16 で
    既定通過) だけ**を対象にする。拒否された確定値 (`simulate_fallback`。
    I16 で既定拒否) を混ぜると分布が汚染され、`FINALIZE_DOWNWARD_TOLERANCE`
    等の閾値設計を誤る (コーディネーター指摘)。拒否側は分布ではなく
    件数・量の集計だけを `rejected_divergence_count`/
    `rejected_divergence_amount_total` (絶対値の合計) に出す。
    """

    divergences: tuple[float, ...]
    gate_held_count: int
    rejected_divergence_count: int
    rejected_divergence_amount_total: float


@dataclass(frozen=True)
class D3ForcedCloseCounters:
    """強制終了カウンタ 3 種 (自己検算は Fix【5】で廃止、2026-08-25)。

    **報告: `episode_forced_close_count` は内訳を区別できない。**
    `ExchangeLedger._force_close` は試合境界 (`match_boundary`) と
    `EPISODE_MAX_SEC` 超過 (`max_sec`) の両方を同じ `forced_close_count`
    に積む。理由文字列は `ClosedEpisodeSummary.close_reason` に残る
    ようになったが (2026-08-24 追加)、`forced_close_count` 自体は
    合算カウンタのままなのでここでは合算値を報告する。

    素点 (掛け算式の左側) の 10 の倍数検算は `ChainEventObservation` に
    `base_score` が無いため実装しない
    (`score_multiple_of_ten_violation_count` は常に `None`)。

    **【2026-08-25 Fix【5】: 自己検算 (`self_check_*`) を廃止】**
    新設計では `SCORE_FINALIZE` (score OCR 確定差分) の値がそのまま
    会計の確定値になるため、それを検算する**独立な第二の権威値が
    存在しない**。旧 `authoritative_ojama` の供給元は `finalized_score`
    と**同一の accumulator** (`OjamaAccountingTracker.
    total_generated_by_pX`) であり、フレーム一致を直しても常に差 0 に
    なるだけ (tautological)。意味のない「0 が合格」は誤読されるため
    残さない。2026-08-25 の実測で `n_authoritative_ojama_present =
    0/20, 0/7` (供給自体がほぼ機能していなかった) も確認済み。
    (前任コーダと診断役が独立に同じ結論に達した。)

    `land_split_count` (2026-08-24 追加): LAND 1 件が
    `OJAMA_MAX_DROP_PER_TURN` (30 個) を超えたため複数イベントに分割
    した回数。フレーム間引き等で複数ターン分の着弾差分がまとまった
    場合に発生する。例外で落とすのでも黙って丸めるのでもなく、
    分割して数える (`_split_land_amount` 参照)。
    """

    episode_forced_close_count: int
    chain_id_force_cut_count: int
    score_multiple_of_ten_violation_count: int | None
    land_split_count: int


@dataclass(frozen=True)
class D4CloseReasonCounts:
    """chain_id をどの理由で閉じたか (CloseReason 別件数) と迷子の終了信号数。"""

    counts_by_reason: dict[str, int]
    orphan_end_signal_count: int


@dataclass(frozen=True)
class D6NetDivergence:
    """net_raw / net_display の乖離が発生した回数と最大乖離量。"""

    divergence_event_count: int
    max_divergence: float


@dataclass(frozen=True)
class D7GenerationStats:
    """生成量そのもの (chain_id 数、段数の分布、growth_observed の割合)。

    段数の内訳 (各段の値) は持たない: `ChainIdResolver.ResolvedChain` は
    `step_count` (段数) と `provisional_score` (最終値) しか返さず、段
    ごとの履歴を保持しない。暫定生成量を `FIRE` 1 件にまとめる設計
    (`_chain_to_events` 参照) とも整合する。

    **【実装3、2026-08-25 追加】自己相殺netting前の生値。**
    `cancel_own_pending_then_send_surplus` (`src/ojama_accounting.py:733`)
    は、発火した側が自分の受け予定を先に打ち消し、余りだけを相手に送る。
    しかし旧実装の FIRE イベントは `gen` 全額を登録しており、発火本人の
    chain の残量が自己相殺した分だけ恒久的に過大だった (実測 v51:
    chain7 は生成19・自己相殺19・送るべき余り0 のはずが、19 のまま
    残っていた)。`_chain_to_events` は FIRE/FINALIZE の量を自己相殺後の
    実送付量 (余剰) に是正したため、台帳へ登録される量は
    `raw_generation_total` より `self_canceled_total` だけ小さくなる。

    - `raw_generation_total`: 自己相殺 netting **前**の生成量 (score OCR
      由来の値そのまま) の総和。台帳の外に置く「元の生成」の値であり、
      独立検算 (スコア合計 ÷ レート) と比較すべきはこちらである。
    - `self_canceled_total`: 発火本人の生成のうち、自分の受け予定の
      打ち消しに使われた量の総和 (別カウンタとして必ず記録)。
      `raw_generation_total == <台帳へ登録された生成量の総和> +
      self_canceled_total` が保存則として成り立つ (§テスト参照)。

    **【コーディネーター指摘、2026-08-25 追加】**
    `self_cancel_clipped_count`/`_amount`: 自己相殺の生値がクリップ上限
    (`provisional_ojama`/`finalized_ojama` の大きい方) を超えて丸められた
    件数・量。実測では 0 が期待値だが、発生したら黙って丸めず必ず数える。
    `self_cancel_eligible_count` はクリップ判定を試みた総数 (母数)。
    `self_cancel_clipped_count`/`self_cancel_eligible_count` を
    `0/N` の形で報告できるようにするための母数である。
    """

    chain_id_count: int
    step_counts: tuple[int, ...]
    growth_observed_ratio: float
    raw_generation_total: float = 0.0
    self_canceled_total: float = 0.0
    self_cancel_clipped_count: int = 0
    self_cancel_clipped_amount: float = 0.0
    self_cancel_eligible_count: int = 0
    # 【P1-1 是正、2026-08-25 Codex レビュー】simulate 由来 (finalized_source
    # == "simulate_fallback") の chain を会計から除外した件数・量。
    # 旧実装は FINALIZE こそ台帳 (I16) が拒否していたが、同じ値が先に
    # provisional FIRE として登録され `net_raw`/`total_generated` に
    # 入っていた (I16 の迂回)。既定 (`allow_simulate_fallback=False`) では
    # イベント化そのものを行わず、ここに件数・量を出す
    # (母数 = `chain_id_count`。`0/N` で読むこと)。
    # `raw_generation_total` (独立検算との比較値 = score OCR 由来の総和)
    # からも同じ条件で除外する。
    simulate_excluded_chain_count: int = 0
    simulate_excluded_amount: float = 0.0


@dataclass(frozen=True)
class ExchangeEpisodeDiagnostics:
    """tracker が出す診断値一式 (D1〜D4, D6, D7)。D5 は出さない (責務外)。

    `negative_generation_delta_count` (2026-08-25 追加): `observe_generation()`
    が負の増分 (累積カウンタの減少。物理的にありえない) を受け取った件数。
    0 が正常。非 0 は試合境界のリセット漏れかバグの証拠 (黙って捨てない)。
    """

    unknown_mechanism_count: int
    unknown_side_count: int
    negative_generation_delta_count: int
    d1: D1EpisodeTotals
    d2: D2FinalizeDivergence
    d3: D3ForcedCloseCounters
    d4: D4CloseReasonCounts
    d6: D6NetDivergence
    d7: D7GenerationStats


_ZERO_DIAGNOSTICS = ExchangeEpisodeDiagnostics(
    unknown_mechanism_count=0,
    unknown_side_count=0,
    negative_generation_delta_count=0,
    d1=D1EpisodeTotals(
        total_generated=0.0, total_canceled=0.0, total_landed=0.0,
        total_unreconciled=0.0, conservation_violation_count=0,
        episodes_without_settlement_input=0,
        oversettled_total=0.0, oversettled_chain_count=0,
        unattributed_settlement_total=0.0,
        open_episode_outstanding=0.0, ledger_residual_all=0.0,
        retired_unreconciled=0.0, retired_canceled=0.0,
        retired_landed=0.0, retired_generated=0.0,
        post_close_settlement_dropped_count=0,
        post_close_settlement_dropped_amount=0.0,
    ),
    d2=D2FinalizeDivergence(
        divergences=(), gate_held_count=0,
        rejected_divergence_count=0, rejected_divergence_amount_total=0.0,
    ),
    d3=D3ForcedCloseCounters(
        episode_forced_close_count=0, chain_id_force_cut_count=0,
        score_multiple_of_ten_violation_count=None,
        land_split_count=0,
    ),
    d4=D4CloseReasonCounts(counts_by_reason={}, orphan_end_signal_count=0),
    d6=D6NetDivergence(divergence_event_count=0, max_divergence=0.0),
    d7=D7GenerationStats(
        chain_id_count=0, step_counts=(), growth_observed_ratio=0.0,
        raw_generation_total=0.0, self_canceled_total=0.0,
        self_cancel_clipped_count=0, self_cancel_clipped_amount=0.0,
        self_cancel_eligible_count=0,
        simulate_excluded_chain_count=0, simulate_excluded_amount=0.0,
    ),
)


# ============================
# tracker 本体
# ============================

class ExchangeEpisodeTracker:
    """動画 1 本ぶんの連鎖イベント列を受けて診断値を出す (観測のみ)。

    `enabled=False` (既定) のとき `observe` は即 return し、何も計算しない。
    """

    def __init__(
        self, *, enabled: bool = False, allow_simulate_fallback: bool = False,
    ) -> None:
        """
        Args:
            enabled: False (既定) なら全 observe が no-op。
            allow_simulate_fallback: I16 (`src/exchange_ledger.py`) と対になる
                P1-1 のゲート (2026-08-25 追加、既存呼び出し互換のため
                keyword-only + 既定値あり)。既定 `False` では simulate 由来
                (`finalized_source == "simulate_fallback"`) の chain を
                **どのイベント種別でも**台帳へ登録しない (FIRE の provisional
                も含む。I16 は FINALIZE しか拒否できず迂回されていた)。
                `True` を明示した場合のみ従来どおり登録し、台帳側 I16 も
                同時に緩める。
        """
        self._enabled = enabled
        self._allow_simulate_fallback = allow_simulate_fallback
        self._resolver = ChainIdResolver()
        self._ledger = ExchangeLedger(allow_simulate_fallback=allow_simulate_fallback)
        self._last_game_idx: int | None = None
        self._context_by_t: dict[float, _ObservationContext] = {}
        self._unknown_mechanism_count: int = 0
        self._unknown_side_count: int = 0
        self._net_divergence_event_count: int = 0
        self._max_net_divergence: float = 0.0
        self._settlement_buffer: list[SettlementObservation] = []
        self._wipe_buffer: list[tuple[float, Side, int]] = []
        self._unattributed_settlement_total: float = 0.0
        self._land_split_count: int = 0
        self._negative_generation_delta_count: int = 0
        # 【実装3、2026-08-25 追加】自己相殺で FIRE/FINALIZE から差し引いた量の
        # 総和 (`_chain_to_events` 参照)。
        self._self_canceled_total: float = 0.0
        # 【コーディネーター指摘、2026-08-25 追加】自己相殺の生値が
        # クリップ上限 (`provisional_ojama`/`finalized_ojama` の大きい方) を
        # 超えて丸められた件数・量と、クリップ判定を試みた総数 (母数)。
        self._self_cancel_clipped_count: int = 0
        self._self_cancel_clipped_amount: float = 0.0
        self._self_cancel_eligible_count: int = 0
        # 【P1-1 是正、2026-08-25】simulate 由来 chain を会計から除外した
        # 件数・量 (母数 = D7 の chain_id_count)。
        self._simulate_excluded_chain_count: int = 0
        self._simulate_excluded_amount: float = 0.0
        self._diagnostics: ExchangeEpisodeDiagnostics = _ZERO_DIAGNOSTICS

    # ------------------------------
    # 受付
    # ------------------------------

    def observe(self, obs: ChainEventObservation) -> None:
        """観測を 1 件受け付ける。t_sec 昇順で呼ぶこと。"""
        if not self._enabled:
            return
        self._maybe_inject_match_boundary(obs)
        self._context_by_t[obs.t_sec] = _ObservationContext(
            game_idx=obs.game_idx, elapsed_sec=obs.elapsed_sec,
            ojama_sent=obs.ojama_sent,
        )
        kind = _MECHANISM_TO_KIND.get(obs.mechanism)
        if kind is None:
            self._unknown_mechanism_count += 1
            return
        self._resolver.push(ChainObservation(
            side=obs.side, t_sec=obs.t_sec, kind=kind,
            chain_count=obs.chain_count, total_score=obs.total_score,
            mechanism=obs.mechanism,
        ))

    def _maybe_inject_match_boundary(
        self, obs: ChainEventObservation | GenerationObservation,
    ) -> None:
        """game_idx が変わったら resolver へ MATCH_BOUNDARY を注入する。

        **`observe()` と `observe_generation()` の共有ロジック** (2026-08-25
        是正)。`_last_game_idx` を 2 つの入力チャネルで共有することで、
        「試合境界の直後、次の `observe()` が来る前に `observe_generation()`
        だけが届く」順序でも境界を取り逃さない。**呼び出し順序を呼び出し側
        の注意力に頼らない**という方針 (コーディネーター指摘) をここで
        構造的に満たす。
        """
        if self._last_game_idx is not None and obs.game_idx != self._last_game_idx:
            self._resolver.push(ChainObservation(
                side=obs.side, t_sec=obs.t_sec, kind=ObservationKind.MATCH_BOUNDARY,
            ))
        self._last_game_idx = obs.game_idx

    def observe_settlement(self, obs: SettlementObservation) -> None:
        """相殺・着弾の観測を 1 件受け付ける (バッファリングのみ)。

        `finish()` で FIRE/FINALIZE イベントと t_sec 昇順にマージしてから
        台帳へ反映する。観測時点ではまだ対応する chain_id が台帳に存在
        しない場合がある (chain_id は resolver の flush 後にしか確定
        しない) ため、ここでは即座に処理しない。
        """
        if not self._enabled:
            return
        self._settlement_buffer.append(obs)

    def observe_wipe(self, side: Side, t_sec: float, game_idx: int) -> None:
        """ラウンド境界での片側の予告消滅 (ワイプ) の観測を 1 件受け付ける
        (Fix【2】、2026-08-25 追加、バッファリングのみ)。

        `side` は「予告を受け取れなくなった側」(負けた側)。
        `ExchangeLedger.retire_side_chains` は `finish()` (`_replay_timeline`)
        で chain イベント・相殺着弾観測と同じタイムライン上、正しい
        t_sec 順で反映する (chain_id 解決の完了を待つ必要があるため、
        `observe_settlement` と同様に即座には処理しない)。
        """
        if not self._enabled:
            return
        self._wipe_buffer.append((t_sec, side, game_idx))

    def observe_generation(self, obs: GenerationObservation) -> None:
        """score OCR 確認済みの生成増分を 1 件受け付ける。

        `ObservationKind.SCORE_FINALIZE` として resolver へ**即座に**反映する
        (`observe()` の FORMULA_STEP/CHAIN_SETTLED と同じ経路。chain_id
        解決の途中で必要な入力のため、`SettlementObservation` と違い
        バッファリングしない)。

        **試合境界の注入・診断コンテキストの更新は `observe()` と同じ**
        (2026-08-25 是正): このチャネル単独で境界を跨いでも、前の試合の
        chain へ確定値が紛れ込まない (今回潰した「試合を跨ぐ汚染」と同じ形の
        欠陥を、このチャネルでも構造的に防ぐ)。

        `generated_delta == 0` (増分なし・OCR ノイズ等) は黙って無視する
        (`_attribute` の `amount <= 0.0` ガードと同じ扱い。実際に何も
        生成されていないので捨てる情報自体が存在しない)。
        `generated_delta < 0` は**累積カウンタが減った**ことを意味し、
        物理的にありえない (試合境界のリセット漏れ、またはバグの証拠)。
        黙って捨てず `negative_generation_delta_count` に記録して無視する。
        """
        if not self._enabled:
            return
        if obs.generated_delta < 0:
            self._negative_generation_delta_count += 1
            return
        if obs.generated_delta == 0:
            return
        self._maybe_inject_match_boundary(obs)
        self._merge_generation_context(obs)
        self._resolver.push(ChainObservation(
            side=obs.side, t_sec=obs.t_sec, kind=ObservationKind.SCORE_FINALIZE,
            total_score=obs.generated_delta, mechanism=_SCORE_FINALIZE_MECHANISM_LABEL,
        ))

    def _merge_generation_context(self, obs: GenerationObservation) -> None:
        """`_context_by_t` の `game_idx` を `GenerationObservation` の情報で
        更新する (`_chain_to_events` が境界直後の game_idx を正しく引けるように)。

        `elapsed_sec`/`ojama_sent` はこのチャネルには無い情報であり、
        既存 (同一 t_sec の `observe()` 由来) のエントリがあればそのまま
        保持する。既存が無ければ空値で新規作成する。
        """
        existing = self._context_by_t.get(obs.t_sec)
        self._context_by_t[obs.t_sec] = _ObservationContext(
            game_idx=obs.game_idx,
            elapsed_sec=existing.elapsed_sec if existing else 0.0,
            ojama_sent=existing.ojama_sent if existing else 0,
        )

    # ------------------------------
    # 確定
    # ------------------------------

    def finish(self) -> None:
        """入力ストリーム終端。resolver を flush し、台帳へ反映して診断値を確定する。"""
        if not self._enabled:
            return
        self._resolver.flush()
        resolved = self._resolver.resolved()
        ojama_by_chain = self._convert_resolved_to_ojama(resolved)
        self_cancel_lookup = self._build_self_cancel_lookup()
        chain_events = self._build_events(resolved, ojama_by_chain, self_cancel_lookup)
        final_snapshot = self._replay_timeline(chain_events)
        self._diagnostics = self._build_diagnostics(resolved, ojama_by_chain, final_snapshot)

    def _build_self_cancel_lookup(self) -> dict[tuple[Side, float], float]:
        """自己相殺ルックアップ (実装3、2026-08-25 追加)。

        `SettlementObservation.canceled_by_1p`/`canceled_by_2p` は、その
        side 自身の `pending_uncapped` が「自分の連鎖確定と同時」に減少
        した量 (`_classify_side_delta` の CANCEL 分岐) であり、
        `cancel_own_pending_then_send_surplus` (`src/ojama_accounting.py:733`)
        が発火した側自身の受け予定を先に打ち消す量と物理的に一致する
        (同一フレームの score OCR 確定差分が両方の判定を駆動するため)。
        キーは (発火した側, 発火が確定した時刻)。同一 side・同一時刻の
        観測が複数あっても (通常は無い) 合算して安全側に倒す。
        """
        lookup: dict[tuple[Side, float], float] = {}
        for obs in self._settlement_buffer:
            for side, amount in (
                (Side.P1, obs.canceled_by_1p), (Side.P2, obs.canceled_by_2p),
            ):
                if amount > 0.0:
                    key = (side, obs.t_sec)
                    lookup[key] = lookup.get(key, 0.0) + amount
        return lookup

    def diagnostics(self) -> ExchangeEpisodeDiagnostics:
        """診断値のスナップショット。呼び出しによって内部状態は変わらない。"""
        return self._diagnostics

    # ------------------------------
    # 単位換算 (スコア点 -> おじゃま個数)
    # ------------------------------

    def _to_ojama(self, score: int, elapsed_sec: float) -> int:
        """スコア点をおじゃま個数へ換算する (§4.1.2: 暫定・確定を同一規約で)。"""
        return score_to_ojama(
            score, prev_leftover=_PREV_LEFTOVER_FOR_TRACKER,
            elapsed_sec=elapsed_sec, rate_base=OJAMA_RATE_STANDARD,
        ).ojama_count

    def _convert_resolved_to_ojama(
        self, resolved: list[ResolvedChain],
    ) -> dict[int, _ChainOjama]:
        """各 chain_id の暫定・確定生成量をおじゃま個数へ換算し、自己検算する。"""
        result: dict[int, _ChainOjama] = {}
        for rc in resolved:
            prov = self._provisional_ojama_for(rc)
            fin = self._finalized_ojama_for(rc) if self._has_finalized_value(rc) else None
            result[rc.chain_id] = _ChainOjama(provisional_ojama=prov, finalized_ojama=fin)
        return result

    def _has_finalized_value(self, rc: ResolvedChain) -> bool:
        return rc.was_finalized and rc.finalized_score is not None

    def _provisional_ojama_for(self, rc: ResolvedChain) -> int:
        """暫定生成量をおじゃま個数へ換算する。

        **【2026-08-25 追加の例外】** growth_observed=False (成長フェーズを
        一度も観測できず即クローズした経路) かつ
        `finalized_source==FINALIZED_SOURCE_SCORE_OCR_DIFF` のときだけ、
        `provisional_score` は `finalized_score` と同じ値 (`_close_immediately_
        without_growth` が両方に同じ値を入れるため) で、**既におじゃま個数**
        (`GenerationObservation.generated_delta`、換算不要)。それ以外は
        常にスコア点として `score_to_ojama` で変換する。

        **なぜ二重換算を避けるか**: `score_to_ojama` はスコア点をレート
        (既定 70) で割っておじゃま個数にする関数。既に個数の値 (例: 367)
        へもう一度適用すると `367 // 70 = 5` のように**再び割ってしまい**、
        本物の確定生成量が黙って約 1/70 に消える。これは今回の主題だった
        「推定値が確定値として紛れ込む」バグとは別方向だが、同じ根
        (単位の取り違え) を持つ**第二の静かなバグ**になる。
        """
        if not rc.growth_observed and rc.finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF:
            return rc.provisional_score
        fire_ctx = self._context_by_t.get(rc.opened_at_sec)
        fire_elapsed = fire_ctx.elapsed_sec if fire_ctx else 0.0
        return self._to_ojama(rc.provisional_score, fire_elapsed)

    def _finalized_ojama_for(self, rc: ResolvedChain) -> int:
        """確定生成量をおじゃま個数へ換算する (自己検算は Fix【5】で廃止)。

        `finalized_source==FINALIZED_SOURCE_SCORE_OCR_DIFF` の値は
        `GenerationObservation.generated_delta` 由来で**既におじゃま個数**
        (換算不要、コーディネーター指定)。二重変換すると壊れた値になる
        (理由は `_provisional_ojama_for` の docstring 参照: レートで
        もう一度割ってしまい、確定生成量が黙って約 1/70 に消える)。
        それ以外 (`simulate_fallback`) はスコア点なので変換する。
        """
        if rc.finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF:
            return rc.finalized_score
        fire_ctx = self._context_by_t.get(rc.opened_at_sec)
        fin_ctx = self._context_by_t.get(rc.closed_at_sec)
        fin_elapsed = (
            fin_ctx.elapsed_sec if fin_ctx
            else (fire_ctx.elapsed_sec if fire_ctx else 0.0)
        )
        return self._to_ojama(rc.finalized_score, fin_elapsed)

    # ------------------------------
    # ResolvedChain -> ExchangeEvent への変換
    # ------------------------------

    def _build_events(
        self, resolved: list[ResolvedChain], ojama_by_chain: dict[int, _ChainOjama],
        self_cancel_lookup: dict[tuple[Side, float], float],
    ) -> list[tuple[ExchangeEvent, int]]:
        """ResolvedChain の一覧を FIRE/FINALIZE イベント列へ変換する (t_sec 昇順)。

        **【P1-1 是正、2026-08-25 Codex レビュー】** simulate 由来の chain は
        既定 (`allow_simulate_fallback=False`) でイベント化そのものを行わない。
        旧実装では FINALIZE こそ台帳 (I16) が拒否したが、同じ値が先に
        provisional FIRE として登録され会計 (`net_raw`/`total_generated`) に
        入っていた (Codex 最小再現: baseline-only 52150 点で
        finalize_rejected_amount=745 なのに net_raw=745)。除外は黙って行わず
        件数・量を `simulate_excluded_*` に必ず記録する。
        """
        out: list[tuple[ExchangeEvent, int]] = []
        for rc in resolved:
            side = _SIDE_TO_LEDGER_SIDE.get(rc.side)
            if side is None:
                self._unknown_side_count += 1
                continue
            if self._is_simulate_source_excluded(rc):
                self._simulate_excluded_chain_count += 1
                self._simulate_excluded_amount += float(
                    ojama_by_chain[rc.chain_id].provisional_ojama,
                )
                continue
            out.extend(self._chain_to_events(
                rc, side, ojama_by_chain[rc.chain_id], self_cancel_lookup,
            ))
        out.sort(key=lambda pair: (pair[0].t_sec, pair[0].kind is EventKind.FINALIZE))
        return out

    def _is_simulate_source_excluded(self, rc: ResolvedChain) -> bool:
        """P1-1: この chain の値が simulate 由来で、会計から除外すべきか。

        副作用なし (カウンタ更新は呼び出し側)。`_build_events` (台帳への
        イベント化) と `_build_d7` (独立検算と比較する raw 総和) が
        **同じ条件**で除外することを 1 箇所に固定するための述語。
        """
        return (
            not self._allow_simulate_fallback
            and rc.finalized_source == FINALIZED_SOURCE_SIMULATE_FALLBACK
        )

    def _self_canceled_amount_for_netting(
        self, rc: ResolvedChain, side: Side, ojama: _ChainOjama,
        self_cancel_lookup: dict[tuple[Side, float], float],
    ) -> float:
        """1 本の chain が自分の受け予定を打ち消すのに使った量を求め、
        `self_canceled_total` (D7) へ加算する (実装3)。

        score OCR 確定差分 (`FINALIZED_SOURCE_SCORE_OCR_DIFF`) で確定した
        chain だけを対象にする。`simulate_fallback`/未確定の chain には
        対応する `observe_generation()` 呼び出しが無いため、
        `self_cancel_lookup` に偶然同じ t_sec のキーが無い限り 0 になる
        (誤って紐付かないための明示ガード)。

        **【2026-08-25 実データ検証で発覚した修正】** 当初は
        `provisional_ojama` だけを上限にクリップしていたが、
        `provisional_ojama` (掛け算式の成長フェーズ由来) と
        `finalized_ojama` (score OCR 確定差分由来) は独立な観測経路であり、
        前者が OCR の乱れ等で小さい/0 のまま確定後に大きく乖離することが
        実測で確認された (v51 chain6: provisional=0 点相当・finalized=620個。
        chain11: provisional=40点→ojama換算0個・finalized=30個)。
        `provisional_ojama` だけでクリップすると、FINALIZE 側へ差し引く
        自己相殺まで 0 に潰れ、**自己相殺が丸ごと無視される**という
        実装3の趣旨に反する結果になっていた (chain6/chain11 で自己相殺
        145/30 が両方とも 0 として扱われていた、v51 実測で発覚)。
        `provisional_ojama`/`finalized_ojama` の**大きい方**を上限にする
        ことで、この取りこぼしを防ぐ (FIRE/FINALIZE それぞれの netting は
        呼び出し側の `max(0.0, amount - self_canceled)` で独立にクリップ
        されるため、上限を大きい方に広げても負の量が生まれる心配は無い)。
        """
        if not (rc.was_finalized and rc.finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF):
            return 0.0
        self._self_cancel_eligible_count += 1
        raw = self_cancel_lookup.get((side, rc.closed_at_sec), 0.0)
        cap = max(float(ojama.provisional_ojama), float(ojama.finalized_ojama or 0.0))
        if raw > cap:
            # 【コーディネーター指摘、2026-08-25 追加】実測では未発生だが、
            # 自己相殺の生値がこの上限を超える壊れたデータが来たら
            # 黙って丸めず必ず数える (発生すれば `保存則テスト
            # (元の生成=送付分+自己相殺) が緩む方向にずれる可能性がある証拠)。
            self._self_cancel_clipped_count += 1
            self._self_cancel_clipped_amount += raw - cap
        self_canceled = min(raw, cap)
        self._self_canceled_total += self_canceled
        return self_canceled

    def _chain_to_events(
        self, rc: ResolvedChain, side: Side, ojama: _ChainOjama,
        self_cancel_lookup: dict[tuple[Side, float], float],
    ) -> list[tuple[ExchangeEvent, int]]:
        """1 本の ResolvedChain を FIRE (+ 確定していれば FINALIZE) へ変換する。

        **STEP は使わず FIRE 1 件に暫定生成量の全量をまとめる。** 台帳側の
        `provisional_amount` は加算なので、1 回でまとめても合計は変わらない。
        ただし副作用として、lazy open (§9.4.1) が判定するタイミングは
        「段階的に 4 個を超えた瞬間」ではなく「その連鎖が chain_id として
        確定した (FIRE イベントが生成された) 瞬間」にずれる。段の内訳が
        必要になったら `ChainIdResolver` 側に段の履歴を持たせる拡張が要る。

        **【実装3、2026-08-25 追加】自己相殺 netting。** 台帳へ登録する
        FIRE/FINALIZE の量から自己相殺分を差し引く。理由と検算式は
        `D7GenerationStats` docstring 参照 (`self_canceled_total`/
        `raw_generation_total`)。
        """
        self_canceled = self._self_canceled_amount_for_netting(
            rc, side, ojama, self_cancel_lookup,
        )
        net_provisional = max(0.0, ojama.provisional_ojama - self_canceled)
        fire_ctx = self._context_by_t.get(rc.opened_at_sec)
        fire_game_idx = fire_ctx.game_idx if fire_ctx else 0
        events = [(ExchangeEvent(
            kind=EventKind.FIRE, side=side, t_sec=rc.opened_at_sec,
            amount=float(net_provisional), chain_id=rc.chain_id,
            chain_count=rc.step_count, source="exchange_episode_tracker",
        ), fire_game_idx)]
        if rc.was_finalized and ojama.finalized_ojama is not None:
            net_finalized = max(0.0, ojama.finalized_ojama - self_canceled)
            fin_ctx = self._context_by_t.get(rc.closed_at_sec)
            fin_game_idx = fin_ctx.game_idx if fin_ctx else fire_game_idx
            events.append((ExchangeEvent(
                kind=EventKind.FINALIZE, side=side, t_sec=rc.closed_at_sec,
                amount=float(net_finalized), chain_id=rc.chain_id,
                # I16 (`src/exchange_ledger.py`): FINALIZE の値供給源をここで
                # 正直に申告する。`rc.finalized_source` が
                # `FINALIZED_SOURCE_SCORE_OCR_DIFF` (score OCR 確定差分) なら
                # 台帳が既定で受け入れ、`simulate_fallback` (低信頼推定) なら
                # 台帳が既定で拒否する (`allow_simulate_fallback=True` を
                # 明示しない限り)。`was_finalized=True` のとき
                # `finalized_source` は必ずどちらかの文字列になる
                # (`ChainIdResolver._close_side`/`_close_immediately_without_growth`
                # 参照。None にはならない)。
                source=rc.finalized_source or "",
            ), fin_game_idx))
        return events

    def _replay_timeline(
        self, chain_events: list[tuple[ExchangeEvent, int]],
    ) -> LedgerSnapshot:
        """chain イベント・settlement・wipe 観測を t_sec 昇順にマージして
        台帳へ流し込む (Fix【2】で wipe を追加、2026-08-25)。

        settlement の FIFO 帰属 (`_attribute`) はその時点で台帳に存在する
        未決着 chain を見て決めるため、chain 由来のイベントより先に
        処理してはいけない。同一 t_sec の優先順位は
        chain イベント (0) -> wipe (1) -> settlement (2)。wipe を
        settlement より先にするのは、ワイプで退役した分には
        (もう受け取らない、という確定事象のため) 帰属させないようにする。
        """
        timeline = self._build_timeline(chain_events)
        snapshot = self._ledger.snapshot()
        for _, kind, payload in timeline:
            game_idx = self._apply_timeline_entry(kind, payload)
            snapshot = self._ledger.snapshot(PhysicalContext(game_idx=game_idx))
            self._track_net_divergence(snapshot)
        return snapshot

    def _apply_timeline_entry(self, kind: int, payload: object) -> int:
        """timeline 1 件を台帳へ反映し、その時点の game_idx を返す。"""
        if kind == _TIMELINE_KIND_SETTLEMENT:
            self._process_settlement(payload)
            return payload.game_idx
        if kind == _TIMELINE_KIND_WIPE:
            t_sec, side, game_idx = payload
            self._ledger.retire_side_chains(side, t_sec, PhysicalContext(game_idx=game_idx))
            return game_idx
        ev, game_idx = payload
        self._ledger.push(ev, PhysicalContext(game_idx=game_idx))
        return game_idx

    def _build_timeline(
        self, chain_events: list[tuple[ExchangeEvent, int]],
    ) -> list[tuple[float, int, object]]:
        """chain イベント・settlement・wipe 観測を (t_sec, kind, payload) でまとめる。

        同一 t_sec の優先順位は `kind` の昇順
        (chain=`_TIMELINE_KIND_CHAIN` -> wipe=`_TIMELINE_KIND_WIPE` ->
        settlement=`_TIMELINE_KIND_SETTLEMENT`、`_replay_timeline` docstring
        参照)。Fix【2】で wipe 系列を追加した (2026-08-25、以前は
        settlement だけの bool フラグだった)。
        """
        timeline: list[tuple[float, int, object]] = [
            (ev.t_sec, _TIMELINE_KIND_CHAIN, (ev, game_idx))
            for ev, game_idx in chain_events
        ]
        timeline += [
            (t_sec, _TIMELINE_KIND_WIPE, (t_sec, side, game_idx))
            for t_sec, side, game_idx in self._wipe_buffer
        ]
        timeline += [
            (obs.t_sec, _TIMELINE_KIND_SETTLEMENT, obs)
            for obs in self._settlement_buffer
        ]
        timeline.sort(key=lambda item: (item[0], item[1]))
        return timeline

    def _track_net_divergence(self, snapshot: LedgerSnapshot) -> None:
        divergence = abs(snapshot.net_raw - snapshot.net_display)
        if divergence <= _FLOAT_EPS:
            return
        self._net_divergence_event_count += 1
        self._max_net_divergence = max(self._max_net_divergence, divergence)

    # ------------------------------
    # 相殺・着弾の帰属 (SettlementObservation -> CANCEL/LAND イベント)
    # ------------------------------

    def _process_settlement(self, obs: SettlementObservation) -> None:
        """1 件の SettlementObservation を 4 方向 (相殺 x2、着弾 x2) に分けて帰属する。"""
        self._attribute(Side.P1, obs.canceled_by_1p, EventKind.CANCEL, obs.t_sec, obs.game_idx)
        self._attribute(Side.P2, obs.canceled_by_2p, EventKind.CANCEL, obs.t_sec, obs.game_idx)
        self._attribute(Side.P1, obs.landed_on_1p, EventKind.LAND, obs.t_sec, obs.game_idx)
        self._attribute(Side.P2, obs.landed_on_2p, EventKind.LAND, obs.t_sec, obs.game_idx)

    def _attribute(
        self, observer_side: Side, amount: float, kind: EventKind,
        t_sec: float, game_idx: int,
    ) -> None:
        """observer_side が相殺/着弾した量を、相手側の未決着 chain へ帰属する。

        **帰属規則 (向きに注意)**: 台帳の `ChainRecord.canceled`/`.landed` は
        「その連鎖が生成した量のうち打ち消された/降った量」である。
        `observer_side` が相殺した/着弾を受けたのは、
        **常に相手側 (`observer_side.other`) の連鎖が生成した量**である
        (1P が相殺した量 -> 消えたのは 2P の連鎖が生成した量。
         1P に着弾した量 -> 降ったのは 2P の連鎖が生成した量)。
        2026-08-10 の「side 非依存の絶対量を無条件反転した」バグの教訓
        により、帰属先を `observer_side` 自身にしないことをここに明記する。

        配分は FIFO (chain_id 昇順 = 発火が古い順、おじゃまは先に
        送られたものから順に処理されるため)。帰属先が尽きたら残りを
        `_unattributed_settlement_total` へ計上し、黙って捨てない。
        """
        if amount <= 0.0:
            return
        attributee = observer_side.other
        remaining = amount
        ctx = PhysicalContext(game_idx=game_idx)
        for chain_id in self._ledger.open_chain_ids(attributee):
            if remaining <= 0.0:
                break
            take = min(remaining, self._ledger.outstanding_of(chain_id))
            if take <= 0.0:
                continue
            remaining -= take
            self._push_settlement_event(
                kind, observer_side, attributee, chain_id, take, t_sec, ctx,
            )
        if remaining > 0.0:
            self._unattributed_settlement_total += remaining

    def _push_settlement_event(
        self, kind: EventKind, observer_side: Side, attributee: Side,
        chain_id: int, amount: float, t_sec: float, ctx: PhysicalContext,
    ) -> None:
        """CANCEL/LAND イベントを台帳へ push する。

        CANCEL の `side` は帰属先 chain の side (相殺で消えた連鎖の持ち主、
        既存テストの慣習 `tests/test_exchange_ledger.py` の `_cancel` に合わせる)。
        LAND の `side` は「受けた側」(`observer_side`。台帳の規約、
        `ExchangeEvent.side` docstring 参照)。LAND は上限超過時に分割する。

        **時刻は偽装しない。** 分割された複数イベントは同一の本物の
        `t_sec` を持ち、`ExchangeEvent.seq` (2026-08-24 追加) で区別する。
        時刻をずらして重複排除 (I4) をすり抜けるのは、断片化による
        二重計上を止める機構そのものを無効化する禁じ手 (fable アーキ裁定)。
        """
        ev_side = attributee if kind is EventKind.CANCEL else observer_side
        chunks = (
            self._split_land_amount(amount) if kind is EventKind.LAND else [amount]
        )
        for i, chunk in enumerate(chunks):
            self._ledger.push(ExchangeEvent(
                kind=kind, side=ev_side, t_sec=t_sec, seq=i,
                chain_id=chain_id, amount=chunk, source="settlement_observation",
            ), ctx)

    def _split_land_amount(self, amount: float) -> list[float]:
        """LAND 1 回分が OJAMA_MAX_DROP_PER_TURN を超えたら 30 個以下に分割する。

        台帳は 1 件の LAND がこの上限を超えると例外を投げる (物理則:
        1 ターンに 30 個より多く降ることはない)。フレーム間引き等で
        複数ターン分の差分がまとまった場合、例外で落とすのでも黙って
        丸めるのでもなく、複数イベントに分割して `land_split_count` に数える。
        """
        if amount <= OJAMA_MAX_DROP_PER_TURN:
            return [amount]
        chunks: list[float] = []
        remaining = amount
        while remaining > OJAMA_MAX_DROP_PER_TURN:
            chunks.append(float(OJAMA_MAX_DROP_PER_TURN))
            remaining -= OJAMA_MAX_DROP_PER_TURN
        if remaining > 0.0:
            chunks.append(remaining)
        self._land_split_count += 1
        return chunks

    # ------------------------------
    # 診断値の組み立て
    # ------------------------------

    def _build_diagnostics(
        self,
        resolved: list[ResolvedChain],
        ojama_by_chain: dict[int, _ChainOjama],
        final_snapshot: LedgerSnapshot,
    ) -> ExchangeEpisodeDiagnostics:
        resolver_stats = self._resolver.stats()
        return ExchangeEpisodeDiagnostics(
            unknown_mechanism_count=self._unknown_mechanism_count,
            unknown_side_count=self._unknown_side_count,
            negative_generation_delta_count=self._negative_generation_delta_count,
            d1=self._build_d1(),
            d2=self._build_d2(resolved, ojama_by_chain),
            d3=self._build_d3(resolver_stats, final_snapshot),
            d4=self._build_d4(resolved, resolver_stats),
            d6=D6NetDivergence(
                divergence_event_count=self._net_divergence_event_count,
                max_divergence=self._max_net_divergence,
            ),
            d7=self._build_d7(resolved, ojama_by_chain),
        )

    def _sum_closed_episode_totals(
        self, closed_episodes: list,
    ) -> dict[str, float]:
        """closed episode の要約を合算し、保存則違反件数も同時に数える
        (`_build_d1` から分離、関数長規約のため2026-08-25追加)。"""
        total_generated = total_canceled = total_landed = total_unreconciled = 0.0
        violation_count = no_settlement_count = 0
        for summary in closed_episodes:
            total_generated += summary.total_generated
            total_canceled += summary.total_canceled
            total_landed += summary.total_landed
            total_unreconciled += summary.unreconciled
            if not summary.has_settlement_input:
                no_settlement_count += 1
                continue
            if summary.status is EpisodeStatus.CLOSED:
                settled = summary.total_canceled + summary.total_landed + summary.unreconciled
                if abs(summary.total_generated - settled) > _FLOAT_EPS:
                    violation_count += 1
        return {
            "total_generated": total_generated, "total_canceled": total_canceled,
            "total_landed": total_landed, "total_unreconciled": total_unreconciled,
            "violation_count": violation_count, "no_settlement_count": no_settlement_count,
        }

    def _build_d1(self) -> D1EpisodeTotals:
        """D1: episode 単位の累計と保存則違反件数。

        settlement input (CANCEL/LAND) が 1 件も無い episode は、この
        tracker の入力からは構造的に相殺・着弾を作れないため保存則の
        検査対象から除外し、件数だけ `episodes_without_settlement_input`
        に出す (クラス docstring 参照。Gate 3-2b で配線が入るまでの措置)。
        CLOSED_FORCED は仕様 I7 により保存則検査そのものの対象外。

        `oversettled_total` / `oversettled_chain_count` は上記の除外とは
        **独立に**全 episode を対象にする。超過決済は
        `保存則違反件数が 0 の状態でも」起こり得る (クラス docstring 参照)
        ため、violation の判定条件に混ぜてはいけない。
        """
        closed_episodes = self._ledger.closed_episodes()
        totals = self._sum_closed_episode_totals(closed_episodes)
        ledger_snapshot = self._ledger.snapshot()
        return D1EpisodeTotals(
            total_generated=totals["total_generated"],
            total_canceled=totals["total_canceled"],
            total_landed=totals["total_landed"],
            total_unreconciled=totals["total_unreconciled"],
            conservation_violation_count=totals["violation_count"],
            episodes_without_settlement_input=totals["no_settlement_count"],
            oversettled_total=sum(s.oversettled for s in closed_episodes),
            oversettled_chain_count=sum(s.oversettled_chain_count for s in closed_episodes),
            unattributed_settlement_total=self._unattributed_settlement_total,
            # 【実装1、2026-08-25 追加】測定器の是正: 「まだ数えていない」を
            # 検出するための2枠 (クラス docstring 参照)。
            open_episode_outstanding=self._ledger.open_episode_outstanding(),
            ledger_residual_all=self._ledger.total_outstanding_all_chains(),
            # 【実装2、2026-08-25 追加】退役で丸ごと消えていた分の転記。
            retired_unreconciled=ledger_snapshot.retired_unreconciled,
            retired_canceled=ledger_snapshot.retired_canceled,
            retired_landed=ledger_snapshot.retired_landed,
            retired_generated=ledger_snapshot.retired_generated,
            post_close_settlement_dropped_count=(
                ledger_snapshot.post_close_settlement_dropped_count
            ),
            post_close_settlement_dropped_amount=(
                ledger_snapshot.post_close_settlement_dropped_amount
            ),
        )

    def _build_d2(
        self, resolved: list[ResolvedChain], ojama_by_chain: dict[int, _ChainOjama],
    ) -> D2FinalizeDivergence:
        """D2: chain_id ごとの finalize 乖離 (確定 - 暫定、単位はおじゃま個数)。

        台帳が既定で拒否する `simulate_fallback` 出所の乖離を分布
        (`divergences`) に混ぜない (2026-08-25 是正)。件数・量だけ別枠で出す。
        """
        accepted: list[float] = []
        rejected: list[float] = []
        for rc in resolved:
            fin = ojama_by_chain[rc.chain_id].finalized_ojama
            if not (rc.was_finalized and fin is not None):
                continue
            divergence = float(fin - ojama_by_chain[rc.chain_id].provisional_ojama)
            if rc.finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF:
                accepted.append(divergence)
            else:
                rejected.append(divergence)
        gate_held = sum(1 for d in accepted if d < -FINALIZE_DOWNWARD_TOLERANCE)
        return D2FinalizeDivergence(
            divergences=tuple(accepted), gate_held_count=gate_held,
            rejected_divergence_count=len(rejected),
            rejected_divergence_amount_total=sum(abs(d) for d in rejected),
        )

    def _build_d3(
        self, resolver_stats: ResolverStats, snapshot: LedgerSnapshot,
    ) -> D3ForcedCloseCounters:
        """D3: 強制終了カウンタ 3 種。素点検算は None (未実装)。自己検算は Fix【5】で廃止。"""
        return D3ForcedCloseCounters(
            episode_forced_close_count=snapshot.forced_close_count,
            chain_id_force_cut_count=resolver_stats.force_cut_count,
            score_multiple_of_ten_violation_count=None,
            land_split_count=self._land_split_count,
        )

    def _build_d4(
        self, resolved: list[ResolvedChain], resolver_stats: ResolverStats,
    ) -> D4CloseReasonCounts:
        """D4: CloseReason 別件数と迷子の CHAIN_END_SIGNAL 件数。"""
        counts: dict[str, int] = {}
        for rc in resolved:
            key = rc.close_reason.name
            counts[key] = counts.get(key, 0) + 1
        return D4CloseReasonCounts(
            counts_by_reason=counts,
            orphan_end_signal_count=resolver_stats.orphan_end_signal_count,
        )

    def _build_d7(
        self, resolved: list[ResolvedChain], ojama_by_chain: dict[int, _ChainOjama],
    ) -> D7GenerationStats:
        """D7: chain_id 数、段数の分布、growth_observed の割合。

        **【実装3、2026-08-25 追加】** `raw_generation_total` は自己相殺
        netting **前**の生成量 (`_chain_to_events` が FIRE/FINALIZE へ
        差し引く前の値) の総和。独立検算 (スコア合計 ÷ レート) と
        比較すべきは「台帳へ登録された生成量」ではなくこちらである
        (クラス docstring 参照)。
        """
        step_counts = tuple(rc.step_count for rc in resolved)
        growth_ratio = (
            sum(1 for rc in resolved if rc.growth_observed) / len(resolved)
            if resolved else 0.0
        )
        # P1-1: simulate 由来 chain は raw 総和 (score OCR 由来の値の総和、
        # docstring 参照) からも除外する (`_is_simulate_source_excluded` と
        # `_build_events` の除外条件は同一)。除外の件数・量は
        # `simulate_excluded_*` に別枠で出す (母数 = chain_id_count)。
        raw_total = sum(
            ojama_by_chain[rc.chain_id].finalized_ojama
            if rc.was_finalized and ojama_by_chain[rc.chain_id].finalized_ojama is not None
            else ojama_by_chain[rc.chain_id].provisional_ojama
            for rc in resolved
            if not self._is_simulate_source_excluded(rc)
        )
        return D7GenerationStats(
            chain_id_count=len(resolved),
            step_counts=step_counts,
            growth_observed_ratio=growth_ratio,
            raw_generation_total=float(raw_total),
            self_canceled_total=self._self_canceled_total,
            self_cancel_clipped_count=self._self_cancel_clipped_count,
            self_cancel_clipped_amount=self._self_cancel_clipped_amount,
            self_cancel_eligible_count=self._self_cancel_eligible_count,
            simulate_excluded_chain_count=self._simulate_excluded_chain_count,
            simulate_excluded_amount=self._simulate_excluded_amount,
        )


__all__ = [
    "ChainEventObservation",
    "D1EpisodeTotals",
    "D2FinalizeDivergence",
    "D3ForcedCloseCounters",
    "D4CloseReasonCounts",
    "D6NetDivergence",
    "D7GenerationStats",
    "ExchangeEpisodeDiagnostics",
    "ExchangeEpisodeTracker",
    "GenerationObservation",
    "PendingDeltaClassification",
    "PendingUncappedFrame",
    "SettlementObservation",
    "classify_pending_uncapped_delta",
]
