"""連鎖数の外部正解 (真値) モジュール (C-1a, docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md)。

## 背景

連鎖数は OCR 単体・盤面シミュレーション単体のいずれも信頼不可と実測済み
(MEMORY.md `project_chain_count_both_untrustworthy_2026-07-30`: 真値 8 連鎖を
simulate() が 1 連鎖と過小評価する壊滅例)。時間軸指標ファミリー
(C-1b: E[残り連鎖|N到達] 等の統計テーブル) には、これより信頼できる
連鎖数の外部正解が必要。

## 設計方針 (fail-safe: 数値だけで採否を決めない)

既存の2系統を**独立に**評価してから突き合わせる:
    ① テロップ読み: `src.chain_count_ocr.ChainCountOcr.read_max_in_window()` を
       `delta_score` 省略 (=既存の連続列方式のみ) で呼ぶ。孤立誤検出を排除する
       ロジックは実データ (video_c54) で検証済みのものをそのまま流用する。
    ② 得点逆算: 観測 `delta_score` から、連鎖数の**全域** (CHAIN_COUNT_MIN..MAX)
       を候補として `scoring.chain_power` ベースの下限近似期待得点と比較し、
       最も整合する連鎖数を選ぶ。**テロップの検出値には一切依存しない**
       (テロップが割れた場合の候補に引きずられないよう、独立系統として保つ)。

**2系統が完全一致した場合のみ**「真値」として採用する。不一致・いずれかが
不明 (None) の場合は fail-safe で `chain_count=None` (不明) を返す。
これは、`chain_count_ocr.py` の docstring に実データで記録されている
「テロップの連続列方式が video_c54 2P 実9連鎖イベントで 3 に過小評価した」
事例のように、単一系統だけでは静かに間違える (fail-silent) リスクがある
ため — 測定器自体の健全性を数値の見かけより優先する
(feedback_viz_eval_required.md / 測定器事故6件の教訓)。

## 得点逆算候補を「全域」にする理由 (独立性の担保)

`chain_count_ocr._select_chain_count_by_score()` はテロップ検出値の集合を
候補として受け取る設計 (得点裏取り方式、2026-07-29 追加) だが、本モジュールは
その候補集合を「テロップ検出値」ではなく `CHAIN_COUNT_MIN..CHAIN_COUNT_MAX`
の**全域**にして呼び出す。これにより系統②はテロップの読み取り結果に一切
依存しない独立推定になり、系統①と系統②の一致は本当の意味でのクロス
チェックになる (テロップが割れているときに割れた値を候補に含めてしまうと、
「両系統が同じ誤りに合意する」false agreement のリスクが生じるため)。

下限近似・許容比率 [0.5, 2.0] の妥当性は `chain_count_ocr.py` で実データ
(video_c54 game_idx=1,9) を用いて検証済みのため、重複実装によるバグ混入を
避けるためそのロジックをそのまま re-export して再利用する (private 関数だが
同一パッケージ内の内部実装として扱う。tests/test_chain_count_ocr.py 自体が
同様の private 関数を直接テストしている既存の慣行と揃える)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.chain_count_ocr import (
    CHAIN_COUNT_MAX,
    CHAIN_COUNT_MIN,
    CHAIN_WINDOW_SAMPLE_INTERVAL_SEC,
    ChainCountOcr,
    ChainCountWindowResult,
    Side,
    _select_chain_count_by_score,
)

if TYPE_CHECKING:
    import cv2

# 得点逆算 (系統②) の候補範囲。テロップ検出値に依存しない独立系統にするため
# 連鎖数の全域を候補とする (上記モジュール docstring 「独立性の担保」参照)。
FULL_CHAIN_COUNT_CANDIDATES: frozenset[int] = frozenset(
    range(CHAIN_COUNT_MIN, CHAIN_COUNT_MAX + 1)
)


@dataclass(frozen=True)
class ChainCountTruthResult:
    """2系統 (テロップ読み / 得点逆算) の突合結果。

    Attributes:
        chain_count: 高信頼の連鎖数真値。2系統が完全一致した場合のみ非 None。
            不一致・いずれかの系統が不明な場合は None (fail-safe)。
        telop_chain_count: テロップ読み (系統①、連続列方式) 単独の結果。
        telop_n_hits: テロップ読みの生検出サンプル数 (デバッグ用)。
        score_chain_count: 得点逆算 (系統②、テロップ非依存) 単独の結果。
        score_ratio: 系統②で採用した候補の score_consistency_ratio
            (デバッグ用、系統②が不明なら None)。
        reason: 判定理由。
            "agree"          = 2系統一致 (真値採用)
            "disagree"       = 2系統とも値が出たが不一致 (fail-safe で不明)
            "telop_missing"  = テロップ側が不明
            "score_missing"  = 得点逆算側が不明 (許容比率を満たす候補なし)
            "both_missing"   = 両系統とも不明
    """

    chain_count: int | None
    telop_chain_count: int | None
    telop_n_hits: int
    score_chain_count: int | None
    score_ratio: float | None
    reason: str


def resolve_chain_count_truth(
    telop_window: ChainCountWindowResult,
    delta_score: int,
) -> ChainCountTruthResult:
    """テロップ読み結果 (系統①) と得点逆算 (系統②) を突き合わせて真値を出す。

    Args:
        telop_window: `ChainCountOcr.read_max_in_window(..., delta_score=None)`
            の結果 (=既存の連続列方式のみ、得点は未使用のテロップ単独読み)。
        delta_score: 連鎖イベントの実測得点差分 (score OCR の差分値)。

    Returns:
        ChainCountTruthResult (stateless、入力のみに依存する純粋関数)。
    """
    telop_n = telop_window.max_chain_count
    score_n, score_ratio = _select_chain_count_by_score(
        FULL_CHAIN_COUNT_CANDIDATES, delta_score,
    )
    if telop_n is None and score_n is None:
        reason = "both_missing"
    elif telop_n is None:
        reason = "telop_missing"
    elif score_n is None:
        reason = "score_missing"
    elif telop_n == score_n:
        reason = "agree"
    else:
        reason = "disagree"
    truth = telop_n if reason == "agree" else None
    return ChainCountTruthResult(
        chain_count=truth,
        telop_chain_count=telop_n,
        telop_n_hits=telop_window.n_hits,
        score_chain_count=score_n,
        score_ratio=score_ratio,
        reason=reason,
    )


def read_chain_count_truth(
    ocr: ChainCountOcr,
    cap: "cv2.VideoCapture",
    side: Side,
    t_start: float,
    t_end: float,
    delta_score: int,
    sample_interval_sec: float = CHAIN_WINDOW_SAMPLE_INTERVAL_SEC,
) -> ChainCountTruthResult:
    """video window から直接、連鎖数の真値を読み取る便利関数 (video I/O込み)。

    内部で系統①(テロップ、delta_score 不使用) を読み取ってから
    `resolve_chain_count_truth` に渡すだけの薄いラッパー。video I/O 自体は
    `ChainCountOcr.read_max_in_window` に委譲するため、本関数自体は状態を
    持たない (stateless)。

    Args:
        ocr: テンプレ読込済み ChainCountOcr インスタンス。
        cap: cv2.VideoCapture (呼び出し側でオープン済み)。
        side: "1P" or "2P"。
        t_start, t_end: 連鎖イベントの window (秒)。
        delta_score: 実測の得点差分 (score OCR)。
        sample_interval_sec: テロップ読み取りのサンプリング間隔。

    Returns:
        ChainCountTruthResult。
    """
    telop_window = ocr.read_max_in_window(
        cap, side, t_start, t_end, sample_interval_sec=sample_interval_sec,
    )
    return resolve_chain_count_truth(telop_window, delta_score)


__all__ = [
    "FULL_CHAIN_COUNT_CANDIDATES",
    "ChainCountTruthResult",
    "resolve_chain_count_truth",
    "read_chain_count_truth",
]
