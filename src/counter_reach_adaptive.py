"""時間予算内で K を伸ばし、統計的な保証を付けた打ち合い応手判定 (2026-08-09).

user 指示:
> 性能に余裕がある限り K を増やし、 **モンテカルロ法で優位な数が保証できる形**で
> 実装してください

## 設計の要点

### 0. K=1,2 は実際のツモを使う (色の全列挙をしない)
user 指摘 (2026-08-09):
> そもそも K1 と 2 はネクストとダブルネクストを使えばよく、 置き方だけで良い

**次と次々のツモは画面に表示されている** (NextDetector は認識精度 100%)。
したがって K=1,2 で色を 4×4=16 通り列挙するのは無駄であり、 **実際のツモ色で
置き方 22 通りだけ**調べればよい。

| | 色を全列挙 | 実ツモを使う | 削減 |
|---|---|---|---|
| K=1 | 16ツモ × 22配置 = 352 | **1 × 22 = 22** | 94% |
| K=2 | 256 × 22 = 5,632 | **22 × 22 = 484** | 91% |

推測でなく実測の色を使うので **精度も上がる**。 K>=3 は未知なので従来通り
色をサンプリングする。

### 1. K は固定しない — 時間予算内で伸ばせるだけ伸ばす
既存の `counter_reach_probability_fast` は K を引数で受け取るが、
**呼び出し側が K を決め打ちする**設計だった。 本モジュールは逆に
**時間予算 (秒) を受け取り、 予算が尽きるまで K を 1 つずつ深くする**。

user 確定: 有利不利の更新は **1 秒 2 回で十分** なので予算は 500ms 級。
30fps (33ms) に縛られる必要はない。

### 2. モンテカルロの精度を保証する
K>=3 はモンテカルロ近似になるため、 サンプル数が足りないと確率が信用できない。
本モジュールは **二項比率の正規近似**で信頼区間を出し、

    半幅 = z * sqrt(p*(1-p)/n)

が目標精度 (既定 ±5%) に収まるまでサンプルを追加する。 収まらないまま予算が
尽きた場合は **その K を採用しない** (浅くても信頼できる方を優先する)。

p が 0 / 1 に張り付く場合は正規近似が使えないので、 Wilson 区間を使う。

### 3. 結果に「どこまで保証できたか」を必ず含める
`achieved_k` (到達した手数) と `half_width` (信頼区間の半幅)、
`exact` (全列挙で誤差ゼロか) を返す。 **黙って浅い結果を返さない**。

## 既存資産との関係
- 探索と得点計算は `src.indicators_v2` の既存関数をそのまま使う (再実装しない)
- K=1,2 は全列挙なので誤差ゼロ (信頼区間は 0)
- K>=3 は既存の MC 経路を、 サンプル数を制御しながら繰り返し呼ぶ
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.board import Board

# 目標精度 (信頼区間の半幅)。 ±5% で「返せる/返せない」の判定に足りる。
DEFAULT_TARGET_HALF_WIDTH: float = 0.05
# 信頼水準 95% の z 値。
Z_95: float = 1.96
# 1 回の追加で足すサンプル数 (小刻みにして予算超過を防ぐ)。
SAMPLE_CHUNK: int = 24
# **最低サンプル数** (2026-08-09 修正)。
# Wilson 区間は p が 0 や 1 に張り付くと幅が急に狭くなるため、 サンプルが
# 少なくても「精度十分」と誤判定してしまう (実測: 73 件で ±5% 達成と判定)。
# 統計的に意味のある結論を出すための下限を別途設ける。
# 30 は「二項比率の正規近似が使える目安 (np>=5, n(1-p)>=5)」を p=1/6 程度でも
# 満たす水準として置く。 これはヒューリスティックであり、 精度保証の本体は
# Wilson 区間の方である。
MIN_SAMPLES: int = 30
# 全列挙で誤差ゼロになる K (それ以上は MC 近似)。
EXACT_K_MAX: int = 2
# K の上限 (これ以上は探索空間が爆発し、 実戦の手数も超える)。
K_HARD_MAX: int = 8


@dataclass(frozen=True)
class AdaptiveCounterResult:
    """時間予算内で到達した応手判定の結果。

    Attributes:
        probability: 閾値以上を返せる確率 (achieved_k での値)。
        achieved_k: 実際に到達した手数。 **予算内で保証できた深さ**。
        requested_budget_sec: 与えられた時間予算。
        elapsed_sec: 実際に使った時間。
        half_width: 確率の信頼区間の半幅 (95%)。 全列挙なら 0.0。
        exact: 全列挙で求めたか (True なら誤差ゼロ)。
        n_samples: MC のサンプル数 (全列挙なら列挙件数)。
        truncated_by_budget: 予算切れでこれ以上深くできなかったか。
        truncated_by_precision: 精度が目標に届かず K を採用しなかったか。
    """

    probability: float
    achieved_k: int
    requested_budget_sec: float
    elapsed_sec: float
    half_width: float
    exact: bool
    n_samples: int
    truncated_by_budget: bool
    truncated_by_precision: bool


def wilson_half_width(p: float, n: int, z: float = Z_95) -> float:
    """Wilson 区間の半幅を返す (p が 0/1 に張り付いても破綻しない)。

    正規近似 (p±z*sqrt(p(1-p)/n)) は p=0 や p=1 で幅 0 になってしまい、
    「サンプルが足りないのに精度十分」と誤判定する。 Wilson 区間はこの場合も
    有限の幅を持つため、 サンプル数の判定に使える。
    """
    if n <= 0:
        return 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    # 半幅は中心からの距離。 上下非対称なので広い方を返す (保守的)。
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return max(hi - p, p - lo)


def required_samples(target_half_width: float, z: float = Z_95) -> int:
    """目標半幅を満たすのに必要な最悪ケースのサンプル数を返す。

    p=0.5 のとき分散が最大 (0.25) なので、 n = (z/target)^2 * 0.25。
    例: target=0.05, z=1.96 → n ≈ 384。
    """
    if target_half_width <= 0:
        raise ValueError("target_half_width must be > 0")
    return int(math.ceil((z / target_half_width) ** 2 * 0.25))


def _reach_with_known_pairs(
    board: Board,
    threshold_ojama: float,
    pairs: "tuple[tuple[int, int], ...]",
    elapsed_sec: float,
) -> "tuple[float, int]":
    """実際のツモ列で置き方を全探索し、 (到達確率, 評価件数) を返す。

    ツモの色が確定しているので確率的な要素は無い。 到達確率は
    **0.0 か 1.0 のどちらか** (「その置き方が存在するか」の判定になる)。
    これは「相手が最善を打てば返せるか」という意味であり、 応手判定として
    正しい (相手の技量は最善と仮定する)。
    """
    import src.indicators_v2 as iv
    from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score

    frontier: "list[Board]" = [board]
    n_eval = 0
    for pair in pairs:
        nxt: "list[Board]" = []
        for b in frontier:
            for placed in iv._enumerate_placement_boards(b, pair):
                nxt.append(placed)
                n_eval += 1
        if not nxt:
            return 0.0, n_eval
        frontier = nxt
    # 最終盤面のどれかが閾値に届けば「返せる」
    sim = iv._SHARED_SIMULATOR if hasattr(iv, "_SHARED_SIMULATOR") else None
    if sim is None:
        # フォールバック (通常到達しない、iv._SHARED_SIMULATOR が既に ON 済)。
        # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
        from src.chain import ChainSimulator
        from src.production_config import GHOST_CHAIN_RULE_ENABLED
        sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for b in frontier:
        res = sim.simulate(b)
        if res.chain_count < 1:
            continue
        ojama = calculate_chain_score(res).total_score / OJAMA_RATE_STANDARD
        if ojama >= threshold_ojama:
            return 1.0, n_eval
    return 0.0, n_eval


def estimate_with_budget(
    board: Board,
    threshold_ojama: float,
    budget_sec: float,
    target_half_width: float = DEFAULT_TARGET_HALF_WIDTH,
    elapsed_sec: float = 0.0,
    active_colors: "tuple[int, ...] | None" = None,
    k_hard_max: int = K_HARD_MAX,
    known_pairs: "tuple[tuple[int, int], ...]" = (),
) -> AdaptiveCounterResult:
    """時間予算内で K を伸ばしつつ、精度を保証して応手確率を返す。

    Args:
        board: 応手側 (相手) の確定盤面。
        threshold_ojama: 「返せた」とみなす閾値 (お邪魔換算個数)。
        budget_sec: 使ってよい時間 [秒]。 これを超えたら打ち切る。
        target_half_width: 目標とする信頼区間の半幅 (既定 ±5%)。
        elapsed_sec: 試合の経過秒 (マージンタイム補正用)。
        active_colors: 試合で使われている色 (省略時は盤面から推定)。
        k_hard_max: K の上限。
        known_pairs: **画面に出ている実際のツモ** (次、 次々の順)。
            K=1,2 はこれを使い、 色の全列挙をしない (user指摘 2026-08-09)。
            渡さない場合は従来通り色を全列挙する (後方互換)。

    Returns:
        AdaptiveCounterResult。 **どこまで保証できたかを必ず含む**。
    """
    # 遅延 import (循環参照回避 + 起動コスト削減)
    import src.indicators_v2 as iv

    t_start = time.perf_counter()
    best_p = 0.0
    best_k = 0
    best_hw = 1.0
    best_exact = False
    best_n = 0
    by_budget = False
    by_precision = False

    last_k_cost = 0.0  # 直前の K にかかった時間 [秒]
    for k in range(1, k_hard_max + 1):
        now = time.perf_counter() - t_start
        # 次の K は前の K より重くなるので、 同じ時間すら残っていなければ
        # 進まない (予算超過の抑制、 2026-08-09 修正)。
        if now >= budget_sec or (last_k_cost > 0 and now + last_k_cost > budget_sec):
            by_budget = True
            break
        t_k = time.perf_counter()
        if k <= EXACT_K_MAX:
            # K=1,2: 実ツモが分かっていればそれを使い、 置き方だけ調べる。
            # 分かっていなければ従来通り色を全列挙する。
            if len(known_pairs) >= k:
                p, n = _reach_with_known_pairs(
                    board, threshold_ojama, known_pairs[:k], elapsed_sec,
                )
            else:
                res = iv.counter_reach_probability_fast(
                    board, threshold_ojama, elapsed_sec=elapsed_sec,
                    k_levels=(k,), active_colors=active_colors,
                )
                p = float(res.probabilities.get(k, 0.0))
                n = int(res.n_evaluated.get(k, 0))
            # どちらも全列挙なので誤差ゼロ
            best_p, best_k, best_hw, best_exact, best_n = p, k, 0.0, True, n
            last_k_cost = time.perf_counter() - t_k
            continue
        # K>=3: MC。 目標精度に届くまでサンプルを足す。
        need = required_samples(target_half_width)
        n_done = 0
        hits = 0.0
        reached = False
        chunk_cost = 0.0  # 直前チャンクの所要時間 [秒]
        while n_done < need:
            now = time.perf_counter() - t_start
            # **次のチャンクが予算内に収まるか**を事前に見る (2026-08-09 修正)。
            # 「実行してから超過に気づく」設計だと必ず予算をまたぐ。
            if now >= budget_sec or (chunk_cost > 0 and now + chunk_cost > budget_sec):
                by_budget = True
                break
            chunk = min(SAMPLE_CHUNK, need - n_done)
            t_chunk = time.perf_counter()
            res = iv.counter_reach_probability_fast(
                board, threshold_ojama, elapsed_sec=elapsed_sec,
                k_levels=(k,), active_colors=active_colors,
                mc_n_samples=chunk, rng_seed=(k * 1000 + n_done),
            )
            chunk_cost = time.perf_counter() - t_chunk
            p_chunk = float(res.probabilities.get(k, 0.0))
            n_chunk = max(1, int(res.n_evaluated.get(k, chunk)))
            hits += p_chunk * n_chunk
            n_done += n_chunk
            p_hat = hits / n_done if n_done else 0.0
            # 最低サンプル数を満たさないうちは「精度十分」と判定しない
            # (p が 0/1 に張り付くと Wilson 幅が狭く出るため)。
            if (n_done >= MIN_SAMPLES
                    and wilson_half_width(p_hat, n_done) <= target_half_width):
                reached = True
                break
        p_hat = hits / n_done if n_done else 0.0
        hw = wilson_half_width(p_hat, n_done)
        if reached or (n_done >= MIN_SAMPLES and hw <= target_half_width):
            # 精度を満たしたのでこの K を採用する
            best_p, best_k, best_hw, best_exact, best_n = (
                p_hat, k, hw, False, n_done,
            )
        else:
            # 精度が足りない K は採用しない (浅くても信頼できる方を残す)
            by_precision = True
            break
        last_k_cost = time.perf_counter() - t_k
    return AdaptiveCounterResult(
        probability=best_p,
        achieved_k=best_k,
        requested_budget_sec=budget_sec,
        elapsed_sec=time.perf_counter() - t_start,
        half_width=best_hw,
        exact=best_exact,
        n_samples=best_n,
        truncated_by_budget=by_budget,
        truncated_by_precision=by_precision,
    )
