"""scripts/_analyze_false_event_verdict_2026-08-24.py の単体テスト (Q-02 対応)。

Codex 品質精査 Q-02: 「偽イベント率の合否判定が実装・実行されていない」を
閉じるための後段分類器の回帰テスト。実ログではなく合成データで、
TP/FP/FN/判定不能/重複/合否判定の各ロジックを個別に固定する。

ハイフン入りファイル名のためモジュールとして直接ロードする
(tests/test_ab_stage_compare_2026-08-18.py と同じ方式)。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mod():
    path = Path(__file__).resolve().parent.parent / "scripts" / (
        "_analyze_false_event_verdict_2026-08-24.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_analyze_false_event_verdict_for_test", path,
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_analyze_false_event_verdict_for_test"] = m
    spec.loader.exec_module(m)
    return m


# ===========================================================================
# 1. classify_trigger: TP / FP / INDETERMINATE の 3 分類
# ===========================================================================


def test_classify_tp_when_score_increases_in_window(mod) -> None:
    """窓内に +MIN_CHAIN_SCORE 以上の score 上昇があれば TP。"""
    trig = mod.Trigger(side="1P", t_seen=10.0, trigger_sec=10.0, mechanism="formula_read")
    scores = [
        mod.ScoreSample(t=9.0, side="1P", value=1000),
        mod.ScoreSample(t=11.0, side="1P", value=1000 + mod.MIN_CHAIN_SCORE),
    ]
    result = mod.classify_trigger(trig, scores)
    assert result.label == "TP"


def test_classify_fp_when_score_read_but_not_increased(mod) -> None:
    """窓内に score の print はあるが閾値未満なら FP (判定不能に混ぜない)。"""
    trig = mod.Trigger(side="1P", t_seen=10.0, trigger_sec=10.0, mechanism="baseline")
    scores = [
        mod.ScoreSample(t=9.0, side="1P", value=1000),
        mod.ScoreSample(t=11.0, side="1P", value=1010),  # +10 のみ、閾値未満
    ]
    result = mod.classify_trigger(trig, scores)
    assert result.label == "FP"


def test_classify_indeterminate_when_no_score_sample_in_window(mod) -> None:
    """窓内に score の print が一切無ければ判定不能 (FP と混同しない)。

    score OCR の last_score は forward-fill されるため、print が無いのは
    「不変」と「欠測」を区別できない。Codex 明示要求: これを FP に混ぜない。
    """
    trig = mod.Trigger(side="1P", t_seen=10.0, trigger_sec=10.0, mechanism="baseline")
    scores = [mod.ScoreSample(t=9.0, side="1P", value=1000)]  # 窓の前にのみ存在
    result = mod.classify_trigger(trig, scores)
    assert result.label == "INDETERMINATE"
    assert result.reason == "no_score_sample_in_window"


def test_classify_indeterminate_when_no_prior_baseline(mod) -> None:
    """trigger より前に score が一度も読めていなければ判定不能。"""
    trig = mod.Trigger(side="1P", t_seen=1.0, trigger_sec=1.0, mechanism="landing")
    scores = [mod.ScoreSample(t=2.0, side="1P", value=100)]  # trigger より後
    result = mod.classify_trigger(trig, scores)
    assert result.label == "INDETERMINATE"
    assert result.reason == "no_prior_score_baseline"


def test_classify_ignores_other_side_scores(mod) -> None:
    """相手 side の score 変化で TP 判定してはいけない (side 混同防止)。"""
    trig = mod.Trigger(side="1P", t_seen=10.0, trigger_sec=10.0, mechanism="baseline")
    scores = [
        mod.ScoreSample(t=9.0, side="1P", value=1000),
        mod.ScoreSample(t=11.0, side="2P", value=99999),  # 相手側は無関係
    ]
    result = mod.classify_trigger(trig, scores)
    assert result.label == "INDETERMINATE"


# ===========================================================================
# 2. find_false_negatives: 多段連鎖の後半をカバーする (回帰テスト)
# ===========================================================================


def test_fn_detected_when_score_jump_has_no_trigger(mod) -> None:
    """trigger が無いまま score が跳ねたら FN。"""
    scores = [
        mod.ScoreSample(t=0.0, side="1P", value=0),
        mod.ScoreSample(t=1.0, side="1P", value=mod.MIN_CHAIN_SCORE),
    ]
    fns = mod.find_false_negatives([], scores)
    assert len(fns) == 1
    assert fns[0]["side"] == "1P"


def test_fn_not_raised_when_long_chain_covers_late_score_jump(mod) -> None:
    """多段連鎖で t_seen から SUPPORT_WINDOW_SEC を超えた後半の得点上昇は、
    trigger がまだ活動中 (t_last_active) なら FN にしない (回帰: 実データで
    多段連鎖の後半を誤って FN 扱いしていたバグの固定)。
    """
    long_gap = mod.SUPPORT_WINDOW_SEC + 5.0
    trig = mod.Trigger(
        side="1P", t_seen=0.0, trigger_sec=0.0, mechanism="formula_read",
        t_last_active=long_gap,
    )
    scores = [
        mod.ScoreSample(t=0.0, side="1P", value=0),
        mod.ScoreSample(t=long_gap, side="1P", value=mod.MIN_CHAIN_SCORE),
    ]
    fns = mod.find_false_negatives([trig], scores)
    assert fns == []


def test_fn_raised_when_jump_is_after_trigger_fully_ended(mod) -> None:
    """trigger 終了 (t_last_active) + SUPPORT_WINDOW_SEC を過ぎた得点上昇は
    別イベントとみなし、FN として検出する。
    """
    trig = mod.Trigger(
        side="1P", t_seen=0.0, trigger_sec=0.0, mechanism="formula_read",
        t_last_active=1.0,
    )
    far_t = 1.0 + mod.SUPPORT_WINDOW_SEC + 10.0
    scores = [
        mod.ScoreSample(t=0.0, side="1P", value=0),
        mod.ScoreSample(t=far_t, side="1P", value=mod.MIN_CHAIN_SCORE),
    ]
    fns = mod.find_false_negatives([trig], scores)
    assert len(fns) == 1


# ===========================================================================
# 3. find_duplicates: 近接 trigger を候補として検出
# ===========================================================================


def test_duplicate_detected_for_close_same_side_triggers(mod) -> None:
    trig_a = mod.Trigger(side="1P", t_seen=0.0, trigger_sec=0.0, mechanism="formula_read")
    trig_b = mod.Trigger(side="1P", t_seen=1.0, trigger_sec=1.0, mechanism="formula_read")
    dups = mod.find_duplicates([trig_a, trig_b])
    assert dups == [trig_b]


def test_no_duplicate_for_well_separated_triggers(mod) -> None:
    gap = mod.DUPLICATE_TRIGGER_GAP_SEC + 1.0
    trig_a = mod.Trigger(side="1P", t_seen=0.0, trigger_sec=0.0, mechanism="formula_read")
    trig_b = mod.Trigger(side="1P", t_seen=gap, trigger_sec=gap, mechanism="formula_read")
    dups = mod.find_duplicates([trig_a, trig_b])
    assert dups == []


def test_duplicate_check_is_per_side(mod) -> None:
    """side が違えば近接していても重複扱いしない。"""
    trig_a = mod.Trigger(side="1P", t_seen=0.0, trigger_sec=0.0, mechanism="formula_read")
    trig_b = mod.Trigger(side="2P", t_seen=0.5, trigger_sec=0.5, mechanism="formula_read")
    dups = mod.find_duplicates([trig_a, trig_b])
    assert dups == []


# ===========================================================================
# 4. parse_probe_log: 実ログ形式の最小サンプルを解析する
# ===========================================================================


def test_parse_probe_log_extracts_new_trigger_once(mod, tmp_path: Path) -> None:
    """同一 trigger_sec の連続行は 1 件の trigger にまとめ、
    None を挟んだ再出現・trigger_sec の直接変化はそれぞれ新規カウントする。
    """
    log_text = "\n".join([
        "[score] t=10.000 1P 100",
        "[ev] t=10.100 ev1 (10.1, 'formula_read', 1, 40)",
        "[ev] t=10.200 ev1 (10.1, 'formula_read', 2, 360)",  # 継続 (新規でない)
        "[ev] t=10.300 ev1 None",
        "[ev] t=12.000 ev1 (12.0, 'baseline', 1, 100)",  # None を挟んだ新規
        "[score] t=12.500 1P 200",
    ])
    log_path = tmp_path / "probe_sample.log"
    log_path.write_text(log_text, encoding="utf-8")

    triggers, scores = mod.parse_probe_log(log_path)

    assert [tr.trigger_sec for tr in triggers] == [10.1, 12.0]
    assert triggers[0].t_last_active == 10.2  # 継続行で更新されている
    assert triggers[0].chain_count == 2
    assert [s.value for s in scores] == [100, 200]


def test_parse_probe_log_flags_legacy_mechanism(mod, tmp_path: Path) -> None:
    """旧機構名 'formula' もそのまま mechanism として抽出できる
    (w2 ON 側の混入検出に使う経路の健全性確認)。
    """
    log_path = tmp_path / "probe_legacy.log"
    log_path.write_text(
        "[ev] t=1.000 ev2 (1.0, 'formula', 1, 0)\n", encoding="utf-8",
    )
    triggers, _ = mod.parse_probe_log(log_path)
    assert len(triggers) == 1
    assert triggers[0].mechanism == mod.LEGACY_MECHANISM_NAME


# ===========================================================================
# 5. compute_verdict: ACCEPTED / REJECTED の合否判定
# ===========================================================================


def _base_window_result(mod, window: str, mode: str, **overrides) -> dict:
    base = {
        "window": window, "mode": mode, "source": "probe_log",
        "n_triggers": 10, "tp": 8, "fp": 2, "indeterminate": 0, "fn": 0,
        "duplicates": 0, "duplicates_empty": 0, "duplicates_with_score": 0,
        "chain_clusters": 5, "legacy_mechanism_count": 0,
        "classified": [], "false_negatives": [],
    }
    base.update(overrides)
    return base


def test_verdict_accepted_when_no_regression(mod) -> None:
    results = [
        _base_window_result(mod, "w1", "off", fp=5, duplicates=3),
        _base_window_result(mod, "w1", "on", fp=4, duplicates=2),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "ACCEPTED"
    assert verdict["reasons"] == []


def test_verdict_rejected_when_fp_increases(mod) -> None:
    results = [
        _base_window_result(mod, "w1", "off", fp=2),
        _base_window_result(mod, "w1", "on", fp=5),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "REJECTED"
    assert any("FP 増加" in reason for reason in verdict["reasons"])


def test_verdict_rejected_when_empty_event_duplicates_increase(mod) -> None:
    """**実体のない (score=0) 再トリガー**が増えたら REJECTED。

    【2026-08-24 再訂正】判定軸を 2 度直している。経緯:

    1. 当初: 近接再トリガーの**素の件数**で比較 → ON は段単位で検出するので
       改善するほど数字が悪化した (構造的に逆向き)。
    2. 次に: **率**で比較 → ON は trigger も物理連鎖も統合して減らすので、
       分子・分母の両方が改善で動き、どの正規化でも率が上がった。
    3. 現在: **「偽イベント」と「二重計上」を分離**。
       Codex の要件は「偽イベントを増やしていないこと」なので、
       実体のない `score=0` の再トリガーだけを判定に使う。

    実測でこの分離が効くことを確認済み (c0BQoMJwwQU の w1/w2):
    ON 側で増えた再トリガーは `formula_read cc=2 score=2340` の直後に
    `baseline cc=2 score=2340` のように**同じ連鎖を 2 機構が同じ値で報告**
    しているもので、偽イベントではなく機構間の一致だった。
    一方 `score=0` の再トリガーは w1 で 8→3、w2 で 19→7 と 63% 減っている。
    """
    results = [
        _base_window_result(mod, "w1", "off", fp=1, duplicates_empty=1),
        _base_window_result(mod, "w1", "on", fp=1, duplicates_empty=9),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "REJECTED"
    assert any("偽イベント増加" in reason for reason in verdict["reasons"])


def test_verdict_tolerates_real_valued_duplicates(mod) -> None:
    """実値を持つ再トリガー (機構間の一致) が増えても落とさない。

    これは「偽イベント」ではなく「同じ実イベントを 2 回数える」問題であり、
    `chain_id` による統合 (交換エピソード会計) で解く別の課題。
    判定には使わず `duplicates_with_score` として報告する。
    """
    results = [
        _base_window_result(mod, "w1", "off", fp=2,
                            duplicates_empty=5, duplicates_with_score=1),
        _base_window_result(mod, "w1", "on", fp=1,
                            duplicates_empty=2, duplicates_with_score=15),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "ACCEPTED", verdict["reasons"]


def test_is_empty_event_uses_total_score(mod) -> None:
    """score=0 のイベントだけを『実体なし』と判定する。"""
    T = mod.Trigger
    zero = T(side="1P", t_seen=1.0, trigger_sec=1.0, mechanism="formula",
             chain_count=1, total_score=0, t_last_active=1.0)
    real = T(side="1P", t_seen=1.0, trigger_sec=1.0, mechanism="formula_read",
             chain_count=1, total_score=40, t_last_active=1.0)
    assert mod.is_empty_event(zero) is True
    assert mod.is_empty_event(real) is False


def test_verdict_rejected_when_chain_clusters_increase(mod) -> None:
    """物理連鎖クラスタ数が増えたら REJECTED (多重計上の直接指標)。

    同じ物理連鎖を多重計上すれば必ずクラスタ数の増加として現れる。
    段単位検出の増加はクラスタ内に吸収されるので、この指標は
    「段を細かく取れるようになったこと」に影響されない。
    """
    results = [
        _base_window_result(mod, "w1", "off", n_triggers=100, chain_clusters=30),
        _base_window_result(mod, "w1", "on", n_triggers=100, chain_clusters=45),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "REJECTED"
    assert any("クラスタ増加" in reason for reason in verdict["reasons"])


def test_step_progress_pairs_are_not_counted_as_duplicates(mod) -> None:
    """段が進んでいる近接ペアは重複に数えない。

    user 伝授 (memory reference_chain_formula_per_step_2026-08-22):
    「掛け算式は消えるたびに出る。回数を数えれば連鎖数、値を足せば火力」
    → 連鎖数か素点が単調増加している近接ペアは「段の進行」であって、
    同じ物理連鎖の二重計上ではない。
    """
    T = mod.Trigger
    # 同一 side、段周期 1.4 秒間隔で連鎖数と素点が増えていく = 段の進行
    trigs = [
        T(side="1P", t_seen=10.0, trigger_sec=10.0, mechanism="formula_read",
          chain_count=1, total_score=40, t_last_active=10.5),
        T(side="1P", t_seen=11.4, trigger_sec=11.4, mechanism="formula_read",
          chain_count=2, total_score=540, t_last_active=11.9),
        T(side="1P", t_seen=12.8, trigger_sec=12.8, mechanism="formula_read",
          chain_count=3, total_score=1180, t_last_active=13.3),
    ]
    assert mod.find_duplicates(trigs) == []
    # 一方、物理連鎖としては 1 本にまとまる
    assert mod.count_chain_clusters(trigs) == 1


def test_flicker_pairs_are_counted_as_duplicates(mod) -> None:
    """連鎖数も素点も同じままの再トリガーは重複に数える (再計上の疑い)。"""
    T = mod.Trigger
    trigs = [
        T(side="2P", t_seen=20.0, trigger_sec=20.0, mechanism="formula",
          chain_count=1, total_score=0, t_last_active=20.1),
        T(side="2P", t_seen=20.1, trigger_sec=20.1, mechanism="formula",
          chain_count=1, total_score=0, t_last_active=20.2),
    ]
    dups = mod.find_duplicates(trigs)
    assert len(dups) == 1 and dups[0].t_seen == 20.1


def test_verdict_tolerates_low_rate_legacy_fallback(mod) -> None:
    """ON 側に旧機構名が少数出ても、それだけでは REJECTED にしない。

    【2026-08-24 訂正】当初「1 件でも混入したら無条件 REJECTED」としていたが
    誤りだった。`src/recognition_pipeline.py:6178-6205` のとおり

        read_fire = (self._enable_chain_formula_read_verify
                     and read_res is not None
                     and bool(getattr(read_res, "valid", False)))
        if read_fire:  mechanism = CHAIN_MECHANISM_FORMULA_READ
        else:          mechanism = CHAIN_MECHANISM_FORMULA

    であり、フラグ ON でも「そのフレームで掛け算式が読めなかった」場合は
    旧経路へ落ちる。これは**設計どおりの fail-safe** であって
    ゲーティングの失敗ではない。件数ではなく率で判定する。
    """
    results = [
        _base_window_result(mod, "w1", "off", fp=5),
        _base_window_result(
            mod, "w1", "on", fp=1, n_triggers=100, legacy_mechanism_count=5,
        ),  # フォールバック率 5% << 上限 40%
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "ACCEPTED", verdict["reasons"]


def test_verdict_rejected_when_legacy_fallback_rate_too_high(mod) -> None:
    """実読フォールバック率が上限を超えたら REJECTED (フラグが効いていない)。"""
    results = [
        _base_window_result(mod, "w1", "off", fp=5),
        _base_window_result(
            mod, "w1", "on", fp=1, n_triggers=10, legacy_mechanism_count=8,
        ),  # フォールバック率 80% > 上限 40%
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "REJECTED"
    assert any("フォールバック率" in reason for reason in verdict["reasons"])


def test_legacy_fallback_rate_is_not_checked_for_off_mode(mod) -> None:
    """OFF 側は旧機構で動くのが正しいので、フォールバック率を判定しない。"""
    results = [
        _base_window_result(
            mod, "w1", "off", fp=5, n_triggers=10, legacy_mechanism_count=10,
        ),
        _base_window_result(mod, "w1", "on", fp=1, n_triggers=100),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "ACCEPTED", verdict["reasons"]


def test_verdict_accepted_with_coarse_c62_metric(mod) -> None:
    """c62 (FP 分離不可) は fp_or_indeterminate_combined の上限値で比較する。"""
    results = [
        _base_window_result(
            mod, "c62", "off", source="ab_json_coarse", fp=None,
            indeterminate=None, fp_or_indeterminate_combined=5,
        ),
        _base_window_result(
            mod, "c62", "on", source="ab_json_coarse", fp=None,
            indeterminate=None, fp_or_indeterminate_combined=5,
        ),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "ACCEPTED"


def test_verdict_rejected_when_coarse_metric_increases(mod) -> None:
    results = [
        _base_window_result(
            mod, "c62", "off", source="ab_json_coarse", fp=None,
            indeterminate=None, fp_or_indeterminate_combined=5,
        ),
        _base_window_result(
            mod, "c62", "on", source="ab_json_coarse", fp=None,
            indeterminate=None, fp_or_indeterminate_combined=9,
        ),
    ]
    verdict = mod.compute_verdict(results)
    assert verdict["verdict"] == "REJECTED"
