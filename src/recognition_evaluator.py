"""強化アナリスト: 物理推論ベース自動評価 (2026-05-19、 cycle 33+).

ユーザー指示「評価者強化 = プロジェクト成功」 に基づく、 viz 目視を超える
自動評価フレームワーク。 既存 confirmed_board 履歴と物理推論 (= 連鎖判定 /
配置整合性) を組み合わせて fail-silent モードを検知する。

入力: visualize_recognition.py の --dump-board-log で生成された JSONL
出力: 評価違反 (Violation) のリスト + サマリレポート

評価メトリクス:
    1. puyo_count_consistency: 直近 STABLE 間で puyo 数が常識的範囲か
    2. chain_no_disappear: 4 連結成立しているのに次 STABLE で消えていない
    3. sudden_drop: 1 STABLE 期間で puyo 数が大幅減少 (= puyo→empty fail-silent)
    4. retrospective_chain_validation: 連鎖発生時に過去盤面が連鎖成立必要条件を
       満たしていたか
    5. auto_correction_signal: 同位置 cell の色が STABLE 間で変化 (= 連鎖/落下を除く)
    6. floating_puyo: 浮いている puyo (= 物理法則違反)
    7. background_color_distribution: 全 cell の色分布、 特定色が異常に多い
    8. stable_short_burst: STABLE 期間が異常に短い連続 (= 認識崩壊)

各 cycle 評価:
    evaluator = RecognitionEvaluator()
    evaluator.load_jsonl(Path("logs/cycle_32d_board_log.jsonl"))
    report = evaluator.generate_report()
    # report = {"violations": [...], "summary": {...}, "verdict": "REJECT" | "ACCEPT" | "REVIEW"}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT


# ============================
# 定数定義 (= マジックナンバー禁止)
# ============================

# Violation severity
SEVERITY_INFO: str = "info"
SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"

# puyo_count_consistency 閾値
# 1 STABLE 期間で puyo 数が変動する想定範囲。 1 ツモ = 2 puyo 増加が標準。
# おじゃま落下で +X、 連鎖消滅で -Y、 全消しで 0 化。 STABLE 間で +0~+72 増、
# 連鎖直後は -大量減も正常 (= ただし通常は CHAIN 状態を挟む)。
# cycle 40 (2026-05-20、 ユーザー指摘 受け強化): -5 → -2 に感度向上。
# 直前 STABLE → 現 STABLE で連鎖を挟まないため、 -2 以上の減少は puyo→empty
# 誤認の signal。 -3 や -4 の中規模 drop も catch する。
PUYO_COUNT_DROP_THRESHOLD_NORMAL: int = 2  # 通常 STABLE 間で -2 超は異常
PUYO_COUNT_INCREASE_THRESHOLD_NORMAL: int = 30  # +30 超は異常 (ojama 落下相当)

# chain_no_disappear: 4 連結が N 連続 STABLE で残存したら違反
CHAIN_NO_DISAPPEAR_PERSISTENCE_FRAMES: int = 30  # ~0.5 秒 @ 60fps

# sudden_drop: 1 STABLE 期間で puyo 数が大幅減
# cycle 40 (= 強化): -10 → -5 で中規模消失も catch
SUDDEN_DROP_THRESHOLD: int = 5  # -5 超で警告

# stable_short_burst: STABLE 期間が短すぎる連続
STABLE_MIN_FRAMES: int = 3  # 3 frame 未満の STABLE が 5 回連続なら認識崩壊
STABLE_BURST_COUNT_THRESHOLD: int = 5

# cycle 36+ (2026-05-20): 散発的誤認 catch (= sparse_color_pop)
# 「EMPTY → 色 → EMPTY」 が短期間で起きる cell を flag。
# 正常事象 (= ツモ着地、 連鎖消滅) では puyo は持続的に存在 (= 数十 frame)、
# 散発的誤認は 1-5 STABLE frame で消える特徴がある。
SPARSE_POP_MAX_FRAMES: int = 10  # 出現後 10 STABLE frame 以内に消えれば散発的

# cycle 47 (2026-05-20): chain 誤判定二重 catch metrics
# M1 chain_state_too_short: 実連鎖は最低 ~30 frame (=0.5 秒) 持続する。
#   state=chain が短すぎる連続フレーム未満で stable に戻るケース = false positive 候補。
CHAIN_STATE_MIN_FRAMES: int = 5
# M2 chain_no_puyo_loss: 実連鎖は MIN_ERASE_COUNT (= 4) 以上の puyo 消去を伴う。
#   chain 状態前後 STABLE で puyo 数が 4 個以上減っていなければ false positive。
CHAIN_MIN_PUYO_LOSS: int = 4

# background_color_distribution: 特定色の比率が異常
# cycle 36+ (2026-05-20) tuning: 35% → 25%
# ユーザー目視の「散発的 1-3 cell 青誤認」 は 35% threshold に達せず取りこぼし。
# 25% threshold で中規模誤認も catch する。 ojama は除外済 (false positive 防止)。
BG_COLOR_DOMINANT_THRESHOLD: float = 0.25

# cycle 33+ (2026-05-20): 動画全体累積メトリクス追加。 frame 単位の 35% threshold では
# 中規模誤認 (10-30%) を取りこぼすため、 動画全体での特定色累積比率を別途 catch。
BG_COLOR_CUMULATIVE_THRESHOLD: float = 0.20  # STABLE 全 frame 累積で特定色 20% 超は異常
BG_COLOR_CUMULATIVE_MIN_FRAMES: int = 30  # 累積メトリクスを発火するには最低 30 STABLE frame

# floating_puyo: 浮き puyo 検出は STABLE 中のみ (= TSUMO_FALL/CHAIN 中は浮きが正常)

# cycle 56 KB (= 2026-05-22): ojama 認識退行 catch 用 metric.
# cycle 56_v2 で「ojama 0 件 seed で fine-tune → ojama 認識能力消去 (-99.6%)」 が
# 評価ツールで catch できず、 critical -26.5% 改善 (= 5 色のみ) で「採用」 判定が
# 出た事故 (= 2026-05-22 朝)。 ユーザー目視で初めて発覚。
# 評価ツールで ojama 退行を自動検知するため 2 metric 追加。
#
# M1 ojama_disappearance: STABLE 間で ojama → EMPTY 遷移 (= 連鎖外、 自然消失でない)。
#   既存 check_auto_correction は new=EMPTY を許容 (= 連鎖消滅) しているため
#   ojama → EMPTY 退行を取りこぼす。 本 metric は ojama 限定で catch。
OJAMA_DISAPPEARANCE_PER_FRAME_THRESHOLD: int = 3  # 1 STABLE 間で ojama 3 個以上消失 = CRITICAL
#
# M2 ojama_global_scarcity: 全 STABLE frame で ojama 認識率が極小 (= 一切認識しない).
#   baseline 比較不要、 絶対的な「ojama 認識率 < 0.5%」 を WARNING。
#   (= 通常試合の ojama 平均率 3-10%、 0.5% 未満は ojama 認識能力喪失の signal)
OJAMA_GLOBAL_SCARCITY_THRESHOLD: float = 0.005  # 全 cell 中 ojama 0.5% 未満
OJAMA_GLOBAL_SCARCITY_MIN_FRAMES: int = 100  # 最低 100 STABLE frame で発火

# KC (= 2026-05-22): 静止中の色ブレ catch (= 5 色相互誤認).
# テスター案 A 採用。 ぷよが動いていない STABLE→STABLE 間で同位置 cell の
# 色が変化した回数を動画全体で集計し、 ペア別 (= 赤↔紫、 青↔緑 等) も
# 数える。 ユーザー目視「ぷよ誤認ちょっとある」 を数値で見える化する。
# 既存 check_auto_correction は frame ごとに violation を発行するが、
# 「動画全体での総数」 と「どのペアで誤認しているか」 が見えづらいため
# サマリ metric として追加。
STATIC_COLOR_FLICKER_WARNING_THRESHOLD: int = 50  # 動画全体で 50 件以上 = WARNING
STATIC_COLOR_FLICKER_CRITICAL_THRESHOLD: int = 200  # 動画全体で 200 件以上 = CRITICAL
STATIC_COLOR_FLICKER_MIN_FRAMES: int = 30  # 最低 30 STABLE frame で発火

# THREE_WAY_SUDDEN_DROP (2026-06-03): 3者一致ぷよの突然消失検知。
# raw_cnn == raw_hsv == confirmed かつ非 EMPTY・非 UNKNOWN の cell (= 3者合意ぷよ) が
# STABLE 間で大幅に減少しているのに chain / ojama_fall が介在しないケースは
# 「全員同じ誤り値」の fail-silent 盲点。 既存 sudden_drop は confirmed のみ対象で
# この全員誤りパターンを catch できない。
#
# 条件:
#   1. 両 STABLE frame で raw_cnn / raw_hsv / confirmed の 3 者が利用可能
#   2. prev_stable → cur_stable の間に chain も ojama_fall も介在しない
#   3. tsumo_fall が介在する場合も除外 (= physics_fix 大量書換えを誤検知防止)
#   4. 3者一致ぷよ数の差分が -THREE_WAY_DROP_THRESHOLD 以下
THREE_WAY_DROP_THRESHOLD: int = 8  # -8 個以上の減少で CRITICAL

# C1 (Phase 1 = 2026-05-28): avg_puyo_count_per_stable_frame baseline 比閾値.
# STABLE 確定盤面の平均ぷよ数 (1P+2P 合算) が baseline 比 85% 未満なら
# 「ぷよを消す経路の fail-silent」 (= puyo→empty 大量誤認) を検知して REJECT。
# cycle 56_v2 で confirmed puyo 数が -30% 落ちていたのに mismatch 改善で見過ごした
# パターンを構造的に catch する。
AVG_PUYO_COUNT_CRITICAL_RATIO: float = 0.85  # baseline 比 85% 未満で CRITICAL

# C3 (Phase 1 = 2026-05-28): 複合 verdict の critical 悪化許容幅。
# baseline 比 +10% 以内の critical 増加は NEEDS_REVIEW、 +10% 超は AUTO_REJECT。
JUDGE_CYCLE_CRITICAL_REJECT_RATIO: float = 1.10
# AUTO_ACCEPT_PROVISIONAL とするための critical 許容上限 (= baseline + 2 件以内)
JUDGE_CYCLE_CRITICAL_ACCEPT_DELTA: int = 2
# p_to_e_count が baseline 比 +20% かつ > 0 なら AUTO_REJECT
JUDGE_CYCLE_P_TO_E_REJECT_RATIO: float = 1.20


# ============================
# データクラス
# ============================

@dataclass
class FrameEntry:
    """1 frame の認識結果 (= JSONL 1 行)."""
    frame_idx: int
    t_sec: float
    p1_state: str
    p2_state: str
    p1_confirmed: list[list[int]] | None
    p2_confirmed: list[list[int]] | None
    # T4 PuyoErasureMonitor: 当該 frame での新規 alert [(row, col), ...]
    # backwards compat: 古い board_log は本フィールドを持たないため default []。
    p1_erasure_alerts: list[list[int]] = field(default_factory=list)
    p2_erasure_alerts: list[list[int]] = field(default_factory=list)
    # 3 者一致 DROP 検知用 raw 盤面 (= Phase I 以降の board_log に含まれる)。
    # backwards compat: 古い board_log はこれらフィールドを持たないため default None。
    p1_raw_cnn_board: list[list[int]] | None = None
    p2_raw_cnn_board: list[list[int]] | None = None
    p1_raw_hsv_board: list[list[int]] | None = None
    p2_raw_hsv_board: list[list[int]] | None = None

    @classmethod
    def from_jsonable(cls, obj: dict[str, Any]) -> "FrameEntry":
        return cls(
            frame_idx=int(obj["frame_idx"]),
            t_sec=float(obj["t_sec"]),
            p1_state=str(obj["p1_state"]),
            p2_state=str(obj["p2_state"]),
            p1_confirmed=obj.get("p1_confirmed"),
            p2_confirmed=obj.get("p2_confirmed"),
            # backwards compat: キーなし (古い board_log) は空リストで処理継続。
            p1_erasure_alerts=obj.get("p1_erasure_alerts", []),
            p2_erasure_alerts=obj.get("p2_erasure_alerts", []),
            # 3 者一致 DROP 検知用 raw 盤面 (= 古い board_log では None 維持)。
            p1_raw_cnn_board=obj.get("p1_raw_cnn_board"),
            p2_raw_cnn_board=obj.get("p2_raw_cnn_board"),
            p1_raw_hsv_board=obj.get("p1_raw_hsv_board"),
            p2_raw_hsv_board=obj.get("p2_raw_hsv_board"),
        )


@dataclass
class Violation:
    """評価違反 (= 認識誤認の signal)."""
    frame_idx: int
    t_sec: float
    side: str  # "1P" or "2P"
    metric: str
    severity: str  # info / warning / critical
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_sec": round(self.t_sec, 2),
            "side": self.side,
            "metric": self.metric,
            "severity": self.severity,
            "detail": self.detail,
            "extra": self.extra,
        }


# ============================
# 強化アナリスト本体
# ============================

class RecognitionEvaluator:
    """物理推論ベース自動評価。 stateless で履歴は受け取る."""

    def __init__(self) -> None:
        self.entries: list[FrameEntry] = []
        self._chain_sim = ChainSimulator()

    # ========================
    # 入力読み込み
    # ========================

    def load_jsonl(self, path: Path) -> None:
        """JSONL ファイルから FrameEntry リストを構築."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self.entries.append(FrameEntry.from_jsonable(obj))
                except Exception:
                    continue

    # ========================
    # ユーティリティ
    # ========================

    @staticmethod
    def _to_board(grid_list: list[list[int]] | None) -> Board | None:
        """JSON grid → Board オブジェクト変換."""
        if grid_list is None:
            return None
        board = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                board.set(r, c, int(grid_list[r][c]))
        return board

    def _side_iter(self, side: str):
        """1P or 2P の (frame_idx, t_sec, state, board) を yield."""
        # side="1P" → "p1", "2P" → "p2" (FrameEntry フィールド名と整合)
        prefix = "p1" if side == "1P" else "p2"
        attr_state = f"{prefix}_state"
        attr_board = f"{prefix}_confirmed"
        for e in self.entries:
            state = getattr(e, attr_state)
            grid = getattr(e, attr_board)
            board = self._to_board(grid)
            yield e.frame_idx, e.t_sec, state, board

    # ========================
    # 評価メトリクス
    # ========================

    def check_puyo_count_consistency(self, side: str) -> list[Violation]:
        """STABLE 間 puyo 数の変動を check.

        通常: +0~+2 (= ツモ着地)。 ojama_fall 後は +N (= おじゃま大量)。
        連鎖後は -M (= 消滅)。 STABLE → 次 STABLE の差分が常識的か。
        """
        violations: list[Violation] = []
        prev_count: int | None = None
        prev_state: str = "menu"
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                prev_state = state
                continue
            count = board.count_puyos()
            if prev_count is not None and prev_state == "stable":
                # 直前 STABLE → 現 STABLE 間で間に CHAIN/OJAMA を挟まずに大幅変動
                diff = count - prev_count
                if diff < -PUYO_COUNT_DROP_THRESHOLD_NORMAL:
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="puyo_count_drop",
                        severity=SEVERITY_CRITICAL,
                        detail=f"STABLE → STABLE で puyo 数 {prev_count}→{count} ({diff})",
                        extra={"diff": diff, "prev": prev_count, "cur": count},
                    ))
                elif diff > PUYO_COUNT_INCREASE_THRESHOLD_NORMAL:
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="puyo_count_surge",
                        severity=SEVERITY_WARNING,
                        detail=f"STABLE → STABLE で puyo 数 {prev_count}→{count} (+{diff})",
                        extra={"diff": diff, "prev": prev_count, "cur": count},
                    ))
            prev_count = count
            prev_state = state
        return violations

    def check_chain_no_disappear(self, side: str) -> list[Violation]:
        """4 連結成立しているのに N frame 以上消えないなら誤認."""
        violations: list[Violation] = []
        # (color, frozenset(positions)) → 持続 frame 数 + 最初の frame
        tracked: dict[tuple, dict[str, Any]] = {}
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            erasable = self._chain_sim.find_erasable_groups(board)
            current_keys: set[tuple] = set()
            for g in erasable:
                positions = frozenset(g.cells)
                key = (g.color, positions)
                current_keys.add(key)
                if key not in tracked:
                    tracked[key] = {"start_frame": fi, "start_t": t, "count": 1}
                else:
                    tracked[key]["count"] += 1
            # 消えたグループを除外
            for key in list(tracked.keys()):
                if key not in current_keys:
                    info = tracked.pop(key)
                    if info["count"] >= CHAIN_NO_DISAPPEAR_PERSISTENCE_FRAMES // 10:
                        # 短時間で消えたが、 一度違反として記録 (= info severity)
                        pass
            # 持続 frame 数が threshold 超えたら違反
            for key, info in tracked.items():
                if info["count"] == CHAIN_NO_DISAPPEAR_PERSISTENCE_FRAMES:
                    color, positions = key
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="chain_no_disappear",
                        severity=SEVERITY_CRITICAL,
                        detail=f"色 {color} の {len(positions)} 連結が "
                               f"{info['count']} STABLE frame 残存 "
                               f"(= 色誤認の証拠)",
                        extra={"color": color, "size": len(positions),
                               "positions": list(positions)},
                    ))
        return violations

    def check_sudden_drop(self, side: str) -> list[Violation]:
        """連続 STABLE 間で puyo 数が突然大幅減 (= puyo→empty fail-silent)."""
        # check_puyo_count_consistency に内包済だが、 ここでは異なる threshold で
        # 「警告」 レベルを別途 catch
        violations: list[Violation] = []
        history: list[tuple[int, float, int]] = []  # (frame_idx, t_sec, count)
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            count = board.count_puyos()
            history.append((fi, t, count))
        for i in range(1, len(history)):
            prev_fi, prev_t, prev_c = history[i-1]
            cur_fi, cur_t, cur_c = history[i]
            diff = cur_c - prev_c
            # 直近 STABLE 同士で大幅減 (= ChainSimulator が介在していないのに)
            if diff < -SUDDEN_DROP_THRESHOLD:
                violations.append(Violation(
                    frame_idx=cur_fi, t_sec=cur_t, side=side,
                    metric="sudden_drop",
                    severity=SEVERITY_CRITICAL,
                    detail=f"STABLE 間で puyo {diff} 個消失 "
                           f"(= chain_no_disappear と相関する fail-silent signal)",
                    extra={"prev_frame": prev_fi, "diff": diff},
                ))
        return violations

    def check_retrospective_chain(self, side: str) -> list[Violation]:
        """連鎖発生 frame の盤面が「連鎖成立可能」 だったか.

        STABLE → CHAIN 遷移時に、 STABLE 末尾の盤面で 4 連結が必須。
        4 連結が無いのに CHAIN に遷移していたら、 STABLE 時の認識に誤りあり。
        """
        violations: list[Violation] = []
        prev_stable_board: Board | None = None
        prev_stable_frame: int = -1
        prev_stable_t: float = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state == "stable" and board is not None:
                prev_stable_board = board
                prev_stable_frame = fi
                prev_stable_t = t
            elif state == "chain" and prev_stable_board is not None:
                # 直前 STABLE 盤面で 4 連結 erasable があるか
                erasable = self._chain_sim.find_erasable_groups(prev_stable_board)
                if not erasable:
                    violations.append(Violation(
                        frame_idx=prev_stable_frame, t_sec=prev_stable_t,
                        side=side,
                        metric="retrospective_chain_missing",
                        severity=SEVERITY_CRITICAL,
                        detail=f"STABLE → CHAIN 遷移時 (frame={fi}) に "
                               f"前 STABLE 盤面で 4 連結なし (= 色認識誤りの証拠)",
                        extra={"chain_frame": fi},
                    ))
                # 一度 check したら次の STABLE まで skip
                prev_stable_board = None
        return violations

    def check_auto_correction(self, side: str) -> list[Violation]:
        """同位置 cell の色が STABLE 間で変化した = 過去 STABLE の誤認可能性.

        ただし: ツモ着地 (= EMPTY → 色) と連鎖消滅 (= 色 → EMPTY) は除外。
        色 → 別の色 への変化は誤認 signal。
        """
        violations: list[Violation] = []
        prev_board: Board | None = None
        prev_state: str = "menu"
        prev_fi: int = -1
        prev_t: float = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                prev_state = state
                continue
            if prev_board is not None and prev_state == "stable":
                # STABLE → STABLE で色変化を check
                changes: list[tuple[int, int, int, int]] = []
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        old = int(prev_board.get(r, c))
                        new = int(board.get(r, c))
                        if old == new:
                            continue
                        # EMPTY → 色 (= ツモ着地相当) は許容
                        if old == COLOR_EMPTY:
                            continue
                        # 色 → EMPTY (= 連鎖消滅相当) は許容 (= sudden_drop で別途 catch)
                        if new == COLOR_EMPTY:
                            continue
                        # UNKNOWN 関連は除外
                        if old == COLOR_UNKNOWN or new == COLOR_UNKNOWN:
                            continue
                        changes.append((r, c, old, new))
                if changes:
                    # 色変化が 1 つでもあれば warning、 5+ なら critical
                    severity = SEVERITY_CRITICAL if len(changes) >= 5 else SEVERITY_WARNING
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="auto_correction",
                        severity=severity,
                        detail=f"STABLE 間で同位置 {len(changes)} cell の "
                               f"色変化 (= 過去 STABLE の誤認可能性)",
                        extra={"changes": changes[:10]},
                    ))
            prev_board = board
            prev_state = state
            prev_fi = fi
            prev_t = t
        return violations

    def check_floating_puyo(self, side: str) -> list[Violation]:
        """STABLE 中に浮き puyo (= 物理法則違反) を検出.

        各列で、 EMPTY の上に puyo が乗っているなら浮き = 認識誤り。
        """
        violations: list[Violation] = []
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            for col in range(BOARD_COLS):
                # 下から上に向かって、 一度 EMPTY を見たらそれ以降 puyo はあり得ない
                empty_seen = False
                for row in range(BOARD_ROWS - 1, -1, -1):  # 下→上
                    color = int(board.get(row, col))
                    if color in (COLOR_EMPTY, COLOR_UNKNOWN):
                        empty_seen = True
                    else:
                        if empty_seen:
                            violations.append(Violation(
                                frame_idx=fi, t_sec=t, side=side,
                                metric="floating_puyo",
                                severity=SEVERITY_CRITICAL,
                                detail=f"col={col} row={row} 色 {color} "
                                       f"が浮いている (= 物理違反)",
                                extra={"row": row, "col": col, "color": color},
                            ))
                            break  # この列の他は冗長なので 1 件で打ち切り
        return violations

    def check_background_color_distribution(self, side: str) -> list[Violation]:
        """全 cell 中の特定色比率が異常に高い = 背景誤認の signal.

        cycle 36+ (2026-05-20) 修正: ojama (color=9) を除外。 試合終了直前の
        ojama 大量降下 (= 盤面の 50%) を背景誤認と誤判定していた false positive
        バグの修正。 ojama dominant は正常な試合進行なので、 別 metric にすべき
        だが当面は除外で対応。
        """
        violations: list[Violation] = []
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            grid = board._grid
            non_empty_mask = (grid != COLOR_EMPTY) & (grid != COLOR_UNKNOWN)
            non_empty_count = int(non_empty_mask.sum())
            total = BOARD_ROWS * BOARD_COLS
            if non_empty_count == 0:
                continue
            # 色別比率 (= 非空 cell 中、 ojama 除外で背景誤認のみ catch)
            for color in (1, 2, 3, 4, 5):  # red/blue/green/yellow/purple (ojama 除外)
                color_count = int((grid == color).sum())
                ratio_total = color_count / total  # 全 cell 中の比率
                if ratio_total >= BG_COLOR_DOMINANT_THRESHOLD:
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="bg_color_dominant",
                        severity=SEVERITY_CRITICAL,
                        detail=f"色 {color} が全 cell の {ratio_total:.1%} "
                               f"({color_count}/{total}) = 背景誤認 signal",
                        extra={"color": color, "ratio": ratio_total,
                               "count": color_count},
                    ))
        return violations

    def check_bg_color_cumulative(self, side: str) -> list[Violation]:
        """STABLE 全 frame 累積で特定色の比率が異常に高い = 中規模背景誤認 catch.

        frame 単位の bg_color_dominant (= 35% threshold) では取りこぼす中規模誤認
        (= 10-30% で動画全体に持続) を catch するための累積メトリクス。
        """
        violations: list[Violation] = []
        # 全 STABLE frame の cell 色分布を累積 (= cycle 36+ ojama 除外)
        color_counts: dict[int, int] = {c: 0 for c in (1, 2, 3, 4, 5)}
        total_cells = 0
        stable_frame_count = 0
        last_stable_fi = 0
        last_stable_t = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            stable_frame_count += 1
            last_stable_fi = fi
            last_stable_t = t
            grid = board._grid
            for color in color_counts.keys():
                color_counts[color] += int((grid == color).sum())
            total_cells += BOARD_ROWS * BOARD_COLS
        if stable_frame_count < BG_COLOR_CUMULATIVE_MIN_FRAMES or total_cells == 0:
            return violations
        for color, count in color_counts.items():
            ratio = count / total_cells
            if ratio >= BG_COLOR_CUMULATIVE_THRESHOLD:
                violations.append(Violation(
                    frame_idx=last_stable_fi, t_sec=last_stable_t, side=side,
                    metric="bg_color_cumulative",
                    severity=SEVERITY_CRITICAL,
                    detail=f"色 {color} が STABLE 全 frame 累積 cell の "
                           f"{ratio:.1%} ({count}/{total_cells}) "
                           f"= 動画全体での背景誤認",
                    extra={"color": color, "ratio": ratio,
                           "count": count, "stable_frames": stable_frame_count},
                ))
        return violations

    def check_chain_state_too_short(self, side: str) -> list[Violation]:
        """M1 (cycle 47): state=chain が短すぎる連続 = false positive chain.

        実連鎖は最低 ~30 frame 持続する。 5 frame 未満で stable に戻るのは
        chain への誤遷移の証拠。 retrospective_chain_missing と直交した観点。
        """
        violations: list[Violation] = []
        prefix = "p1" if side == "1P" else "p2"
        attr_state = f"{prefix}_state"
        # state=chain の連続区間を抽出
        chain_runs: list[tuple[int, int, float]] = []  # (start_frame, length, start_t)
        run_start: int | None = None
        run_start_t: float = 0.0
        for e in self.entries:
            if getattr(e, attr_state) == "chain":
                if run_start is None:
                    run_start = e.frame_idx
                    run_start_t = e.t_sec
            else:
                if run_start is not None:
                    length = e.frame_idx - run_start
                    chain_runs.append((run_start, length, run_start_t))
                    run_start = None
        # 末尾でも chain 状態なら追加
        if run_start is not None and self.entries:
            length = self.entries[-1].frame_idx - run_start
            chain_runs.append((run_start, length, run_start_t))
        # 短い run を flag
        for start, length, t in chain_runs:
            if length < CHAIN_STATE_MIN_FRAMES:
                violations.append(Violation(
                    frame_idx=start, t_sec=t, side=side,
                    metric="chain_state_too_short",
                    severity=SEVERITY_CRITICAL,
                    detail=f"state=chain が {length} frame だけ持続 "
                           f"(< {CHAIN_STATE_MIN_FRAMES} = 実連鎖最低未満) "
                           f"= chain 誤判定の signal",
                    extra={"length": length, "min": CHAIN_STATE_MIN_FRAMES},
                ))
        return violations

    def check_chain_no_puyo_loss(self, side: str) -> list[Violation]:
        """M2 (cycle 47): chain 前後 STABLE で puyo 数変化なし = false positive.

        実連鎖は MIN_ERASE_COUNT (= 4) 以上の puyo 消去を伴う。 chain 状態を
        挟んだ前後 STABLE で puyo 数が 4 個以上減っていなければ偽 chain。
        retrospective_chain_missing と直交した観点 (= 連鎖後の結果を見る)。
        """
        violations: list[Violation] = []
        prefix = "p1" if side == "1P" else "p2"
        # state 列 + puyo 数列を構築
        events = []  # (frame_idx, t_sec, state, puyo_count)
        for fi, t, state, board in self._side_iter(side):
            count = board.count_puyos() if board is not None else None
            events.append((fi, t, state, count))
        # state=chain の前後 STABLE puyo 数を確認
        last_stable_before_chain: int | None = None
        last_stable_before_chain_fi: int = 0
        last_stable_before_chain_t: float = 0.0
        in_chain: bool = False
        chain_start_fi: int = 0
        chain_start_t: float = 0.0
        for fi, t, state, count in events:
            if state == "chain":
                if not in_chain:
                    chain_start_fi = fi
                    chain_start_t = t
                    in_chain = True
            elif state == "stable":
                if in_chain and count is not None and last_stable_before_chain is not None:
                    # chain 直後の stable
                    loss = last_stable_before_chain - count
                    if loss < CHAIN_MIN_PUYO_LOSS:
                        violations.append(Violation(
                            frame_idx=chain_start_fi, t_sec=chain_start_t,
                            side=side,
                            metric="chain_no_puyo_loss",
                            severity=SEVERITY_CRITICAL,
                            detail=f"chain 前後 STABLE puyo 数 "
                                   f"{last_stable_before_chain}→{count} "
                                   f"(loss={loss} < {CHAIN_MIN_PUYO_LOSS}) "
                                   f"= 偽 chain signal",
                            extra={"prev": last_stable_before_chain,
                                   "cur": count, "loss": loss},
                        ))
                    in_chain = False
                if count is not None:
                    last_stable_before_chain = count
                    last_stable_before_chain_fi = fi
                    last_stable_before_chain_t = t
        return violations

    def check_sparse_color_pop(self, side: str) -> list[Violation]:
        """散発的誤認検出: EMPTY → 色 → EMPTY が短期間で起きる cell を flag.

        正常な puyo 出現 (= ツモ着地、 ojama 落下) は数十 STABLE frame 持続する。
        N frame 以内 (= SPARSE_POP_MAX_FRAMES=10) に消える cell は **散発的誤認**
        signal = ユーザー目視「青誤認大量」 の核心。
        """
        violations: list[Violation] = []
        # 各 (row, col) ごとに「empty → 色」 の出現履歴を追跡
        cell_state: dict[tuple[int, int], tuple[int, int, int]] = {}
        # value = (color, appeared_frame, appeared_t_sec)
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    cur = int(board.get(r, c))
                    prev = cell_state.get((r, c))
                    if prev is None:
                        # 初出
                        if cur not in (COLOR_EMPTY, COLOR_UNKNOWN):
                            cell_state[(r, c)] = (cur, fi, t)
                        continue
                    prev_color, prev_fi, prev_t = prev
                    if cur == prev_color:
                        # 色維持、 何もしない
                        continue
                    if cur in (COLOR_EMPTY, COLOR_UNKNOWN):
                        # 色 → EMPTY 遷移、 持続 frame 数を check
                        duration = fi - prev_fi
                        if duration <= SPARSE_POP_MAX_FRAMES:
                            violations.append(Violation(
                                frame_idx=fi, t_sec=t, side=side,
                                metric="sparse_color_pop",
                                severity=SEVERITY_CRITICAL,
                                detail=f"row={r} col={c} 色 {prev_color} が "
                                       f"{duration} frame だけ出現して消失 "
                                       f"(= 散発的誤認 signal)",
                                extra={"row": r, "col": c,
                                       "color": prev_color,
                                       "duration": duration,
                                       "appeared_frame": prev_fi},
                            ))
                        cell_state[(r, c)] = (cur, fi, t)
                    else:
                        # 色 → 別の色 (= auto_correction が別途 catch)
                        cell_state[(r, c)] = (cur, fi, t)
        return violations

    def check_stable_short_burst(self, side: str) -> list[Violation]:
        """STABLE 期間が異常に短い連続 = 認識崩壊 signal."""
        violations: list[Violation] = []
        # STABLE → 非 STABLE → STABLE の周期を測る
        prefix = "p1" if side == "1P" else "p2"
        attr_state = f"{prefix}_state"
        stable_runs: list[tuple[int, int]] = []  # (start_frame, length)
        run_start: int | None = None
        for e in self.entries:
            state = getattr(e, attr_state)
            if state == "stable":
                if run_start is None:
                    run_start = e.frame_idx
            else:
                if run_start is not None:
                    length = e.frame_idx - run_start
                    stable_runs.append((run_start, length))
                    run_start = None
        if run_start is not None and self.entries:
            length = self.entries[-1].frame_idx - run_start
            stable_runs.append((run_start, length))
        # 短い STABLE 連続を探す
        short_count = 0
        burst_start: int | None = None
        for start, length in stable_runs:
            if length < STABLE_MIN_FRAMES:
                if burst_start is None:
                    burst_start = start
                short_count += 1
                if short_count >= STABLE_BURST_COUNT_THRESHOLD:
                    violations.append(Violation(
                        frame_idx=start, t_sec=start / 60.0,
                        side=side,
                        metric="stable_short_burst",
                        severity=SEVERITY_WARNING,
                        detail=f"短 STABLE ({STABLE_MIN_FRAMES} frame 未満) が "
                               f"{short_count} 連続 (= 認識崩壊 signal)",
                        extra={"burst_start": burst_start},
                    ))
                    burst_start = None
                    short_count = 0
            else:
                burst_start = None
                short_count = 0
        return violations

    def check_ojama_disappearance(self, side: str) -> list[Violation]:
        """KB (cycle 56): ojama → EMPTY 遷移 (= 連鎖外) を CRITICAL.

        既存 check_auto_correction は new=EMPTY を許容するため、
        「ojama 認識退行 (= ojama → EMPTY)」 を取りこぼす。 本 metric は
        ojama 限定で STABLE 間 ojama 数減少を catch する。
        連鎖発火 (= STABLE → CHAIN 遷移) を介する ojama 消失は許容。
        """
        violations: list[Violation] = []
        prev_board: Board | None = None
        prev_state: str = "menu"
        prev_fi: int = -1
        prev_t: float = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                prev_state = state
                continue
            if prev_board is not None and prev_state == "stable":
                # 直前 STABLE → 現 STABLE で ojama 消失数を集計
                ojama_lost = 0
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        old = int(prev_board.get(r, c))
                        new = int(board.get(r, c))
                        if old == COLOR_OJAMA and new != COLOR_OJAMA:
                            ojama_lost += 1
                if ojama_lost >= OJAMA_DISAPPEARANCE_PER_FRAME_THRESHOLD:
                    violations.append(Violation(
                        frame_idx=fi, t_sec=t, side=side,
                        metric="ojama_disappearance",
                        severity=SEVERITY_CRITICAL,
                        detail=f"STABLE 間で ojama {ojama_lost} 個消失 "
                               f"(= 連鎖外、 ojama 認識退行 signal)",
                        extra={"prev_frame": prev_fi, "lost": ojama_lost},
                    ))
            prev_board = board
            prev_state = state
            prev_fi = fi
            prev_t = t
        return violations

    def check_ojama_global_scarcity(self, side: str) -> list[Violation]:
        """KB (cycle 56): 全 STABLE frame で ojama 認識率が極小なら WARNING.

        「ojama を一切認識しない」 状態 (= cycle 56_v2 の -99.6% 退行) を
        絶対値ベースで catch する。 通常試合の ojama 平均率は 3-10%。
        0.5% 未満は ojama 認識能力喪失の強い signal。
        """
        violations: list[Violation] = []
        total_cells = 0
        ojama_cells = 0
        stable_frames = 0
        first_fi = -1
        first_t = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                continue
            if first_fi == -1:
                first_fi = fi
                first_t = t
            stable_frames += 1
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    total_cells += 1
                    if int(board.get(r, c)) == COLOR_OJAMA:
                        ojama_cells += 1
        if stable_frames < OJAMA_GLOBAL_SCARCITY_MIN_FRAMES:
            return violations
        if total_cells == 0:
            return violations
        ratio = ojama_cells / total_cells
        if ratio < OJAMA_GLOBAL_SCARCITY_THRESHOLD:
            violations.append(Violation(
                frame_idx=first_fi, t_sec=first_t, side=side,
                metric="ojama_global_scarcity",
                severity=SEVERITY_CRITICAL,
                detail=f"全 STABLE 中 ojama 認識率 {ratio*100:.3f}% "
                       f"(< {OJAMA_GLOBAL_SCARCITY_THRESHOLD*100:.1f}%) "
                       f"= ojama 認識能力喪失の signal "
                       f"(stable_frames={stable_frames}, ojama_cells={ojama_cells})",
                extra={"ratio": ratio, "ojama_cells": ojama_cells,
                       "total_cells": total_cells, "stable_frames": stable_frames},
            ))
        return violations

    def check_static_color_flicker(self, side: str) -> list[Violation]:
        """KC (cycle 56 G): 静止中の色ブレ集計 (= 5 色相互誤認の見える化).

        STABLE → STABLE 間で同位置 cell の色 → 別色 変化を動画全体で集計。
        EMPTY ↔ 色、 ojama 関連、 UNKNOWN は除外 (= 5 色相互誤認のみ catch)。
        ペア別カウント (= "1-2" = 赤→青、 "5-1" = 紫→赤 等) を extra に出す。

        既存 check_auto_correction は frame ごとに violation を発行するため
        動画全体での総数 + ペア別が見えづらかった。 本 metric は動画 1 本
        あたり 1 件の Violation で集計値を出す (= 動画 cycle 採否判定用)。
        """
        violations: list[Violation] = []
        prev_board: Board | None = None
        prev_state: str = "menu"
        total_flips: int = 0
        pair_counts: dict[str, int] = {}
        stable_frames: int = 0
        first_fi: int = -1
        first_t: float = 0.0
        for fi, t, state, board in self._side_iter(side):
            if state != "stable" or board is None:
                prev_state = state
                continue
            if first_fi == -1:
                first_fi = fi
                first_t = t
            stable_frames += 1
            if prev_board is not None and prev_state == "stable":
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        old = int(prev_board.get(r, c))
                        new = int(board.get(r, c))
                        if old == new:
                            continue
                        # 5 色相互のみ catch (= 1-5、 ojama=9 と EMPTY=0 と UNKNOWN 除外)
                        if old not in (1, 2, 3, 4, 5) or new not in (1, 2, 3, 4, 5):
                            continue
                        total_flips += 1
                        pair_key = f"{old}-{new}"
                        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            prev_board = board
            prev_state = state
        if stable_frames < STATIC_COLOR_FLICKER_MIN_FRAMES:
            return violations
        if total_flips == 0:
            return violations
        # severity 判定
        if total_flips >= STATIC_COLOR_FLICKER_CRITICAL_THRESHOLD:
            severity = SEVERITY_CRITICAL
        elif total_flips >= STATIC_COLOR_FLICKER_WARNING_THRESHOLD:
            severity = SEVERITY_WARNING
        else:
            severity = SEVERITY_INFO
        # 上位ペア top 5 を detail に
        top_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:5]
        pair_str = ", ".join(f"{k}:{v}" for k, v in top_pairs)
        violations.append(Violation(
            frame_idx=first_fi, t_sec=first_t, side=side,
            metric="static_color_flicker",
            severity=severity,
            detail=f"STABLE 中 色 flip 計 {total_flips} 件 "
                   f"(stable_frames={stable_frames}). 上位: {pair_str}",
            extra={
                "total_flips": total_flips,
                "stable_frames": stable_frames,
                "pair_counts": pair_counts,
            },
        ))
        return violations

    def _count_three_way_agree(
        self,
        cnn: list[list[int]] | None,
        hsv: list[list[int]] | None,
        conf: list[list[int]] | None,
    ) -> int | None:
        """3 者 (raw_cnn / raw_hsv / confirmed) が一致する非 EMPTY・非 UNKNOWN cell 数を返す。

        いずれかが None の場合は None を返す (= raw 情報なし = 評価不能)。
        """
        if cnn is None or hsv is None or conf is None:
            return None
        count = 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                a = cnn[r][c]
                b = hsv[r][c]
                x = conf[r][c]
                if a == b == x and a not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    count += 1
        return count

    @staticmethod
    def _make_three_way_drop_violation(
        entry_frame_idx: int,
        entry_t_sec: float,
        side: str,
        prev_n: int,
        cur_n: int,
        prev_frame: int,
    ) -> Violation:
        """check_three_way_sudden_drop の Violation オブジェクトを生成するヘルパー."""
        diff = cur_n - prev_n
        return Violation(
            frame_idx=entry_frame_idx, t_sec=entry_t_sec, side=side,
            metric="three_way_sudden_drop",
            severity=SEVERITY_CRITICAL,
            detail=(
                f"3 者一致ぷよ数 {prev_n}→{cur_n} ({diff:+d}) "
                f"連鎖・お邪魔・ツモ落下介在なし "
                f"(= fail-silent 盲点誤認 signal)"
            ),
            extra={
                "prev_3way_n": prev_n,
                "cur_3way_n": cur_n,
                "diff": diff,
                "prev_frame": prev_frame,
                "chain_intervened": False,
            },
        )

    def check_three_way_sudden_drop(self, side: str) -> list[Violation]:
        """3 者一致ぷよの突然消失検知 (= fail-silent 盲点炙り出し).

        raw_cnn == raw_hsv == confirmed が全員同じ誤り値の場合、 既存 sudden_drop は
        confirmed のみ対象のため検知できない。 本 metric は 3 者一致ぷよ数を追跡し、
        連鎖・お邪魔・ツモ落下が介在しない STABLE 間で大幅減少した場合を CRITICAL。

        除外条件 (= 誤検知防止):
            1. chain / ojama_fall が介在する → 正当な消去なので除外
            2. tsumo_fall が介在する → physics_fix 大量書換えの可能性があるので除外
            3. raw_cnn / raw_hsv / confirmed のいずれかが None → 評価不能なので skip
        """
        violations: list[Violation] = []
        prefix = "p1" if side == "1P" else "p2"
        attr_cnn = f"{prefix}_raw_cnn_board"
        attr_hsv = f"{prefix}_raw_hsv_board"
        attr_conf = f"{prefix}_confirmed"
        attr_state = f"{prefix}_state"
        prev_stable_frame: int | None = None
        prev_stable_n: int | None = None
        intervene_states: set[str] = set()

        for e in self.entries:
            state = getattr(e, attr_state)
            if state != "stable":
                intervene_states.add(state)
                continue
            # STABLE フレーム: 3 者一致数を計算
            cur_n = self._count_three_way_agree(
                getattr(e, attr_cnn),
                getattr(e, attr_hsv),
                getattr(e, attr_conf),
            )
            if cur_n is not None and prev_stable_n is not None:
                diff = cur_n - prev_stable_n
                # chain / ojama_fall / tsumo_fall が介在しない大幅減少を CRITICAL
                chain_intervened = bool(
                    intervene_states & {"chain", "ojama_fall", "tsumo_fall"}
                )
                if not chain_intervened and diff <= -THREE_WAY_DROP_THRESHOLD:
                    violations.append(self._make_three_way_drop_violation(
                        e.frame_idx, e.t_sec, side,
                        prev_stable_n, cur_n, prev_stable_frame,  # type: ignore[arg-type]
                    ))
            # 現 STABLE を prev として更新、 介在 state をリセット
            if cur_n is not None:
                prev_stable_frame = e.frame_idx
                prev_stable_n = cur_n
            intervene_states = set()
        return violations

    # ========================
    # 統合実行
    # ========================

    def evaluate_all(self) -> list[Violation]:
        """全メトリクスを 1P/2P 両方で実行."""
        all_violations: list[Violation] = []
        for side in ("1P", "2P"):
            all_violations.extend(self.check_puyo_count_consistency(side))
            all_violations.extend(self.check_chain_no_disappear(side))
            all_violations.extend(self.check_sudden_drop(side))
            all_violations.extend(self.check_retrospective_chain(side))
            all_violations.extend(self.check_auto_correction(side))
            all_violations.extend(self.check_floating_puyo(side))
            all_violations.extend(self.check_background_color_distribution(side))
            all_violations.extend(self.check_bg_color_cumulative(side))
            all_violations.extend(self.check_chain_state_too_short(side))
            all_violations.extend(self.check_chain_no_puyo_loss(side))
            all_violations.extend(self.check_sparse_color_pop(side))
            all_violations.extend(self.check_stable_short_burst(side))
            # KB (cycle 56): ojama 認識退行 catch
            all_violations.extend(self.check_ojama_disappearance(side))
            all_violations.extend(self.check_ojama_global_scarcity(side))
            # KC (cycle 56 G): 静止中の色ブレ集計 (= 5 色相互誤認)
            all_violations.extend(self.check_static_color_flicker(side))
            # 3 者一致 DROP (2026-06-03): fail-silent 盲点炙り出し
            all_violations.extend(self.check_three_way_sudden_drop(side))
        all_violations.sort(key=lambda v: (v.frame_idx, v.side, v.metric))
        return all_violations

    def count_erasure_alerts(self) -> dict[str, int]:
        """全 frame の erasure_alerts を 1P/2P 別に集計する。

        board_log JSONL の各行に埋め込まれた p1/p2_erasure_alerts を合計し、
        _eval_static_mask.sh の verdict ロジック (REJECT_P_TO_E) に使う。

        Returns:
            dict: {
                "p1": int,  # 1P 側の alert 総数
                "p2": int,  # 2P 側の alert 総数
                "total": int,  # 合計
            }
        """
        p1_total = sum(len(e.p1_erasure_alerts) for e in self.entries)
        p2_total = sum(len(e.p2_erasure_alerts) for e in self.entries)
        return {
            "p1": p1_total,
            "p2": p2_total,
            "total": p1_total + p2_total,
        }

    def generate_report(
        self,
        baseline_avg_puyo_count: float | None = None,
    ) -> dict[str, Any]:
        """評価サマリレポートを生成 (= 採否判定材料).

        Args:
            baseline_avg_puyo_count: baseline の avg_puyo_count_per_stable_frame。
                指定時は ratio check を行い、 0.85 未満で CRITICAL アラートを追加。
                backwards compat のため optional (= None なら ratio check skip)。
        """
        violations = self.evaluate_all()
        # メトリクス別集計
        by_metric: dict[str, list[Violation]] = {}
        for v in violations:
            by_metric.setdefault(v.metric, []).append(v)
        # severity 別集計
        critical_count = sum(1 for v in violations if v.severity == SEVERITY_CRITICAL)
        warning_count = sum(1 for v in violations if v.severity == SEVERITY_WARNING)
        # T4 PuyoErasureMonitor: STABLE 中「色→EMPTY」遷移 alert 集計
        erasure_counts = self.count_erasure_alerts()
        p_to_e_count = erasure_counts["total"]
        # C1: avg_puyo_count 集計 (= baseline 比チェック)
        avg_puyo_stats = compute_avg_puyo_count(
            [e.__dict__ for e in self.entries]
        )
        avg_puyo = avg_puyo_stats["avg_puyo_count_per_stable_frame"]
        n_stable = avg_puyo_stats["n_stable_frames"]
        avg_puyo_ratio: float | None = None
        if baseline_avg_puyo_count is not None and baseline_avg_puyo_count > 0:
            avg_puyo_ratio = avg_puyo / baseline_avg_puyo_count
        # 採否判定: critical が一定数以上 OR 特定メトリクスがある = REJECT
        verdict = "ACCEPT"
        if critical_count >= 20:
            verdict = "REJECT"
        elif critical_count >= 5 or warning_count >= 30:
            verdict = "REVIEW"
        # C1: ratio チェックで REJECT 追加 (= baseline 指定時のみ)
        if avg_puyo_ratio is not None and avg_puyo_ratio < AVG_PUYO_COUNT_CRITICAL_RATIO:
            verdict = "REJECT"
        return {
            "total_frames": len(self.entries),
            "violations": [v.to_dict() for v in violations],
            "summary": {
                "total_violations": len(violations),
                "critical": critical_count,
                "warning": warning_count,
                "info": sum(1 for v in violations if v.severity == SEVERITY_INFO),
                "by_metric": {
                    m: len(vs) for m, vs in by_metric.items()
                },
                "by_metric_critical": {
                    m: sum(1 for v in vs if v.severity == SEVERITY_CRITICAL)
                    for m, vs in by_metric.items()
                },
                # C1: avg_puyo_count メトリクス (= fail-silent 経路の新規 catch)
                "avg_puyo_count_per_stable_frame": avg_puyo,
                "n_stable_frames": n_stable,
                "avg_puyo_count_ratio": avg_puyo_ratio,
            },
            "verdict": verdict,
            # T4 PuyoErasureMonitor: fail-silent 自動検知カウンタ。
            # _eval_static_mask.sh の verdict ロジック (REJECT_P_TO_E) で参照。
            "p_to_e_count": p_to_e_count,
            "p_to_e_detail": erasure_counts,
        }


# ============================
# C1: モジュールレベル関数 (stateless)
# ============================


def compute_avg_puyo_count(entries: list[dict]) -> dict[str, Any]:
    """STABLE フレームの平均ぷよ数 (1P+2P 合算) を計算する。

    Args:
        entries: board_log JSONL を読み込んだ dict リスト。
            各 dict は FrameEntry.from_jsonable() と同等の構造を持つ。
            "p1_state" / "p2_state" が "stable" かつ "p1_confirmed" /
            "p2_confirmed" が非 None のフレームのみを集計対象とする。

    Returns:
        dict:
            "avg_puyo_count_per_stable_frame": float  # 1P+2P 合算 STABLE 平均ぷよ数
            "n_stable_frames": int  # 集計対象フレーム数 (1P STABLE OR 2P STABLE)
            "total_puyo_sum": int   # 合計ぷよ数 (デバッグ用)

    Note:
        COLOR_EMPTY (0) と COLOR_UNKNOWN (10) 以外のすべての色を「ぷよあり」 として
        カウントする (= COLOR_OJAMA=9 も含む)。
    """
    total_sum: int = 0
    n_stable: int = 0
    for e in entries:
        p1_state = str(e.get("p1_state", ""))
        p2_state = str(e.get("p2_state", ""))
        p1_grid = e.get("p1_confirmed")
        p2_grid = e.get("p2_confirmed")
        frame_count = _count_puyo_in_grids(p1_state, p1_grid, p2_state, p2_grid)
        if frame_count is not None:
            total_sum += frame_count
            n_stable += 1
    avg = total_sum / n_stable if n_stable > 0 else 0.0
    return {
        "avg_puyo_count_per_stable_frame": avg,
        "n_stable_frames": n_stable,
        "total_puyo_sum": total_sum,
    }


def _count_puyo_in_grids(
    p1_state: str,
    p1_grid: list[list[int]] | None,
    p2_state: str,
    p2_grid: list[list[int]] | None,
) -> int | None:
    """1P + 2P の STABLE ぷよ数を返す。両方 non-STABLE なら None。"""
    total = 0
    has_stable = False
    if p1_state == "stable" and p1_grid is not None:
        total += _count_grid_puyos(p1_grid)
        has_stable = True
    if p2_state == "stable" and p2_grid is not None:
        total += _count_grid_puyos(p2_grid)
        has_stable = True
    return total if has_stable else None


def _count_grid_puyos(grid: list[list[int]]) -> int:
    """grid (list[list[int]]) の非 EMPTY・非 UNKNOWN cell 数を返す。"""
    count = 0
    for row in grid:
        for cell in row:
            if int(cell) not in (COLOR_EMPTY, COLOR_UNKNOWN):
                count += 1
    return count


# ============================
# C3: 複合 verdict ロジック (module-level export)
# ============================


def judge_cycle(
    baseline_stats: dict,
    candidate_stats: dict,
) -> Literal["AUTO_ACCEPT_PROVISIONAL", "AUTO_REJECT", "NEEDS_REVIEW"]:
    """cycle 採否の複合 verdict を返す。

    評価ロジック:
        AUTO_REJECT: 以下の 1 つでも該当
            1. avg_puyo_count_ratio < 0.85
            2. p_to_e_count > 0 かつ baseline 比 +20% 以上
            3. critical > baseline_critical × 1.10

        AUTO_ACCEPT_PROVISIONAL: 以下を全て満たす
            1. critical <= baseline_critical + 2
            2. avg_puyo_count_ratio >= 0.85 (or baseline 未指定)
            3. p_to_e_count <= baseline p_to_e (増加なし)
            4. transition_drop_alerts 増加なし (or 未提供)

        NEEDS_REVIEW: それ以外

    Args:
        baseline_stats: baseline の generate_report() 出力 dict。
        candidate_stats: candidate cycle の generate_report() 出力 dict。

    Returns:
        "AUTO_ACCEPT_PROVISIONAL" / "AUTO_REJECT" / "NEEDS_REVIEW"
    """
    base_critical = int(baseline_stats.get("summary", {}).get("critical", 0))
    cand_critical = int(candidate_stats.get("summary", {}).get("critical", 0))
    base_p2e = int(baseline_stats.get("p_to_e_count", 0))
    cand_p2e = int(candidate_stats.get("p_to_e_count", 0))
    base_avg = baseline_stats.get("summary", {}).get(
        "avg_puyo_count_per_stable_frame", None
    )
    cand_avg = candidate_stats.get("summary", {}).get(
        "avg_puyo_count_per_stable_frame", None
    )
    base_drops = baseline_stats.get("transition_drop_alert_count", None)
    cand_drops = candidate_stats.get("transition_drop_alert_count", None)

    # --- AUTO_REJECT 判定 ---
    if base_avg is not None and cand_avg is not None and base_avg > 0:
        ratio = cand_avg / base_avg
        if ratio < AVG_PUYO_COUNT_CRITICAL_RATIO:
            return "AUTO_REJECT"
    if cand_p2e > 0 and base_p2e > 0:
        if cand_p2e > base_p2e * JUDGE_CYCLE_P_TO_E_REJECT_RATIO:
            return "AUTO_REJECT"
    if base_critical > 0 and cand_critical > base_critical * JUDGE_CYCLE_CRITICAL_REJECT_RATIO:
        return "AUTO_REJECT"

    # --- AUTO_ACCEPT_PROVISIONAL 判定 ---
    critical_ok = cand_critical <= base_critical + JUDGE_CYCLE_CRITICAL_ACCEPT_DELTA
    ratio_ok = True
    if base_avg is not None and cand_avg is not None and base_avg > 0:
        ratio_ok = (cand_avg / base_avg) >= AVG_PUYO_COUNT_CRITICAL_RATIO
    p2e_ok = cand_p2e <= base_p2e
    drops_ok = True
    if base_drops is not None and cand_drops is not None:
        drops_ok = cand_drops <= base_drops
    if critical_ok and ratio_ok and p2e_ok and drops_ok:
        return "AUTO_ACCEPT_PROVISIONAL"

    return "NEEDS_REVIEW"
