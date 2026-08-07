"""背景誤検出 (でっち上げ) 全体頻度測定器。

3744セル人手検証で確定した誤り類型の罪の序列 (memory
project_yardstick_first_results_2026-07-31.md):
    背景誤検出 (空セルに色をでっち上げ→偽連鎖生成の恐れ) >> おじゃま誤認 (一時的) > ツモ写り込み (軽微)
のうち最悪の「背景誤検出」の全体頻度を測定する (99.99%目標に対する優先度判断の土台)。

でっち上げ候補の定義 (STABLE snapshot間の cell 変化):
    直前 STABLE で空(COLOR_EMPTY)だったセルが、次の STABLE で色 (非空・非UNKNOWN) になった。
    ただしその間に TSUMO_FALL/OJAMA_FALL (設置/着弾) や CHAIN/GRAVITY_SETTLE (連鎖再充填) の
    state が観測されていれば「正当な出現」として除外する
    (scripts/physics_violation_audit.py の類型3 conservation_gain と同じ除外設計を
    cell 単位に絞って再利用、盤面レベルの保存則違反エンジンとは独立に運用する)。

入力は2系統:
    1. 既存 board_log JSONL (p1_state/p1_confirmed 形式、scripts/visualize_recognition.py
       --dump-board-log[-detailed] で生成される形式と同一) を data/verify, data/viz 配下から
       自動棚卸しして解析する。
    2. --videos 指定時は scripts/recognition_physics_review._capture_frames を直接呼び、
       (動画render無しで) live capture して同じ検出ロジックに通す
       (capture 層の重複実装を避ける、既存資産の再利用)。

測定すること:
    a. でっち上げ候補の頻度 (動画別・1P/2P別、件/分 と 件/1000snapshot)
    b. 持続時間分布 (TRANSIENT_DURATION_SEC 以下=一時的自己修復 / それ以上=持続的)
    c. 偽連鎖誘発の有無 (即simulate発火 or 直後 lookahead 秒以内の CHAIN state 観測)
    d. 人手確定済でっち上げ実例3件 (KNOWN_CALIBRATION_CASES) が検出できるかの較正

Usage:
    PYTHONPATH=. python -m scripts.measure_background_fp_frequency --smoke
    PYTHONPATH=. python -m scripts.measure_background_fp_frequency --videos c62,30,35,38
    PYTHONPATH=. python -m scripts.measure_background_fp_frequency --skip-calibration
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from scripts.recognition_physics_review import _capture_frames  # noqa: E402
from scripts.physics_violation_audit import (  # noqa: E402
    DEFAULT_VIDEO_STEMS, DIAG_WINDOW_BY_STEM,
)

# ============================
# 定数
# ============================
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "background_fp_2026-08-01"
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
# 既存 board_log JSONL の棚卸し対象ルート (p1_state/p1_confirmed スキーマを持つファイルのみ採用)。
LOG_SEARCH_ROOTS: tuple[Path, ...] = (
    PROJ_ROOT / "data" / "verify",
    PROJ_ROOT / "data" / "viz",
)

# 一時的 (自己修復) とみなす持続時間の閾値 (秒)。
# memory project_yardstick_first_results_2026-07-31 の実測 (≤1秒で自己修復) に準拠。
TRANSIENT_DURATION_SEC: float = 1.0

# 偽連鎖誘発の lookahead 窓 (秒)。連鎖アニメーション開始までの猶予として採用
# (プロジェクト既存の「連鎖後3秒除外」規約と同じ値、独自のマジックナンバーを増やさない)。
INDUCED_CHAIN_LOOKAHEAD_SEC: float = 3.0

# 設置 (着地/着弾) で説明可能な出現 → でっち上げ候補から除外する state 集合。
PLACEMENT_STATES: frozenset[BoardState] = frozenset(
    {BoardState.TSUMO_FALL, BoardState.OJAMA_FALL},
)
# 連鎖 (消去+重力再充填) で説明可能な出現 → でっち上げ候補から除外する state 集合。
REFILL_STATES: frozenset[BoardState] = frozenset(
    {BoardState.CHAIN, BoardState.GRAVITY_SETTLE},
)

# 設置/連鎖state終了直後の1枚目STABLE snapshotが未収束 (実測: おじゃま着地直後の
# 1枚目STABLEが着地個数を拾いきれない境界ケース) を吸収するための猶予窓 (秒)。
PLACEMENT_LOOKBACK_GRACE_SEC: float = 1.0

# --smoke 用の既知安全窓 (physics_violation_audit と同一の実測済み安全区間を流用)。
SMOKE_VIDEO_STEM: str = "30"
SMOKE_START_SEC: float = 225.0
SMOKE_MAX_SEC: float = 21.0

# DIAG_WINDOW_BY_STEM に未登録の --videos 指定 stem 向けフォールバック窓
# (未知動画で動画全体を舐める事故を防ぐため短く固定する)。
DEFAULT_CAPTURE_START_SEC: float = 120.0
DEFAULT_CAPTURE_MAX_SEC: float = 90.0

_VIDEO_STEM_RE = re.compile(r"video_([a-zA-Z0-9]+)")


# ============================
# データ構造
# ============================


@dataclass
class SideFrame:
    """1 (side, frame) 分の正規化された観測値 (JSONL / live-capture 共通表現)。"""

    frame_idx: int
    t_sec: float
    state: BoardState
    grid: np.ndarray | None


@dataclass
class BackgroundFpCandidate:
    """1 件の背景誤検出 (でっち上げ) 候補。"""

    source: str
    video_stem: str
    side: str
    row: int
    col: int
    fabricated_color: int
    appear_t_sec: float
    appear_frame_idx: int
    persistent_sec: float
    n_snapshots_observed: int
    resolved_within_log: bool
    is_transient: bool
    induced_immediate_fire: bool
    induced_chain_followed: bool


@dataclass
class CalibrationCase:
    """人手確定済 でっち上げ実例 (較正用)。"""

    label: str
    video_stem: str
    side: str
    frame_idx: int
    row: int
    col: int
    note: str


# 人手確定済 でっち上げ実例3件 (data/verify/board_labels_2026-07-31_v2/v3 の labels.tsv より)。
KNOWN_CALIBRATION_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        label="v2#004", video_stem="c12", side="1P", frame_idx=80803, row=10, col=3,
        note="背景赤→空(r10c3)、0.25秒で自然消滅と時間追跡済み(project_yardstick_first_results)",
    ),
    CalibrationCase(
        label="v2#030", video_stem="c11", side="1P", frame_idx=138171, row=1, col=2,
        note="×印を紫と誤認(r1c2)",
    ),
    CalibrationCase(
        label="v3#027", video_stem="c23", side="2P", frame_idx=74554, row=1, col=2,
        note="×印を紫と誤認・満杯盤面(r1c2)、v2#030と同型2例目",
    ),
)

# 較正ケースの capture 窓 (対象フレームの前後を挟む秒数)。
CALIBRATION_WINDOW_PRE_SEC: float = 8.0
CALIBRATION_WINDOW_POST_SEC: float = 8.0

# force_in_match=True の live-capture 窓冒頭は confirmed_board 収束前の cold-start
# 揺れを含む (実測: warmup無しで較正窓が174件等の異常値、2026-08-01対処)。
# この秒数分は候補集計・母数の両方から除外する。
CAPTURE_WARMUP_SEC: float = 3.0


# ============================
# 検出ロジック (stateless、JSONL / live-capture 共通)
# ============================


def find_empty_to_color_cells(
    prev_grid: np.ndarray, curr_grid: np.ndarray,
) -> list[tuple[int, int]]:
    """空(COLOR_EMPTY)から非空(非UNKNOWN)へ変化したセル一覧を返す (隠し段は除外)。"""
    cells: list[tuple[int, int]] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            pv, cv = int(prev_grid[row, col]), int(curr_grid[row, col])
            if pv == COLOR_EMPTY and cv not in (COLOR_EMPTY, COLOR_UNKNOWN):
                cells.append((row, col))
    return cells


def distinct_stable_snapshots(frames: list[SideFrame]) -> list[tuple[int, SideFrame]]:
    """STABLE かつ grid ありのフレームのうち直前と異なる盤面のみ (frames内index, frame) で返す。"""
    out: list[tuple[int, SideFrame]] = []
    last_bytes: bytes | None = None
    for i, f in enumerate(frames):
        if f.state != BoardState.STABLE or f.grid is None:
            continue
        gb = f.grid.tobytes()
        if gb == last_bytes:
            continue
        last_bytes = gb
        out.append((i, f))
    return out


def has_state_in_range(
    frames: list[SideFrame], i_lo: int, i_hi: int, states: frozenset[BoardState],
) -> bool:
    """(i_lo, i_hi] 区間 (frames内index) に states のいずれかが観測されているか。"""
    return any(frames[k].state in states for k in range(i_lo + 1, i_hi + 1))


def has_state_in_range_with_lookback(
    frames: list[SideFrame], prev_idx: int, curr_idx: int, prev_f: SideFrame,
    states: frozenset[BoardState],
) -> bool:
    """(prev_idx, curr_idx] に加え、prev直前 PLACEMENT_LOOKBACK_GRACE_SEC 以内も判定に含める。

    実測で特定した境界ケース対応 (2026-08-01): OJAMA_FALL 終了直後1枚目のSTABLE
    snapshot は着地したおじゃまを全部拾いきれていないことがあり、2枚目の
    STABLE snapshot との差分が「でっち上げ」に誤認識される。OJAMA_FALL/TSUMO_FALL/
    CHAIN/GRAVITY_SETTLE が prev の直前ごく短時間に終わっていた場合も
    「設置/連鎖で説明可能」として扱う。
    """
    if has_state_in_range(frames, prev_idx, curr_idx, states):
        return True
    lo_t = prev_f.t_sec - PLACEMENT_LOOKBACK_GRACE_SEC
    for k in range(prev_idx, -1, -1):
        if frames[k].t_sec < lo_t:
            break
        if frames[k].state in states:
            return True
    return False


def measure_persistence(
    snaps: list[tuple[int, SideFrame]], start_pos: int, row: int, col: int, fabricated_color: int,
) -> tuple[float, int, bool]:
    """候補セルが後続 STABLE snapshot で何秒/何回 持続するかを測る。

    Returns:
        (persistent_sec, n_snapshots_observed, resolved_within_log)。
        resolved_within_log=True: ログ内で別の値に変わった (自己修復含む)。
        False: ログ終端まで同じ値のまま (観測窓を超えて持続する可能性があり断定しない)。
    """
    start_frame = snaps[start_pos][1]
    n_observed = 1
    for pos in range(start_pos + 1, len(snaps)):
        f = snaps[pos][1]
        if int(f.grid[row, col]) != fabricated_color:
            return f.t_sec - start_frame.t_sec, n_observed, True
        n_observed += 1
    last_t = snaps[-1][1].t_sec
    return last_t - start_frame.t_sec, n_observed, False


def check_induced_chain(
    frames: list[SideFrame], curr_idx: int, curr_frame: SideFrame, sim: ChainSimulator,
) -> tuple[bool, bool]:
    """でっち上げ候補を含む盤面が偽連鎖を誘発したかを2経路で判定する。

    経路1 (immediate_fire): 候補セルを含む confirmed_board をそのまま simulate し
        4連結以上の消去 (連鎖) が成立するか (盤面レベルで既に不整合が連鎖を生みうる状態)。
    経路2 (chain_followed): 直後 INDUCED_CHAIN_LOOKAHEAD_SEC 秒以内に実際に CHAIN state が
        観測されたか (認識パイプライン側で本当に連鎖処理へ入ったか、疑似発火との弁別はしない
        近似値であることに留意)。
    """
    try:
        board = Board.from_list(curr_frame.grid.tolist())
        sim_result = sim.simulate(board)
        immediate_fire = sim_result.chain_count >= 1
    except Exception:
        immediate_fire = False
    deadline = curr_frame.t_sec + INDUCED_CHAIN_LOOKAHEAD_SEC
    chain_followed = any(
        curr_frame.t_sec < frames[k].t_sec <= deadline and frames[k].state == BoardState.CHAIN
        for k in range(curr_idx, len(frames))
    )
    return immediate_fire, chain_followed


def find_background_fp_candidates(
    frames: list[SideFrame], video_stem: str, side: str, source: str, sim: ChainSimulator,
    *, min_appear_t_sec: float = float("-inf"),
) -> list[BackgroundFpCandidate]:
    """1 (video, side) 分の背景誤検出 (でっち上げ) 候補を全件検出する。

    min_appear_t_sec: 2026-08-01 追加 (既定 -inf = 従来通り無効・後方互換維持)。
        force_in_match=True の live-capture は動画中盤から認識を強制的に「試合中」
        扱いで開始するため、窓の冒頭は confirmed_board が収束するまでの cold-start
        揺れを拾ってしまう (実測: 較正窓で最初の1ペアだけで数十セルが同時に空->色化する
        バーストが発生、warmup区間の「収束前snapshot」と「収束後snapshot」を比較して
        しまうのが原因と実測で特定)。prev/curr 双方が min_appear_t_sec 以降の
        snapshot であるペアのみを候補対象にする (どちらか一方だけの判定では不十分、
        prev がwarmup中だとcurrとの差分が「収束による見かけの変化」になるため)。
    """
    snaps = distinct_stable_snapshots(frames)
    candidates: list[BackgroundFpCandidate] = []
    for pos in range(len(snaps) - 1):
        prev_idx, prev_f = snaps[pos]
        curr_idx, curr_f = snaps[pos + 1]
        if prev_f.t_sec < min_appear_t_sec:
            continue  # prev が cold-start 収束待ち区間 (warmup) 中 → ペア全体を除外
        cells = find_empty_to_color_cells(prev_f.grid, curr_f.grid)
        if not cells:
            continue
        if has_state_in_range_with_lookback(frames, prev_idx, curr_idx, prev_f, PLACEMENT_STATES):
            continue  # 設置/着弾で説明可能な正当な出現
        if has_state_in_range_with_lookback(frames, prev_idx, curr_idx, prev_f, REFILL_STATES):
            continue  # 連鎖再充填で説明可能な正当な出現
        immediate_fire, chain_followed = check_induced_chain(frames, curr_idx, curr_f, sim)
        for row, col in cells:
            color = int(curr_f.grid[row, col])
            dur, n_obs, resolved = measure_persistence(snaps, pos + 1, row, col, color)
            candidates.append(BackgroundFpCandidate(
                source=source, video_stem=video_stem, side=side, row=row, col=col,
                fabricated_color=color, appear_t_sec=curr_f.t_sec,
                appear_frame_idx=curr_f.frame_idx, persistent_sec=dur,
                n_snapshots_observed=n_obs, resolved_within_log=resolved,
                is_transient=dur <= TRANSIENT_DURATION_SEC,
                induced_immediate_fire=immediate_fire, induced_chain_followed=chain_followed,
            ))
    return candidates


# ============================
# 入力ローダ (JSONL / live-capture)
# ============================


def load_side_frames_from_jsonl(path: Path) -> dict[str, list[SideFrame]]:
    """board_log JSONL (p1_state/p1_confirmed 形式) を読み side 別に正規化する。"""
    out: dict[str, list[SideFrame]] = {"1P": [], "2P": []}
    prefix_by_side = {"1P": "p1", "2P": "p2"}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for side, prefix in prefix_by_side.items():
                state_raw = row.get(f"{prefix}_state")
                if state_raw is None:
                    continue
                try:
                    state = BoardState(state_raw)
                except ValueError:
                    continue
                grid_raw = row.get(f"{prefix}_confirmed")
                grid = np.array(grid_raw, dtype=np.int64) if grid_raw is not None else None
                out[side].append(SideFrame(
                    frame_idx=int(row["frame_idx"]), t_sec=float(row["t_sec"]),
                    state=state, grid=grid,
                ))
    return out


def load_side_frames_from_capture(
    stem: str, start_sec: float, max_sec: float,
) -> dict[str, list[SideFrame]]:
    """_capture_frames (recognition_physics_review、動画render無し) で正規化データを得る。"""
    by_side_raw = _capture_frames(stem, start_sec, max_sec)
    out: dict[str, list[SideFrame]] = {}
    for side, records in by_side_raw.items():
        out[side] = [
            SideFrame(frame_idx=r.frame_idx, t_sec=r.t_sec, state=BoardState[r.state], grid=r.grid)
            for r in records
        ]
    return out


def looks_like_board_log(path: Path) -> bool:
    """1行目に p1_state/p1_confirmed キーがあるか (board_log JSONL スキーマ判定)。"""
    try:
        with path.open(encoding="utf-8") as f:
            first = f.readline()
        row = json.loads(first)
        return "p1_state" in row and "p1_confirmed" in row
    except Exception:
        return False


def discover_board_logs() -> list[Path]:
    """既知の探索先配下から使えるスキーマの board_log JSONL を棚卸しする。"""
    found: list[Path] = []
    for root in LOG_SEARCH_ROOTS:
        if not root.exists():
            continue
        found.extend(p for p in sorted(root.rglob("*.jsonl")) if looks_like_board_log(p))
    return found


def guess_video_stem(path: Path) -> str:
    """ファイル名から動画stemを推測する (best-effort、不明ならファイル名そのもの)。"""
    m = _VIDEO_STEM_RE.search(path.stem)
    return m.group(1) if m else path.stem


# ============================
# 集計
# ============================


@dataclass
class SourceStats:
    """1 source (1 board_log JSONL または 1 live-capture 窓) 分の集計基礎値。"""

    source: str
    video_stem: str
    side: str
    duration_sec: float
    n_stable_snapshots: int
    candidates: list[BackgroundFpCandidate]


def analyze_source(
    by_side: dict[str, list[SideFrame]], video_stem: str, source: str, sim: ChainSimulator,
    *, min_appear_t_sec: float = float("-inf"),
) -> list[SourceStats]:
    """1 source 分を両 side で解析し SourceStats のリストを返す。

    min_appear_t_sec: find_background_fp_candidates と同じ warmup 除外を、
        測定母数 (duration_sec/n_stable_snapshots) 側にも一貫して適用する
        (warmup区間を候補からは除くのに母数の分母には含めるとレートが過小評価される)。
    """
    stats: list[SourceStats] = []
    for side in ("1P", "2P"):
        frames = by_side.get(side, [])
        if not frames:
            continue
        snaps = distinct_stable_snapshots(frames)
        measured_snaps = [f for _i, f in snaps if f.t_sec >= min_appear_t_sec]
        cands = find_background_fp_candidates(
            frames, video_stem, side, source, sim, min_appear_t_sec=min_appear_t_sec,
        )
        measured_start = max(frames[0].t_sec, min_appear_t_sec)
        duration_sec = max(1e-6, frames[-1].t_sec - measured_start)
        stats.append(SourceStats(
            source=source, video_stem=video_stem, side=side, duration_sec=duration_sec,
            n_stable_snapshots=len(measured_snaps), candidates=cands,
        ))
    return stats


# 1 STABLE遷移で同時に空->色化するセル数がこれ以上の場合、単発の背景誤検出ではなく
# baseline-reset (内部復旧機構) 等による大規模再ポピュレーションの疑いが強い
# (実測 2026-08-01: 件/分が突出する動画は例外なく数箇所の巨大バーストに起因し、
# そのタイムスタンプは [baseline-reset] ログと一致した)。真の設置は最大2セル、
# 4連結消去+重力再充填でも通常は数セル規模のため、4を境界に区別する。
BURST_CELL_THRESHOLD: int = 4


def _group_events(
    candidates: list[BackgroundFpCandidate],
) -> dict[tuple[str, str, float], list[BackgroundFpCandidate]]:
    """候補セルを (source, side, appear_t_sec) = 同一transitionイベント単位でまとめる。"""
    groups: dict[tuple[str, str, float], list[BackgroundFpCandidate]] = {}
    for c in candidates:
        groups.setdefault((c.source, c.side, c.appear_t_sec), []).append(c)
    return groups


def classify_events(candidates: list[BackgroundFpCandidate]) -> tuple[int, int]:
    """(n_isolated_events, n_bulk_events) を transition-event 単位で返す (cell単位ではない)。

    bulk (BURST_CELL_THRESHOLD セル以上同時出現) は baseline-reset 等の内部復旧機構
    由来の疑いが強く、素朴な「背景誤検出」件数として数えると過大評価になるため分離する。
    """
    groups = _group_events(candidates)
    n_isolated = sum(1 for g in groups.values() if len(g) < BURST_CELL_THRESHOLD)
    n_bulk = sum(1 for g in groups.values() if len(g) >= BURST_CELL_THRESHOLD)
    return n_isolated, n_bulk


def _group_by_video(all_stats: list[SourceStats]) -> dict[str, dict]:
    """動画別・1P/2P別の内訳 (件数・分・件/分・件/1000snapshot・isolated/bulk内訳) を作る。"""
    by_key: dict[str, dict] = {}
    for s in all_stats:
        key = f"{s.video_stem}_{s.side}"
        row = by_key.setdefault(key, {
            "video_stem": s.video_stem, "side": s.side, "n_candidates": 0,
            "duration_min": 0.0, "n_stable_snapshots": 0, "n_persistent": 0,
        })
        row["n_candidates"] += len(s.candidates)
        row["duration_min"] += s.duration_sec / 60.0
        row["n_stable_snapshots"] += s.n_stable_snapshots
        row["n_persistent"] += sum(1 for c in s.candidates if not c.is_transient)
    for key, row in by_key.items():
        stem, side = row["video_stem"], row["side"]
        cells = [c for s in all_stats if s.video_stem == stem and s.side == side for c in s.candidates]
        row["n_isolated_events"], row["n_bulk_events"] = classify_events(cells)
        row["rate_per_min"] = row["n_candidates"] / row["duration_min"] if row["duration_min"] > 0 else 0.0
        row["rate_per_1000_snapshots"] = (
            row["n_candidates"] / row["n_stable_snapshots"] * 1000 if row["n_stable_snapshots"] else 0.0
        )
    return by_key


def build_summary(all_stats: list[SourceStats]) -> dict:
    """全 source 分の集計結果 (全体・動画別・1P/2P別) を構築する。"""
    total_min = sum(s.duration_sec for s in all_stats) / 60.0
    total_snapshots = sum(s.n_stable_snapshots for s in all_stats)
    all_candidates = [c for s in all_stats for c in s.candidates]
    n_total = len(all_candidates)
    n_persistent = sum(1 for c in all_candidates if not c.is_transient)
    n_induced = sum(1 for c in all_candidates if c.induced_immediate_fire or c.induced_chain_followed)
    n_isolated_events, n_bulk_events = classify_events(all_candidates)
    return {
        "n_sources": len(all_stats),
        "total_duration_min": total_min,
        "total_stable_snapshots": total_snapshots,
        "n_total_candidates": n_total,
        "rate_per_min": n_total / total_min if total_min > 0 else 0.0,
        "rate_per_1000_snapshots": (n_total / total_snapshots * 1000) if total_snapshots else 0.0,
        "n_persistent": n_persistent,
        "n_transient": n_total - n_persistent,
        "n_induced_fake_chain": n_induced,
        "n_isolated_events": n_isolated_events,
        "n_bulk_events": n_bulk_events,
        "by_video": _group_by_video(all_stats),
    }


# ============================
# 較正 (人手確定済 でっち上げ実例3件)
# ============================


def _read_video_fps(video_stem: str) -> float:
    """動画fpsを読む (0.0=失敗)。"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"video_{video_stem}.mp4"))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps


def _run_one_calibration_case(case: CalibrationCase, sim: ChainSimulator) -> dict:
    """1 較正ケース分: 短窓を live capture し候補一覧に含まれるかを判定する。"""
    fps = _read_video_fps(case.video_stem)
    if fps <= 0:
        return {"case": case.label, "measured": False, "reason": "動画fps取得失敗", "note": case.note}
    center_sec = case.frame_idx / fps
    start_sec = max(0.0, center_sec - CALIBRATION_WINDOW_PRE_SEC)
    max_sec = CALIBRATION_WINDOW_PRE_SEC + CALIBRATION_WINDOW_POST_SEC
    try:
        by_side = load_side_frames_from_capture(case.video_stem, start_sec, max_sec)
    except Exception as e:  # noqa: BLE001 - 較正は失敗しても他ケースの測定を止めない
        return {"case": case.label, "measured": False, "reason": f"capture失敗: {e}", "note": case.note}
    frames = by_side.get(case.side, [])
    if not frames:
        return {"case": case.label, "measured": False, "reason": "sideデータなし", "note": case.note}
    source = f"calibration:{case.label}"
    cands = find_background_fp_candidates(
        frames, case.video_stem, case.side, source, sim,
        min_appear_t_sec=start_sec + CAPTURE_WARMUP_SEC,
    )
    hit = any(c.row == case.row and c.col == case.col for c in cands)
    return {
        "case": case.label, "measured": True, "detected": hit,
        "n_candidates_in_window": len(cands), "note": case.note,
    }


def run_calibration_checks(sim: ChainSimulator) -> list[dict]:
    """人手確定済 でっち上げ実例3件が検出器に引っかかるかを較正する。"""
    return [_run_one_calibration_case(case, sim) for case in KNOWN_CALIBRATION_CASES]


# ============================
# 出力
# ============================


def write_candidates_tsv(candidates: list[BackgroundFpCandidate], path: Path) -> None:
    """候補明細を TSV に書き出す。"""
    header = [
        "source", "video_stem", "side", "row", "col", "fabricated_color",
        "appear_t_sec", "appear_frame_idx", "persistent_sec", "n_snapshots_observed",
        "resolved_within_log", "is_transient", "induced_immediate_fire", "induced_chain_followed",
    ]
    lines = ["\t".join(header)]
    for c in candidates:
        lines.append("\t".join(str(getattr(c, h)) for h in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_calibration_section(calibration: list[dict]) -> list[str]:
    """summary.md の較正セクション行を作る。"""
    lines = ["", "## 検出器の較正 (人手確定済 でっち上げ実例3件)", ""]
    if not calibration:
        lines.append("- (--skip-calibration 指定のため未実施)")
        return lines
    for r in calibration:
        if not r.get("measured"):
            lines.append(f"- {r['case']}: 未測定 ({r.get('reason', '不明')}) — {r.get('note', '')}")
            continue
        mark = "検出" if r["detected"] else "未検出"
        lines.append(
            f"- {r['case']}: {mark} (窓内候補{r['n_candidates_in_window']}件) — {r['note']}",
        )
    return lines


def format_summary_md(summary: dict, calibration: list[dict]) -> str:
    """summary.md 用の平易な日本語サマリを組み立てる。"""
    lines = [
        "# 背景誤検出(でっち上げ) 全体頻度測定 summary",
        "",
        f"- 測定 source 数: {summary['n_sources']}",
        f"- 測定できた総時間: {summary['total_duration_min']:.1f} 分 "
        f"(STABLE snapshot {summary['total_stable_snapshots']} 件、これが測れた母数)",
        f"- でっち上げ候補 セル単位件数: {summary['n_total_candidates']} 件 "
        f"({summary['rate_per_min']:.3f} 件/分、{summary['rate_per_1000_snapshots']:.2f} 件/1000snapshot)",
        f"- 上記をイベント単位 (同一transitionでのセル群=1件) で数え直すと: "
        f"孤立イベント(<{BURST_CELL_THRESHOLD}セル/回) {summary['n_isolated_events']} 件、"
        f"バーストイベント(>={BURST_CELL_THRESHOLD}セル/回、baseline-reset等の"
        f"内部復旧混入疑いが強い) {summary['n_bulk_events']} 件",
        f"  - うち持続的 (>{TRANSIENT_DURATION_SEC:.0f}秒、自己修復せず): {summary['n_persistent']} 件",
        f"  - うち一時的 (<={TRANSIENT_DURATION_SEC:.0f}秒で自己修復): {summary['n_transient']} 件",
        f"  - うち偽連鎖誘発疑い (即simulate発火 or 直後{INDUCED_CHAIN_LOOKAHEAD_SEC:.0f}秒以内にCHAIN state観測): "
        f"{summary['n_induced_fake_chain']} 件",
        "",
        "## 動画別・1P/2P別 内訳",
        "",
        "| video | side | 候補件数(セル) | 孤立イベント | バーストイベント | うち持続的 | 分 | 件/分 | 件/1000snapshot | STABLE snapshot数 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _key, row in sorted(summary["by_video"].items()):
        lines.append(
            f"| {row['video_stem']} | {row['side']} | {row['n_candidates']} | "
            f"{row['n_isolated_events']} | {row['n_bulk_events']} | {row['n_persistent']} | "
            f"{row['duration_min']:.1f} | {row['rate_per_min']:.3f} | "
            f"{row['rate_per_1000_snapshots']:.2f} | {row['n_stable_snapshots']} |",
        )
    lines += _format_calibration_section(calibration)
    lines += _format_known_limitations_section()
    return "\n".join(lines)


def _format_known_limitations_section() -> list[str]:
    """summary.md の「既知の限界」セクション (2026-08-01 実測で判明した注意点)。"""
    return [
        "",
        "## 既知の限界 (この測定結果を読む上での注意点)",
        "",
        "- **バーストイベント (>=" + str(BURST_CELL_THRESHOLD) + "セル同時出現) は "
        "baseline-reset 等の内部復旧機構混入の疑いが強い**: 実測で件/分が突出した"
        "動画・sideは例外なく数箇所の巨大バースト (最大42セル同時) に起因し、そのタイムスタンプは"
        "内部ログ `[baseline-reset] ...` と時刻が一致した。単発の背景誤検出とは別現象であり、"
        "上表の「バーストイベント」列で分離した (孤立イベント列が本来知りたい値に近い)。",
        "- **旧世代ログ (video_124系、2026-06生成) は現行認識と乖離**: data/verify, data/viz "
        "配下の自動棚卸しでヒットした既存 board_log JSONL はいずれも2026-06生成の古いスモークログで、"
        "多数の認識修正 (2026-07以降) を経ていない。現行品質の代表値としては使えないため、"
        "全体集計に混ぜず動画別内訳で個別に確認すること。",
        "- **較正3例は検出器が捕捉できなかった (0/3)**: 人手確定済みの3実例は全て"
        "満杯/準満杯盤面かつ実際の設置(TSUMO_FALL)・おじゃま着弾(OJAMA_FALL)が"
        "頻発する終盤の込み合った区間で発生していた。本検出器の除外ロジック"
        "(prev/curr間に設置/連鎖stateが1回でもあれば「説明可能」と判定) は、"
        "そうした区間では真の誤読も設置イベントと同時に起きたとみなして除外して"
        "しまう。**したがって本測定値は下限であり、実際のでっち上げ頻度"
        "(特に満杯盤面・終盤の込み合った局面) はこれより高い可能性が高い。**",
        "- **live-capture (force_in_match=True) の cold-start**: 窓の先頭"
        f"{CAPTURE_WARMUP_SEC:.0f}秒は confirmed_board 収束前の揺れを含みうるため、"
        "候補・母数の両方から除外済み (min_appear_t_sec)。",
        "- **vetted4動画 (c62/30/35/38) の孤立イベントは1件のみ、かつ別類型の疑い**: "
        "video_30 1P で唯一検出された孤立イベントは列0の3セルが同時に色9(おじゃま)化する"
        "もので、おじゃま着弾の6列均等分配パターン(1列3個)と一致する。OJAMA_FALL state"
        "は間に観測されなかったが、内容から scripts/physics_violation_audit.py の"
        "類型16 ojama_drop_desync (#45、担当別・別途追跡中) の疑いが強く、"
        "純粋な「背景誤検出」とは別現象の可能性がある。",
    ]


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser(description="背景誤検出(でっち上げ)全体頻度測定器")
    ap.add_argument(
        "--logs", nargs="*", default=None,
        help="解析対象 board_log JSONL パス (省略時は data/verify, data/viz 配下を自動棚卸し)。",
    )
    ap.add_argument(
        "--videos", type=str, default="",
        help="追加で live capture 解析する動画stemのカンマ区切り "
             "(例: c62,30,35,38。physics_violation_audit の既知安全短窓 DIAG_WINDOW_BY_STEM を "
             "優先採用し、未登録stemは固定フォールバック短窓を使う)。",
    )
    ap.add_argument(
        "--use-default-videos", action="store_true", dest="use_default_videos",
        help="physics_violation_audit.DEFAULT_VIDEO_STEMS (c62,30,35,38) を --videos の既定として使う。",
    )
    ap.add_argument("--smoke", action="store_true", help="動作確認用: video_30の短窓のみ処理する。")
    ap.add_argument(
        "--skip-calibration", action="store_true", dest="skip_calibration",
        help="人手確定済み3例の較正チェックをスキップする (高速反復用)。",
    )
    ap.add_argument(
        "--skip-log-scan", action="store_true", dest="skip_log_scan",
        help="既存 board_log JSONL の自動棚卸しをスキップする (--videos のみで測定)。",
    )
    return ap.parse_args()


def _resolve_log_paths(args: argparse.Namespace) -> list[Path]:
    """--logs 指定 or 自動棚卸しで対象 JSONL パス一覧を決める。"""
    if args.logs is not None:
        return [Path(p) for p in args.logs]
    if args.skip_log_scan or args.smoke:
        return []
    return discover_board_logs()


def _resolve_capture_windows(args: argparse.Namespace) -> list[tuple[str, float, float]]:
    """--videos / --smoke / --use-default-videos から live capture 対象窓を決める。"""
    if args.smoke:
        return [(SMOKE_VIDEO_STEM, SMOKE_START_SEC, SMOKE_MAX_SEC)]
    stems_str = args.videos or (",".join(DEFAULT_VIDEO_STEMS) if args.use_default_videos else "")
    if not stems_str:
        return []
    windows: list[tuple[str, float, float]] = []
    for stem in (s.strip() for s in stems_str.split(",") if s.strip()):
        if stem in DIAG_WINDOW_BY_STEM:
            start, dur = DIAG_WINDOW_BY_STEM[stem]
            windows.append((stem, start, dur))
        else:
            windows.append((stem, DEFAULT_CAPTURE_START_SEC, DEFAULT_CAPTURE_MAX_SEC))
    return windows


def main() -> None:
    """メイン処理: 棚卸し/live capture → 検出 → 較正 → summary.md/tsv 出力。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    sim = ChainSimulator()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_paths = _resolve_log_paths(args)
    print(f"[INFO] 既存 board_log JSONL 棚卸し結果: {len(log_paths)} 件")
    all_stats: list[SourceStats] = []
    for path in log_paths:
        by_side = load_side_frames_from_jsonl(path)
        all_stats += analyze_source(by_side, guess_video_stem(path), str(path), sim)

    capture_windows = _resolve_capture_windows(args)
    for stem, start_sec, max_sec in capture_windows:
        print(f"[INFO] live capture: video_{stem} start={start_sec:.1f}s dur={max_sec:.1f}s")
        by_side = load_side_frames_from_capture(stem, start_sec, max_sec)
        all_stats += analyze_source(
            by_side, stem, f"capture:{stem}:{start_sec:.0f}", sim,
            min_appear_t_sec=start_sec + CAPTURE_WARMUP_SEC,
        )

    if not log_paths and not capture_windows:
        print("[WARN] 対象ログ0件・live capture窓0件 (母数ゼロで測定不能)。--videos か --smoke を指定してください。")

    calibration = [] if args.skip_calibration else run_calibration_checks(sim)
    summary = build_summary(all_stats)
    all_candidates = [c for s in all_stats for c in s.candidates]

    write_candidates_tsv(all_candidates, OUTPUT_DIR / "candidates_detail.tsv")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "calibration": calibration}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md = format_summary_md(summary, calibration)
    (OUTPUT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[DONE] {OUTPUT_DIR} に保存しました。")


if __name__ == "__main__":
    main()
