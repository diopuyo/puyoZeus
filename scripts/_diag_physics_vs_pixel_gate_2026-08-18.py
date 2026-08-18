"""物理制約違反 vs 画素差分ゲートの検出力比較 (2026-08-18、デバッガエージェント計装)。

user依頼: `--enable-stable-persistence-gate` (画素差分による静止判定、B) が
学習データを61%減らし欠落が重要局面に偏っていた問題に対し、
「物理制約違反 (消去可能グループが残っている/浮きぷよ) を汚染判定基準にすれば、
画素差分より正確に、かつ局面に偏らずに汚染を除去できるか」を実測する。

対象: data/indicators_v2/boards_lean_phase_l_2026-08-11/{36,52,c100}.npz
       (旧構成、STABLE dedup済みlean npz。3動画で十分、高速優先とのuser指示)

A: 物理制約違反
   (a) 消去可能グループが残っている: ChainSimulator.find_erasable_groups (src/chain.py)
   (b) 重力違反 (浮きぷよ): check_gravity_rule (src/self_supervised/physical_consistency.py)
   どちらも既存実装をそのまま流用 (実装しない、計装のみ)。

B: 画素差分による持続静止判定 (src/board_motion.py: is_raw_pixel_stable、
   window=0.25秒・閾値1.0、collect_boards_lean.py の
   _update_raw_pixel_stable と同一ロジック)。
   新構成 npz (boards_lean_phase_l_2026-08-18) にはこの対象3動画が含まれず
   (14本は別動画セット)、また `stable_persistence_confidence` 列も
   未収録 (収集時点の script version が旧い) と判明したため、
   生動画 (data/frames/video_{vid}.mp4) を密デコードして同ロジックを
   その場で再現する (--sample-interval 0 と同じ、全フレーム処理)。

局面カテゴリ (自分が連鎖中/相手が連鎖中(バースト)/通常設置中/おじゃま着弾中/その他)
は _diag_physics_only_prototype_2026-08-18.py の measure_non_chain_segments と
同じ判定式を、各記録行 (transition の「after」側) に付与する形で再利用する。

実行: 1動画ずつ CLI 引数で回す (並列化のため)。
  PYTHONPATH=. python scripts/_diag_physics_vs_pixel_gate_2026-08-18.py 36
  PYTHONPATH=. python scripts/_diag_physics_vs_pixel_gate_2026-08-18.py 52
  PYTHONPATH=. python scripts/_diag_physics_vs_pixel_gate_2026-08-18.py c100
出力: logs/_diag_physics_vs_pixel_gate_2026-08-18_{vid}.json
統合: --merge で3本まとめてレポート出力。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board,
)
from src.board_motion import (  # noqa: E402
    STABLE_PERSISTENCE_DIFF_THRESHOLD,
    STABLE_PERSISTENCE_WINDOW_SEC,
    board_roi_gray,
    frame_diff_mean,
    is_raw_pixel_stable,
)
from src.chain import ChainSimulator  # noqa: E402
from src.production_config import GHOST_CHAIN_RULE_ENABLED  # noqa: E402
from src.self_supervised.physical_consistency import check_gravity_rule  # noqa: E402

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("logs")
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")

TARGET_W: int = 1920
TARGET_H: int = 1080

_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})


@dataclass
class RowRecord:
    video: str
    side: str
    game_idx: int
    frame_idx: int
    t_sec: float
    category: str
    a_erasable: bool
    a_gravity: bool
    a_violation: bool
    b_stable: "bool | None"  # None = 未計算 (動画側で解決不能)
    b_diff_max_in_window: "float | None"


# =============================================================================
# A: 物理制約違反
# =============================================================================


def _physics_violation(sim: ChainSimulator, grid: np.ndarray) -> "tuple[bool, bool]":
    """(has_erasable, has_gravity_violation) を返す。"""
    board = Board.from_list(grid.tolist())
    has_erasable = len(sim.find_erasable_groups(board)) > 0
    gravity_valid, _violations = check_gravity_rule(board)
    return has_erasable, (not gravity_valid)


# =============================================================================
# 局面カテゴリ (前スクリプトの measure_non_chain_segments と同一判定式)
# =============================================================================


def _opponent_chain_overlaps(
    other_idxs: list[int], t_sec: np.ndarray, trigger: np.ndarray,
    mechanism: np.ndarray, t_lo: float, t_hi: float,
) -> bool:
    for j in other_idxs:
        tj = float(t_sec[j])
        if t_lo < tj <= t_hi and np.isfinite(trigger[j]):
            tag = str(mechanism[j]).strip().lower()
            if tag not in _NO_CHAIN_TAG_VALUES:
                return True
    return False


def _row_category(
    grids: np.ndarray, t_sec: np.ndarray, trigger: np.ndarray, mechanism: np.ndarray,
    idxs: list[int], pos: int, other_idxs: list[int],
) -> str:
    """記録行 idxs[pos] のカテゴリを付与する (前行との遷移 or own_chain タグ)。"""
    j = idxs[pos]
    tag = str(mechanism[j]).strip().lower()
    if np.isfinite(trigger[j]) and tag not in _NO_CHAIN_TAG_VALUES:
        return "own_chain"  # 自分が連鎖中 (chain_trigger_sec タグあり)
    if pos == 0:
        return "game_start"
    i = idxs[pos - 1]
    a = grids[i]
    b = grids[j]
    new_color_mask = (a == COLOR_EMPTY) & (b >= 1) & (b <= 5)
    new_ojama_mask = (a == COLOR_EMPTY) & (b == COLOR_OJAMA)
    removed_mask = (a != COLOR_EMPTY) & (a != COLOR_UNKNOWN) & (b == COLOR_EMPTY)
    changed_existing_mask = (
        (a != COLOR_EMPTY) & (a != COLOR_UNKNOWN)
        & (b != COLOR_EMPTY) & (b != COLOR_UNKNOWN) & (a != b)
    )
    if not (new_color_mask.any() or new_ojama_mask.any() or removed_mask.any() or changed_existing_mask.any()):
        return "no_change"
    if removed_mask.any():
        return "erasure_evidence_untagged"
    if new_ojama_mask.any() and new_color_mask.any():
        return "mixed_ojama_and_placement"
    if new_ojama_mask.any():
        return "ojama_landing"
    t_lo, t_hi = float(t_sec[i]), float(t_sec[j])
    if _opponent_chain_overlaps(other_idxs, t_sec, trigger, mechanism, t_lo, t_hi):
        return "opponent_burst_placement"  # 相手が連鎖中/相手バースト中
    return "normal_placement"


# =============================================================================
# B: 画素差分持続静止 (生動画デコードで再現)
# =============================================================================


def _compute_b_stable_for_video(
    video_id: str, needed: "dict[str, set[int]]",
) -> "dict[tuple[str, int], tuple[bool, float]]":
    """1動画を密デコードし、needed[side] の frame_idx 時点の
    is_raw_pixel_stable 判定を再現する (collect_boards_lean.py の
    _update_raw_pixel_stable と同一ロジック)。

    Returns:
        (side, frame_idx) -> (b_stable, window内diffの最大値)
    """
    path = FRAMES_DIR / f"video_{video_id}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[警告] {path} を開けません。B計測スキップ。")
        return {}

    max_frame_idx = max(
        (max(s) for s in needed.values() if s), default=-1,
    )
    if max_frame_idx < 0:
        cap.release()
        return {}

    out: "dict[tuple[str, int], tuple[bool, float]]" = {}
    prev_gray: "dict[str, np.ndarray | None]" = {"1P": None, "2P": None}
    diff_hist: "dict[str, list[tuple[float, float]]]" = {"1P": [], "2P": []}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fi = 0
    n_report = max(1, max_frame_idx // 10)
    while fi <= max_frame_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        for side in ("1P", "2P"):
            gray = board_roi_gray(frame, side)
            pg = prev_gray[side]
            if pg is not None:
                diff = frame_diff_mean(pg, gray)
                diff_hist[side].append((t_sec, diff))
                diff_hist[side] = [
                    (t, d) for t, d in diff_hist[side]
                    if t_sec - t <= STABLE_PERSISTENCE_WINDOW_SEC
                ]
            prev_gray[side] = gray
            if fi in needed.get(side, ()):
                diffs_now = [d for _t, d in diff_hist[side]]
                stable = is_raw_pixel_stable(
                    diffs_now, diff_threshold=STABLE_PERSISTENCE_DIFF_THRESHOLD,
                )
                worst = max(diffs_now) if diffs_now else 0.0
                out[(side, fi)] = (stable, worst)
        if fi % n_report == 0:
            print(f"  [{video_id}] frame {fi}/{max_frame_idx} ({fi / max_frame_idx * 100:.0f}%)")
        fi += 1
    cap.release()
    return out


# =============================================================================
# メイン処理 (1動画)
# =============================================================================


def process_video(video_id: str, compute_b: bool = True) -> list[RowRecord]:
    npz_path = NPZ_DIR / f"{video_id}.npz"
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    frame_idx = d["frame_idx"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"] if "chain_mechanism" in d else np.array(
        [""] * len(grids),
    )

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(grids)):
        groups[(str(side[i]), int(game_idx[i]))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: float(t_sec[i]))

    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)

    rows: list[RowRecord] = []
    row_meta: list[tuple] = []  # (side, game, frame_idx, t_sec, category)

    for (s, g), idxs in groups.items():
        other_s = "2P" if s == "1P" else "1P"
        other_idxs = groups.get((other_s, g), [])
        for pos, i in enumerate(idxs):
            cat = _row_category(grids, t_sec, trigger, mechanism, idxs, pos, other_idxs)
            has_erasable, has_gravity = _physics_violation(sim, grids[i])
            row_meta.append((s, g, int(frame_idx[i]), float(t_sec[i]), cat, has_erasable, has_gravity))

    needed: "dict[str, set[int]]" = {"1P": set(), "2P": set()}
    for s, _g, fi, _t, _cat, _e, _gr in row_meta:
        needed[s].add(fi)

    b_map: "dict[tuple[str, int], tuple[bool, float]]" = {}
    if compute_b:
        print(f"[{video_id}] B計測 (生動画密デコード) 開始 — 必要frame数: "
              f"1P={len(needed['1P'])} 2P={len(needed['2P'])}")
        b_map = _compute_b_stable_for_video(video_id, needed)

    for s, g, fi, t, cat, has_erasable, has_gravity in row_meta:
        b_entry = b_map.get((s, fi))
        b_stable = b_entry[0] if b_entry is not None else None
        b_diff_max = b_entry[1] if b_entry is not None else None
        rows.append(RowRecord(
            video=video_id, side=s, game_idx=g, frame_idx=fi, t_sec=t,
            category=cat, a_erasable=has_erasable, a_gravity=has_gravity,
            a_violation=(has_erasable or has_gravity),
            b_stable=b_stable, b_diff_max_in_window=b_diff_max,
        ))
    return rows


# =============================================================================
# 集計・レポート
# =============================================================================


def build_report(rows: list[RowRecord]) -> str:
    lines: list[str] = []
    n_total = len(rows)
    lines.append(f"=== 総行数: {n_total} ===")

    n_a = sum(1 for r in rows if r.a_violation)
    n_a_erasable = sum(1 for r in rows if r.a_erasable)
    n_a_gravity = sum(1 for r in rows if r.a_gravity)
    lines.append(
        f"A(物理制約違反): {n_a}/{n_total} ({n_a / n_total * 100:.2f}%) "
        f"[内訳: 消去可能グループ残存={n_a_erasable} ({n_a_erasable / n_total * 100:.2f}%), "
        f"重力違反={n_a_gravity} ({n_a_gravity / n_total * 100:.2f}%)]"
    )

    b_rows = [r for r in rows if r.b_stable is not None]
    n_b_computed = len(b_rows)
    n_b_unstable = sum(1 for r in b_rows if not r.b_stable)
    if n_b_computed:
        lines.append(
            f"B(画素差分で不安定=旧ゲートなら弾かれる): {n_b_unstable}/{n_b_computed} "
            f"({n_b_unstable / n_b_computed * 100:.2f}%) (B計測不能行={n_total - n_b_computed})"
        )
    else:
        lines.append("B: 計測不能 (動画デコード失敗)")

    # 4象限
    if n_b_computed:
        both = sum(1 for r in b_rows if r.a_violation and not r.b_stable)
        a_only = sum(1 for r in b_rows if r.a_violation and r.b_stable)
        b_only = sum(1 for r in b_rows if not r.a_violation and not r.b_stable)
        neither = sum(1 for r in b_rows if not r.a_violation and r.b_stable)
        lines.append(
            f"\n=== 4象限 (n={n_b_computed}) ===\n"
            f"  A且つB(両方が汚染判定): {both} ({both / n_b_computed * 100:.2f}%)\n"
            f"  Aのみ(物理違反だが画素は静止): {a_only} ({a_only / n_b_computed * 100:.2f}%)\n"
            f"  Bのみ(画素不安定だが物理は整合=不当に捨てられていた候補): "
            f"{b_only} ({b_only / n_b_computed * 100:.2f}%)\n"
            f"  どちらでもない(綺麗): {neither} ({neither / n_b_computed * 100:.2f}%)"
        )

        # Bのみ行のカテゴリ内訳
        b_only_rows = [r for r in b_rows if not r.a_violation and not r.b_stable]
        cat_b_only = Counter(r.category for r in b_only_rows)
        lines.append(f"\n=== Bのみ (不当に捨てられていた候補, n={len(b_only_rows)}) カテゴリ内訳 ===")
        for cat, cnt in cat_b_only.most_common():
            lines.append(f"  {cat}: {cnt} ({cnt / len(b_only_rows) * 100:.1f}%)" if b_only_rows else "")

    # カテゴリ別 A率・B率 (局面偏り比較)
    lines.append("\n=== カテゴリ別 A率・B率 (局面偏り比較) ===")
    cats = sorted(set(r.category for r in rows))
    for cat in cats:
        sub = [r for r in rows if r.category == cat]
        sub_b = [r for r in sub if r.b_stable is not None]
        a_rate = sum(1 for r in sub if r.a_violation) / len(sub) * 100 if sub else 0.0
        if sub_b:
            b_rate = sum(1 for r in sub_b if not r.b_stable) / len(sub_b) * 100
            lines.append(
                f"  {cat}: n={len(sub)}, A率={a_rate:.1f}%, B率(不安定)={b_rate:.1f}% (n_b={len(sub_b)})"
            )
        else:
            lines.append(f"  {cat}: n={len(sub)}, A率={a_rate:.1f}%, B計測なし")

    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        all_rows: list[RowRecord] = []
        for vid in TARGET_VIDEOS:
            p = OUT_DIR / f"_diag_physics_vs_pixel_gate_2026-08-18_{vid}.json"
            if not p.exists():
                print(f"[skip] {p} なし")
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for rd in data["rows"]:
                all_rows.append(RowRecord(**rd))
        print(build_report(all_rows))
        merged_path = OUT_DIR / "_diag_physics_vs_pixel_gate_2026-08-18_merged.json"
        merged_path.write_text(
            json.dumps({"rows": [asdict(r) for r in all_rows]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[保存] {merged_path}")
        return

    if not args:
        print("使い方: python _diag_physics_vs_pixel_gate_2026-08-18.py <video_id|--merge> [--no-b]")
        sys.exit(1)
    video_id = args[0]
    compute_b = "--no-b" not in args
    rows = process_video(video_id, compute_b=compute_b)
    print(build_report(rows))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"_diag_physics_vs_pixel_gate_2026-08-18_{video_id}.json"
    out_path.write_text(
        json.dumps({"video": video_id, "rows": [asdict(r) for r in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
