"""バーストガード v2: 設置→confirmed色確定 反映遅延分布の測定 (2026-08-05)。

`docs/BURST_GUARD_DESIGN_2026-08-05.md` §7.2 に対応する。OFF (アンカー npz) と
(v2) (バーストガード再認識 npz) を比較し、「設置→confirmed_board 反映」の
フレーム遅延が (v2) でどれだけ増えたか (= バーストガードのコスト) を、
バーストWindow活性中/非活性中で分けて測定する。

## `_diag_placement_confirm_frames_2026-07-25.py` (feedback_placement_
reflection_8frames_2026-07-25) との違い (npz-only制約への対応)
旧スクリプトは動画を再走行し、生CNNの逐フレーム出力から「設置 (空→色に
安定して初めて変化した瞬間)」を検出していた。本スクリプトは **既存 npz
(STABLE確定スナップショットのみ、逐フレームCNN履歴は保持しない)** を使う
ため、同じ検出方式は使えない (npz-only制約、再走行コストを避けるための
意図的な設計判断)。代わりに:

1. **設置イベントの近似検出**: npz の連続する STABLE 行の grid 差分が
   1〜2セルかつ全て 空/UNKNOWN→有効ぷよ色(1-5) であるものを「設置らしい
   反映イベント」とみなす (近似、生CNN逐フレーム検出の代理指標)。
2. **OFF→(v2) の対応イベント特定**: OFF側イベントの grid が (v2) npz 中に
   最初 (最小 t_sec) に現れる行を bit-exact 探索する (決定論的認識なら
   いつかは同じ内容が出現するはず、`project_step0_fire_board_alignment`
   の決定論性実証を前提とする)。遅延 = (v2)のframe_idx - OFFのframe_idx。
3. **バーストWindow活性判定の近似**: v2 npz には Window ON/OFF の bit が
   残らないため (§1.1 参照)、コスト低い方の近似として
   `_diag_c_zero_effect_2026-08-04.py` の相手 chain_trigger_sec ベース窓
   再構成 (`_build_trigger_windows`、既存メソッドの再利用) を使う。
   **限界**: これは相手の「連鎖式検知」時刻からの固定窓近似であり、実際の
   視覚バーストWindow (Schmitt trigger) の開閉と厳密には一致しない
   (過大/過小評価の可能性がある、数値的な誤差は数十%程度を想定)。
   自分のお邪魔着弾直後 window (smoke layer 相当) は本スクリプトでは
   判定しない (Stage1 のスコープ外、Stage2 で対応)。
4. **§1.1 残存リスク (fallback経路) の検出は本スクリプトでは行わない**:
   TSUMO_FALL 遷移検知失敗による fallback 経路の発生自体は npz からは
   判別不能 (state machine 内部の遷移履歴が失われているため)。これは
   本測定の限界として明記し、必要なら再走行計装 (コスト高) が別途必要。

## 使い方 (npz 着弾後)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts.measure_burst_guard_reflection_delay_2026-08-05

## smokeテスト (npz着弾前でも実行可能、OFF vs OFF で遅延差ゼロを確認)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts.measure_burst_guard_reflection_delay_2026-08-05 --smoke-test
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_PURPLE, COLOR_RED,
    COLOR_UNKNOWN, COLOR_YELLOW,
)

# ファイル名にハイフンを含むため動的 import (コピペ禁止指示への対応)。
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# 設置らしい反映イベントの差分セル数 (ツモは常に2色ペア、片方だけ先に反映
# されるフレームもあり得るため 1〜2 を許容する)。
PLACEMENT_DIFF_CELLS_MIN: int = 1
PLACEMENT_DIFF_CELLS_MAX: int = 2
VALID_PUYO_COLORS: "frozenset[int]" = frozenset(
    {COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE}
)
_EMPTY_LIKE: "frozenset[int]" = frozenset({COLOR_EMPTY, COLOR_UNKNOWN})

# OFF→(v2) 対応イベント探索窓 (秒)。バーストガードの最大凍結時間
# (BURST_GATE_MAX_WINDOW_SEC=30秒) にマージンを加えた値。
CONTENT_MATCH_WINDOW_SEC: float = 40.0
# 探索窓を「center_t_sec 以降」に限定する片側窓の許容誤差 (浮動小数点誤差
# 吸収用、OFF自身の行を確実に候補に含めるため負の微小マージンを持たせる)。
_CAUSAL_EPSILON_SEC: float = 1e-6

OUTPUT_CSV_PATH: Path = Path(
    "data/verify/burst_guard_2026-08-05/reflection_delay_breakdown.csv"
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class PlacementEvent:
    """OFF npz 中の1件の「設置らしい反映イベント」。"""

    frame_idx: int
    t_sec: float
    grid: "np.ndarray"


@dataclass
class ReflectionDelayResult:
    """1イベント分の OFF→(v2) 反映遅延測定結果。"""

    video: str
    side: str
    game_idx: int
    off_frame_idx: int
    off_t_sec: float
    v2_frame_idx: "int | None"
    delay_frames: "int | None"
    in_opponent_window: bool


# =============================================================================
# 1. 設置イベント検出 (npz-only 近似)
# =============================================================================


def _diff_placement_cells(
    prev_grid: "np.ndarray", grid: "np.ndarray",
) -> bool:
    """prev_grid→grid の差分が「設置らしい」パターンか判定する。

    差分セル数が 1〜2 個、かつ全て 空/UNKNOWN→有効ぷよ色(1-5) の遷移で
    あることを要求する (連鎖・お邪魔着弾等の一斉変化を除外する近似)。
    """
    diffs = np.argwhere(prev_grid != grid)
    n = len(diffs)
    if not (PLACEMENT_DIFF_CELLS_MIN <= n <= PLACEMENT_DIFF_CELLS_MAX):
        return False
    for r, c in diffs:
        prev_v = int(prev_grid[r, c])
        new_v = int(grid[r, c])
        if prev_v not in _EMPTY_LIKE or new_v not in VALID_PUYO_COLORS:
            return False
    return True


def enumerate_placement_events(
    idx: "object", side: str, game_idx: int,
) -> list[PlacementEvent]:
    """(side, game_idx) の STABLE 行を frame_idx 昇順に走査し、設置らしい
    反映イベントを列挙する (npz-only 近似、docstring 差分1 参照)。
    """
    rows = _DIAG._side_game_row_indices(idx, side, game_idx)
    events: list[PlacementEvent] = []
    prev_grid: "np.ndarray | None" = None
    for i in rows:
        grid = idx.grids[i]
        if prev_grid is not None and _diff_placement_cells(prev_grid, grid):
            events.append(PlacementEvent(
                frame_idx=int(idx.frame_idxs[i]), t_sec=float(idx.t_secs[i]),
                grid=grid.copy(),
            ))
        prev_grid = grid
    return events


# =============================================================================
# 2. OFF→(v2) 対応イベント特定 (bit-exact 最速一致)
# =============================================================================


def find_first_occurrence(
    idx: "object", side: str, target_grid: "np.ndarray",
    center_t_sec: float, window_sec: float,
) -> "tuple[int, float] | None":
    """target_grid と bit一致する行のうち、窓内で t_sec が最小の行を返す。

    「そのcontentが最初に現れた瞬間」= 反映確定フレームの近似 (docstring 差分2)。

    2026-08-05 バグ修正: 探索窓は center_t_sec **以降** の片側窓に限定する
    (対称 ±window だと、同一盤面パターンが過去に偶然再出現していた場合に
    それを「最も時刻が近い」として誤って選び、負の delay を生む自己言及
    バグがあった。OFF vs OFF smoke test で発覚、修正済み)。
    """
    mask = (
        (idx.sides == side)
        & (idx.t_secs >= center_t_sec - _CAUSAL_EPSILON_SEC)
        & (idx.t_secs <= center_t_sec + window_sec)
    )
    cand = np.where(mask)[0]
    exact = [i for i in cand if np.array_equal(idx.grids[i], target_grid)]
    if not exact:
        return None
    best = min(exact, key=lambda i: idx.t_secs[i])
    return int(idx.frame_idxs[best]), float(idx.t_secs[best])


# =============================================================================
# 3. Window活性判定 (近似、docstring 差分3)
# =============================================================================


def _opponent_window_active_approx(
    off_idx: "object", chain_trigger_secs: "np.ndarray",
    side: str, game_idx: int, t_sec: float,
) -> bool:
    """相手 chain_trigger_sec ベースの窓再構成で近似する (既存メソッド再利用)。"""
    opp_side = "2P" if side == "1P" else "1P"
    windows = _DIAG._build_trigger_windows(
        off_idx, chain_trigger_secs, opp_side, game_idx,
        _DIAG.OPPONENT_WINDOW_PRE_MARGIN_SEC, _DIAG.OPPONENT_WINDOW_POST_SEC,
    )
    return _DIAG._time_in_any_window(t_sec, windows)


# =============================================================================
# 4. 1動画分の測定
# =============================================================================


def measure_one_video(
    video_stem: str, off_dir: Path, v2_dir: Path,
    window_sec: float = CONTENT_MATCH_WINDOW_SEC,
) -> list[ReflectionDelayResult]:
    """1動画分、1P/2P × 全 game_idx の設置イベントを検出し OFF→(v2) 遅延を測る。"""
    off_path = off_dir / f"{video_stem}.npz"
    v2_path = v2_dir / f"{video_stem}.npz"
    off_idx = _MC._load_npz_index(off_path)
    v2_idx = _MC._load_npz_index(v2_path)
    if off_idx is None or v2_idx is None:
        return []
    chain_trigger_secs = _DIAG._load_chain_trigger_secs(off_path)
    results: list[ReflectionDelayResult] = []
    for side in ("1P", "2P"):
        game_idxs = sorted(set(off_idx.game_idxs[off_idx.sides == side].tolist()))
        for game_idx in game_idxs:
            results.extend(_measure_one_side_game(
                video_stem, side, game_idx, off_idx, v2_idx,
                chain_trigger_secs, window_sec,
            ))
    return results


def _measure_one_side_game(
    video_stem: str, side: str, game_idx: int, off_idx: "object", v2_idx: "object",
    chain_trigger_secs: "np.ndarray", window_sec: float,
) -> list[ReflectionDelayResult]:
    """1 (side, game_idx) 分の設置イベント→(v2)遅延測定 (50行規約回避のため分離)。"""
    out: list[ReflectionDelayResult] = []
    for ev in enumerate_placement_events(off_idx, side, game_idx):
        match = find_first_occurrence(v2_idx, side, ev.grid, ev.t_sec, window_sec)
        v2_frame = match[0] if match else None
        delay = (v2_frame - ev.frame_idx) if match else None
        in_window = _opponent_window_active_approx(
            off_idx, chain_trigger_secs, side, game_idx, ev.t_sec,
        )
        out.append(ReflectionDelayResult(
            video=video_stem, side=side, game_idx=game_idx,
            off_frame_idx=ev.frame_idx, off_t_sec=ev.t_sec,
            v2_frame_idx=v2_frame, delay_frames=delay, in_opponent_window=in_window,
        ))
    return out


def measure_all_videos(
    video_stems: list[str], off_dir: Path, v2_dir: Path,
) -> list[ReflectionDelayResult]:
    """複数動画分をまとめて測定する (着弾済み動画のみ自動集計、未着弾は空リスト)。"""
    results: list[ReflectionDelayResult] = []
    for stem in video_stems:
        results.extend(measure_one_video(stem, off_dir, v2_dir))
    return results


# =============================================================================
# 5. 集計レポート
# =============================================================================


def _delay_distribution(values: list[int]) -> "dict[str, float | int | None]":
    """delay_frames の分布統計 (中央値/p90/最大/件数)。"""
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": len(values), "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)), "max": float(np.max(arr)),
    }


def build_delay_report(results: list[ReflectionDelayResult]) -> str:
    """side × window活性 別の反映遅延分布レポート (layer別必須と同じ層別思想)。"""
    lines = ["--- OFF→(v2) 反映遅延分布 (side × 相手連鎖窓活性 別) ---"]
    n_total = len(results)
    n_unmatched = sum(1 for r in results if r.delay_frames is None)
    lines.append(
        f"総イベント数: {n_total} 件 (うち (v2) 側で対応イベント見つからず: "
        f"{n_unmatched} 件、fail-silent回避のため明示報告)"
    )
    for side in ("1P", "2P"):
        for in_window in (True, False):
            subset = [
                r.delay_frames for r in results
                if r.side == side and r.in_opponent_window == in_window
                and r.delay_frames is not None
            ]
            stats = _delay_distribution(subset)
            label = f"{side} / 相手連鎖窓{'内' if in_window else '外'}"
            lines.append(
                f"  {label}: n={stats['n']} 中央値={stats['median']} "
                f"p90={stats['p90']} 最大={stats['max']}"
            )
    return "\n".join(lines)


def write_delay_csv(results: list[ReflectionDelayResult], out_path: Path) -> None:
    """イベント単位の内訳 CSV を出力する。"""
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video", "side", "game_idx", "off_frame_idx", "off_t_sec",
        "v2_frame_idx", "delay_frames", "in_opponent_window",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "video": r.video, "side": r.side, "game_idx": r.game_idx,
                "off_frame_idx": r.off_frame_idx, "off_t_sec": f"{r.off_t_sec:.3f}",
                "v2_frame_idx": r.v2_frame_idx if r.v2_frame_idx is not None else "unmatched",
                "delay_frames": r.delay_frames if r.delay_frames is not None else "unmatched",
                "in_opponent_window": r.in_opponent_window,
            })


# =============================================================================
# 6. smokeテスト (OFF vs OFF で遅延差ゼロを確認、npz着弾前でも実行可能)
# =============================================================================


def run_smoke_test(sample_video: "str | None" = None) -> bool:
    """OFF npz を off_dir にも v2_dir にも指定し、全イベントの delay_frames==0
    になることを確認する (自己一致の健全性確認、npz着弾前でも実行可能)。
    """
    candidates = (
        [sample_video] if sample_video
        else sorted(p.stem for p in _MC.ANCHOR_NPZ_DIR.glob("*.npz"))[:3]
    )
    all_ok = True
    for stem in candidates:
        results = measure_one_video(stem, _MC.ANCHOR_NPZ_DIR, _MC.ANCHOR_NPZ_DIR)
        bad = [r for r in results if r.delay_frames != 0]
        n = len(results)
        ok = len(bad) == 0
        all_ok = all_ok and ok
        print(
            f"[smoke] {stem}: イベント{n}件, delay!=0 の件数={len(bad)} "
            f"→ {'OK' if ok else '★NG★'}"
        )
        if not ok:
            for r in bad[:5]:
                print(f"    NG例: {r}")
    return all_ok


# =============================================================================
# 7. main
# =============================================================================


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--off-npz-dir", type=Path, default=_MC.ANCHOR_NPZ_DIR,
        help=f"OFF (アンカー) npz ディレクトリ (既定: {_MC.ANCHOR_NPZ_DIR})",
    )
    parser.add_argument(
        "--v2-npz-dir", type=Path, default=_MC.NPZ_DIR_V2,
        help=f"バーストガード v2 npz ディレクトリ (既定: {_MC.NPZ_DIR_V2})",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="OFF vs OFF で遅延差ゼロになることを確認するだけ (npz着弾前でも実行可能)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        ok = run_smoke_test()
        print(f"\n[smoke] 総合判定: {'合格' if ok else '★不合格★'}")
        return

    video_stems = sorted(p.stem for p in args.off_npz_dir.glob("*.npz"))
    landed = [
        s for s in video_stems if (args.v2_npz_dir / f"{s}.npz").exists()
    ]
    print(f"[1/2] (v2) 着弾状況: {len(landed)}/{len(video_stems)} 動画着弾済み")
    if not landed:
        print("  未着弾のため測定をスキップします (着弾後に再実行してください)。")
        return

    results = measure_all_videos(landed, args.off_npz_dir, args.v2_npz_dir)
    write_delay_csv(results, OUTPUT_CSV_PATH)
    print(f"[出力] イベント単位内訳 CSV: {OUTPUT_CSV_PATH}")
    print("\n[2/2] " + build_delay_report(results))


if __name__ == "__main__":
    main()
