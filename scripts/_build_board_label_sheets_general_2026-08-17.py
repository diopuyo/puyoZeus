"""一般分布ラベルセット (55盤面) のラベリングシート生成 (ラベル規約v2、2026-08-17)。

## 目的
構成F (本番採用構成) の「一般分布」正解率測定用に、難所条件で選別しない
無作為・層別 (動画×試合内進行率3分位×1P/2P) サンプル55盤面のラベリング
シートを生成する。アンカー計画は `scripts._select_general_yardstick_anchors_
2026-08-17` が生成した `anchor_plan.tsv` + 構成F収集済み npz を使う。

## ラベル規約v2 (W8根治、`docs/LABEL_ANCHOR_SPEC_2026-08-17.md`)
各盤面の主キーは「参照画像の平均ハッシュ+周辺±2フレームの縮小シグネチャ」
(`scripts._label_anchor_lib_2026-08-17`)。`video`/`frame_idx`/`t_sec` は
補助キー (参考値) として記録するが、突合の主経路には使わない。

## 既存資産の再利用 (コピペ禁止指示への対応)
- 実画面クロップ/認識グリッド描画/左右結合: `_build_board_label_sheets_
  2026-07-31.py` の `_crop_board`/`_render_grid`/`_compose` を直接呼ぶ。
- npz 読込: `measure_effect_gate_c_2026-08-04.py` の `_load_npz_index`。
- 主キー計算: `_label_anchor_lib_2026-08-17.py`。

## 使い方 (WSL、動画ファイルが要るため)
    PYTHONPATH=. ./venv/bin/python -m scripts._build_board_label_sheets_general_2026-08-17
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.production_config import collect_flags  # noqa: E402

_SHEETS = importlib.import_module("scripts._build_board_label_sheets_2026-07-31")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_LIB = importlib.import_module("scripts._label_anchor_lib_2026-08-17")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

ANCHOR_PLAN_TSV: Path = Path("data/verify/board_labels_general_2026-08-17/anchor_plan.tsv")
VIDEO_DIR_WSL: Path = Path.home() / "frames"
OUT_DIR: Path = Path("data/verify/board_labels_general_2026-08-17")
SHEETS_DIR: Path = OUT_DIR / "sheets"
ANCHORS_DIR: Path = OUT_DIR / "anchors"
LABELS_TSV: Path = OUT_DIR / "labels.tsv"
README_PATH: Path = OUT_DIR / "README.md"

SIDES: "tuple[str, ...]" = ("1P", "2P")
# アンカー時刻からSTABLE snapshotを探す許容窓 (秒)。試合中はSTABLEが
# 頻繁に発生するはずなので3秒あれば十分 (無ければ「無し」と明示して除外)。
NEAREST_TOLERANCE_SEC: float = 3.0

# 動画1本あたり6候補 (3分位×2side) から2つを間引き、4候補/動画×14動画=56 → 55に調整。
DROP_PER_VIDEO: int = 2
TOTAL_TARGET: int = 55
# 「空盤面」判定 (row1-12、72セル中いくつ以上埋まっていれば空でないか)
EMPTY_FILL_MAX: int = 3
MAX_EMPTY_BOARDS: int = 10

BASE_CONFIG_LABEL: str = f"構成F (2026-08-17収集): {collect_flags()}"


@dataclass
class Candidate:
    """1盤面分の候補 (最終55盤面への採否は後段で決める)。"""

    video: str
    game_abs_idx: int
    tertile: str
    side: str
    npz_path: Path
    frame_idx: int
    t_sec: float
    grid: np.ndarray
    fill_count: int
    slot: int  # 0..5 (tertile×side の位置、間引きの決定論的パターン用)


# =============================================================================
# 1. アンカー計画の読込 + 候補構築
# =============================================================================


def _fill_count(grid: np.ndarray) -> int:
    """row1-12 (row0隠し段除く) の非空セル数。"""
    return int((grid[1:, :] != 0).sum())


def load_candidates() -> "list[Candidate]":
    """anchor_plan.tsv + npz から候補 (最大14動画×6=84件) を構築する。"""
    rows = list(csv.DictReader(ANCHOR_PLAN_TSV.open(encoding="utf-8"), delimiter="\t"))
    by_npz: "dict[str, list[dict]]" = {}
    for r in rows:
        by_npz.setdefault(r["npz_path"], []).append(r)
    cands: "list[Candidate]" = []
    tertile_order = {"early": 0, "mid": 1, "late": 2}
    for npz_rel, group in by_npz.items():
        idx = _MC._load_npz_index(Path(npz_rel))
        if idx is None:
            print(f"[skip] npz未着弾: {npz_rel}")
            continue
        for r in group:
            for side in SIDES:
                mask = idx.sides == side
                cand_i = np.where(mask)[0]
                if len(cand_i) == 0:
                    continue
                dt = np.abs(idx.t_secs[cand_i] - float(r["anchor_t_sec"]))
                best = int(cand_i[int(np.argmin(dt))])
                if float(dt[int(np.argmin(dt))]) > NEAREST_TOLERANCE_SEC:
                    print(f"[skip] STABLE無し: {r['video']} {side} t={r['anchor_t_sec']}")
                    continue
                grid = idx.grids[best]
                slot = tertile_order[r["tertile"]] * 2 + (0 if side == "1P" else 1)
                cands.append(Candidate(
                    r["video"], int(r["game_abs_idx"]), r["tertile"], side,
                    Path(npz_rel), int(idx.frame_idxs[best]), float(idx.t_secs[best]),
                    grid, _fill_count(grid), slot,
                ))
    return cands


# =============================================================================
# 2. 最終55盤面の選定 (層別間引き + 空盤面キャップ)
# =============================================================================


def select_final_55(cands: "list[Candidate]") -> "list[Candidate]":
    """動画ごとに決定論的に2件間引いて56件にし、空盤面過多なら差し替える。"""
    by_video: "dict[str, list[Candidate]]" = {}
    for c in cands:
        by_video.setdefault(c.video, []).append(c)
    kept: "list[Candidate]" = []
    spare: "dict[str, list[Candidate]]" = {}
    for i, (video, group) in enumerate(sorted(by_video.items())):
        drop_slots = {(i * DROP_PER_VIDEO + k) % 6 for k in range(DROP_PER_VIDEO)}
        keep = [c for c in group if c.slot not in drop_slots]
        spare[video] = [c for c in group if c.slot in drop_slots]
        kept.extend(keep)
    kept.sort(key=lambda c: (c.video, c.tertile, c.side))
    kept = kept[:TOTAL_TARGET] if len(kept) > TOTAL_TARGET else kept
    n_empty = sum(1 for c in kept if c.fill_count <= EMPTY_FILL_MAX)
    if n_empty > MAX_EMPTY_BOARDS:
        kept = _swap_out_excess_empty(kept, spare, n_empty - MAX_EMPTY_BOARDS)
    return kept


def _swap_out_excess_empty(
    kept: "list[Candidate]", spare: "dict[str, list[Candidate]]", n_to_swap: int,
) -> "list[Candidate]":
    """空盤面 (fill_count<=EMPTY_FILL_MAX) を、同一動画の間引き候補 (非空優先) に差し替える。"""
    out = list(kept)
    swapped = 0
    for i, c in enumerate(out):
        if swapped >= n_to_swap:
            break
        if c.fill_count > EMPTY_FILL_MAX:
            continue
        alt_pool = [a for a in spare.get(c.video, []) if a.fill_count > EMPTY_FILL_MAX]
        if not alt_pool:
            continue
        out[i] = alt_pool[0]
        spare[c.video].remove(alt_pool[0])
        swapped += 1
    if swapped < n_to_swap:
        print(f"[warn] 空盤面キャップ未達成: {n_to_swap - swapped}件は差し替え候補が無かった")
    return out


# =============================================================================
# 3. シート生成 (実画面クロップ+認識グリッド+主キー保存)
# =============================================================================


def _video_path(video: str) -> Path:
    stem = video if video.startswith("video_") else f"video_{video}"
    return VIDEO_DIR_WSL / f"{stem}.mp4"


def _region(side: str) -> "tuple[int, int, int, int]":
    return _LIB._RA.region_for_side(side)


def build_sheet(c: Candidate, sheet_idx: int) -> "list[str] | None":
    """1候補分のPNG+シグネチャを生成し、labels.tsv用の行を返す (失敗時None)。"""
    vpath = _video_path(c.video)
    if not vpath.exists():
        print(f"[skip] 動画が無い: {vpath}")
        return None
    frame = _LIB.read_frame_at(vpath, c.frame_idx)
    if frame is None:
        print(f"[skip] フレーム読込失敗: {c.video} f{c.frame_idx}")
        return None
    sheet = _SHEETS._compose(_SHEETS._crop_board(frame, c.side), _SHEETS._render_grid(c.grid))
    name = f"{sheet_idx:03d}_{c.video}_{c.side}_f{c.frame_idx}"
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS_DIR / f"{name}.png"), sheet)
    key = _LIB.build_anchor_key(vpath, c.frame_idx, _region(c.side))
    sig_path = ANCHORS_DIR / f"{name}.npz"
    _LIB.save_anchor_sidecar(sig_path, key, candidate_grid=c.grid)
    return [
        f"{name}.png", key.hash_hex, str(sig_path), c.video, c.side, str(c.frame_idx),
        f"{c.t_sec:.3f}", c.tertile, str(c.fill_count), BASE_CONFIG_LABEL, "",
    ]


# =============================================================================
# 4. README生成 (userがすぐ作業を始められる状態にする)
# =============================================================================


def write_readme(n_made: int, n_target: int, per_video: "dict[str, int]") -> None:
    """記入手順+動画別内訳をREADMEとして書き出す。"""
    breakdown = "\n".join(f"- {v}: {n}盤面" for v, n in sorted(per_video.items()))
    README_PATH.write_text(
        "# 一般分布ラベルセット (55盤面、2026-08-17)\n\n"
        f"生成 {n_made}/{n_target} 盤面。`labels.tsv` に記入してください。\n\n"
        "## 記入方法\n"
        "1. `sheets/` 内の各PNGを開く (左=実画面クロップ、右=認識グリッド)。\n"
        "2. **判断根拠は必ず左側の実画面**。右側は構成Fの下書きなので、"
        "一致していても盲目的に信用しないこと。\n"
        "3. 誤っているセルがあれば `labels.tsv` の該当行 `wrong_cells` 列に "
        "`r3c2=1,r5c0=0` 形式で記入。全部正しければ `ok` と記入。\n"
        "4. 行 (r) は画面内の行 (r1=最上段, r12=最下段)、列 (c) は列 (c0=左端, c5=右端)。\n"
        "5. 色コード: 0=空 1=赤 2=青 3=緑 4=黄 5=紫 9=おじゃま。\n\n"
        "## 動画別内訳\n" + breakdown + "\n\n"
        "## 補足\n"
        "本セットは難所条件で選別していない一般分布サンプル (無作為・層別: "
        "動画×試合内進行率3分位×1P/2P)。`docs/LABEL_ANCHOR_SPEC_2026-08-17.md` "
        "のラベル規約v2に準拠し、`video`/`frame_idx`/`t_sec` は参考値、"
        "`anchor_hash`/`anchor_sig_path` が再アンカリング可能な主キー。\n",
        encoding="utf-8",
    )


# =============================================================================
# 5. main
# =============================================================================


def main() -> None:
    cands = load_candidates()
    print(f"[1/3] 候補: {len(cands)}件")
    final = select_final_55(cands)
    print(f"[2/3] 最終選定: {len(final)}盤面 "
          f"(空盤面={sum(1 for c in final if c.fill_count <= EMPTY_FILL_MAX)}件)")
    header = [
        "# 誤っているセルだけ wrong_cells に記入してください "
        "(例: r3c2=1,r5c0=0)。全部正しければ ok と書いてください。",
        "# 判断根拠は必ずシートPNG左側の実画面クロップにすること。右側のグリッドは"
        " 構成Fの下書きで、盲目的に一致確認しないこと (docs/LABEL_ANCHOR_SPEC_2026-08-17.md)。",
        "# r は画面内の行 (r1=最上段, r12=最下段)、c は列 (c0=左端, c5=右端)。",
        "# 色コード: 0=空 1=赤 2=青 3=緑 4=黄 5=紫 9=おじゃま",
        "sheet\tanchor_hash\tanchor_sig_path\tvideo\tside\tframe_idx\tt_sec\ttertile\tfill_count\tbase_config\twrong_cells",
    ]
    rows_out = list(header)
    made = 0
    per_video: "dict[str, int]" = {}
    for i, c in enumerate(final):
        row = build_sheet(c, i)
        if row is not None:
            rows_out.append("\t".join(row))
            made += 1
            per_video[c.video] = per_video.get(c.video, 0) + 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_TSV.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    write_readme(made, len(final), per_video)
    print(f"[3/3] 生成 {made}/{len(final)} 枚 -> {SHEETS_DIR}")
    print(f"ラベル記入用: {LABELS_TSV}")
    print(f"README: {README_PATH}")


if __name__ == "__main__":
    main()
