"""物差し v2 (W8根治版) の候補抽出 + レビューシート生成 (2026-08-14)。

`scripts._collect_yardstick_v2_2026-08-14` が生成した npz チャンクから
層別サンプリングでレビュー対象を選び、1盤面=1PNGシート + 訂正TSV雛形 +
W8根治のためのアンカーマニフェスト (参照フレームPNG+sha256+ROI縮小画像) を
出力する。認識パイプラインはここでは呼ばない (循環なし、npzの提案値は
下書き表示のみ)。

## W8根治の要点
frame_idx は「収集と同じ実行で読んだのと同じ動画ファイル」に対してのみ有効。
本スクリプトは収集直後の同一動画ファイルへ frame_idx で再アクセスして
参照フレームPNGを保存するので、収集〜シート生成の間にファイルが変わらない
限りズレない。将来動画が再DLされた場合は、この参照フレームPNGを使った
NCC再アンカリング (scripts._reanchor_yardstick_labels_2026-08-14 と同方式)
で復旧できる。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._build_yardstick_v2_sheets_2026-08-14
"""
from __future__ import annotations

import hashlib
import importlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

_COLLECT = importlib.import_module("scripts._collect_yardstick_v2_2026-08-14")
_SHEET = importlib.import_module("scripts._build_board_label_sheets_2026-07-31")

# =============================================================================
# 定数
# =============================================================================

NPZ_DIR: Path = _COLLECT.OUT_NPZ_DIR
VIDEO_DIR: Path = _COLLECT.VIDEO_DIR
OUT_DIR: Path = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SHEETS_DIR: Path = OUT_DIR / "sheets"
ANCHORS_DIR: Path = OUT_DIR / "anchors"

SEED: int = 20260814
N_TARGET_BOARDS: int = 60
MIN_DISTINCT_VIDEOS: int = 12
MAX_PICKS_PER_VIDEO: int = 5
MIN_OJAMA_FRACTION: float = 0.5
MIN_NEAR_FULL_FRACTION: float = 0.2
NEAR_FULL_FILL_THRESHOLD: float = 0.70
MIN_FILLED_CELLS_FOR_CANDIDATE: int = 6
VISIBLE_CELLS: int = (_SHEET.BOARD_ROWS - _SHEET.HIDDEN_ROWS) * _SHEET.BOARD_COLS
TARGET_PER_SIDE: int = N_TARGET_BOARDS // 2
PHASES: tuple[str, ...] = ("序盤", "中盤", "終盤")
TARGET_PER_PHASE: int = N_TARGET_BOARDS // len(PHASES)

STD_WIDTH: int = 1920
STD_HEIGHT: int = 1080

# 手動目視で確認済みの汚染チャンク (2026-08-14): この (video_id, chunk_idx) は
# 試合間のコンボ祝福演出 (非試合画面) が数十秒続く区間に当たっており、
# 認識が「盤面らしい」形をした幻盤面を返すため候補から丸ごと除外する
# (docs/KNOWN_WEAKNESSES.md 「非試合画面の幻盤面」パターン、
# 実例: 001_c109_1P_f367962 / 006_c109_1P_f368132 / 007_c109_2P_f368084)。
EXCLUDED_CHUNKS: frozenset[tuple[str, int]] = frozenset({("c109", 1)})


@dataclass
class CandidateRow:
    """1候補盤面。"""

    video_id: str
    chunk_idx: int
    side: str
    game_idx: int
    frame_idx: int
    t_sec: float
    grid: np.ndarray
    phase: str = ""
    fill_ratio: float = 0.0
    has_ojama: bool = False


# =============================================================================
# 1. npz 読込 + 位相付与
# =============================================================================


def _load_all_rows() -> list[CandidateRow]:
    """全 npz チャンクを読み、候補行 (フィルタ・位相付与前) を返す。"""
    rows: list[CandidateRow] = []
    for f in sorted(NPZ_DIR.glob("*.npz")):
        stem_parts = f.stem.rsplit("_chunk", 1)
        chunk_idx = int(stem_parts[1]) if len(stem_parts) == 2 else 0
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        grids = np.asarray(d["grids"])
        sides = np.asarray(d["side"]).astype(str)
        fidx = np.asarray(d["frame_idx"])
        tsecs = np.asarray(d["t_sec"]).astype(float)
        gidx = np.asarray(d["game_idx"])
        vid = stem_parts[0]
        if (vid, chunk_idx) in EXCLUDED_CHUNKS:
            continue
        for i in range(len(grids)):
            rows.append(CandidateRow(
                video_id=vid, chunk_idx=chunk_idx, side=str(sides[i]),
                game_idx=int(gidx[i]), frame_idx=int(fidx[i]),
                t_sec=float(tsecs[i]), grid=grids[i].copy(),
            ))
    return rows


def _dedup_consecutive_stable(rows: list[CandidateRow]) -> list[CandidateRow]:
    """同一STABLE区間内で盤面が変わらない連続行は先頭の1件だけ残す。

    2026-08-14 是正: collect_boards_lean は connect_indicators_v2 と異なり
    連続同一STABLE区間の重複出力を除去しない (docstring上は「1回のみ出力」が
    仕様だが lean 側は未実装)。重複を残すと同一盤面が層別サンプリングで
    複数枚選ばれ、見かけの多様性を損なう (実例: c109 で3枚が実質同一瞬間)。
    """
    groups: dict[tuple, list[CandidateRow]] = {}
    for r in rows:
        groups.setdefault((r.video_id, r.chunk_idx, r.side, r.game_idx), []).append(r)
    kept: list[CandidateRow] = []
    for members in groups.values():
        members.sort(key=lambda r: r.t_sec)
        prev_grid = None
        for r in members:
            if prev_grid is not None and np.array_equal(r.grid, prev_grid):
                continue
            kept.append(r)
            prev_grid = r.grid
    return kept


def _assign_phase_and_fill(rows: list[CandidateRow]) -> None:
    """試合内相対進行率で位相を切り (確定知見、project_win_eval_regen)、

    充填率・おじゃま有無も付与する (in-place)。
    """
    groups: dict[tuple, list[CandidateRow]] = {}
    for r in rows:
        groups.setdefault((r.video_id, r.chunk_idx, r.side, r.game_idx), []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: r.t_sec)
        n = len(members)
        for rank, r in enumerate(members):
            frac = rank / max(1, n - 1) if n > 1 else 0.5
            r.phase = PHASES[0] if frac < 1 / 3 else (PHASES[1] if frac < 2 / 3 else PHASES[2])
    for r in rows:
        visible = r.grid[_SHEET.HIDDEN_ROWS:, :]
        r.fill_ratio = float((visible != _SHEET.COLOR_EMPTY).sum()) / VISIBLE_CELLS
        r.has_ojama = bool((visible == _SHEET.COLOR_OJAMA).any())


def load_candidates() -> list[CandidateRow]:
    """npz 読込 → 重複除去 → 位相/充填率/おじゃま付与 → 最低充填数フィルタ、を一括で行う。"""
    rows = _dedup_consecutive_stable(_load_all_rows())
    _assign_phase_and_fill(rows)
    filled = [
        r for r in rows
        if (r.grid[_SHEET.HIDDEN_ROWS:, :] != _SHEET.COLOR_EMPTY).sum()
        >= MIN_FILLED_CELLS_FOR_CANDIDATE
    ]
    return filled


# =============================================================================
# 2. 層別サンプリング (量的ゲート、fail-silent回避のため達成状況を必ず報告)
# =============================================================================


def _quota_unmet(r: CandidateRow, counts: dict) -> bool:
    """このcandidateが未達クォータのいずれかに貢献するか。"""
    if r.has_ojama and counts["ojama"] < counts["min_ojama"]:
        return True
    if r.fill_ratio >= NEAR_FULL_FILL_THRESHOLD and counts["full"] < counts["min_full"]:
        return True
    if counts["phase"][r.phase] < TARGET_PER_PHASE:
        return True
    if counts["side"][r.side] < TARGET_PER_SIDE:
        return True
    if r.video_id not in counts["videos_seen"] and len(counts["videos_seen"]) < MIN_DISTINCT_VIDEOS:
        return True
    return False


def _try_add(r: CandidateRow, picked: list, counts: dict) -> bool:
    """per-video上限を守って1件追加する。成功したらTrue。"""
    if counts["per_video"].get(r.video_id, 0) >= MAX_PICKS_PER_VIDEO:
        return False
    picked.append(r)
    counts["per_video"][r.video_id] = counts["per_video"].get(r.video_id, 0) + 1
    counts["phase"][r.phase] += 1
    counts["side"][r.side] += 1
    counts["videos_seen"].add(r.video_id)
    if r.has_ojama:
        counts["ojama"] += 1
    if r.fill_ratio >= NEAR_FULL_FILL_THRESHOLD:
        counts["full"] += 1
    return True


def _fill_pass(
    shuffled: list[CandidateRow], picked: list[CandidateRow], picked_ids: set[int],
    counts: dict, predicate, budget_key: "str | None", min_count: int,
) -> None:
    """1優先度パス: predicate を満たす候補だけを、budget_key が min_count に

    達するまで (or budget_key=None なら制限なく) 追加する。希少資源
    (おじゃま/満杯) を phase/side の一般クォータより先に確保するための
    ヘルパー (2026-08-14 是正: 単一優先度のgreedyだと希少資源が
    per-video上限をphase/side埋めに使い切られて未達になっていた)。
    """
    for r in shuffled:
        if len(picked) >= N_TARGET_BOARDS:
            return
        if budget_key is not None and counts[budget_key] >= min_count:
            return
        if id(r) in picked_ids or not predicate(r):
            continue
        if _try_add(r, picked, counts):
            picked_ids.add(id(r))


def stratified_sample(rows: list[CandidateRow], rng: random.Random) -> list[CandidateRow]:
    """優先度付き多段greedyで N_TARGET_BOARDS 件選ぶ。

    希少資源 (おじゃま含み・満杯/準満杯) を最優先で確保 → 一般クォータ
    (位相/side/動画分散) → 残り埋め、の順。dataclass の既定 `__eq__` は
    grid (numpy配列) を含むため `in` 演算子での重複チェックは使わない
    (配列の真偽値が曖昧というエラーになる)。id() ベースの集合で判定する。
    """
    shuffled = rows[:]
    rng.shuffle(shuffled)
    counts = {
        "per_video": {}, "phase": {p: 0 for p in PHASES}, "side": {"1P": 0, "2P": 0},
        "videos_seen": set(), "ojama": 0, "full": 0,
        "min_ojama": int(np.ceil(N_TARGET_BOARDS * MIN_OJAMA_FRACTION)),
        "min_full": int(np.ceil(N_TARGET_BOARDS * MIN_NEAR_FULL_FRACTION)),
    }
    picked: list[CandidateRow] = []
    picked_ids: set[int] = set()
    _fill_pass(shuffled, picked, picked_ids, counts, lambda r: r.has_ojama, "ojama", counts["min_ojama"])
    _fill_pass(
        shuffled, picked, picked_ids, counts,
        lambda r: r.fill_ratio >= NEAR_FULL_FILL_THRESHOLD, "full", counts["min_full"],
    )
    _fill_pass(shuffled, picked, picked_ids, counts, lambda r: _quota_unmet(r, counts), None, 0)
    _fill_pass(shuffled, picked, picked_ids, counts, lambda r: True, None, 0)
    return picked


# =============================================================================
# 3. アンカー生成 (参照フレームPNG + sha256 + ROI縮小シグネチャ)
# =============================================================================


def _read_native_frame(cap: cv2.VideoCapture, frame_idx: int) -> "np.ndarray | None":
    """frame_idx の生フレームを標準解像度 (1920x1080) へ正規化して返す。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def _sha256_of_png_bytes(img: np.ndarray) -> tuple[str, bytes]:
    """画像を PNG エンコードし (sha256, pngバイト列) を返す。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNGエンコード失敗")
    data = bytes(buf)
    return hashlib.sha256(data).hexdigest(), data


def build_anchor(cap: cv2.VideoCapture, video_filename: str, r: CandidateRow, sheet_id: str) -> dict:
    """1候補のW8アンカー3点セットを保存し、マニフェスト用dictを返す。"""
    frame = _read_native_frame(cap, r.frame_idx)
    if frame is None:
        return {"sheet_id": sheet_id, "anchor_status": "frame_read_failed"}
    frame_sha, frame_png = _sha256_of_png_bytes(frame)
    frame_path = ANCHORS_DIR / f"{sheet_id}_frame.png"
    frame_path.write_bytes(frame_png)
    reg = _SHEET.DEFAULT_P1_REGION if r.side == "1P" else _SHEET.DEFAULT_P2_REGION
    roi = frame[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
    roi_sha, roi_png = _sha256_of_png_bytes(roi)
    roi_path = ANCHORS_DIR / f"{sheet_id}_roi.png"
    roi_path.write_bytes(roi_png)
    return {
        "sheet_id": sheet_id, "anchor_status": "ok",
        "video_filename": video_filename, "frame_idx": r.frame_idx,
        "t_sec_at_collection": round(r.t_sec, 3),
        "reference_frame_png": str(frame_path.relative_to(_ROOT)),
        "reference_frame_sha256": frame_sha,
        "board_roi_png": str(roi_path.relative_to(_ROOT)),
        "board_roi_sha256": roi_sha,
        # review_app.html (採点用HTML) の下書きプリセット用。認識提案であり
        # 正解ではない (下流の全ての採点UIで明示すること)。
        "init_grid": r.grid.tolist(),
        "phase": r.phase, "fill_ratio": round(r.fill_ratio, 4), "has_ojama": r.has_ojama,
        "video_id": r.video_id, "side": r.side, "t_sec": round(r.t_sec, 3),
    }


# =============================================================================
# 4. main
# =============================================================================


def _video_filename_for(video_id: str) -> str:
    """video_id (例: "c96") から実物ファイル名を復元する。"""
    for fn in _COLLECT.VIDEO_FILENAMES:
        if _COLLECT.video_id_of(fn) == video_id:
            return fn
    raise KeyError(video_id)


def _report_quota_status(picked: list[CandidateRow]) -> None:
    """量的ゲートの達成状況を明示的に報告する (fail-silent回避)。"""
    n = len(picked)
    n_videos = len({r.video_id for r in picked})
    n_ojama = sum(1 for r in picked if r.has_ojama)
    n_full = sum(1 for r in picked if r.fill_ratio >= NEAR_FULL_FILL_THRESHOLD)
    phase_counts = {p: sum(1 for r in picked if r.phase == p) for p in PHASES}
    side_counts = {s: sum(1 for r in picked if r.side == s) for s in ("1P", "2P")}
    print(f"選定 {n}枚 / 動画{n_videos}本 (目標>={MIN_DISTINCT_VIDEOS})")
    print(f"おじゃま有り {n_ojama}/{n} ({n_ojama / max(1, n):.1%}, 目標>={MIN_OJAMA_FRACTION:.0%})")
    print(f"満杯/準満杯(fill>={NEAR_FULL_FILL_THRESHOLD}) {n_full}/{n} "
          f"({n_full / max(1, n):.1%}, 目標>={MIN_NEAR_FULL_FRACTION:.0%})")
    print(f"位相分布: {phase_counts} (目標各{TARGET_PER_PHASE})")
    print(f"side分布: {side_counts} (目標各{TARGET_PER_SIDE})")


def _write_labels_tsv(rows_out: list[str]) -> Path:
    """訂正記入用TSVを書き出す。"""
    tsv = OUT_DIR / "labels.tsv"
    tsv.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    return tsv


def main() -> None:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] npz読込+位相/充填率付与...")
    candidates = load_candidates()
    print(f"  候補 {len(candidates)} 件")
    if not candidates:
        print("候補が無い (収集がまだ終わっていない可能性)")
        return
    print("[2/4] 層別サンプリング...")
    picked = stratified_sample(candidates, random.Random(SEED))
    _report_quota_status(picked)
    print("[3/4] シート+アンカー生成...")
    manifest: list[dict] = []
    rows_out = [
        "# 誤っているセルだけ wrong_cells に記入してください (例: r3c2=1,r5c0=0)。"
        "全部正しければ ok と書いてください。",
        "# r は画面内の行 (r1=最上段, r12=最下段)、c は列 (c0=左端, c5=右端)。",
        "# 色コード: 0=空 1=赤 2=青 3=緑 4=黄 5=紫 9=おじゃま",
        "# 左の実画面がそもそも対局中の盤面でない (結果演出/メニュー等) 場合は"
        " wrong_cells に NOT_A_BOARD と書いてください (色の正誤判定はしない)。",
        "sheet\tvideo\tside\tframe_idx_aux\twrong_cells",
    ]
    caps: dict[str, cv2.VideoCapture] = {}
    for i, r in enumerate(picked):
        sheet_id = f"{i:03d}_{r.video_id}_{r.side}_f{r.frame_idx}"
        vfn = _video_filename_for(r.video_id)
        cap = caps.setdefault(vfn, cv2.VideoCapture(str(VIDEO_DIR / vfn)))
        anchor = build_anchor(cap, vfn, r, sheet_id)
        manifest.append(anchor)
        if anchor["anchor_status"] != "ok":
            print(f"  [警告] {sheet_id}: アンカー生成失敗、シート生成をスキップ")
            continue
        frame = _read_native_frame(cap, r.frame_idx)
        sheet = _SHEET._compose(_SHEET._crop_board(frame, r.side), _SHEET._render_grid(r.grid))
        cv2.imwrite(str(SHEETS_DIR / f"{sheet_id}.png"), sheet)
        rows_out.append(f"{sheet_id}\t{r.video_id}\t{r.side}\t{r.frame_idx}\t")
    for cap in caps.values():
        cap.release()
    print("[4/4] マニフェスト+TSV書き出し...")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    tsv = _write_labels_tsv(rows_out)
    print(f"完了: シート{len(manifest)}枚 → {SHEETS_DIR}")
    print(f"マニフェスト → {OUT_DIR / 'manifest.json'}")
    print(f"ラベル記入用 → {tsv}")


if __name__ == "__main__":
    main()
