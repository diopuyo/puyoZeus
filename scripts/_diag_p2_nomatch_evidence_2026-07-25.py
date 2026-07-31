"""P2誤色 NO_MATCH 事例の実画面エビデンス画像生成 (read-only、ピンポイントseek専用)。

背景: 色フリッカの真犯人 P2 (recognition_pipeline.infer_placement による着地推論)
の誤色について、write_trace クロス集計 (scripts._diag_confirmed_write_trace_2026-07-25
が出力した data/verify/write_trace/30_*.jsonl) を再解析すると、訂正後の正しい色が
NEXT queue 記録 (prev_queue_tail/post_queue_tail 直近3件) のどこにも一致しない
「NO_MATCH」ケースが最多 (video30 で 73件中36件)。本スクリプトは、NO_MATCH 代表例と
比較用の一致ケース (2区分) について、P2書き込み時点/訂正時点それぞれの実画面全体
(盤面+NEXT表示が両方写る範囲) を1枚のmontage画像にまとめ、ユーザーが目視で
「NextDetector読み値の誤りか、別経路か」を判定できる材料を出力する。

read-only 原則: src/ は一切変更しない。動画は cv2.VideoCapture の
CAP_PROP_POS_FRAMES によるピンポイントseekのみで読み込み、フル走行 (認識
パイプライン実行) は行わない (write_trace jsonl は既存データを再利用するのみ)。

一致カテゴリの呼称について:
  ユーザー依頼文中の「A2確定」「逆ラグ」という呼称の内部定義はこのセッションの
  contextに無いため、実データで観測された非NO_MATCHの2大区分
  (prev[-3]+post[-3] 一致 / prev[-1]+post[-1] 一致) にそのまま対応付けて
  出力する (banner には実測の match label をそのまま表示し、呼称の当てずっぽうを
  避ける)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_p2_nomatch_evidence_2026-07-25
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.board import COLOR_OJAMA, COLOR_UNKNOWN  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from scripts.physics_violation_audit import (  # noqa: E402
    _draw_cell_highlight_boxes, _read_frame_at,
)

# ============================
# 定数
# ============================
VIDEO_STEM: str = "30"
WRITE_TRACE_DIR: Path = PROJ_ROOT / "data" / "verify" / "write_trace"
VIDEO_PATH: Path = PROJ_ROOT / "data" / "frames" / f"video_{VIDEO_STEM}.mp4"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "p2_nomatch_evidence_2026-07-25"

# write_trace 内の経路ID (scripts._diag_confirmed_write_trace_2026-07-25 が
# 出力するタグと同一文字列。モジュール名にハイフンを含み `import` 文で
# 直接参照できないため、値をここで再定義する)。
ROUTE_P2_INFER_PLACEMENT: str = "P2_infer_placement"
ROUTE_P2_DIAG_QUEUE_CONTEXT: str = "P2_diag_queue_context"

# 判定カテゴリ計算で参照する queue tail 位置 (負のindex, ラベル)。
PREV_TAIL_POSITIONS: tuple[tuple[int, str], ...] = (
    (-3, "prev[-3]"), (-2, "prev[-2]"), (-1, "prev[-1]"),
)
POST_TAIL_POSITIONS: tuple[tuple[int, str], ...] = (
    (-3, "post[-3]"), (-2, "post[-2]"), (-1, "post[-1]"),
)
MATCH_LABEL_NO_DIAG_CONTEXT: str = "NO_DIAG_CONTEXT"
MATCH_LABEL_UNKNOWN_INVOLVED: str = "UNKNOWN_INVOLVED"
MATCH_LABEL_NO_MATCH: str = "NO_MATCH_ANYWHERE"

# 代表例の抽出件数。
N_NOMATCH_SAMPLES: int = 8
N_COMPARISON_PER_GROUP: int = 2
MATCH_LABEL_GROUP_FAR: str = "prev[-3]+post[-3]"   # 依頼文中「A2確定」相当と解釈
MATCH_LABEL_GROUP_NEAR: str = "prev[-1]+post[-1]"  # 依頼文中「逆ラグ」相当と解釈

# 色コード → 短縮記号 (banner表示用)。
COLOR_SYMBOL: dict[int, str] = {
    0: "EMPTY", 1: "RED", 2: "BLUE", 3: "GREEN", 4: "YELLOW", 5: "PURPLE",
    COLOR_OJAMA: "OJAMA", COLOR_UNKNOWN: "UNKNOWN",
}

# montage 描画定数。
BANNER_HEIGHT_PX: int = 190
BANNER_LINE_HEIGHT_PX: int = 26
BANNER_FONT = cv2.FONT_HERSHEY_DUPLEX
BANNER_FONT_SCALE: float = 0.62
BANNER_FONT_THICKNESS: int = 1
BANNER_TEXT_COLOR: tuple[int, int, int] = (0, 255, 255)
PANEL_LABEL_COLOR: tuple[int, int, int] = (0, 255, 0)


@dataclass
class P2Event:
    """1件のP2誤色書き込み〜訂正イベント (montage生成に必要な情報一式)。"""

    side: str
    p2_frame: int
    correction_frame: int
    cells: list[tuple[int, int]]      # 着地セル (row, col) のみ (赤枠描画用)
    wrong_colors: list[int]           # P2書き込み時の色 (誤色ペア)
    final_colors: list[int]           # 訂正後の色 (正色ペア)
    falling_pair: list[int] | None
    prev_tail: list[list[int]] | None
    match_label: str


def _load_events(side: str) -> list[dict]:
    """write_trace jsonl (1 side分) を読み込む。"""
    path = WRITE_TRACE_DIR / f"{VIDEO_STEM}_{side}.jsonl"
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _build_by_frame(events: list[dict]) -> dict[int, list[dict]]:
    """frame_idx をキーにしたイベント辞書を作る。"""
    by_frame: dict[int, list[dict]] = {}
    for e in events:
        by_frame.setdefault(e["frame_idx"], []).append(e)
    return by_frame


def _find_p2_for_violation(
    by_frame: dict[int, list[dict]], f_frame: int, p_frame: int, route: str,
) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
    """1件の color_flicker 違反 (訂正フレームF) の元になった P2 書き込みを探す。

    scratchpad/p2_queue_crosstab2.py の find_p2_for_violation と同一ロジック
    (訂正セルの before 値と一致する直前 P2 書き込みの after 値を突合する)。
    """
    corr_events = [e for e in by_frame.get(f_frame, []) if e["route_id"] == route]
    if not corr_events:
        corr_events = by_frame.get(f_frame, [])
    corrected_cells: list[tuple[int, int, int, int]] = []
    for ce in corr_events:
        corrected_cells.extend(tuple(cell) for cell in ce.get("cells", []))
    if not corrected_cells:
        return [], corrected_cells

    prev_events = by_frame.get(p_frame, [])
    matched: list[dict] = []
    seen: set[int] = set()
    for (r, c, before, _after) in corrected_cells:
        for pe in prev_events:
            if pe["route_id"] != ROUTE_P2_INFER_PLACEMENT or id(pe) in seen:
                continue
            for (pr, pc, _pbefore, pafter) in pe.get("cells", []):
                if pr == r and pc == c and pafter == before:
                    seen.add(id(pe))
                    matched.append(pe)
    return matched, corrected_cells


def _compute_match_label(
    final_colors: list[int], wrong_colors: list[int], diag: dict | None,
) -> str:
    """訂正後の色ペアが queue tail (prev/post 直近3件) のどこに一致するか判定する。"""
    has_unknown = COLOR_UNKNOWN in final_colors or COLOR_UNKNOWN in wrong_colors
    if diag is None:
        return MATCH_LABEL_NO_DIAG_CONTEXT
    if has_unknown:
        return MATCH_LABEL_UNKNOWN_INVOLVED
    prev_tail, post_tail = diag["meta"]["prev_queue_tail"], diag["meta"]["post_queue_tail"]
    target = tuple(sorted(final_colors))
    matched_names: list[str] = []
    for i, name in PREV_TAIL_POSITIONS:
        if len(prev_tail) >= abs(i) and tuple(sorted(prev_tail[i])) == target:
            matched_names.append(name)
    for i, name in POST_TAIL_POSITIONS:
        if len(post_tail) >= abs(i) and tuple(sorted(post_tail[i])) == target:
            matched_names.append(name)
    return "+".join(matched_names) if matched_names else MATCH_LABEL_NO_MATCH


def _build_records_for_side(
    side: str, by_frame: dict[int, list[dict]], flicker_rows: list[dict],
) -> list[P2Event]:
    """1 side分の color_flicker 違反から P2Event 一覧を組み立てる。"""
    records: list[P2Event] = []
    for row in flicker_rows:
        if row["side"] != side:
            continue
        f_frame, p_frame, route = row["frame_idx"], row["prev_frame_idx"], row["attributed_route"]
        p2_evs, corrected_cells = _find_p2_for_violation(by_frame, f_frame, p_frame, route)
        if not p2_evs:
            continue
        corr_map = {(r, c): after for (r, c, _b, after) in corrected_cells}
        for pe in p2_evs:
            p2_cells = pe["cells"]
            final_colors = [
                corr_map.get((r, c), after) for (r, c, _b, after) in p2_cells
            ]
            wrong_colors = [after for (_r, _c, _b, after) in p2_cells]
            cells = [(r, c) for (r, c, _b, _a) in p2_cells]
            diag = next(
                (e for e in by_frame.get(pe["frame_idx"], [])
                 if e["route_id"] == ROUTE_P2_DIAG_QUEUE_CONTEXT),
                None,
            )
            records.append(P2Event(
                side=side, p2_frame=pe["frame_idx"], correction_frame=f_frame,
                cells=cells, wrong_colors=wrong_colors, final_colors=final_colors,
                falling_pair=pe["meta"].get("falling_pair"),
                prev_tail=diag["meta"]["prev_queue_tail"] if diag else None,
                match_label=_compute_match_label(final_colors, wrong_colors, diag),
            ))
    return records


def _build_all_records() -> list[P2Event]:
    """video30 の 1P/2P 両方について P2Event 一覧を組み立てる。"""
    xt = json.loads((WRITE_TRACE_DIR / f"{VIDEO_STEM}_crosstab_summary.json").read_text(encoding="utf-8"))
    flicker_rows = [r for r in xt["detail_rows"] if r["type"] == "color_flicker"]
    records: list[P2Event] = []
    for side in ("1P", "2P"):
        by_frame = _build_by_frame(_load_events(side))
        records.extend(_build_records_for_side(side, by_frame, flicker_rows))
    return records


def _evenly_spaced_indices(n_total: int, n_pick: int) -> list[int]:
    """0..n_total-1 から n_pick 件を均等間隔で (重複無く) 選ぶ index 一覧を返す。"""
    if n_total <= n_pick:
        return list(range(n_total))
    return sorted(set(int(round(i)) for i in np.linspace(0, n_total - 1, n_pick)))


def _select_candidates(records: list[P2Event]) -> list[tuple[str, P2Event]]:
    """NO_MATCH代表8件 + 一致比較2区分x2件を時系列均等サンプリングで選ぶ。"""
    groups = {
        "nomatch": (MATCH_LABEL_NO_MATCH, N_NOMATCH_SAMPLES),
        "matchfar": (MATCH_LABEL_GROUP_FAR, N_COMPARISON_PER_GROUP),
        "matchnear": (MATCH_LABEL_GROUP_NEAR, N_COMPARISON_PER_GROUP),
    }
    selected: list[tuple[str, P2Event]] = []
    for group_name, (label, n_pick) in groups.items():
        pool = sorted(
            (r for r in records if r.match_label == label), key=lambda r: r.p2_frame,
        )
        for idx in _evenly_spaced_indices(len(pool), n_pick):
            selected.append((group_name, pool[idx]))
    return selected


def _draw_landing_boxes(frame: np.ndarray, side: str, cells: list[tuple[int, int]]) -> None:
    """該当sideの着地セルに赤枠を描画する (in-place、frame全体座標系)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    _draw_cell_highlight_boxes(frame, region, cells, x1=0, y1=0)


def _format_color_pair(colors: list[int]) -> str:
    """色ペアを 'RED+BLUE(1,2)' のような ASCII 表記にする。"""
    names = "+".join(COLOR_SYMBOL.get(c, str(c)) for c in colors)
    raw = ",".join(str(c) for c in colors)
    return f"{names}({raw})"


def _format_queue_tail(tail: list[list[int]] | None) -> str:
    """queue tail (直近3手) を ASCII 表記にする。"""
    if tail is None:
        return "N/A"
    return " ".join(f"[{_format_color_pair(pair)}]" for pair in tail)


def _banner_lines(group_name: str, ev: P2Event) -> list[str]:
    """banner に描画する行一覧 (ASCII のみ、CJKグリフ問題回避) を組み立てる。"""
    return [
        f"P2 COLOR-FLICKER EVIDENCE  video=30  group={group_name}  match_category={ev.match_label}",
        f"side={ev.side}  P2_write_frame={ev.p2_frame}  correction_frame={ev.correction_frame}",
        f"LEFT = P2 write moment (wrong color committed)   RIGHT = correction moment",
        f"wrong_color_pair(P2 write)  = {_format_color_pair(ev.wrong_colors)}",
        f"correct_color_pair(final)  = {_format_color_pair(ev.final_colors)}",
        f"falling_pair(from P2 meta) = {_format_color_pair(ev.falling_pair) if ev.falling_pair else 'N/A'}",
        f"queue_prev_tail(newest 3, oldest to newest) = {_format_queue_tail(ev.prev_tail)}",
    ]


def _build_banner(group_name: str, ev: P2Event, width: int) -> np.ndarray:
    """montage上部の情報banner画像を作る (黒背景+ASCII文字)。"""
    banner = np.zeros((BANNER_HEIGHT_PX, width, 3), dtype=np.uint8)
    for i, line in enumerate(_banner_lines(group_name, ev)):
        y = 24 + i * BANNER_LINE_HEIGHT_PX
        cv2.putText(
            banner, line, (10, y), BANNER_FONT, BANNER_FONT_SCALE,
            BANNER_TEXT_COLOR, BANNER_FONT_THICKNESS, cv2.LINE_AA,
        )
    return banner


def _label_panel(panel: np.ndarray, text: str) -> None:
    """各パネル左上に簡潔ラベルを描画する (ASCII)。"""
    cv2.putText(
        panel, text, (10, 34), BANNER_FONT, 0.8, PANEL_LABEL_COLOR, 2, cv2.LINE_AA,
    )


def _build_montage(cap: cv2.VideoCapture, group_name: str, ev: P2Event) -> np.ndarray | None:
    """1件分のmontage (banner + 左:P2書込フレーム全体 + 右:訂正フレーム全体) を作る。"""
    left = _read_frame_at(cap, ev.p2_frame)
    right = _read_frame_at(cap, ev.correction_frame)
    if left is None or right is None:
        return None
    _draw_landing_boxes(left, ev.side, ev.cells)
    _label_panel(left, f"P2 WRITE frame={ev.p2_frame}")
    _label_panel(right, f"CORRECTION frame={ev.correction_frame}")
    both = cv2.hconcat([left, right])
    banner = _build_banner(group_name, ev, both.shape[1])
    return cv2.vconcat([banner, both])


def _output_filename(group_name: str, seq: int, ev: P2Event) -> str:
    """出力ファイル名を組み立てる (group/連番/side/frameで一意)。"""
    return f"p2_{group_name}_{seq:02d}_{ev.side}_f{ev.p2_frame}.png"


def main() -> None:
    """NO_MATCH代表 + 比較2区分のmontage画像を生成する (read-only)。"""
    cv2.setNumThreads(1)  # 熱対策 (feedback_thermal_safety_mandatory 準拠)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _build_all_records()
    selected = _select_candidates(records)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {VIDEO_PATH}", file=sys.stderr)
        return

    seq_by_group: dict[str, int] = {}
    generated: list[str] = []
    for group_name, ev in selected:
        seq_by_group[group_name] = seq_by_group.get(group_name, 0) + 1
        montage = _build_montage(cap, group_name, ev)
        if montage is None:
            print(f"[WARN] frame読込失敗: side={ev.side} p2_frame={ev.p2_frame}")
            continue
        fname = _output_filename(group_name, seq_by_group[group_name], ev)
        out_path = OUTPUT_DIR / fname
        cv2.imwrite(str(out_path), montage)
        generated.append(str(out_path))
        print(f"[OK] {fname}  match={ev.match_label}  wrong={ev.wrong_colors}  final={ev.final_colors}")
    cap.release()

    print(f"\n生成件数: {len(generated)} / 選定件数: {len(selected)}")
    print(f"出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
