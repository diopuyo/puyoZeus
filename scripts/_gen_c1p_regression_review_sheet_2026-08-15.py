"""c1p 新規劣化25セルの user目視レビュー用まとめシート生成 (診断専用、修正なし)。

`scripts/_diag_c1p_new_regressions_frames_2026-08-15.py` が出力した
data/verify/yardstick_v2_2026-08-14/diag_c1p/ のROI画像+
_diag_c1p_new_regressions_2026-08-15.json (セル明細) から、
盤面ごとに (A凍結ROI / c1p凍結ROI を並べてグリッド線+誤りセルを赤枠強調した
比較画像) + セル明細テーブルを1枚のHTMLにまとめる。

Usage:
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._gen_c1p_regression_review_sheet_2026-08-15
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "diag_c1p"
REG_JSON = (
    _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "scoring_ablation"
    / "_diag_c1p_new_regressions_2026-08-15.json"
)
OUT_HTML = DIAG_DIR / "review_sheet.html"

# board.py の規約 (13行×6列、行0=隠し段)。ROI 384x720 を等分してグリッド座標を
# 復元する (DEFAULT_P1_REGION/P2_REGION と同じ幅高、src/image_reader.py 参照)。
N_ROWS, N_COLS = 13, 6
ROI_W, ROI_H = 384, 720
CELL_W = ROI_W / N_COLS
CELL_H = ROI_H / N_ROWS

COLOR_NAME = {
    0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "おじゃま", 10: "不明",
}


def _draw_grid_and_marks(img: np.ndarray, marks: list[tuple[int, int]]) -> np.ndarray:
    """13x6グリッド線を薄く描き、 marks (r,c) のセルを赤枠で強調する。"""
    out = img.copy()
    h, w = out.shape[:2]
    sx, sy = w / ROI_W, h / ROI_H
    for r in range(N_ROWS + 1):
        y = int(r * CELL_H * sy)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
    for c in range(N_COLS + 1):
        x = int(c * CELL_W * sx)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
    for r, c in marks:
        x0, y0 = int(c * CELL_W * sx), int(r * CELL_H * sy)
        x1, y1 = int((c + 1) * CELL_W * sx), int((r + 1) * CELL_H * sy)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 3)
    return out


def _find_roi_path(sheet_id: str, tag: str) -> "Path | None":
    matches = sorted(DIAG_DIR.glob(f"{sheet_id}_{tag}_roi_f*.png"))
    return matches[0] if matches else None


def main() -> None:
    cells = json.loads(REG_JSON.read_text(encoding="utf-8"))
    by_sheet: dict[str, list[dict]] = {}
    for c in cells:
        by_sheet.setdefault(c["sheet_id"], []).append(c)

    sections: list[str] = []
    for sid in sorted(by_sheet, key=lambda s: -len(by_sheet[s])):
        sheet_cells = by_sheet[sid]
        meta = sheet_cells[0]
        marks = [(c["r"], c["c"]) for c in sheet_cells]

        a_path = _find_roi_path(sid, "A")
        c1p_path = _find_roi_path(sid, "C1P")
        combined_rel = None
        if a_path is not None and c1p_path is not None:
            img_a = cv2.imread(str(a_path))
            img_c1p = cv2.imread(str(c1p_path))
            ann_a = _draw_grid_and_marks(img_a, marks)
            ann_c1p = _draw_grid_and_marks(img_c1p, marks)
            # ラベル帯を上に追加してから左右結合
            label_h = 36
            def _with_label(im: np.ndarray, text: str) -> np.ndarray:
                band = np.zeros((label_h, im.shape[1], 3), dtype=np.uint8)
                cv2.putText(band, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                return np.vstack([band, im])
            a_frame = a_path.stem.split("_f")[-1]
            c1p_frame = c1p_path.stem.split("_f")[-1]
            left = _with_label(ann_a, f"a (frame {a_frame})")
            right = _with_label(ann_c1p, f"c1p (frame {c1p_frame})")
            gap = np.full((left.shape[0], 12, 3), 128, dtype=np.uint8)
            combined = np.hstack([left, gap, right])
            combined_path = DIAG_DIR / f"{sid}_combined.png"
            cv2.imwrite(str(combined_path), combined)
            combined_rel = combined_path.name

        rows_html = "".join(
            f"<tr><td>{c['r']}</td><td>{c['c']}</td>"
            f"<td>{c['correct']} ({COLOR_NAME.get(c['correct'], '?')})</td>"
            f"<td style='color:#2a7'>{c['a_pred']} ({COLOR_NAME.get(c['a_pred'], '?')})</td>"
            f"<td style='color:#c33'>{c['c1p_pred']} ({COLOR_NAME.get(c['c1p_pred'], '?')})</td>"
            f"</tr>"
            for c in sorted(sheet_cells, key=lambda c: (c["r"], c["c"]))
        )
        img_html = (
            f"<img src='{combined_rel}' style='max-width:100%;border:1px solid #ccc;'>"
            if combined_rel else "<p>(画像抽出失敗)</p>"
        )
        sections.append(f"""
        <section style="margin-bottom:40px;border-bottom:2px solid #888;padding-bottom:24px;">
          <h2>{sid} ({len(sheet_cells)}セル) — video={meta['video_id']} side={meta['side']} phase={meta['phase']}</h2>
          {img_html}
          <table border="1" cellpadding="4" style="border-collapse:collapse;margin-top:8px;">
            <tr style="background:#eee;"><th>行r</th><th>列c</th><th>正解</th><th>a (旧・aは正解)</th><th>c1p (新規劣化)</th></tr>
            {rows_html}
          </table>
        </section>
        """)

    pattern_summary = """
    <div style="background:#fff8e1;border:1px solid #e0c060;padding:16px;margin-bottom:24px;">
      <h3>誤りパターン分類 (実フレーム直接確認、npz grid値と生動画フレームを突合)</h3>
      <p>
      (1) <b>010_c96 (12セル、48%)</b>: nearest match が baseline(a) の exact フレーム(951622)より
      6フレーム前(951616)に landing。 実際の生動画フレーム951616は左3列も色ぷよで<u>視覚的には満杯</u>だが、
      npz に記録された grid 値は該当列が 0 (空) — 既知の一過性ドロップアウト
      (docs/KNOWN_WEAKNESSES.md W9系) と同型の認識側欠陥に、 たまたまこの6フレームのズレで
      巻き込まれた。 placement_override 自体の evidence 判定ロジックの誤りではなく、
      「どのフレームがSTABLE確定に使われるか」の巡り合わせで既知の別欠陥を引いた事故。<br>
      (2) <b>013/026_c13 + 006_c17 (2+3+5=10セル、40%)</b>: 実フレームで直接確認した結果、
      該当セルの列 (c13は列2/3、c17は列4/5) がまだお邪魔/色ぷよの落下・定着途中で、
      隣接列は既に定着済みなのに当該列だけ未定着のまま凍結されていた。
      これは診断当初の「evidence一発判定が未定着フレームを凍結する」という欠陥の
      <u>残存インスタンス</u> — 修正で score_delta 経路(自chainノイズ)は塞いだが、
      slide_motion 経路 (ヒステリシス対象外のまま維持) 側は「ツモを置いた」ことの証拠に
      すぎず「盤面全体が定着完了した」ことの証拠ではないため、 列単位で定着が遅れている
      ケースには同種のリスクが残る。<br>
      (3) <b>019_c23 (3セル、12%)</b>: 列0-1で正解3(緑)に対しc1pが4(黄)と誤読。
      当該フレームは列5側で別の一過性ドロップアウトが同時発生していたが、
      本セル(列0-1)自体はタイミング要因ではなく単純な色誤認(緑↔黄)で、
      OJAMA_FALL/placement_override とは無関係な既存ノイズの可能性が高い。<br>
      (4) <b>付随観察 (013/026_c13)</b>: a と c1p が <u>同一 frame_idx (exact match)</u> なのに
      予測値が異なるケースを確認した。 これは同じ生動画フレームでも収集run間で
      オンラインHSV較正/drift補正等の累積状態が異なる (それまでの状態遷移の履歴が
      a と c1p で異なるため) ことを示唆し、 evidence判定ロジック単体の問題を超えた
      二次的な不安定性の可能性がある — 採用判断とは別に、 アーキ側でのフォロー観察を推奨。
      </p>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>c1p 新規劣化25セル レビューシート (2026-08-15)</title>
<style>body{{font-family:sans-serif;margin:24px;}} h1{{color:#333;}}</style>
</head><body>
<h1>c1p (placement_override 修正版・chain除外のみ) 新規劣化25セル レビューシート</h1>
<p>a (本番構成) が正解だったのに c1p (修正版placement_override) が新規に誤読したセルのみを抽出 (全55盤面ペアワイズ比較)。
黄色グリッド線=6列×13行 (行0=隠し段)、赤枠=当該セル。左=aが突合に使ったフレーム (=物差しground truth基準フレーム)、
右=c1pがSTABLE突合に使ったフレーム (exact一致ならa同一フレーム、nearestなら数フレームずれる)。</p>
<p>合計 {len(cells)} セル / {len(by_sheet)} 盤面。</p>
{pattern_summary}
{"".join(sections)}
</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[done] -> {OUT_HTML}")


if __name__ == "__main__":
    main()
