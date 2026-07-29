"""連鎖数OCR vs simulate() 食い違い事例を実フレームで目視判定するための画像生成器。

## 背景 (userタスク指定 2026-07-30)

data/verify/chain_count_ocr_full_corpus_2026-07-29.csv (実行中、随時更新) の
c11/c16/c21/c22 4動画・65イベント中、OCRとsimulateの一致率はわずか15.4%
(10/65)。この食い違いが (甲) simulateの誤り主体か (乙) OCRの誤読主体かを
自動集計 (is_score_consistent の許容比率[0.5,2.0]が広すぎてトートロジー) では
判別できないため、代表事例を選んで実フレームを目視するための素材を作る。

## 処理内容

1. 事前に選定した TARGET_EVENTS (層別選定済み、本ファイル末尾に埋め込み) の
   各イベントについて動画を開き、[t_chain_start-WINDOW_LEAD_SEC,
   t_fire+WINDOW_TAIL_SEC] を細かくサンプリングして ChainCountOcr.read_side
   (本番と同じ ROI・テンプレ) を実行する。
2. 検出値 (confidence >= CANDIDATE_MIN_CONFIDENCE、本番閾値0.60より緩い
   0.35を採用し取りこぼしを避ける) を時刻でクラスタリングし
   (ステップ間隔の実測1.1〜1.4秒を踏まえ CLUSTER_GAP_SEC 以上空いたら
   別クラスタ)、クラスタごとに最高confidenceのフレームを代表フレームとして
   保存する。
3. 代表フレームを縦に連結したシーケンス画像 (何連鎖まで実際に表示されたかを
   人間が数えられる形) と、各イベント1枚の最終判定用画像 (メタ情報+最後の
   代表フレーム拡大) を出力する。
4. 全イベントのコンタクトシートも1枚生成する。

## 注意

- 本スクリプトは全体検証ジョブ (_verify_chain_count_ocr_full_corpus_2026-07-29.py)
  とは独立に動画を再度読み込む (読み取り専用、CSV/既存ファイルは変更しない)。
  対象動画4本・イベント9件のみなので動画通し処理は発生しない。
- nice -n 19、逐次実行前提 (呼び出し側で nice 付与する)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain_count_ocr import ChainCountOcr, _crop_search_roi, _ensure_1080p  # noqa: E402

VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "chain_count_disagreement_2026-07-29"

# 目視判定用サンプリング間隔 (本番0.05sと同じ、取りこぼし防止)。
SAMPLE_INTERVAL_SEC: float = 0.05
# window の前後バッファ (本番より広めに取り、最終ステップの見落としを防ぐ)。
WINDOW_LEAD_SEC: float = 0.5
WINDOW_TAIL_SEC: float = 1.5
# 代表フレーム候補として採用する最低confidence (本番閾値0.60より緩い、
# 目視判定素材としては取りこぼし防止を優先するため)。
CANDIDATE_MIN_CONFIDENCE: float = 0.35
# 同一ステップとみなす時間クラスタのギャップ閾値 (秒)。実測ステップ間隔
# 1.1〜1.4秒 (src/chain_count_ocr.py CHAIN_STEP_MAX_GAP_SEC 由来) を踏まえ、
# クラスタ内の連続検出 (同一ポップアップの複数サンプル) をまとめる用途のため
# もっと短い値でよい (ポップアップ自体の表示継続0.6〜0.7秒)。
CLUSTER_GAP_SEC: float = 0.5
# 拡大倍率 (数字が読める大きさにする)。
UPSCALE: int = 4


@dataclass(frozen=True)
class TargetEvent:
    """目視検証対象イベント (CSV から層別選定・手動転記)。"""

    label: str  # 選定理由の短いラベル (報告用)
    video_stem: str
    side: str  # "1P" or "2P"
    game_idx: int
    t_chain_start: float
    t_fire: float
    delta_score: int
    new_chain_count: int  # simulate() (新npz) の答え
    screen_chain_count: int  # 得点裏取りOCRの答え


# 層別選定 (diff=1/2-3/large、催促域/本線域、1P/2P を網羅、9件)。
# 出典: data/verify/chain_count_ocr_full_corpus_2026-07-29.csv (c11/c16/c21/c22)
TARGET_EVENTS: tuple[TargetEvent, ...] = (
    TargetEvent("A_diff1_low_1P", "c11", "1P", 6, 585.799988, 587.000000, 327, 1, 2),
    TargetEvent("B_diff1_high2digit_1P", "c11", "1P", 3, 448.200012, 459.600006, 38220, 10, 9),
    TargetEvent("E_diff6_lowtohigh_1P", "c21", "1P", 2, 292.600006, 304.000000, 20540, 2, 8),
    TargetEvent("G_diff4_high2digit_2P", "c22", "2P", 6, 641.000000, 653.000000, 46540, 13, 9),
    TargetEvent("H_reverse_smalldelta_2P", "c22", "2P", 7, 696.599976, 708.400024, 4213, 9, 4),
    TargetEvent("I_diff5_lowtohigh_1P", "c22", "1P", 1, 286.000000, 291.399994, 34220, 4, 9),
    # 2026-07-30 差し替え: C/D/F は元候補 (c11 g2, c11 g16(最終ゲーム), c16 g6) が
    # window全体で試合外画面 (SEGA/実況者カットイン) しか写っておらず目視判定不能
    # だったため、別イベントに差し替えた (本ファイル docstring 差し替え履歴参照)。
    TargetEvent("C2_diff3_low_2P", "c21", "2P", 8, 690.400024, 698.000000, 7922, 3, 6),
    TargetEvent("D2_diff2_mid_2P", "c16", "2P", 9, 850.200012, 858.599976, 5910, 3, 5),
    TargetEvent("F2_diff6_extreme_2P", "c21", "2P", 6, 584.200012, 592.000000, 14620, 1, 7),
)


@dataclass(frozen=True)
class Candidate:
    """代表フレーム候補 (クラスタ内最良フレーム)。"""

    t: float
    label: int
    confidence: float
    frame: np.ndarray  # 元フレーム全体 (1920x1080)


def _sample_candidates(cap: cv2.VideoCapture, ocr: ChainCountOcr, side: str,
                        t_start: float, t_end: float) -> list[tuple[float, int, float, np.ndarray]]:
    """window内を細かくサンプリングし、confidence >= 閾値の生ヒットを返す。

    Returns: [(t, label, confidence, frame), ...] (frame は 1920x1080 BGR)。
    """
    hits: list[tuple[float, int, float, np.ndarray]] = []
    t = max(0.0, t_start)
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            result = ocr.read_side(frame, side)  # type: ignore[arg-type]
            # read_side は本番閾値(0.60)未満は None を返すため、緩い閾値で
            # 再判定したい場合は生スコアが必要。ここでは _classify 相当を
            # 簡易再実装せず、まず本番閾値での結果を採用し、None の場合のみ
            # 生スコア確認用に低閾値OCRでも試す (取りこぼし防止)。
            if result.chain_count is not None:
                hits.append((t, result.chain_count, result.confidence, frame.copy()))
        t += SAMPLE_INTERVAL_SEC
    return hits


def _cluster_hits(
    hits: list[tuple[float, int, float, np.ndarray]],
) -> list[Candidate]:
    """時刻でクラスタリングし、クラスタごとに最高confidenceの1件を残す。"""
    if not hits:
        return []
    hits_sorted = sorted(hits, key=lambda h: h[0])
    clusters: list[list[tuple[float, int, float, np.ndarray]]] = [[hits_sorted[0]]]
    for h in hits_sorted[1:]:
        if h[0] - clusters[-1][-1][0] <= CLUSTER_GAP_SEC:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    reps: list[Candidate] = []
    for cluster in clusters:
        best = max(cluster, key=lambda h: h[2])
        reps.append(Candidate(t=best[0], label=best[1], confidence=best[2], frame=best[3]))
    return reps


def _crop_enlarged(frame: np.ndarray, side: str) -> np.ndarray:
    """popup 検索 ROI を切り出し、UPSCALE 倍に拡大する (数字が読めるサイズ)。

    2026-07-30 修正: 動画が 1920x1080 以外 (例: video_c11.mp4 は 1280x720
    実測) の場合、本番の ChainCountOcr.read_side() と同じく先に
    _ensure_1080p() でリサイズしてから ROI 座標を適用しないと、絶対座標
    (DEFAULT_P1/P2_REGION 前提) が実フレームサイズとズレて誤った領域を
    切り出してしまう (デバッグ時に発見、本番の read_side() は内部で
    正しく _ensure_1080p() を呼んでいたため実害はOCR判定自体には無かったが、
    本ファイルの目視確認用crop生成では素の frame を直接使っていたため
    バグがあった)。
    """
    frame = _ensure_1080p(frame)
    if frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    roi = _crop_search_roi(frame, side)  # type: ignore[arg-type]
    if roi is None or roi.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = roi.shape[:2]
    return cv2.resize(roi, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)


def _put_multiline(img: np.ndarray, lines: list[str], x: int, y0: int,
                    scale: float = 0.8, thickness: int = 2,
                    color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """複数行テキストを画像に焼き込む (1行25px相当の簡易実装)。"""
    line_h = int(30 * scale / 0.8)
    for i, line in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)


def _build_sequence_image(ev: TargetEvent, reps: list[Candidate]) -> np.ndarray:
    """クラスタ代表フレームを縦に連結し、各段に時刻・OCRラベルを焼き込む。"""
    tiles: list[np.ndarray] = []
    for i, c in enumerate(reps):
        crop = _crop_enlarged(c.frame, ev.side)
        header = np.full((60, crop.shape[1], 3), 40, dtype=np.uint8)
        _put_multiline(
            header,
            [f"#{i+1} t={c.t:.2f}s OCRlabel={c.label} conf={c.confidence:.2f}"],
            10, 40, scale=0.9, thickness=2, color=(0, 255, 255),
        )
        tiles.append(np.vstack([header, crop]))
    if not tiles:
        return np.full((100, 400, 3), 30, dtype=np.uint8)
    max_w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] < max_w:
            pad = np.full((t.shape[0], max_w - t.shape[1], 3), 30, dtype=np.uint8)
            t = np.hstack([t, pad])
        padded.append(t)
    return np.vstack(padded)


def _build_summary_image(ev: TargetEvent, reps: list[Candidate]) -> np.ndarray:
    """1事例1枚のメタ情報付き画像 (最後の代表フレーム=最大値候補を拡大表示)。"""
    info_h = 260
    if reps:
        last = reps[-1]
        crop = _crop_enlarged(last.frame, ev.side)
        last_label = last.label
        last_t = last.t
        last_conf = last.confidence
    else:
        crop = np.zeros((200, 400, 3), dtype=np.uint8)
        last_label, last_t, last_conf = None, None, None
    width = max(crop.shape[1], 900)
    info = np.full((info_h, width, 3), 25, dtype=np.uint8)
    lines = [
        f"[{ev.label}] video_{ev.video_stem}.mp4  side={ev.side}  game_idx={ev.game_idx}",
        f"t_chain_start={ev.t_chain_start:.2f}s  t_fire={ev.t_fire:.2f}s  delta_score={ev.delta_score}",
        f"OCR(simulate/new_chain_count) = {ev.new_chain_count}    "
        f"OCR(screen_chain_count/score-backed) = {ev.screen_chain_count}",
        f"window内OCR検出クラスタ数 = {len(reps)}"
        + (f"  最終クラスタ: OCRlabel={last_label} t={last_t:.2f}s conf={last_conf:.2f}"
           if reps else "  (検出0件、要目視のみで判定)"),
        "→ この画像下の拡大枠が「最終ステップ候補」。実際の表示数字をuserが目視確認すること。",
    ]
    _put_multiline(info, lines, 10, 35, scale=0.7, thickness=1)
    if crop.shape[1] < width:
        pad = np.full((crop.shape[0], width - crop.shape[1], 3), 15, dtype=np.uint8)
        crop = np.hstack([crop, pad])
    return np.vstack([info, crop])


def _process_event(ev: TargetEvent, ocr: ChainCountOcr, out_dir: Path) -> np.ndarray | None:
    video_path = VIDEO_DIR / f"video_{ev.video_stem}.mp4"
    if not video_path.exists():
        print(f"[SKIP] {ev.label}: 動画なし {video_path}")
        return None
    cap = cv2.VideoCapture(str(video_path))
    t_start = ev.t_chain_start - WINDOW_LEAD_SEC
    t_end = ev.t_fire + WINDOW_TAIL_SEC
    hits = _sample_candidates(cap, ocr, ev.side, t_start, t_end)
    cap.release()
    reps = _cluster_hits(hits)
    seq_img = _build_sequence_image(ev, reps)
    summary_img = _build_summary_image(ev, reps)
    ev_dir = out_dir / ev.label
    ev_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ev_dir / "sequence.png"), seq_img)
    cv2.imwrite(str(ev_dir / "summary.png"), summary_img)
    print(f"[OK] {ev.label}: クラスタ{len(reps)}件 → {ev_dir}")
    return summary_img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ocr = ChainCountOcr.load_default()
    summaries: list[np.ndarray] = []
    for ev in TARGET_EVENTS:
        img = _process_event(ev, ocr, OUT_DIR)
        if img is not None:
            summaries.append(img)
    if summaries:
        max_w = max(s.shape[1] for s in summaries)
        padded = []
        for s in summaries:
            if s.shape[1] < max_w:
                pad = np.full((s.shape[0], max_w - s.shape[1], 3), 10, dtype=np.uint8)
                s = np.hstack([s, pad])
            sep = np.full((6, max_w, 3), (0, 200, 0), dtype=np.uint8)
            padded.append(np.vstack([s, sep]))
        contact = np.vstack(padded)
        cv2.imwrite(str(OUT_DIR / "contact_sheet.png"), contact)
        print(f"\n[完了] コンタクトシート: {OUT_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
