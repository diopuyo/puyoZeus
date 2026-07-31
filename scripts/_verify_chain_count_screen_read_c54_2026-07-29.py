"""画面「N れんさ!」OCR の自己検証: video_c54 の既知イベント 1 件で確認する。

## 目的 (userタスク指定 2026-07-29)

得点整合性チェック (docs: scripts/_verify_score_consistency_2026-07-29.py) で、
514イベント中 40.3% が simulate(before_grid) 由来の chain_count 誤りで不整合と
判明した。c54 の 1P・game_idx=1・t_chain_start≈252.6秒 のイベントは:
    - 旧 npz (data/indicators_v2/boards_lean_fixed) で simulate() → chain_count=4
      → score_consistency_ratio が許容範囲内 (整合)
    - 新 npz (data/indicators_v2/boards_lean_fixed_regen_2026-07-28) で
      simulate() → chain_count=1 (before盤面の認識誤りにより誤認)
      → 不整合 (ratio が許容範囲外)
既に実フレームで「4 れんさ!」の表示を目視確認済み (本タスクの前段)。
本スクリプトは、画面 OCR (src.chain_count_ocr.ChainCountOcr) が独立に
chain_count=4 を読み取れるかを確認し、simulate() に依存しない検証手段として
成立するかを検証する。

## 自己検証の設計方針 (正直な記録)

calculate_chain_score() は ChainResult (各ステップの erased_count/connection/
color 内訳) を要求するため、整数の chain_count 単体からは expected_score を
厳密に再計算できない (FireEvent は chain_count のみ保持し、ステップ内訳は
保持していない)。そのため本スクリプトは「chain_count を捏造して
score_consistency_ratio に通す」ことはしない。

代わりに、既存の score_consistency_ratio が確立済みの 2 つの事実
(旧npz: chain_count=4 で整合 / 新npz: chain_count=1 で不整合) を
「独立した参照値」として使い、画面 OCR の読み取り結果と突き合わせる
三角測量方式を採る。screen_read == old_npz_chain_count (== 4) かつ
old_npz が整合、new_npz が不整合、という 3 点が揃えば、画面 OCR が
simulate() の誤りを検出・置換できる強い証拠になる。

読み取り専用。src/、scripts/measure_exchange_dynamics.py は import のみで
一切変更しない。CPU負荷は 1 動画・1 イベントの ~4 秒 window のみ (軽量)。

## 訂正 (2026-07-29 追記、digit_5〜9 テンプレ採取後の再検証で判明)

上記「旧npz: chain_count=4 で整合」を三角測量の基準値として扱っていたが、
digit_5〜9 のテンプレ採取後に再実行したところ画面OCRは 5 を返した。
実フレームを t=257.55〜257.70秒 まで目視確認したところ、実際に
「5 れんさ!」のポップアップが (glow演出付きで) 表示されており、
**真の連鎖数は 5 であり、旧npz の chain_count=4 も誤りだった**
(score_consistency_ratio の許容範囲内判定は偶然の一致で、真の
ground truth を保証するものではなかった)。t=258.5秒以降は盤面が
静止しポップアップも消えているため、5 で連鎖終了と確認済み。
本スクリプトの「旧npzを信頼できる参照値とする」という前提は誤りだった
ため、三角測量の結論部分は参考程度に留め、画面OCRの読み取り結果
そのもの (実フレーム目視で直接確認可能) を第一の証拠として扱うこと。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/_verify_chain_count_screen_read_c54_2026-07-29.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from src.chain_count_ocr import ChainCountOcr  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, FireEvent, _process_video,
)

# _verify_score_consistency_2026-07-29.py と同一定義 (ファイル名がハイフン+数字を
# 含み Python の import 文では参照できないため、値をここに複製する)。
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

VIDEO_PATH: Path = PROJ_ROOT / "data" / "frames" / "video_c54.mp4"
TARGET_STEM: str = "c54"
TARGET_SIDE: str = "1P"
TARGET_GAME_IDX: int = 1
TARGET_T_CHAIN_START_APPROX: float = 252.6

# 画面OCRのwindow探索マージン (秒)。t_chain_start 〜 t_fire の前後に余裕を持たせる。
# video_c54 実測でポップアップ演出が t_chain_start+3.7秒程度まで続くため、
# t_fire (発火確定後の時刻) にさらに 1 秒のバッファを足す。
WINDOW_START_LEAD_SEC: float = 0.2
WINDOW_END_BUFFER_SEC: float = 1.0


def _find_target_event(npz_dir: Path) -> FireEvent | None:
    """指定 npz ディレクトリで c54 の対象イベントを検索する。"""
    sim = ChainSimulator()
    npz_path = npz_dir / f"{TARGET_STEM}.npz"
    if not npz_path.exists():
        return None
    _, defrag, _ = _process_video(npz_path, sim, 0)
    cand = [
        e for e in defrag
        if e.fire_side == TARGET_SIDE and e.game_idx == TARGET_GAME_IDX
    ]
    if not cand:
        return None
    return min(cand, key=lambda e: abs(e.t_chain_start - TARGET_T_CHAIN_START_APPROX))


def _report_npz_side(label: str, npz_dir: Path) -> FireEvent | None:
    """指定 npz での対象イベントの simulate() 結果 (chain_count等) を表示する。"""
    ev = _find_target_event(npz_dir)
    if ev is None:
        print(f"[{label}] イベントが見つからない ({npz_dir})")
        return None
    print(f"[{label}] chain_count={ev.chain_count} delta_score={ev.delta_score} "
          f"t_chain_start={ev.t_chain_start:.2f}s t_fire={ev.t_fire:.2f}s "
          f"frag_count={ev.frag_count}")
    return ev


def _run_screen_ocr(ref: FireEvent) -> int | None:
    """参照イベントの時刻レンジで画面 OCR の window 最大値を読み取る。"""
    if not VIDEO_PATH.exists():
        print(f"動画が見つからない: {VIDEO_PATH} (削除済みの可能性、"
              "CLAUDE.mdのストレージ管理ルールに従い処理後削除された場合は要再DL)")
        return None
    t_start = ref.t_chain_start - WINDOW_START_LEAD_SEC
    t_end = ref.t_fire + WINDOW_END_BUFFER_SEC
    ocr = ChainCountOcr.load_default()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    result = ocr.read_max_in_window(cap, TARGET_SIDE, t_start, t_end)
    cap.release()
    print(f"window=[{t_start:.2f}, {t_end:.2f}]秒 サンプル数={len(result.samples)} "
          f"検出あり={result.n_hits}件")
    return result.max_chain_count


def main() -> None:
    print("=== (1) 旧npz (data/indicators_v2/boards_lean_fixed) の simulate() 結果 ===")
    ev_old = _report_npz_side("旧npz", NPZ_DIR)

    print("\n=== (2) 新npz (boards_lean_fixed_regen_2026-07-28) の simulate() 結果 ===")
    ev_new = _report_npz_side("新npz", NPZ_DIR_REGEN)

    print("\n=== (3) 画面 OCR (ChainCountOcr) による独立読み取り ===")
    ref = ev_old or ev_new
    if ref is None:
        print("参照イベントが取得できず window を決定できない。中断。")
        return
    screen_max = _run_screen_ocr(ref)
    print(f"screen_read max_chain_count = {screen_max}")

    print("\n=== (4) 三角測量結論 ===")
    print(f"旧npz chain_count = {ev_old.chain_count if ev_old else 'N/A'} (既知: 整合)")
    print(f"新npz chain_count = {ev_new.chain_count if ev_new else 'N/A'} (既知: 不整合)")
    print(f"画面OCR chain_count = {screen_max}")
    if ev_old is not None and screen_max == ev_old.chain_count:
        print("\n[結論] 画面OCRは旧npz (整合) の chain_count と一致した。"
              "simulate() 依存の chain_count 誤りを、画面OCRで独立に検出・"
              "置換できる強い証拠となる。")
    else:
        print("\n[結論] 画面OCRは旧npzの chain_count と一致しなかった。"
              "要追加調査 (テンプレ・閾値・window範囲の見直し)。")


if __name__ == "__main__":
    main()
