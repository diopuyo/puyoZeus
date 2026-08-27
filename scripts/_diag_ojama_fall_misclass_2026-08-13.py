"""OJAMA_FALL 誤分類の根因診断 (read-only計装, 2026-08-13 userデモレビュー指摘).

## 背景
review_demo_2026-08-12.mp4 (--start-sec 162 で生成) の userレビューで2箇所指摘:

  (A) source t≈188-189 (デモ26-27秒): 2P状態が OJAMA_FALL に張り付いたまま
      なのに 2P score が 213→221 (+8、落下ボーナス) = 実際にツモ設置が
      起きている。
  (B) source t≈196 (デモ34秒): 1P が「40×8」掛け算式表示 (連鎖確定) なのに
      1P状態 = OJAMA_FALL。2秒後 (t≈198) に両者 CHAIN に急反転し判定が
      33%→81% に飛ぶ = OJAMA_FALL 誤分類が連鎖検知 (early-fire-reaction の
      入口) を塞ぎ、致死攻撃が判定に乗らなかった疑い。

## 計装方針 (src/ は一切変更しない、read-only)
1. `RecognitionPipeline.load_default()` の返す `SideResult` (state/score/
   score_delta/chain_event) を frame 単位でそのまま記録する (公開 API なので
   計装不要)。
2. `BoardStateMachine._detectors[0]` (=ChainPhaseDetector) の `chain_sim.
   find_erasable_groups` をラップし、cycle 49 の「4連結ゲート」が
   ctx.confirmed_board (OJAMA_FALL 中は凍結) に対して何を返したかを記録する。
   これで「chain_event は来ていたのに凍結 confirmed_board に erasable が
   無くゲートが却下した」という仮説を直接確認できる。

## 2 設定の比較 (副次発見の切り分け)
実際のデモ生成コマンド (scripts/_gen_demo_final_2026-08-13.sh) は
`src/production_config.py` の `RECOGNITION_ADOPTED` フラグ
(enable_effect_gate / enable_burst_guard_v2 / burst_gate_open_threshold=0.954 /
enable_hidden_row_burst_guard / enable_transition_merge_guard /
enable_match_transition_debounce) を一切 `visualize_advantage_overlay.py`
に渡していない (同スクリプトには CLI 引数自体が存在しない = 配線漏れ)。
本番認識 (collect_boards_lean.py 等) はこれらを ON で使うため、
デモで見えている挙動が (a) OJAMA_FALL state machine 自体の欠陥か、
(b) バーストガード等の recognition-quality 修正が丸ごと欠落した劣化認識の
産物か、を切り分けるため両設定を同一フレームに対して並走させる。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_ojama_fall_misclass_2026-08-13 \
        --video data/frames/review_demo_2026-08-12.mp4 \
        --start-sec 150 --end-sec 205 \
        --out logs/diag_ojama_fall_misclass_2026-08-13.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

DEFAULT_FPS: float = 30.0
SIDES = ("1P", "2P")
CONFIGS = ("asis_demo", "full_prod")

# 本番収集 (collect_boards_lean.py:157-159) の認識前提解像度。
# visualize_advantage_overlay.py は native (本動画は 1280x720) のまま
# `pipe.update()` に渡しており、burst guard 系コンポーネント
# (src/effect_glow_detector.py) は「1920x1080 リサイズ済み」を無条件前提と
# するため、resize せず enable_burst_guard_v2 等を ON にすると
# cv2.cvtColor が空パッチでクラッシュする (本診断で実際に確認、
# 2026-08-13)。full_prod 設定のみ collect_boards_lean.py と同じ resize を
# 適用し、本番の実際の解像度前提を再現する。
PROD_TARGET_W: int = 1920
PROD_TARGET_H: int = 1080


def _resize_for_prod(frame: Any) -> Any:
    """collect_boards_lean.py と同じ resize 方式 (縮小=INTER_AREA/拡大=LANCZOS4)."""
    h, w = frame.shape[:2]
    if (h, w) == (PROD_TARGET_H, PROD_TARGET_W):
        return frame
    interp = cv2.INTER_LANCZOS4 if h < PROD_TARGET_H else cv2.INTER_AREA
    return cv2.resize(frame, (PROD_TARGET_W, PROD_TARGET_H), interpolation=interp)


def _build_pipeline(config: str) -> RecognitionPipeline:
    """config 名に応じた RecognitionPipeline を構築する.

    asis_demo: scripts/_gen_demo_final_2026-08-13.sh が実際に渡している
        引数のみ (production_config.RECOGNITION_ADOPTED は一切渡らない、
        visualize_advantage_overlay.py に該当 CLI 引数自体が無いため)。
    full_prod: 上記 + RECOGNITION_ADOPTED 相当のフラグを明示 ON にしたもの
        (collect_boards_lean.py 等の本番収集と同一構成)。
    """
    base_kwargs: dict[str, Any] = dict(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    if config == "asis_demo":
        return RecognitionPipeline.load_default(**base_kwargs)
    if config == "full_prod":
        return RecognitionPipeline.load_default(
            **base_kwargs,
            enable_effect_gate=True,
            enable_burst_guard_v2=True,
            burst_gate_open_threshold=0.954,
            enable_hidden_row_burst_guard=True,
            enable_transition_merge_guard=True,
            enable_match_transition_debounce=True,
        )
    raise ValueError(f"unknown config: {config}")


class _ChainGateTap:
    """ChainPhaseDetector.chain_sim.find_erasable_groups をラップする計装.

    cycle 49 の 4連結ゲート (src/state_detectors.py ChainPhaseDetector.detect)
    が実際に何を見て何を返したかを frame 単位で記録する。
    """

    def __init__(self, chain_sim: Any, side: str, config: str,
                 frame_cursor: dict[str, Any]) -> None:
        self._orig = chain_sim.find_erasable_groups
        self._side = side
        self._config = config
        self._cursor = frame_cursor
        self.records: list[dict[str, Any]] = []

    def wrapped(self, board: Any) -> Any:
        result = self._orig(board)
        self.records.append({
            "config": self._config,
            "side": self._side,
            "frame_idx": self._cursor["idx"],
            "t_sec": self._cursor["t"],
            "confirmed_puyo_count": int(board.count_puyos()),
            "n_erasable_groups": len(result),
            "gate_rejected": len(result) == 0,
        })
        return result


def _attach_chain_gate_tap(
    pipe: RecognitionPipeline, side: str, config: str,
    frame_cursor: dict[str, Any],
) -> _ChainGateTap:
    """side ("1P"/"2P") の ChainPhaseDetector に gate tap を取り付ける."""
    sm = pipe._sm_1p if side == "1P" else pipe._sm_2p
    chain_det = sm._detectors[0]  # _build_state_machine の登録順で先頭固定
    if type(chain_det).__name__ != "ChainPhaseDetector":
        raise RuntimeError(
            f"detectors[0] は ChainPhaseDetector のはずが "
            f"{type(chain_det).__name__} だった (登録順が変わった疑い)",
        )
    tap = _ChainGateTap(chain_det.chain_sim, side, config, frame_cursor)
    chain_det.chain_sim.find_erasable_groups = tap.wrapped
    return tap


def run(video_path: Path, start_sec: float, end_sec: float, out_path: Path,
        decode_from_sec: float = 0.0, stride: int = 1) -> None:
    """区間を 2 設定で並走させ、frame 単位ログと要約を JSON に書き出す.

    state machine は試合開始 (menu → 初回 STABLE 確定 → score baseline 等) の
    履歴に依存するため、`decode_from_sec` (既定 0.0 = 動画先頭) から連続で
    `pipe.update()` を呼び続ける (シーク禁止、collect_boards_lean.py と同じ
    方式)。ログに記録するのは `start_sec` 以降のみ (--out の肥大化防止)。

    stride: 実デモ生成 (scripts/_gen_demo_final_2026-08-13.sh 経由
        visualize_advantage_overlay.py) は `--normalize-fps-30` が既定 True
        (production_config.OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT) で
        60fps 動画を stride=2 (実効30fps) に間引いて `pipe.update()` に渡す。
        本診断も同じ間引き方式 (`(fi - decode_from_frame) % stride != 0` は
        スキップ) を既定 stride=1 (間引きなし) から再現可能にする
        (2026-08-13 実測: stride=1 では 1P の CHAIN 中に OJAMA_FALL が全く
        出現せず、実デモで観測された挙動と食い違うと判明したための追加)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[diag] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    decode_from_frame = int(decode_from_sec * fps)
    end_frame = int(end_sec * fps)
    if decode_from_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, decode_from_frame)

    pipes: dict[str, RecognitionPipeline] = {c: _build_pipeline(c) for c in CONFIGS}
    cursor = {"idx": -1, "t": -1.0}
    gate_taps: dict[str, dict[str, _ChainGateTap]] = {}
    for cfg, pipe in pipes.items():
        gate_taps[cfg] = {
            side: _attach_chain_gate_tap(pipe, side, cfg, cursor)
            for side in SIDES
        }

    frame_records: dict[str, list[dict[str, Any]]] = {c: [] for c in CONFIGS}

    n_processed = 0
    for frame_idx in range(decode_from_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # 実デモ生成と同じ間引き方式 (visualize_advantage_overlay.py の
        # `(fi - start_frame) % stride != 0: continue` と同一ロジック)。
        # stride=1 (既定) では常に False = 従来通り全フレーム処理。
        if (frame_idx - decode_from_frame) % stride != 0:
            continue
        t_sec = frame_idx / fps
        cursor["idx"] = frame_idx
        cursor["t"] = t_sec
        for cfg, pipe in pipes.items():
            frame_in = _resize_for_prod(frame) if cfg == "full_prod" else frame
            r = pipe.update(frame_idx, t_sec, frame_in.copy())
            if t_sec < start_sec:
                continue
            frame_records[cfg].append({
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "1P_state": r.p1.state.value,
                "2P_state": r.p2.state.value,
                "1P_score": r.p1.score,
                "2P_score": r.p2.score,
                "1P_score_delta": r.p1.score_delta,
                "2P_score_delta": r.p2.score_delta,
                "1P_chain_event": r.p1.chain_event is not None,
                "2P_chain_event": r.p2.chain_event is not None,
                "1P_cnn_puyo_count": int(r.p1.cnn_board.count_puyos()),
                "2P_cnn_puyo_count": int(r.p2.cnn_board.count_puyos()),
                "1P_confirmed_is_none": r.p1.confirmed_board is None,
                "2P_confirmed_is_none": r.p2.confirmed_board is None,
            })
        n_processed += 1
        if n_processed % 600 == 0:
            print(f"[diag] processed {n_processed} frames (t={t_sec:.2f}s)",
                  file=sys.stderr)
    cap.release()

    result = {
        "video": str(video_path),
        "fps": fps,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "n_frames_processed": n_processed,
        "frame_records": frame_records,
        "chain_gate_records": {
            cfg: {side: gate_taps[cfg][side].records for side in SIDES}
            for cfg in CONFIGS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[diag] wrote -> {out_path}")


def main() -> int:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="OJAMA_FALL 誤分類の根因診断")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=150.0)
    parser.add_argument("--end-sec", type=float, default=205.0)
    parser.add_argument("--decode-from-sec", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.start_sec, args.end_sec, args.out, args.decode_from_sec,
        args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
