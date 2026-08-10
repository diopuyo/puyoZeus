"""有利不利の評価が「両者同時 STABLE 待ち」で凍結する時間を実測する (2026-08-08).

## 背景
`scripts/visualize_advantage_overlay.py:1129` は
    settled = r.p1.state == STABLE and r.p2.state == STABLE
のときにしか指標を再計算しない。 片方が連鎖中・おじゃま落下中だと **両者とも
凍結**するため、 盤面が激変しても判定が動かない。 実際 2026-08-08 のデモで
t=54.5〜66 の 11 秒間、 主因の数値が 1 つも変化しなかった (user 指摘)。

コード内には「c56_g3 12連鎖では settled フリーズが約 19 秒続いた」という
実測メモがあるが、 **全体でどれだけ起きているかは測られていない**。 評価の
根幹 (CLAUDE.md「両者 STABLE 時のみ評価」) に手を入れるかの判断材料として、
まず頻度と長さの分布を出す。 本スクリプトは **評価には一切触れない読み取り専用**。

出力: data/verify/settled_freeze_2026-08-08.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

from src.production_config import RECOGNITION_ADOPTED  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

OUT_TSV = _ROOT / "data" / "verify" / "settled_freeze_2026-08-08.tsv"
# 測定対象 (デモと同じクリップ + 学習ライブラリの動画を数本)。
TARGETS: tuple[tuple[str, Path], ...] = (
    ("dio_vs_ts_m01", _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"),
)
# 1 動画あたりの最大処理秒 (全長を回すと時間がかかるため)。0 = 全長。
MAX_SEC: float = 0.0


def _flag_kwargs() -> dict:
    """本番構成のフラグを load_default の引数名へ変換する。

    production_config は CLI 文字列で持っているため、 ここで対応付ける。
    値を取るフラグ (--burst-gate-open-threshold 0.954) も扱う。
    """
    kwargs: dict = {}
    for f in RECOGNITION_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = float(parts[1]) if len(parts) > 1 else True
    return kwargs


def _measure(video: Path) -> tuple[list[float], float, int]:
    """(凍結区間の長さ一覧, 総時間, サンプル数) を返す。"""
    pipeline = RecognitionPipeline.load_default(
        force_in_match=True, **_flag_kwargs(),
    )
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    freezes: list[float] = []
    cur_start: float | None = None
    n = 0
    fi = 0
    last_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if MAX_SEC and t > MAX_SEC:
            break
        r = pipeline.update(fi, t, frame)
        settled = (
            r.p1.state == BoardState.STABLE and r.p2.state == BoardState.STABLE
        )
        if settled:
            if cur_start is not None:
                freezes.append(t - cur_start)
                cur_start = None
        else:
            if cur_start is None:
                cur_start = t
        n += 1
        last_t = t
        fi += 1
    if cur_start is not None:
        freezes.append(last_t - cur_start)
    cap.release()
    return freezes, last_t, n


def main() -> int:
    rows = ["video\ttotal_sec\tfrozen_sec\tfrozen_rate\tn_freezes\tmax_freeze_sec\tp50\tp90"]
    for name, path in TARGETS:
        if not path.exists():
            print(f"skip {name} (not found)")
            continue
        print(f"[freeze] {name} 測定中...")
        freezes, total, _ = _measure(path)
        frozen = float(sum(freezes))
        arr = np.array(freezes) if freezes else np.array([0.0])
        rows.append(
            f"{name}\t{total:.1f}\t{frozen:.1f}\t{frozen / max(total, 1e-9):.4f}\t"
            f"{len(freezes)}\t{arr.max():.2f}\t{np.percentile(arr, 50):.2f}\t"
            f"{np.percentile(arr, 90):.2f}"
        )
        print(f"  総時間 {total:.1f}s / 凍結 {frozen:.1f}s "
              f"({frozen / max(total, 1e-9):.1%})")
        print(f"  凍結回数 {len(freezes)} / 最長 {arr.max():.2f}s / "
              f"中央値 {np.percentile(arr, 50):.2f}s / p90 {np.percentile(arr, 90):.2f}s")
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
