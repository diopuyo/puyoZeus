"""エフェクト有無 セルラベルシート 準備スクリプト (2026-08-03、案B閾値確定用)。

背景: memory project_effect_gate_v1_failure_2026-08-03 / 特性調査
data/verify/effect_signature_study_2026-08-03 (scripts/study_effect_signature_2026-08-03.py)
で s_min (セル内最小彩度) が有望特徴 (AUC0.822) と判明したが、窓タイミングは
連鎖規模/おじゃま増加検知からの機械推定であり「実際にエフェクトが写っているか」
の正解ラベルが無い下限値だった。本スクリプトは次段として、user (ぷよ有段者) が
「このフレームのこのセルにエフェクト (予告おじゃまバースト/煙) が実際に写って
いるか」を人手でラベル付けするための材料 (フレーム画像+クリック式ツール用CSV)
を機械的に準備する (ラベル付け自体は行わない)。

## 既存資産の流用方針
- 窓検出 (どのフレームを候補にするか) は scripts/study_effect_signature_2026-08-03.py
  の collect_burst_samples / collect_smoke_samples / collect_baseline_samples を
  そのまま呼び出して再利用する (importlib 経由、ファイル名に日付ハイフンを含み
  通常のimport文が使えないため)。study側は調査専用スクリプトのため一切変更しない。
- クリック操作UIは scripts/build_full_board_label_tool.py の crop_visible_board_region
  (可視12行の盤面クロップ座標系) を流用する。ラベル付けツール本体は
  scripts/build_effect_cell_label_tool.py が別途生成する。

## 動画分散方針
study側は8動画 (c18/c5/c29/c19/c10/c11/c20/c21) のみを対象にしていたため、
本スクリプトはこれに新規6動画 (npz+mp4キャッシュ存在確認済み) を加えたプールから
選定し、動画あたり上限を設けて偏りを抑える (効果調査と同じ動画への集中回避)。

## 出力 (data/verify/effect_cell_label_2026-08-03/)
    frames/<video>_t<t_sec>_<side>_<layer>_full.png        実画面フルフレーム
    frames/<video>_t<t_sec>_<side>_<layer>_board_crop.png  盤面クロップ(可視12行)
    labeling_sheet.csv     候補一覧 (label_tool.htmlの入力)
    label_sheet.md         説明書き付き一覧 (Windowsパスリンク)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.build_effect_cell_label_sheet
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_full_board_label_tool import crop_visible_board_region  # noqa: E402
from scripts.extract_exchange_event_frames import (  # noqa: E402
    grab_frame, resolve_cached_video_path, to_windows_path,
)

# =============================================================================
# study_effect_signature_2026-08-03 の動的import
# (ファイル名がハイフンを含み `import scripts.study_effect_signature_2026-08-03`
#  という通常のimport文では書けないため importlib.import_module を使う。
#  調査専用スクリプトなので中身は一切変更しない、実行時に一部モジュール定数
#  だけ上書きして動画プール・件数を拡張する)
# =============================================================================

_STUDY_MODULE_NAME: str = "scripts.study_effect_signature_2026-08-03"
_ES = importlib.import_module(_STUDY_MODULE_NAME)

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

OUTPUT_DIR: Path = Path("data/verify/effect_cell_label_2026-08-03")
FRAMES_SUBDIR_NAME: str = "frames"

# 効果特性調査で既に使用済みの動画 (study側定数を再利用、重複定義しない)
STUDY_VIDEO_STEMS: tuple[str, ...] = _ES.CANDIDATE_VIDEO_STEMS
# 効果調査に含まれない新規動画 (npz+mp4キャッシュ存在確認済み、2026-08-03時点)
EXTRA_VIDEO_STEMS: tuple[str, ...] = ("c34", "c40", "c45", "c50", "c55", "c60")
EFFECT_LABEL_VIDEO_STEMS: tuple[str, ...] = STUDY_VIDEO_STEMS + EXTRA_VIDEO_STEMS

# 選定目標件数 (バースト3層×6 + 煙12 + 対照10 = 40)
BURST_TARGET_PER_BIN: int = 6
SMOKE_TARGET_TOTAL: int = 12
BASELINE_TARGET_TOTAL: int = 10

# 動画あたり上限 (全レイヤー合算の共有カウンタで適用、6動画以上への分散を保証)
MAX_CANDIDATES_PER_VIDEO: int = 6

# study側の収集関数から大きめの生プールを取り、その後ラウンドロビンで動画分散
# させた40件に絞り込む (生プールが小さいと動画分散の選択余地が無くなるため)。
BURST_OVERSAMPLE_PER_BIN: int = 18
SMOKE_OVERSAMPLE_TOTAL: int = 30
BASELINE_OVERSAMPLE_TOTAL: int = 30

RANDOM_SEED: int = 20260803

LAYER_BURST: str = "burst"
LAYER_SMOKE: str = "smoke"
LAYER_BASELINE: str = "baseline"
LAYER_LABEL_JA: dict[str, str] = {
    LAYER_BURST: "バースト(予告おじゃま送付エフェクト疑い)",
    LAYER_SMOKE: "煙(おじゃま着弾エフェクト疑い)",
    LAYER_BASELINE: "対照(平穏、エフェクト無しの想定)",
}

CSV_HEADER: tuple[str, ...] = (
    "video_id", "t_sec", "side", "layer", "chain_bin",
    "image_full_frame", "image_board_crop",
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class EffectFrameCandidate:
    """1フレーム分のエフェクトラベル候補 (video_stem+side+t_secで一意)。"""

    video_stem: str
    side: str
    t_sec: float
    layer: str       # "burst" / "smoke" / "baseline"
    chain_bin: str    # burstのみ ("2-3"/"4-6"/"7+")、他は ""

    @property
    def video_id(self) -> str:
        """"c18" -> "video_c18" (既存ツール群の video_id 表記に合わせる)。"""
        return f"video_{self.video_stem}"


# =============================================================================
# 1. study モジュールの一時的な定数上書き (ファイル変更なし)
# =============================================================================


def _patch_study_module_for_extended_pool() -> None:
    """study モジュールのプール定数を実行時だけ上書きする (元ファイルは無変更)。

    collect_smoke_samples/collect_baseline_samples はモジュール定数
    CANDIDATE_VIDEO_STEMS を直接参照するため、呼び出し前にここで拡張する。
    """
    _ES.CANDIDATE_VIDEO_STEMS = EFFECT_LABEL_VIDEO_STEMS
    _ES.BURST_WINDOWS_PER_BIN = BURST_OVERSAMPLE_PER_BIN
    _ES.SMOKE_WINDOWS_TOTAL = SMOKE_OVERSAMPLE_TOTAL
    _ES.BASELINE_WINDOWS_TOTAL = BASELINE_OVERSAMPLE_TOTAL


# =============================================================================
# 2. CellRecord群 -> フレーム単位候補への集約 (重複排除)
# =============================================================================


def group_records_to_frames(records: list, layer: str) -> list[EffectFrameCandidate]:
    """CellRecordのリストから (video_stem, side, t_sec, chain_bin) 一意のフレーム集合を作る。

    layer は呼び出し側で明示指定する (baseline由来のCellRecord.layerは
    "normal"/"empty"/"ojama" 等セル単位の真の色ラベルであり、フレームの
    位置付け=対照群であることとは別軸のため)。
    """
    seen: dict[tuple[str, str, float, str], EffectFrameCandidate] = {}
    for r in records:
        key = (r.video_stem, r.side, round(float(r.t_sec), 2), r.chain_bin)
        if key not in seen:
            seen[key] = EffectFrameCandidate(
                video_stem=r.video_stem, side=r.side, t_sec=float(r.t_sec),
                layer=layer, chain_bin=r.chain_bin,
            )
    return list(seen.values())


# =============================================================================
# 3. ラウンドロビン選定 (動画上限を守りつつ偏りを抑える)
# =============================================================================


def round_robin_select(
    pool: list[EffectFrameCandidate], n_want: int, max_per_video: int,
    usage: dict[str, int],
) -> list[EffectFrameCandidate]:
    """動画あたり上限を守りながらプールから n_want 件をラウンドロビンで選ぶ。

    usage は全レイヤー共有のカウンタ (呼び出し元で使い回すことで、レイヤーを
    跨いだ動画あたり合計件数を制御し、少数動画への集中を防ぐ)。
    """
    picked: list[EffectFrameCandidate] = []
    remaining = list(pool)
    while len(picked) < n_want and remaining:
        used_this_round: set[str] = set()
        next_remaining: list[EffectFrameCandidate] = []
        for c in remaining:
            if len(picked) >= n_want:
                next_remaining.append(c)
                continue
            if c.video_stem in used_this_round or usage.get(c.video_stem, 0) >= max_per_video:
                next_remaining.append(c)
                continue
            picked.append(c)
            used_this_round.add(c.video_stem)
            usage[c.video_stem] = usage.get(c.video_stem, 0) + 1
        if not used_this_round:
            break  # これ以上ラウンドロビンで拾える候補がない
        remaining = next_remaining
    return picked


def collect_and_select_candidates(rng_seed: int = RANDOM_SEED) -> list[EffectFrameCandidate]:
    """study側の窓検出を流用し、動画分散を保証しつつ40フレーム程度を選定する。"""
    _patch_study_module_for_extended_pool()
    rng = np.random.default_rng(rng_seed)
    fire_df = pd.read_csv(_ES.FIRE_EVENTS_CSV)
    target_ids = {f"video_{s}" for s in EFFECT_LABEL_VIDEO_STEMS}
    fire_df = fire_df[fire_df["video_id"].isin(target_ids)].copy()

    frame_cache = _ES.FrameCache()
    try:
        burst_records, _burst_counts = _ES.collect_burst_samples(fire_df, frame_cache, rng)
        smoke_records, _smoke_n = _ES.collect_smoke_samples(frame_cache, rng)
        baseline_records, _baseline_n = _ES.collect_baseline_samples(fire_df, frame_cache, rng)
    finally:
        frame_cache.release_all()

    usage: dict[str, int] = {}
    selected: list[EffectFrameCandidate] = []
    for _lo, _hi, label in _ES.CHAIN_MAGNITUDE_BINS:
        bin_records = [r for r in burst_records if r.chain_bin == label]
        bin_pool = group_records_to_frames(bin_records, LAYER_BURST)
        selected += round_robin_select(bin_pool, BURST_TARGET_PER_BIN, MAX_CANDIDATES_PER_VIDEO, usage)

    smoke_pool = group_records_to_frames(smoke_records, LAYER_SMOKE)
    selected += round_robin_select(smoke_pool, SMOKE_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage)

    baseline_pool = group_records_to_frames(baseline_records, LAYER_BASELINE)
    selected += round_robin_select(baseline_pool, BASELINE_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage)
    return selected


# =============================================================================
# 4. 画像生成
# =============================================================================


def _frame_basename(c: EffectFrameCandidate) -> str:
    """ファイル名の共通部分 (video/t_sec/side/layerを含み衝突しないようにする)。"""
    return f"{c.video_stem}_t{c.t_sec:.2f}_{c.side}_{c.layer}"


def save_candidate_images(
    c: EffectFrameCandidate, frames_dir: Path,
) -> tuple["Path | None", "Path | None"]:
    """1候補分の実画面フルフレームPNG + 盤面クロップPNGを保存する (失敗時はNone)。"""
    video_path = resolve_cached_video_path(c.video_id)
    if video_path is None:
        return None, None
    frame = grab_frame(video_path, c.t_sec)
    if frame is None:
        return None, None
    base = _frame_basename(c)
    full_path = frames_dir / f"{base}_full.png"
    crop_path = frames_dir / f"{base}_board_crop.png"
    cv2.imwrite(str(full_path), frame)
    cv2.imwrite(str(crop_path), crop_visible_board_region(frame, c.side))
    return full_path, crop_path


# =============================================================================
# 5. 出力ファイル生成
# =============================================================================


def write_labeling_csv(rows: list[dict], out_path: Path) -> None:
    """user記入用ではなくツール入力用の labeling_sheet.csv を書き出す。"""
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEADER))
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in CSV_HEADER})


def _row_from_candidate(
    c: EffectFrameCandidate, full_path: "Path | None", crop_path: "Path | None",
) -> dict:
    """CSV/md 出力用の1候補分の辞書を組み立てる (画像パスはWindows形式)。"""
    return {
        "video_id": c.video_id, "t_sec": f"{c.t_sec:.2f}", "side": c.side,
        "layer": c.layer, "chain_bin": c.chain_bin,
        "image_full_frame": to_windows_path(full_path) if full_path else "(取得失敗)",
        "image_board_crop": to_windows_path(crop_path) if crop_path else "(取得失敗)",
    }


def _format_row_line(r: dict) -> str:
    """label_sheet.md の1候補分の行を組み立てる。"""
    layer_ja = LAYER_LABEL_JA.get(r["layer"], r["layer"])
    chain_note = f", 連鎖規模帯:{r['chain_bin']}" if r["chain_bin"] else ""
    return (
        f"- **{r['video_id']} {r['side']} t={r['t_sec']}秒** ({layer_ja}{chain_note}) — "
        f"実画面: {r['image_full_frame']} / 盤面クロップ: {r['image_board_crop']}"
    )


def write_label_sheet_md(rows: list[dict], out_path: Path) -> Path:
    """説明書き付きの label_sheet.md を書き出す。"""
    header = [
        "# エフェクト有無 セルラベルシート (2026-08-03)",
        "",
        "特性調査 (data/verify/effect_signature_study_2026-08-03) で有望特徴 (s_min、"
        "AUC0.822) が見つかったが、窓タイミングが機械推定 (正解ラベル無しの下限値) "
        "だったため、実際にエフェクトが写っているセルの人手ラベルを集めます。",
        "",
        "## お願い",
        "- ラベル付け自体は data/verify/effect_cell_label_2026-08-03/label_tool.html "
        "をブラウザで開いて行ってください (このmdは一覧参考用です)。",
        "- 各フレームの盤面クロップ上で、**予告おじゃまバースト (発光) または"
        "お邪魔落下の煙が実際に被っているセル**をクリックしてマークしてください。",
        "- エフェクトが全く見えないフレームは「エフェクトなし」ボタンを押してください。",
        f"- 総候補数: {len(rows)} 件",
        "",
        "## 候補一覧",
        "",
    ]
    body = [_format_row_line(r) for r in rows]
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    return out_path


# =============================================================================
# 6. 集計レポート
# =============================================================================


def summarize_selection(selected: list[EffectFrameCandidate]) -> str:
    """選定結果の分布 (レイヤー別/動画別) を平易な日本語でまとめる。"""
    by_layer: dict[str, int] = {}
    by_video: dict[str, int] = {}
    for c in selected:
        by_layer[c.layer] = by_layer.get(c.layer, 0) + 1
        by_video[c.video_stem] = by_video.get(c.video_stem, 0) + 1
    lines = [
        f"選定候補数: {len(selected)} 件 (動画数: {len(by_video)} 本)",
        f"レイヤー別: {dict(sorted(by_layer.items()))}",
        f"動画別件数: {dict(sorted(by_video.items()))}",
    ]
    return "\n".join(lines)


# =============================================================================
# メイン
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="エフェクト有無セルラベルシート準備")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """メイン処理: 窓収集(流用) -> 動画分散選定 -> 画像生成 -> CSV/md出力。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    print(f"[1/3] 窓収集(study流用) + 動画分散選定: 対象動画{len(EFFECT_LABEL_VIDEO_STEMS)}本")
    selected = collect_and_select_candidates(rng_seed=args.seed)
    print("  " + summarize_selection(selected).replace("\n", "\n  "))

    print("[2/3] 画像生成")
    frames_dir = args.out_dir / FRAMES_SUBDIR_NAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for c in selected:
        full_path, crop_path = save_candidate_images(c, frames_dir)
        if full_path is None:
            print(f"  [WARN] {c.video_id} t={c.t_sec:.2f} {c.side}: 取得失敗、スキップ")
            continue
        rows.append(_row_from_candidate(c, full_path, crop_path))
    print(f"  画像生成完了: {len(rows)} 件")

    print("[3/3] CSV/md 出力")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_labeling_csv(rows, args.out_dir / "labeling_sheet.csv")
    sheet_path = write_label_sheet_md(rows, args.out_dir / "label_sheet.md")
    print(f"  出力: {sheet_path}")
    print(f"\n[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
