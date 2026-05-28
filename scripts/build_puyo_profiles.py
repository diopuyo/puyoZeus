"""per-video ぷよ色プロファイル DB 構築スクリプト (案 R3 改)。

data/per_video_hsv_ranges/v*.json の online_hsv_state.stats から
per-video プロファイル npz を構築し、全動画統合グローバルプロファイルも生成する。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.build_puyo_profiles

Output:
    data/puyo_profiles/{vid}_profile.npz   (per-video)
    data/puyo_profiles/global_profile.npz  (全動画統合)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# プロジェクトルートが PYTHONPATH に含まれている前提
from src.puyo_color_profile import (
    DEFAULT_PROFILE_DIR,
    GLOBAL_PROFILE_FILENAME,
    PUYO_PROFILE_GLOBAL_MIN_SAMPLES_FACTOR,
    PUYO_PROFILE_MIN_SAMPLES,
    ColorProfile,
    PuyoColorProfileDB,
)

# per-video HSV JSON ディレクトリ
HSV_RANGES_DIR: str = "data/per_video_hsv_ranges"

# online_hsv_state.stats が格納されている JSON キーパス
_KEY_ONLINE_HSV = "online_hsv_state"
_KEY_STATS = "stats"


def _load_stats_from_json(json_path: Path) -> tuple[str | None, dict | None]:
    """JSON から (video_id, stats dict) を返す。stats がなければ None。

    Args:
        json_path: per-video HSV JSON ファイルパス

    Returns:
        (video_id, stats) タプル。stats 不在は (video_id, None)
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {json_path.name}: JSON 読み込み失敗 ({e})")
        return None, None
    video_id: str | None = data.get("video_id") or json_path.stem
    online_hsv = data.get(_KEY_ONLINE_HSV)
    if not online_hsv:
        return video_id, None
    stats = online_hsv.get(_KEY_STATS)
    if not stats:
        return video_id, None
    return video_id, stats


def _build_per_video_profiles(
    hsv_dir: Path,
    profile_dir: Path,
) -> list[PuyoColorProfileDB]:
    """per-video プロファイルを構築・保存し、結果リストを返す。

    Args:
        hsv_dir: per_video_hsv_ranges ディレクトリ
        profile_dir: 保存先ディレクトリ

    Returns:
        構築に成功した PuyoColorProfileDB のリスト
    """
    json_files = sorted(hsv_dir.glob("v*.json"))
    built: list[PuyoColorProfileDB] = []
    skipped = 0
    for json_path in json_files:
        video_id, stats = _load_stats_from_json(json_path)
        if stats is None:
            skipped += 1
            continue
        db = PuyoColorProfileDB.from_online_hsv_state(stats, video_id=video_id)
        if not db.profiles:
            print(f"[skip] {video_id}: プロファイル 0 件 (全色サンプル不足)")
            skipped += 1
            continue
        out_path = profile_dir / f"{video_id}_profile.npz"
        db.save(out_path)
        n_colors = len(db.profiles)
        total_n = sum(p.n_samples for p in db.profiles.values())
        print(f"[save] {video_id}: {n_colors} 色, 合計 {total_n} サンプル → {out_path}")
        built.append(db)
    print(f"\nper-video: {len(built)} 件構築, {skipped} 件 skip")
    return built


def _combine_global_stats(
    dbs: list[PuyoColorProfileDB],
) -> dict[int, dict[str, float]]:
    """複数 DB から加重平均・組み合わせ分散を計算する。

    各色で:
        mean_total = Σ(n_i × mean_i) / Σ(n_i)
        var_total  = Σ(n_i × (var_i + (mean_i - mean_total)^2)) / Σ(n_i)

    PUYO_PROFILE_MIN_SAMPLES × PUYO_PROFILE_GLOBAL_MIN_SAMPLES_FACTOR
    (= 動画 3 本以上相当) 未満の色は除外する。

    Args:
        dbs: per-video PuyoColorProfileDB のリスト

    Returns:
        色コード → {"h_mean", "h_var", "s_mean", "s_var",
                     "v_mean", "v_var", "n"} の dict
    """
    # 色別に統計を集約
    accum: dict[int, list[tuple[int, float, float, float, float, float, float]]] = {}
    for db in dbs:
        for color, profile in db.profiles.items():
            if color not in accum:
                accum[color] = []
            # (n, h_mean, h_var, s_mean, s_var, v_mean, v_var)
            accum[color].append((
                profile.n_samples,
                profile.h_mean, profile.h_std ** 2,
                profile.s_mean, profile.s_std ** 2,
                profile.v_mean, profile.v_std ** 2,
            ))
    global_min = PUYO_PROFILE_MIN_SAMPLES * PUYO_PROFILE_GLOBAL_MIN_SAMPLES_FACTOR
    result: dict[int, dict[str, float]] = {}
    for color, entries in accum.items():
        total_n = sum(e[0] for e in entries)
        if total_n < global_min:
            continue
        # 加重平均 (H/S/V 共通)
        h_mean = sum(e[0] * e[1] for e in entries) / total_n
        s_mean = sum(e[0] * e[3] for e in entries) / total_n
        v_mean = sum(e[0] * e[5] for e in entries) / total_n
        # 組み合わせ分散
        h_var = sum(e[0] * (e[2] + (e[1] - h_mean) ** 2) for e in entries) / total_n
        s_var = sum(e[0] * (e[4] + (e[3] - s_mean) ** 2) for e in entries) / total_n
        v_var = sum(e[0] * (e[6] + (e[5] - v_mean) ** 2) for e in entries) / total_n
        result[color] = {
            "h_mean": h_mean, "h_var": h_var,
            "s_mean": s_mean, "s_var": s_var,
            "v_mean": v_mean, "v_var": v_var,
            "n": float(total_n),
        }
    return result


def _build_global_profile(
    dbs: list[PuyoColorProfileDB],
    profile_dir: Path,
) -> PuyoColorProfileDB:
    """全動画統合グローバルプロファイルを構築・保存する。

    Args:
        dbs: per-video PuyoColorProfileDB のリスト
        profile_dir: 保存先ディレクトリ

    Returns:
        グローバル PuyoColorProfileDB
    """
    global_stats = _combine_global_stats(dbs)
    if not global_stats:
        print("[warn] グローバルプロファイル: 条件を満たす色がゼロ")
        global_db = PuyoColorProfileDB(profiles={}, video_id=None)
    else:
        global_db = PuyoColorProfileDB.from_online_hsv_state(
            {str(k): v for k, v in global_stats.items()},
            video_id=None,
        )
    out_path = profile_dir / GLOBAL_PROFILE_FILENAME
    global_db.save(out_path)
    print(f"\n[global] {len(global_db.profiles)} 色 → {out_path}")
    for color, profile in sorted(global_db.profiles.items()):
        print(
            f"  color={color}: n={profile.n_samples}"
            f"  H={profile.h_mean:.1f}±{profile.h_std:.1f}"
            f"  S={profile.s_mean:.1f}±{profile.s_std:.1f}"
            f"  V={profile.v_mean:.1f}±{profile.v_std:.1f}"
        )
    return global_db


def main() -> None:
    """メインエントリポイント。"""
    hsv_dir = Path(HSV_RANGES_DIR)
    profile_dir = Path(DEFAULT_PROFILE_DIR)
    profile_dir.mkdir(parents=True, exist_ok=True)
    if not hsv_dir.exists():
        print(f"[error] HSV ディレクトリが見つかりません: {hsv_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"=== ぷよ色プロファイル構築 (案 R3 改) ===")
    print(f"入力: {hsv_dir}")
    print(f"出力: {profile_dir}")
    print()
    built_dbs = _build_per_video_profiles(hsv_dir, profile_dir)
    if not built_dbs:
        print("[error] per-video プロファイルが 1 件も構築できませんでした", file=sys.stderr)
        sys.exit(1)
    global_db = _build_global_profile(built_dbs, profile_dir)
    print(f"\n=== 完了 ===")
    print(f"  per-video: {len(built_dbs)} 件")
    print(f"  global: {len(global_db.profiles)} 色")


if __name__ == "__main__":
    main()
