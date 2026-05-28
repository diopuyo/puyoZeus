"""per-video HSV 範囲の yellow (color=4) を拡大するスクリプト。

処理内容:
  1. data/per_video_hsv_ranges/v*.json 全動画の yellow (key "4") 範囲を読み込む
  2. H/S/V 各方向を指定割合で拡大 (両端に幅の EXPAND_RATIO 分を追加)
  3. 他色との境界衝突回避: H_min の下限 (= 赤側境界) を H_MIN_LOWER_LIMIT でガード
  4. バックアップ: _backup_pre_expand_yellow/ に元ファイルを退避 (revert 可能)
  5. 上書きで拡大版を生成

拡大後の使用:
  - measure_stable_cell_acc.py で B0 比較 → yellow 改善 + 他色退行を確認
  - 採用 → そのまま (バックアップ削除は手動)
  - 却下 → scripts/revert_yellow_hsv_ranges.py でバックアップから復元

使い方:
    python scripts/expand_yellow_hsv_ranges.py
    python scripts/expand_yellow_hsv_ranges.py --expand-ratio 0.15 --dry-run
    python scripts/expand_yellow_hsv_ranges.py --revert

制約:
  - H_min 下限: H_MIN_LOWER_LIMIT (= 赤との境界保護)
  - H_max 上限: H_MAX_UPPER_LIMIT (= 緑との境界保護)
  - S/V min 下限: 0, max 上限: 255
  - _merged_default.json は yellow キーなしのため処理対象外
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from datetime import datetime
from typing import Optional

# ============================
# 定数定義
# ============================

# 拡大割合: 範囲幅の EXPAND_RATIO 分を両端に追加
# 例: H=[15,20] (幅5) で 0.10 → H=[14,21] に拡大
EXPAND_RATIO_DEFAULT: float = 0.10

# yellow H の下限ガード: 赤との重複防止
# DEFAULT_COLOR_RANGES 赤 H_max=13 → yellow H_min >= 14 を推奨
# ただし per-video red の H_max も考慮して引き下げは最小限に
H_MIN_LOWER_LIMIT: int = 10  # H=10 未満には下げない (赤 H~0-13 との衝突回避)

# yellow H の上限ガード: 緑との重複防止
# DEFAULT_COLOR_RANGES 緑 H_min=50 → yellow H_max <= 45 を推奨
H_MAX_UPPER_LIMIT: int = 45

# S の最小下限: 低 S は OJAMA (灰色) と混同リスク
# DEFAULT yellow s_min=100 なので 大幅な引き下げは避ける
S_MIN_LOWER_LIMIT: int = 0   # per-video 範囲はもともと多様なため制限しない

# 最小拡大幅 (= 幅が 1-2 と極端に狭い動画に対して最低保証)
MIN_EXPAND_PIXELS_H: int = 2   # H 両端最低 2 ずつ追加
MIN_EXPAND_PIXELS_S: int = 5   # S 両端最低 5 ずつ追加
MIN_EXPAND_PIXELS_V: int = 3   # V 両端最低 3 ずつ追加

# color key
YELLOW_KEY: str = "4"

# パス定義
_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HSV_DIR = _PROJ_ROOT / "data" / "per_video_hsv_ranges"
_BACKUP_DIR = _HSV_DIR / "_backup_pre_expand_yellow"


# ============================
# コア処理
# ============================

def _expand_range_1d(
    lo: int, hi: int,
    expand_ratio: float,
    min_expand: int,
    lower_limit: int,
    upper_limit: int,
) -> tuple[int, int]:
    """1 次元の [lo, hi] 範囲を拡大する。

    拡大量 = max(ceil(幅 * expand_ratio), min_expand)。
    結果を [lower_limit, upper_limit] にクリップする。

    Args:
        lo: 現在の下限値。
        hi: 現在の上限値。
        expand_ratio: 幅に対する拡大割合。
        min_expand: 最低拡大量 (幅が狭い場合の保証)。
        lower_limit: 下限クリップ値。
        upper_limit: 上限クリップ値。

    Returns:
        (new_lo, new_hi): 拡大後の範囲。
    """
    width = max(0, hi - lo)
    delta = max(min_expand, int(width * expand_ratio))
    new_lo = max(lower_limit, lo - delta)
    new_hi = min(upper_limit, hi + delta)
    return new_lo, new_hi


def _expand_yellow(
    yellow_range: list[int],
    expand_ratio: float,
) -> list[int]:
    """yellow (color=4) の [h_min, h_max, s_min, s_max, v_min, v_max] を拡大する。

    Args:
        yellow_range: [h_min, h_max, s_min, s_max, v_min, v_max] の 6 要素リスト。
        expand_ratio: 拡大割合 (0.0 ~ 1.0)。

    Returns:
        拡大後の 6 要素リスト。
    """
    h_min, h_max, s_min, s_max, v_min, v_max = yellow_range

    new_h_min, new_h_max = _expand_range_1d(
        h_min, h_max, expand_ratio, MIN_EXPAND_PIXELS_H,
        H_MIN_LOWER_LIMIT, H_MAX_UPPER_LIMIT,
    )
    new_s_min, new_s_max = _expand_range_1d(
        s_min, s_max, expand_ratio, MIN_EXPAND_PIXELS_S,
        S_MIN_LOWER_LIMIT, 255,
    )
    new_v_min, new_v_max = _expand_range_1d(
        v_min, v_max, expand_ratio, MIN_EXPAND_PIXELS_V,
        0, 255,
    )
    return [new_h_min, new_h_max, new_s_min, new_s_max, new_v_min, new_v_max]


def _process_file(
    json_path: pathlib.Path,
    expand_ratio: float,
    dry_run: bool,
) -> Optional[dict]:
    """1 ファイルの yellow 範囲を拡大し、変更差分 dict を返す。

    yellow キーが存在しない場合は None を返す (= _merged_default.json 等)。

    Returns:
        {"video_id": ..., "before": [...], "after": [...]} または None。
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ranges = data.get("per_video_ranges", {})
    if YELLOW_KEY not in ranges:
        return None

    before = list(ranges[YELLOW_KEY])
    after = _expand_yellow(before, expand_ratio)

    diff = {
        "video_id": data.get("video_id", json_path.stem),
        "before": before,
        "after": after,
        "h_delta": [before[0] - after[0], after[1] - before[1]],
        "s_delta": [before[2] - after[2], after[3] - before[3]],
        "v_delta": [before[4] - after[4], after[5] - before[5]],
    }

    if not dry_run:
        data["per_video_ranges"][YELLOW_KEY] = after
        data.setdefault("expand_yellow_history", []).append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "expand_ratio": expand_ratio,
            "before": before,
            "after": after,
        })
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return diff


def _backup_files(json_files: list[pathlib.Path], dry_run: bool) -> pathlib.Path:
    """バックアップディレクトリに元ファイルをコピーする。

    Args:
        json_files: バックアップ対象ファイルリスト。
        dry_run: True ならコピーしない (パスのみ表示)。

    Returns:
        バックアップディレクトリパス。
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_dir = _BACKUP_DIR / ts
    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in json_files:
            shutil.copy2(f, backup_dir / f.name)
        print(f"[expand_yellow] バックアップ完了: {backup_dir} ({len(json_files)} files)")
    else:
        print(f"[expand_yellow] [dry-run] バックアップ先: {backup_dir}")
    return backup_dir


def _collect_target_files() -> list[pathlib.Path]:
    """処理対象 JSON ファイルリストを返す (v*.json のみ)。"""
    return sorted(_HSV_DIR.glob("v*.json"))


# ============================
# revert 処理
# ============================

def _revert(dry_run: bool) -> int:
    """最新バックアップから JSON ファイルを復元する。

    Returns:
        0 = 成功、1 = バックアップなし。
    """
    if not _BACKUP_DIR.exists():
        print(f"[expand_yellow] バックアップなし: {_BACKUP_DIR}", file=sys.stderr)
        return 1
    backups = sorted(_BACKUP_DIR.iterdir(), reverse=True)
    if not backups:
        print(f"[expand_yellow] バックアップなし", file=sys.stderr)
        return 1
    latest = backups[0]
    print(f"[expand_yellow] revert 元: {latest}")
    restored = 0
    for src in sorted(latest.glob("*.json")):
        dst = _HSV_DIR / src.name
        if not dry_run:
            shutil.copy2(src, dst)
        restored += 1
        print(f"  {'[dry-run] ' if dry_run else ''}restore: {src.name}")
    print(f"[expand_yellow] revert 完了: {restored} files")
    return 0


# ============================
# サマリ表示
# ============================

def _print_summary(diffs: list[dict], expand_ratio: float, dry_run: bool) -> None:
    """拡大前後の差分サマリを表示する。"""
    prefix = "[dry-run] " if dry_run else ""
    print()
    print(f"{prefix}=== yellow HSV 拡大サマリ (expand_ratio={expand_ratio}) ===")
    print(
        f"{'video':8s} | "
        f"H_before        | H_after         | "
        f"S_before        | S_after         | "
        f"V_before        | V_after"
    )
    print("-" * 100)
    for d in diffs:
        b = d["before"]
        a = d["after"]
        print(
            f"{d['video_id']:8s} | "
            f"[{b[0]:3d},{b[1]:3d}](w={b[1]-b[0]:2d}) | "
            f"[{a[0]:3d},{a[1]:3d}](w={a[1]-a[0]:2d}) | "
            f"[{b[2]:3d},{b[3]:3d}](w={b[3]-b[2]:3d}) | "
            f"[{a[2]:3d},{a[3]:3d}](w={a[3]-a[2]:3d}) | "
            f"[{b[4]:3d},{b[5]:3d}](w={b[5]-b[4]:2d}) | "
            f"[{a[4]:3d},{a[5]:3d}](w={a[5]-a[4]:2d})"
        )
    print()
    print(f"{prefix}対象動画数: {len(diffs)}")
    print(
        f"{prefix}次のステップ: python scripts/measure_stable_cell_acc.py "
        f"--videos v29,v40,v89,v97 --output data/verify/stable_cell_acc/"
        f"yellow_expand_<datetime>.json"
    )


# ============================
# CLI
# ============================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="per-video HSV 範囲の yellow を拡大する",
    )
    p.add_argument(
        "--expand-ratio", type=float, default=EXPAND_RATIO_DEFAULT,
        help=f"拡大割合 (default={EXPAND_RATIO_DEFAULT})。幅の expand_ratio 分を両端に追加。",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="実際には書き込まず、変更内容のみ表示する。",
    )
    p.add_argument(
        "--revert", action="store_true",
        help="最新バックアップから JSON を復元する (expand の取り消し)。",
    )
    return p.parse_args()


def main() -> int:
    """終了コード: 0=成功、1=エラー。"""
    args = _parse_args()

    if args.revert:
        return _revert(dry_run=args.dry_run)

    target_files = _collect_target_files()
    if not target_files:
        print(f"[expand_yellow] 対象ファイルなし: {_HSV_DIR}", file=sys.stderr)
        return 1

    # バックアップ (dry_run でなければ実行)
    _backup_files(target_files, dry_run=args.dry_run)

    # yellow 範囲拡大
    diffs: list[dict] = []
    skipped = 0
    for f in target_files:
        diff = _process_file(f, expand_ratio=args.expand_ratio, dry_run=args.dry_run)
        if diff is None:
            skipped += 1
        else:
            diffs.append(diff)

    if skipped > 0:
        print(f"[expand_yellow] yellow キーなし (スキップ): {skipped} files")

    _print_summary(diffs, expand_ratio=args.expand_ratio, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
