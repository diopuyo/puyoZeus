"""連鎖開始直前の設置未確定 (STABLE凍結漏れ) 頻度の実測 (2026-08-13)。

user懸念: 上部に置いて即連鎖が始まると、3フレーム安定の確定窓に間に合わず
連鎖直前の盤面 (=引き起こした設置) が未確定のまま連鎖に入ることがあるはず。
本スクリプトは data/indicators_v2/boards_lean_phase_l_2026-08-11/ の npz
(mtime が古い安定 63 本のみ、収集走行中の新規分は対象外) から、各 side の
連鎖トリガー (chain_trigger_sec が非NaNの行) ごとに直前の STABLE 確定
snapshot との時刻差を計測し、N 秒以内に確定盤面が無ければ未確定として頻度集計する。

重要な注意 (実データ確認済 2026-08-13): chain_trigger_sec/chain_mechanism は
「この snapshot 時点で有効な検知結果」を毎行キャリーフォワードする列であり、
同じ trigger_sec 値が連鎖の hold 期間中に生成される複数の STABLE snapshot に
重複して載る (実測で同じ値が最大10重複超)。そのため生の非NaN行数をそのまま
イベント数に使うと同一連鎖を10重以上に重複計上してしまう。本スクリプトは
側+ゲームごとに直前の値と異なる場合のみ新規イベントとして数える (連続重複除去)。


追加の重要な注意 (分析中に発覚、2026-08-13): chain_trigger_sec は「直前の
STABLE 確定 board の t_sec」そのものと数値的に一致するケースが baseline
(VideoChainTracker) / landing (着地直後即時連鎖判定) の両方で確認された
(spot check で複数件、誤差1e-2秒未満)。つまり本スクリプトの gap 計測
(trigger_sec 未満で直前の行を探索) は、baseline 機構については
「trigger_sec が指す参照 board 自体はもう1つ手前の行」という定義上の
ズレを検出しているだけで、真に「設置が未確定だったか」とは異なる量に
なっている可能性が高い (baseline は ChainSimulator.simulate で
chain_count>=1 が出る board しか参照できない設計のため、参照 board は
定義上すでに引き起こした設置を含む=常に「確定済」。未確定ケースは
むしろ baseline 自体が fail-silent で検知漏れになる側に出る可能性がある)。
gap ベースの数値 (N別未確定率・高さ層別) は方向性の参考値として報告するが、
定量的な採否根拠にはしない。最も defensible な数値は mechanism 別の
出現割合 (landing 機構=既存の特例ルートが実際に発火した頻度) とする。
設計思想: 判断や採否は行わない。頻度と層別だけを出す。
実行: WSL venv シングルスレッドで実行 (収集10並列と衝突しないよう軽量)。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOARD_ROWS = 13
BOARD_COLS = 6
COLOR_EMPTY = 0
COLOR_UNKNOWN = 10

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
STABLE_FILE_COUNT = 63

N_THRESHOLDS_SEC = (0.3, 0.5, 1.0)
HEIGHT_BINS = ((0, 3), (4, 6), (7, 9), (10, 13))
TRIGGER_DEDUP_EPS = 1e-4


def _pick_stable_files(npz_dir: Path, count: int) -> list[Path]:
    files = sorted(npz_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    return files[:count]


def _column_height(grid: np.ndarray, col: int) -> int:
    column = grid[:, col]
    concrete = np.where((column != COLOR_EMPTY) & (column != COLOR_UNKNOWN))[0]
    if len(concrete) == 0:
        return 0
    return BOARD_ROWS - int(concrete[0])


def _max_column_height(grid: np.ndarray) -> int:
    return max(_column_height(grid, c) for c in range(BOARD_COLS))


def _colored_puyo_total(grid: np.ndarray) -> int:
    return int(np.sum((grid >= 1) & (grid <= 5)))


def _height_bin_label(height: int) -> str:
    for lo, hi in HEIGHT_BINS:
        if lo <= height <= hi:
            return f"{lo}-{hi}"
    return "13+"


@dataclass
class ChainEventRecord:
    video_id: str
    side: str
    game_idx: int
    trigger_sec: float
    mechanism: str
    gap_sec: float
    has_prev: bool
    prev_max_height: int
    prev_puyo_total: int
    score_delta: float


def _extract_events_for_file(path: Path) -> list[ChainEventRecord]:
    with np.load(path, allow_pickle=True) as d:
        video_id = d["video_id"]
        side_arr = d["side"]
        t_sec = d["t_sec"].astype(np.float64)
        game_idx = d["game_idx"]
        grids = d["grids"]
        score = d["score"].astype(np.float64)
        if "chain_trigger_sec" not in d.files:
            return []
        trig = d["chain_trigger_sec"].astype(np.float64)
        mech = d["chain_mechanism"] if "chain_mechanism" in d.files else None

        n = len(t_sec)
        groups: dict[tuple[str, int], list[int]] = {}
        for i in range(n):
            key = (str(side_arr[i]), int(game_idx[i]))
            groups.setdefault(key, []).append(i)

        records: list[ChainEventRecord] = []
        for key, idxs in groups.items():
            side, gidx = key
            group_t = [t_sec[i] for i in idxs]
            last_trig_value: float | None = None
            for pos, i in enumerate(idxs):
                trigger_sec = trig[i]
                if np.isnan(trigger_sec):
                    continue
                if (
                    last_trig_value is not None
                    and abs(trigger_sec - last_trig_value) < TRIGGER_DEDUP_EPS
                ):
                    continue  # 同一連鎖のキャリーフォワード重複 (新規イベントでない)
                last_trig_value = trigger_sec
                mechanism = str(mech[i]) if mech is not None else ""
                prev_pos = None
                for p in range(pos - 1, -1, -1):
                    if group_t[p] < trigger_sec - 1e-6:
                        prev_pos = p
                        break
                if prev_pos is None:
                    records.append(
                        ChainEventRecord(
                            video_id=str(video_id[i]), side=side, game_idx=gidx,
                            trigger_sec=float(trigger_sec), mechanism=mechanism,
                            gap_sec=float("inf"), has_prev=False,
                            prev_max_height=-1, prev_puyo_total=-1,
                            score_delta=float("nan"),
                        )
                    )
                    continue
                prev_i = idxs[prev_pos]
                gap = float(trigger_sec - group_t[prev_pos])
                prev_grid = grids[prev_i]
                records.append(
                    ChainEventRecord(
                        video_id=str(video_id[i]), side=side, game_idx=gidx,
                        trigger_sec=float(trigger_sec), mechanism=mechanism,
                        gap_sec=gap, has_prev=True,
                        prev_max_height=_max_column_height(prev_grid),
                        prev_puyo_total=_colored_puyo_total(prev_grid),
                        score_delta=float(score[i] - score[prev_i]),
                    )
                )
        return records


def _summarize_overall(records: list[ChainEventRecord]) -> None:
    n = len(records)
    print(f"\n=== 全連鎖イベント数 (重複除去済): {n} "
          f"(has_prev=False={sum(1 for r in records if not r.has_prev)}) ===")
    print("\n--- N別 未確定率 (直前N秒以内にSTABLE確定盤面が無い割合) ---")
    for n_th in N_THRESHOLDS_SEC:
        unconfirmed = sum(1 for r in records if r.gap_sec > n_th)
        rate = unconfirmed / n if n else 0.0
        print(f"  N={n_th:.1f}s: {unconfirmed}/{n} = {rate*100:.2f}%")

    gaps_finite = [r.gap_sec for r in records if np.isfinite(r.gap_sec)]
    if gaps_finite:
        gaps_sorted = sorted(gaps_finite)
        print(f"\n  gap分布 (has_prev=Trueのみ, n={len(gaps_finite)}): "
              f"median={statistics.median(gaps_finite):.3f}s "
              f"p90={gaps_sorted[int(len(gaps_sorted)*0.9)]:.3f}s "
              f"max={max(gaps_finite):.3f}s")

    print("\n--- mechanism別 未確定率 (N=0.5s) ---")
    by_mech: dict[str, list[ChainEventRecord]] = {}
    for r in records:
        by_mech.setdefault(r.mechanism or "(空)", []).append(r)
    for mech, rs in sorted(by_mech.items(), key=lambda kv: -len(kv[1])):
        unconfirmed = sum(1 for r in rs if r.gap_sec > 0.5)
        print(f"  {mech:12s}: n={len(rs):5d}  未確定率={unconfirmed/len(rs)*100:.2f}%")


def _summarize_by_height(records: list[ChainEventRecord]) -> None:
    print("\n--- 高さ層別 未確定率 (直前STABLE盤面のmax_column_height, N=0.5s) ---")
    print("  (height_of は隠し段row0含む0-13、10以上=可視上部到達=危険域)")
    with_prev = [r for r in records if r.has_prev]
    by_bin: dict[str, list[ChainEventRecord]] = {}
    for r in with_prev:
        by_bin.setdefault(_height_bin_label(r.prev_max_height), []).append(r)
    for label in [f"{lo}-{hi}" for lo, hi in HEIGHT_BINS]:
        rs = by_bin.get(label, [])
        if not rs:
            print(f"  height {label:6s}: n=0")
            continue
        unconfirmed = sum(1 for r in rs if r.gap_sec > 0.5)
        avg_gap = statistics.mean(r.gap_sec for r in rs)
        print(f"  height {label:6s}: n={len(rs):5d}  未確定率={unconfirmed/len(rs)*100:.2f}%  "
              f"平均gap={avg_gap:.3f}s")

    puyo_totals = [r.prev_puyo_total for r in with_prev]
    gaps = [r.gap_sec for r in with_prev]
    if len(puyo_totals) > 2:
        try:
            corr = np.corrcoef(puyo_totals, gaps)[0, 1]
            print(f"\n  相関 (直前盤面の色ぷよ総数 vs gap_sec): r={corr:.3f} (n={len(puyo_totals)})")
        except Exception:
            pass


def _summarize_by_video(records: list[ChainEventRecord]) -> None:
    print("\n--- 動画別 未確定率 (N=0.5s、中央値と最悪ケース) ---")
    by_video: dict[str, list[ChainEventRecord]] = {}
    for r in records:
        by_video.setdefault(r.video_id, []).append(r)
    rates = []
    for vid, rs in by_video.items():
        n = len(rs)
        unconfirmed = sum(1 for r in rs if r.gap_sec > 0.5)
        rate = unconfirmed / n if n else 0.0
        rates.append((rate, vid, n, unconfirmed))
    rates.sort(reverse=True)
    if rates:
        rate_values = [r for r, _, _, _ in rates]
        print(f"  動画数={len(rates)}  中央値={statistics.median(rate_values)*100:.2f}%  "
              f"平均={statistics.mean(rate_values)*100:.2f}%")
        print("  ワースト10:")
        for rate, vid, n, unconfirmed in rates[:10]:
            print(f"    {vid:20s} n={n:4d}  未確定={unconfirmed:4d}  率={rate*100:.2f}%")
    else:
        print("  イベントなし")


def main() -> int:
    files = _pick_stable_files(NPZ_DIR, STABLE_FILE_COUNT)
    print(f"対象ファイル数: {len(files)}")
    print(f"  最古: {files[0].name} ({files[0].stat().st_mtime:.0f})")
    print(f"  最新: {files[-1].name} ({files[-1].stat().st_mtime:.0f})")

    all_records: list[ChainEventRecord] = []
    for path in files:
        try:
            all_records.extend(_extract_events_for_file(path))
        except Exception as exc:
            print(f"  [WARN] {path.name} 読込失敗: {exc}")

    if not all_records:
        print("連鎖イベントが1件も抽出できませんでした。")
        return 1

    _summarize_overall(all_records)
    _summarize_by_height(all_records)
    _summarize_by_video(all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
