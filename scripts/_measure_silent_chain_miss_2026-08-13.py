"""
連鎖検知のサイレント取り逃し測定 (2026-08-13)

目的:
    baseline (VideoChainTracker 事後検知) + landing (着地直後の設置推論による
    即時判定) の2経路が「両方とも」取り逃す連鎖検知イベントの頻度を、
    検知経路と独立なシグナル (score OCR 跳躍 + 盤面ぷよ総数変化) から測る。

原理 (docs/PUYO_RULES 準拠):
    連鎖が起きると score が跳躍する。
      - 最小: 4連結x10点x最小倍率1 = 40点 (MIN_CHAIN_SCORE)
      - 得点は常に BASE_SCORE_PER_PUYO=10 の倍数 (src/scoring.py)
      - 全消しボーナス (2100点) は「連鎖そのもの」ではなく、次の連鎖発火時に
        加算される (連鎖外の持ち越し)。単独では出現しない
    独立シグナルの条件:
        delta_score = score[j] - score[i] (同一 video/side/game_idx の
        連続STABLEスナップショット間) が 40 以上 かつ 10 の倍数
        なら「実際に連鎖が起きた」候補とみなす。
    npz の chain_trigger_sec / chain_mechanism 列 (baseline/landing/formula
    が書き込む) を同区間 (前後 TRIGGER_MATCH_WINDOW_SEC 秒のゆとり込み) で
    突き合わせ、記録が無ければサイレント取り逃しと判定する。

既知の制約 (解釈時に必読、報告にも明記):
    - score == -1 (OCR無効/試合開始前センチネル) の区間は判定不能でスコープ外
    - score OCR が動画全体で機能していない動画 (無効率50%超) は測定対象外
      (既知事例: c26/c58/c69, memory project_video_difficulty_3broken)
    - 試合最後の連鎖は次のSTABLEスナップショットが無く原理的に検出不可能。
      j がgroup内最後の行のペアは game末尾ペアとして別集計する
    - 短時間の連続連鎖は1ペアに融合されうる (粗い計測、下限的性質)
    - おじゃま流入等でpuyo数変化(drop)は相殺されうるため補助情報止まり
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY

MIN_CHAIN_SCORE = 40
SCORE_MULTIPLE = 10
ALL_CLEAR_BONUS = 2100
TRIGGER_MATCH_WINDOW_SEC = 5.0
SCORE_INVALID = -1
SCORE_OCR_BROKEN_VIDEO_RATIO = 0.5
UPPER_DANGER_ROWS = 4
CONTAM_CHECK_STEPS = 3  # 汚染区間判定: 候補ペア前後何行を確認するか

MAGNITUDE_SMALL_MAX = 100
MAGNITUDE_MEDIUM_MAX = 500

DEFAULT_NPZ_DIR = "data/indicators_v2/boards_lean_phase_l_2026-08-11"
DEFAULT_OUT_JSON = "logs/silent_chain_miss_measure_2026-08-13.json"


@dataclass
class CandidatePair:
    video_id: str
    side: str
    game_idx: int
    t_before: float
    t_after: float
    delta_score: int
    nz_before: int
    nz_after: int
    drop: int
    max_height_before: int
    upper_used_before: bool
    is_game_tail_pair: bool
    matched_mechanism: str
    all_clear_suspect: bool


def _height_and_upper(grid):
    """1枚の盤面grid(13x6)から(最大スタック高さ, 上部使用中フラグ)を返す。"""
    occ = grid != COLOR_EMPTY
    col_has = occ.any(axis=0)
    if not col_has.any():
        return 0, False
    topmost = np.where(col_has, occ.argmax(axis=0), BOARD_ROWS)
    heights = np.where(col_has, BOARD_ROWS - topmost, 0)
    upper_used = bool(occ[:UPPER_DANGER_ROWS, :].any())
    return int(heights.max()), upper_used


def _find_match(t_before, t_after, events, window):
    lo, hi = t_before - window, t_after + window
    for trig, mech in events:
        if lo <= trig <= hi:
            return mech
    return None


def _is_contaminated_neighborhood(order, score, pos, n):
    """周辺行が「非10の倍数の正deltaが続く」= 集計演出/リプレイ等の非試合
    汚染区間かどうかを判定する (実スコアは常に10の倍数、CONTAM_CHECK_STEPS
    行分を候補ペアの前後で確認する)。
    """
    lo = max(0, pos - CONTAM_CHECK_STEPS)
    hi = min(n - 1, pos + 1 + CONTAM_CHECK_STEPS)
    for k in range(lo, hi):
        a, b = order[k], order[k + 1]
        sa, sb = int(score[a]), int(score[b])
        if sa == SCORE_INVALID or sb == SCORE_INVALID:
            continue
        d = sb - sa
        if d > 0 and d % SCORE_MULTIPLE != 0:
            return True
    return False


def _process_group(video_id, side, game_idx, order, t_sec, score, grids,
                    trigger_sec, mechanism, stats):
    """(video_id, side, game_idx) 1グループ分を処理する。"""
    events = [
        (float(trigger_sec[k]), str(mechanism[k]))
        for k in order
        if str(mechanism[k]) != ""
    ]
    candidates = []
    n = len(order)
    for pos in range(n - 1):
        i, j = order[pos], order[pos + 1]
        si, sj = int(score[i]), int(score[j])
        if si == SCORE_INVALID or sj == SCORE_INVALID:
            stats["score_unavailable_pairs"] += 1
            continue
        delta = sj - si
        if delta < 0:
            stats["score_decrease_anomaly_pairs"] += 1
            continue
        if delta == 0:
            continue
        if delta < MIN_CHAIN_SCORE or delta % SCORE_MULTIPLE != 0:
            stats["noise_subthreshold_pairs"] += 1
            continue

        if _is_contaminated_neighborhood(order, score, pos, n):
            stats["contamination_excluded_pairs"] += 1
            continue

        nz_before = int(np.count_nonzero(grids[i]))
        nz_after = int(np.count_nonzero(grids[j]))
        max_h, upper_used = _height_and_upper(grids[i])
        is_tail = (pos + 1 == n - 1)
        all_clear_suspect = (
            (delta - ALL_CLEAR_BONUS) >= MIN_CHAIN_SCORE
            and (delta - ALL_CLEAR_BONUS) % SCORE_MULTIPLE == 0
        )
        matched = _find_match(
            float(t_sec[i]), float(t_sec[j]), events, TRIGGER_MATCH_WINDOW_SEC
        )
        candidates.append(
            CandidatePair(
                video_id=video_id, side=side, game_idx=game_idx,
                t_before=float(t_sec[i]), t_after=float(t_sec[j]),
                delta_score=delta, nz_before=nz_before, nz_after=nz_after,
                drop=nz_before - nz_after, max_height_before=max_h,
                upper_used_before=upper_used, is_game_tail_pair=is_tail,
                matched_mechanism=matched, all_clear_suspect=all_clear_suspect,
            )
        )
    return candidates


def process_video(path, stats):
    """1npzファイルを処理し、(候補ペア一覧, score無効率)を返す。"""
    d = np.load(path, allow_pickle=True)
    score = d["score"]
    invalid_ratio = float(np.mean(score == SCORE_INVALID))
    video_ids = d["video_id"]
    sides = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    grids = d["grids"]
    trigger_sec = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]

    if invalid_ratio >= SCORE_OCR_BROKEN_VIDEO_RATIO:
        return [], invalid_ratio

    vid = str(video_ids[0])
    all_candidates = []
    keys = defaultdict(list)
    for idx in range(len(video_ids)):
        keys[(str(sides[idx]), int(game_idx[idx]))].append(idx)

    for (side, g_idx), idxs in keys.items():
        order = np.array(sorted(idxs, key=lambda k: (float(t_sec[k]), k)))
        cands = _process_group(
            vid, side, g_idx, order, t_sec, score, grids,
            trigger_sec, mechanism, stats,
        )
        all_candidates.extend(cands)
    return all_candidates, invalid_ratio


def magnitude_bucket(delta):
    if delta <= MAGNITUDE_SMALL_MAX:
        return "small(<=100)"
    if delta <= MAGNITUDE_MEDIUM_MAX:
        return "medium(101-500)"
    return "large(>500)"


def height_bucket(max_h, upper_used):
    if upper_used:
        return "upper_used(row0-3有)"
    if max_h >= 6:
        return "mid(高さ6以上)"
    return "low(高さ6未満)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    stats = defaultdict(int)
    all_candidates = []
    broken_videos = []
    used_videos = []

    for p in paths:
        cands, invalid_ratio = process_video(p, stats)
        if invalid_ratio >= SCORE_OCR_BROKEN_VIDEO_RATIO:
            broken_videos.append((os.path.basename(p), invalid_ratio))
            continue
        used_videos.append(os.path.basename(p))
        all_candidates.extend(cands)

    non_tail = [c for c in all_candidates if not c.is_game_tail_pair]
    tail = [c for c in all_candidates if c.is_game_tail_pair]

    def miss_rate(cands):
        n_total = len(cands)
        n_miss = sum(1 for c in cands if c.matched_mechanism is None)
        rate = (n_miss / n_total * 100.0) if n_total else 0.0
        return n_total, n_miss, rate

    n_total, n_miss, rate = miss_rate(non_tail)
    n_total_tail, n_miss_tail, rate_tail = miss_rate(tail)

    mech_counter = defaultdict(int)
    for c in non_tail:
        mech_counter[c.matched_mechanism or "MISS"] += 1

    by_height = defaultdict(lambda: [0, 0])
    for c in non_tail:
        b = height_bucket(c.max_height_before, c.upper_used_before)
        by_height[b][0] += 1
        if c.matched_mechanism is None:
            by_height[b][1] += 1

    by_magnitude = defaultdict(lambda: [0, 0])
    for c in non_tail:
        b = magnitude_bucket(c.delta_score)
        by_magnitude[b][0] += 1
        if c.matched_mechanism is None:
            by_magnitude[b][1] += 1

    by_video = defaultdict(lambda: [0, 0])
    for c in non_tail:
        by_video[c.video_id][0] += 1
        if c.matched_mechanism is None:
            by_video[c.video_id][1] += 1

    print("=== 対象 ===")
    print(f"使用動画数: {len(used_videos)} / 全npz数: {len(paths)}")
    print(f"score OCR破綻で除外: {[v for v, _ in broken_videos]}")
    print()
    print("=== 独立シグナル検出 (game末尾ペアを除く) ===")
    print(f"候補ペア総数: {n_total}")
    print(f"記録あり(baseline/landing/formulaいずれか): {n_total - n_miss}")
    print(f"サイレント取り逃し (記録なし): {n_miss}  ({rate:.2f}%)")
    print(f"内訳: {dict(mech_counter)}")
    print()
    print("=== game末尾ペア (別カウント、終局連鎖欠落と混同しないこと) ===")
    print(f"候補ペア総数: {n_total_tail}, 記録なし: {n_miss_tail} ({rate_tail:.2f}%)")
    print()
    print("=== 診断カウンタ ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
    print("=== 層別: 盤面高さ (miss件数/総数) ===")
    for k, (tot, miss) in sorted(by_height.items()):
        pct = (miss / tot * 100.0) if tot else 0.0
        print(f"  {k}: {miss}/{tot} ({pct:.1f}%)")
    print()
    print("=== 層別: 連鎖規模 score跳躍幅 (miss件数/総数) ===")
    for k, (tot, miss) in sorted(by_magnitude.items()):
        pct = (miss / tot * 100.0) if tot else 0.0
        print(f"  {k}: {miss}/{tot} ({pct:.1f}%)")
    print()
    print("=== 層別: 動画別 (miss件数/総数, miss>0のみ表示) ===")
    for k, (tot, miss) in sorted(by_video.items(), key=lambda kv: -kv[1][1]):
        if miss > 0:
            print(f"  {k}: {miss}/{tot} ({miss/tot*100:.1f}%)")

    out = {
        "used_videos": used_videos,
        "broken_videos": broken_videos,
        "n_total": n_total,
        "n_miss": n_miss,
        "miss_rate_pct": rate,
        "mechanism_breakdown": dict(mech_counter),
        "tail_pairs": {
            "n_total": n_total_tail, "n_miss": n_miss_tail,
            "miss_rate_pct": rate_tail,
        },
        "diagnostics": dict(stats),
        "by_height": {k: {"total": v[0], "miss": v[1]} for k, v in by_height.items()},
        "by_magnitude": {k: {"total": v[0], "miss": v[1]} for k, v in by_magnitude.items()},
        "by_video": {k: {"total": v[0], "miss": v[1]} for k, v in by_video.items()},
        "all_miss_candidates": [
            {
                "video_id": c.video_id, "side": c.side, "game_idx": c.game_idx,
                "t_before": c.t_before, "t_after": c.t_after,
                "delta_score": c.delta_score, "drop": c.drop,
                "max_height_before": c.max_height_before,
                "upper_used_before": c.upper_used_before,
                "is_game_tail_pair": c.is_game_tail_pair,
                "all_clear_suspect": c.all_clear_suspect,
            }
            for c in all_candidates
            if c.matched_mechanism is None
        ],
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n詳細JSON: {args.out_json}")


if __name__ == "__main__":
    main()
