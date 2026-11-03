"""評価器切替時の表示値の飛びを、保存済み表示列(display.npz)と events.jsonl から数える。

認識も評価も再実行しない純粋な後処理。表示値 = display_adv (EMA後の実表示) を勝率へ戻して比較する。
飛び = 同一試合内で 1 秒以内に |Δ勝率| >= JUMP_THRESHOLD となる変化。
各飛びを「直前の由来の切替 (from→to)」へ帰属し、物理イベント (発火・連鎖終了・着地・死亡・試合境界)
の近傍かどうかで 正当/切替のみ に分ける。母数 (フレーム数・切替数) を必ず併記する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JUMP_THRESHOLD = 0.15        # 勝率 (0〜1) の変化幅
JUMP_WINDOW_SEC = 1.0        # 変化を測る時間幅
CLUSTER_GAP_SEC = 1.0        # この間隔以内の連続検出は 1 件の飛びにまとめる
EVENT_BEFORE_SEC = 0.3       # 飛びの起点より前に物理イベントがあってよい幅
EVENT_AFTER_SEC = 0.3        # 飛びの起点より後に物理イベントがあってよい幅
FRAMES_PER_SEC = 30          # 表示列の標本化 (1/30 秒)
DEATH_SOURCES = ("confirmed_death", "unavoidable_death")
SOURCE_NONE = "(同一由来内)"


def winprob_k() -> float | None:
    """本番表示と同じ adv→勝率 変換の傾きを取り出す。"""
    from scripts.visualize_advantage_overlay import _WINPROB_K
    return _WINPROB_K


def adv_to_prob(adv: np.ndarray, k: float | None) -> np.ndarray:
    """本番 adv_to_winprob と同じ式を配列へ適用する。"""
    if k is not None:
        return 1.0 / (1.0 + np.exp(-k * adv))
    return np.clip(0.5 + adv / 200.0, 0.0, 1.0)


def physical_events(events: list[dict]) -> dict[int, dict[str, np.ndarray]]:
    """試合ごとに 発火/連鎖終了/着地/死亡 の時刻を種別別に集める。"""
    found: dict[int, dict[str, list[float]]] = {}
    for exchange in events:
        kinds = found.setdefault(exchange["game_idx"], {k: [] for k in
                                 ("fire", "chain_end", "landing", "death")})
        kinds["fire"].append(exchange["trigger_sec"])
        for chain in exchange["chains"]:
            kinds["fire"].append(chain["trigger_sec"])
            for key in ("end_signal_sec", "score_finalize_sec"):
                if chain.get(key) is not None:
                    kinds["chain_end"].append(chain[key])
        kinds["landing"] += [x["t_sec"] for x in exchange.get("landings", []) if x["t_sec"] is not None]
        kinds["death"] += [v["t_sec"] for v in exchange["values"] if v["source"] in DEATH_SOURCES]
    return {g: {k: np.sort(np.asarray(v, dtype=float)) for k, v in d.items()} for g, d in found.items()}


def near(times: np.ndarray, t: float, before: float, after: float) -> bool:
    """t の [-before, +after] に times が 1 つでもあるか。"""
    if len(times) == 0:
        return False
    index = int(np.searchsorted(times, t - before))
    return index < len(times) and times[index] <= t + after


def flag_frames(t: np.ndarray, p: np.ndarray, game: np.ndarray) -> np.ndarray:
    """各フレームで、直前 JUMP_WINDOW_SEC 内の同一試合の値との最大差が閾値以上か。"""
    flagged = np.zeros(len(t), dtype=bool)
    start = 0
    for i in range(len(t)):
        while t[i] - t[start] > JUMP_WINDOW_SEC:
            start += 1
        seg = slice(start, i)
        same = game[seg] == game[i]
        if same.any() and np.abs(p[seg][same] - p[i]).max() >= JUMP_THRESHOLD:
            flagged[i] = True
    return flagged


def cluster(flagged: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """検出フレームを時間的に近いもの同士でまとめ (開始, 終了) の添字区間にする。"""
    spans: list[tuple[int, int]] = []
    for i in np.flatnonzero(flagged):
        if spans and t[i] - t[spans[-1][1]] <= CLUSTER_GAP_SEC:
            spans[-1] = (spans[-1][0], int(i))
        else:
            spans.append((int(i), int(i)))
    return spans


def transitions(source: np.ndarray, game: np.ndarray) -> list[int]:
    """由来が変わったフレーム添字 (試合境界は除く)。"""
    return [i for i in range(1, len(source)) if source[i] != source[i - 1] and game[i] == game[i - 1]]


def step_sizes(p: np.ndarray, game: np.ndarray) -> np.ndarray:
    """隣接フレームの |Δ勝率|。試合境界をまたぐ差は 0 (別試合の値の差は飛びではない)。"""
    steps = np.abs(np.diff(p))
    steps[game[1:] != game[:-1]] = 0.0
    return np.concatenate([[0.0], steps])


def attribute(span: tuple[int, int], steps: np.ndarray, t: np.ndarray, source: np.ndarray,
              switches: list[int]) -> tuple[str, int]:
    """飛びの区間で最も急な 1 フレームを探し、直前 1 秒内の最後の切替 (from→to) を返す。"""
    lo = max(1, span[0] - int(JUMP_WINDOW_SEC * FRAMES_PER_SEC))
    steep = lo + int(np.argmax(steps[lo:span[1] + 1]))
    cands = [s for s in switches if t[steep] - JUMP_WINDOW_SEC <= t[s] <= t[steep] + 1e-9]
    if not cands:
        return SOURCE_NONE + str(source[steep]), steep
    s = cands[-1]
    return f"{source[s - 1]}→{source[s]}", steep


def classify(t_event: float, game: int, events: dict[int, dict[str, np.ndarray]],
             game_change: bool) -> str:
    """飛びの起点近傍の物理イベント種別 (なければ 'none')。優先: 試合境界>死亡>発火>連鎖終了>着地。"""
    if game_change:
        return "boundary"
    kinds = events.get(game, {})
    for kind in ("death", "fire", "chain_end", "landing"):
        if near(kinds.get(kind, np.empty(0)), t_event, EVENT_BEFORE_SEC, EVENT_AFTER_SEC):
            return kind
    return "none"


def measure(display: Path, events_path: Path) -> dict:
    """1 本の記録の飛びを集計する。"""
    data = np.load(display)
    t, game, source = data["t_sec"], data["game_idx"], data["source"]
    p = adv_to_prob(data["display_adv"], winprob_k())
    events = physical_events([json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()])
    switches = transitions(source, game)
    spans = cluster(flag_frames(t, p, game), t)
    steps = step_sizes(p, game)
    rows = []
    for span in spans:
        label, steep = attribute(span, steps, t, source, switches)
        # 起点 = 最急フレームの直前の切替時刻 (切替が無ければ最急フレーム)
        origin = float(t[steep])
        near_game = bool(game[max(0, span[0] - FRAMES_PER_SEC):span[1] + 1].min() != game[span[1]])
        rows.append(dict(t=origin, game=int(game[steep]), transition=label,
                         kind=classify(origin, int(game[steep]), events, near_game),
                         size=float(steps[steep]), switched=not label.startswith(SOURCE_NONE)))
    return dict(frames=len(t), switches=len(switches), jumps=rows,
                switch_pairs=_count_pairs(source, switches))


def _count_pairs(source: np.ndarray, switches: list[int]) -> dict[str, int]:
    """切替 (from→to) ごとの回数 = 切替側の母数。"""
    pairs: dict[str, int] = {}
    for s in switches:
        key = f"{source[s - 1]}→{source[s]}"
        pairs[key] = pairs.get(key, 0) + 1
    return pairs


def summarize(results: dict[str, dict]) -> dict:
    """全記録を合算し、遷移別・種別別の件数と 切替のみの飛び数 を出す。"""
    by_transition: dict[str, dict[str, int]] = {}
    kinds: dict[str, int] = {}
    total_jumps = total_frames = total_switches = non_event_switch = 0
    for r in results.values():
        total_frames += r["frames"]
        total_switches += r["switches"]
        for j in r["jumps"]:
            total_jumps += 1
            if j["kind"] == "none" and j["switched"]:
                non_event_switch += 1
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
            row = by_transition.setdefault(j["transition"], {})
            row[j["kind"]] = row.get(j["kind"], 0) + 1
    return dict(frames=total_frames, switches=total_switches, jumps=total_jumps,
                by_kind=kinds, non_event_jumps=kinds.get("none", 0),
                non_event_switch_jumps=non_event_switch,
                by_transition=dict(sorted(by_transition.items(), key=lambda kv: -sum(kv[1].values()))))


def main() -> None:
    """記録ディレクトリ群 (display.npz + events.jsonl) を測り JSON で出す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="名前=ディレクトリ (複数可)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for item in args.run:
        name, directory = item.split("=", 1)
        root = Path(directory)
        results[name] = measure(root / "display.npz", root / "events.jsonl")
    summary = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, runs=results), ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
