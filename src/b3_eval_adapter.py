"""B3 新評価器の出力を、既存の M_time / M_agree / 誤確定 採点器が読む形へ変換する薄いアダプタ (事前登録 §4, §6)。

既存採点器 (exev: scripts/eval_set_score_20261001.py, scripts/_diag_set2_falsedeath_audit.py) は
再生原票ディレクトリ <dir>/display.npz (t_sec, display_p1, display_adv, source) と <dir>/events.jsonl を読む。
本モジュールはそれを書くだけで、採点式には触れない (定義の固定。別の LL 指標へ読み替えない)。

評価器の出力は時刻ごとの p (勝率) と死亡証明状態。行は更新契機ごとの不均一な時刻なので、
30Hz の表示格子へ「直前値の保持」で再標本化する (R の表示と同じ 30Hz 等間隔の物差しに合わせる)。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

DISPLAY_HZ = 30.0
ADV_SCALE = 100.0
SOURCE_PROB = "prob"
SOURCE_NULL = "null"
MAX_HOLD_SEC = 2.0                   # 行生成器の HELD_CAP_SEC と同じ。超えたら null
NULL_P = 0.5
SOURCE_UNAVOIDABLE = "unavoidable_death"     # 既存採点器 CERTAIN の語彙
SOURCE_CONFIRMED = "confirmed_death"
CERTAIN_SOURCES = (SOURCE_UNAVOIDABLE, SOURCE_CONFIRMED)
EXEV_SCORER = "scripts/eval_set_score_20261001.py"
EXEV_SCAN_MODULE = "scripts._diag_set2_falsedeath_audit"


@dataclass
class EvaluatorTrace:
    """新評価器の出力 (時刻順)。死亡確定は side ('1P'/'2P') と source で持つ。"""
    t_sec: np.ndarray            # 行の評価時刻 (単調非減少)
    p1: np.ndarray               # 平滑前の P(1P 勝ち)
    dead_side: Optional[np.ndarray] = None    # 各行: 0=なし, 1='1P'死亡確定, 2='2P'死亡確定
    dead_source: Optional[np.ndarray] = None  # 各行: 既存採点器の source 語彙 (確定行のみ)

    def __post_init__(self) -> None:
        self.t_sec = np.asarray(self.t_sec, dtype=np.float64)
        self.p1 = np.asarray(self.p1, dtype=np.float64)
        if len(self.t_sec) != len(self.p1) or len(self.t_sec) == 0:
            raise ValueError("t_sec と p1 の長さが不一致/空")
        if np.any(np.diff(self.t_sec) < 0):
            raise ValueError("t_sec が単調でない")
        if not np.isfinite(self.p1).all():
            raise ValueError("p1 に非有限値 (null は呼出側で 0.5 に課点してから渡す)")


def resample_hold(trace: EvaluatorTrace, start: float, end: float, hz: float = DISPLAY_HZ,
                  max_hold_sec: float = MAX_HOLD_SEC,
                  grid_times: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """[start, end) を hz 格子で直前値保持。

    最初の評価行より前、または直近の評価行から max_hold_sec を超えた格子点は null として p=0.5 で課点する
    (採点から消さない、事前登録 §2.4)。source は "null"。
    """
    grid = np.arange(start, end, 1.0 / hz) if grid_times is None else np.asarray(grid_times, dtype=float)
    idx = np.searchsorted(trace.t_sec, grid, side="right") - 1
    before = idx < 0
    idx = np.where(before, 0, idx)
    stale = before | (grid - trace.t_sec[idx] > max_hold_sec)
    p = np.where(stale, NULL_P, trace.p1[idx])
    source = np.full(len(grid), SOURCE_PROB, dtype="U24")
    if trace.dead_side is not None and trace.dead_source is not None:
        conf = (np.asarray(trace.dead_side)[idx] > 0) & ~stale
        source[conf] = np.asarray(trace.dead_source)[idx][conf]
    source[stale] = SOURCE_NULL
    return {"t_sec": grid, "display_p1": p, "display_adv": (p - 0.5) * 2.0 * ADV_SCALE, "source": source}


def write_replay_record(directory: Path, trace: EvaluatorTrace, windows: Sequence[tuple[float, float]],
                        events: Optional[Sequence[Mapping]] = None,
                        display_arrays: Optional[Mapping[str, np.ndarray]] = None) -> dict:
    """display.npz と events.jsonl を既存採点器の形式で書く。原票の hash を返す (受領票用)。"""
    import hashlib
    directory.mkdir(parents=True, exist_ok=True)
    if display_arrays is None:
        parts = [resample_hold(trace, a, b) for a, b in windows]
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    else:
        cat = dict(display_arrays)
    if len(cat["t_sec"]) == 0:
        raise ValueError("表示格子が空 (窓内に評価行が無い)")
    np.savez_compressed(directory / "display.npz", **cat)
    ev = [dict(e) for e in (events or ())]
    (directory / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev),
                                            encoding="utf-8")
    h = lambda f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
    return {"display_sha256": h("display.npz"), "events_sha256": h("events.jsonl"), "frames": int(len(cat["t_sec"]))}


def death_events_from_trace(trace: EvaluatorTrace, game_idx_of_time) -> list[dict]:
    """確定の初出を既存 events.jsonl 形式 ({game_idx, values:[{t_sec, source, dead_sides}]}) にする。"""
    if trace.dead_side is None or trace.dead_source is None:
        return []
    side_name = {1: "1P", 2: "2P"}
    out: dict[tuple, dict] = {}
    for t, s, src in zip(trace.t_sec, trace.dead_side, trace.dead_source):
        if int(s) <= 0:
            continue
        key = (int(game_idx_of_time(float(t))), side_name[int(s)], str(src))
        out.setdefault(key, {"game_idx": key[0], "values": [{"t_sec": float(t), "source": key[2], "dead_sides": [key[1]]}]})
    return list(out.values())


def load_module_from_file(path: Path, name: str):
    """自己完結した exev 採点器を、パッケージ名衝突なしで読み込む (importlib、ファイルパス指定)。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_m_time_agree(exev_root: Path, display_paths: Sequence[Path], labels_json: Path) -> dict:
    """既存 eval_set_score (M_time/M_agree/M_cp) をそのまま呼ぶ。式は変更しない。"""
    mod = load_module_from_file(exev_root / EXEV_SCORER, "b3_exev_eval_set_score")
    return mod.score([Path(p) for p in display_paths], Path(labels_json))


def run_false_death_scan(exev_root: Path, record_dirs: Sequence[Path], games: Sequence[Mapping],
                         python: str = sys.executable, timeout: int = 600) -> dict:
    """既存の誤確定走査 (_diag_set2_falsedeath_audit.scan) を exev を cwd にして子プロセスで呼ぶ。

    scripts パッケージ名が wt_b3 と衝突するため in-process では呼ばない。
    """
    code = ("import json,sys\nfrom pathlib import Path\n"
            f"from {EXEV_SCAN_MODULE} import scan\n"
            "a=json.loads(sys.stdin.read())\n"
            "print(json.dumps(scan([Path(p) for p in a['paths']], a['games']),ensure_ascii=False))\n")
    payload = json.dumps({"paths": [str(p) for p in record_dirs], "games": list(games)})
    r = subprocess.run([python, "-c", code], input=payload, capture_output=True, text=True, cwd=str(exev_root),
                       timeout=timeout, env=_clean_env())
    if r.returncode != 0:
        raise RuntimeError(f"誤確定走査が失敗: {r.stderr[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _clean_env() -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    return env
