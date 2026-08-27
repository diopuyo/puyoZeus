"""長時間ジョブの汎用 停滞・死亡検知 番人 (2026-08-18)。

## なぜ必要か

2026-08-18 に、切り離し起動した長時間ジョブ (番人 `_watchdog_regen148`、
先回りDL `_prefetch_dl_failures`、148本体 `_regen148_recollect`) が **3件とも
黙って死んだ**。いずれもログが更新されないまま停止しており、Monitor は
「ログに出た内容」を拾う仕組みのため、ログそのものが止まると何も通知できない。

実測した根因: `setsid -f ./venv/bin/python script.py > log 2>&1` という
python 直接起動は launch すらしないことがある (ログ0バイトのまま停止)。
148本体と同じ「シェルスクリプトを `setsid -f bash script.sh` で起動し、
スクリプト自身が `exec >> log 2>&1` でリダイレクトする」方式に統一すれば
起動は安定するが、**起動後に本当に生き続けているか・進捗しているか**は
別途 継続監視 しないと分からない。本スクリプトはその継続監視を担う。

## 「意図的な停止」と「異常な停止」の区別 (2026-08-18 追加、事故を受けて)

本スクリプトの初版を検証のため実行したところ、既存の旧番人
`_watchdog_regen148_2026-08-18.py` が生き残っていて再起動され、その旧番人が
「本体が異常終了した」と誤判定して 148本体 (収集方式を作り替え中で **意図的に
停止していた**) を勝手に再開してしまう事故が起きた (旧構成のまま16並列で
走り出し、途中で発見して手動停止)。

自動再起動は「止まっている＝異常」という前提だが、運用上は方式見直し・設定
変更・調査のために **意図的に止める場面がある**。この区別がないと同じ事故を
繰り返す。そのため:

- 対象ごとに `pause_flag` (ファイルパス) を設定でき、**そのファイルが存在する
  間は再起動しない**。存在確認のみ行い、本スクリプトは絶対にそのファイルを
  作成・削除しない (人間が置く/消すもの)。
- 加えて全対象共通の `global_pause_flag` (既定 `logs/.stall_watchdog_paused`)
  も用意した。本スクリプト自体の様子見・停止時に使う。
- フラグが立っている間も **無音にはしない** (無音だと「番人が死んでいる」の
  と区別がつかない)。`[PAUSED] ... 意図的停止中のため再起動しません` を
  間隔を空けて出し続ける。
- 「本体が異常終了したら再起動する」という旧番人 `_watchdog_regen148` の
  役割は本スクリプトの `regen148_recollect` 対象がそのまま引き継ぐ。
  **旧番人には停止フラグを後付けせず、今後は起動しない** (このファイルの
  設定に一本化し、旧スクリプトは履歴・参考用にファイルのみ残す)。

## 設計

- 監視対象は JSON 設定 (既定 `scripts/_stall_watchdog_targets_2026-08-18.json`)
  で外出しし、対象ごとに以下を持つ:
  - `process_pattern`: `pgrep -f` で生存確認するパターン
  - `progress` (省略可): 進捗の見方。`mtime` (ファイル更新時刻) /
    `line_count` (行数、`match` で部分一致フィルタ可) /
    `file_count` (ディレクトリ内ファイル数、`glob` パターン指定可) /
    `tsv_count` (TSV の特定列が特定値の行数)。**省略した場合は生存監視のみ**
    (正常運転中に長時間無出力になる番人プロセス向け)。
  - `stall_threshold_sec`: 進捗が増えない状態がこれを超えたら STALL
  - `restart_cmd` (省略可): プロセス死亡時に実行する再起動コマンド (list)
  - `min_restart_interval_sec` / `max_restarts`: 再起動の空回り・無限
    再起動を防ぐガード
  - `pause_flag` (省略可): このファイルが存在する間は死亡していても
    再起動しない (意図的停止の合図。人間が作成/削除する)

- 進捗値は「前回チェック時からの最大値を上回ったか」で判定する。TSV行数・
  ファイル数・mtime はいずれも正常運転下では単調非減少なので、同一ロジックで
  扱える。ロールバック (ログのローテーション等で値が下がる) は「基準リセット」
  として扱い、誤って STALL 扱いしない。

- STALL (プロセスは生きているが進捗が止まっている) は **ログ出力のみ**
  (自動再起動しない)。プロセスが本当にハングしているのか、正常な長時間処理
  なのか自動判断できないため。DEAD (プロセスが消えている) は設定があれば
  自動再起動する。

- 状態 (前回進捗値・最終再起動時刻・再起動回数) は `state_file` に永続化する。
  本スクリプト自身が再起動されても、直近の再起動履歴を引き継いで無限
  再起動ループを防ぐため。

## 使い方 (切り離し起動、CLAUDE.md プロセス管理ルール準拠)

    wsl -d Ubuntu -- bash -c "cd /mnt/c/.../puyo_analyzer && \
      setsid -f bash scripts/_run_stall_watchdog_2026-08-18.sh < /dev/null"

検証用に1周だけ実行して終了する `--once` と、設定ファイルを差し替える
`--config` を用意した。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "_stall_watchdog_targets_2026-08-18.json"

# 設定側で省略された場合のフォールバック既定値。
FALLBACK_STALL_THRESHOLD_SEC = 1800.0
FALLBACK_MIN_RESTART_INTERVAL_SEC = 1800.0
FALLBACK_MAX_RESTARTS = 10
FALLBACK_POLL_INTERVAL_SEC = 300.0
# 進捗省略ターゲット (生存監視のみ) の [ALIVE] / [PAUSED] ログを間引く間隔。
# ポーリング間隔設定に依存しない時間ベースにし、poll_interval を変えても
# 通知頻度が破綻しないようにする。
HEARTBEAT_LOG_INTERVAL_SEC = 3600.0  # 約1時間に1回
DEFAULT_GLOBAL_PAUSE_FLAG = "logs/.stall_watchdog_paused"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        with state_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # 破損していても監視を止めない。初期状態から再開する。
        return {}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """壊れた state で監視が止まらないよう、一時ファイル経由で原子的に書く。"""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp_path.replace(state_path)


def process_alive(pattern: str) -> bool:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(proc.stdout.strip())


def find_active_pause_flag(target: dict[str, Any], global_pause_flag: str) -> Path | None:
    """有効な停止フラグがあればそのパスを返す。**このファイルは絶対に
    作成・削除しない** (人間が置く/消す運用)。global を優先してチェックする。
    """
    global_path = PROJECT_ROOT / global_pause_flag
    if global_path.exists():
        return global_path
    target_flag = target.get("pause_flag")
    if target_flag:
        target_path = PROJECT_ROOT / target_flag
        if target_path.exists():
            return target_path
    return None


def _log_throttled(state_key: str, st: dict[str, Any], now: float, message: str) -> None:
    """同じ状態が続く間、`HEARTBEAT_LOG_INTERVAL_SEC` 間隔でのみ再掲する。
    状態が変わった直後は必ず出す (呼び出し側で was_* フラグを見て判断済み前提)。
    """
    last_log = st.get(state_key, 0.0)
    if now - last_log >= HEARTBEAT_LOG_INTERVAL_SEC:
        print(message, flush=True)
        st[state_key] = now


def _progress_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _progress_line_count(path: Path, match: str | None) -> float:
    if not path.exists():
        return 0.0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if match is None or match in line:
                count += 1
    return float(count)


def _progress_file_count(dir_path: Path, glob_pattern: str) -> float:
    if not dir_path.is_dir():
        return 0.0
    return float(len(list(dir_path.glob(glob_pattern))))


def _progress_tsv_count(path: Path, column: str, value: str) -> float:
    if not path.exists():
        return 0.0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get(column) == value:
                count += 1
    return float(count)


def get_progress_value(progress_cfg: dict[str, Any]) -> float:
    """進捗定義から現在値を取得する。値が大きいほど「進んでいる」とみなす。"""
    kind = progress_cfg["type"]
    path = PROJECT_ROOT / progress_cfg["path"]
    if kind == "mtime":
        return _progress_mtime(path)
    if kind == "line_count":
        return _progress_line_count(path, progress_cfg.get("match"))
    if kind == "file_count":
        return _progress_file_count(path, progress_cfg.get("glob", "*"))
    if kind == "tsv_count":
        return _progress_tsv_count(path, progress_cfg["column"], progress_cfg["value"])
    raise ValueError(f"未知の progress.type: {kind}")


def _try_restart(name: str, target: dict[str, Any], st: dict[str, Any], now: float) -> None:
    """DEAD 検知後の再起動処理。ガード (最低間隔・最大回数) を必ず通す。"""
    restart_cmd = target.get("restart_cmd")
    if not restart_cmd:
        print(f"[DEAD] {name}: 再起動コマンド未設定、警告のみ", flush=True)
        return
    max_restarts = target.get("max_restarts", FALLBACK_MAX_RESTARTS)
    min_interval = target.get("min_restart_interval_sec", FALLBACK_MIN_RESTART_INTERVAL_SEC)
    if st.get("restart_count", 0) >= max_restarts:
        print(
            f"[DEAD] {name}: 再起動上限({max_restarts}回)到達、自動再起動を停止 "
            f"(手動確認が必要)", flush=True,
        )
        return
    since_last = now - st.get("last_restart_time", 0.0)
    if st.get("last_restart_time") and since_last < min_interval:
        print(
            f"[DEAD] {name}: 再起動最低間隔未満 ({since_last:.0f}s < {min_interval:.0f}s)、待機",
            flush=True,
        )
        return
    print(f"[RESTART] {name}: 起動コマンド実行 {restart_cmd}", flush=True)
    subprocess.run(
        restart_cmd, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, check=False,
    )
    st["last_restart_time"] = now
    st["restart_count"] = st.get("restart_count", 0) + 1
    # 再起動直後は進捗の基準をリセットし、旧プロセスの停滞時間を引きずらない。
    st["last_value"] = None
    st["last_progress_time"] = now


def check_dead(
    name: str, target: dict[str, Any], st: dict[str, Any], now: float, global_pause_flag: str,
) -> bool:
    """プロセス死亡を確認し、意図的停止でなければ再起動する。死亡していれば True。

    停止フラグ (`pause_flag`/`global_pause_flag`) が存在する間は **絶対に
    再起動しない**。ただし無音にはせず `[PAUSED]` を間隔を空けて出し続ける
    (無音だと「番人自体が死んでいる」のと見分けがつかないため)。
    """
    if process_alive(target["process_pattern"]):
        st["was_paused"] = False
        return False
    pause_path = find_active_pause_flag(target, global_pause_flag)
    if pause_path is not None:
        message = f"[PAUSED] {name}: 意図的停止中のため再起動しません (flag={pause_path})"
        if not st.get("was_paused"):
            print(message, flush=True)
            st["last_paused_log_time"] = now
        else:
            _log_throttled("last_paused_log_time", st, now, message)
        st["was_paused"] = True
        return True
    st["was_paused"] = False
    print(f"[DEAD] {name}: プロセス未検出 (pattern={target['process_pattern']})", flush=True)
    _try_restart(name, target, st, now)
    return True


def check_stall(name: str, target: dict[str, Any], st: dict[str, Any], now: float) -> None:
    """生存中の進捗停滞を確認する。自動再起動はしない (警告のみ)。"""
    progress_cfg = target.get("progress")
    if progress_cfg is None:
        return
    value = get_progress_value(progress_cfg)
    prev_value = st.get("last_value")
    if prev_value is None or value > prev_value:
        if st.get("was_stalled"):
            print(f"[RECOVER] {name}: 進捗再開 (value={value})", flush=True)
        st["last_value"] = value
        st["last_progress_time"] = now
        st["was_stalled"] = False
        return
    threshold = target.get("stall_threshold_sec", FALLBACK_STALL_THRESHOLD_SEC)
    elapsed = now - st.get("last_progress_time", now)
    if elapsed > threshold:
        print(
            f"[STALL] {name}: 進捗が{elapsed / 60:.0f}分停滞 "
            f"(閾値{threshold / 60:.0f}分、value={value})", flush=True,
        )
        st["was_stalled"] = True


def run_once(config: dict[str, Any], state: dict[str, Any], global_pause_flag: str) -> None:
    now = time.time()
    for target in config["targets"]:
        name = target["name"]
        st = state.setdefault(name, {})
        if check_dead(name, target, st, now, global_pause_flag):
            continue
        check_stall(name, target, st, now)
        if target.get("progress") is None:
            _log_throttled(
                "last_heartbeat_time", st, now,
                f"[ALIVE] {name}: 生存確認のみ (進捗監視なし)",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--once", action="store_true", help="1周だけ実行して終了 (検証用)")
    parser.add_argument("--poll-interval", type=float, default=None, help="ポーリング間隔上書き")
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = PROJECT_ROOT / config.get("state_file", "logs/_stall_watchdog_state.json")
    poll_interval = args.poll_interval or config.get("poll_interval_sec", FALLBACK_POLL_INTERVAL_SEC)
    global_pause_flag = config.get("global_pause_flag", DEFAULT_GLOBAL_PAUSE_FLAG)

    print(f"[stall_watchdog] 開始 対象={[t['name'] for t in config['targets']]}", flush=True)
    state = load_state(state_path)
    while True:
        run_once(config, state, global_pause_flag)
        save_state(state_path, state)
        if args.once:
            return 0
        time.sleep(poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
