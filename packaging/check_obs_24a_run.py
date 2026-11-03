"""24a の各走行を PREV / NEW の期待と照合する。標準ライブラリだけで動く。

10 フラグの記録・環境変数・pipeline 設定、認識側の R37・組判定、暫定 X と先読みの痕跡を検査。
製品側の腕分類 ALL / CUSTOM は維持し、PREV / NEW へ対応付ける。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

X_FLAGS = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire')
SPEED_FLAGS = ('live_frame_drop_robust', 'b3_warning_batch', 'b3_warning_prefetch')
NEW_FLAGS = ('live_lockdown_score_zero_then_move_release', 'live_match_end_pair', 'live_verify_reason_log')
TRIAL_FLAGS = X_FLAGS+SPEED_FLAGS+NEW_FLAGS+('b3_warning_async', 'live_native_hsv_main', 'live_verify_async')
TRIAL_FLAGS += ('live_match_end_feed_on_hold', 'match_end_yatta_text_template')
TRIAL_FLAGS += ('live_capture_reconnect', 'b3_worker_graceful_stop')
TRIAL_FLAGS += ('live_board_vectorized',)
TRIAL_FLAGS += ('live_verify_soft_hold',)
ARM_FLAGS = {'PREV': X_FLAGS+SPEED_FLAGS, 'NEW': (X_FLAGS[0],)+SPEED_FLAGS+NEW_FLAGS}
RECORD_ARMS = {'PREV': 'ALL', 'NEW': 'CUSTOM'}
RECOGNITION_FLAGS = dict(live_lockdown_score_zero_then_move_release='enable_lockdown_score_zero_then_move_release',
                         live_match_end_pair='enable_match_end_pair')
FLAGS_FILE = 'live_trial_flags.json'
PIPELINE_FILE = 'pipeline_config.json'
RECOGNITION_FILE = 'recognition_config.json'
V49F_STATUS_FILE = 'v49f_display_status.json'
PREFIRE_FILES = ('live_x_prefire.jsonl', 'live_x_prefire_summary.json')
TIMING_FILE = 'spool/frame_timing_recognition.jsonl'
PREFETCH_STAGE = '"b3_warning_prefetch"'
TIMING_SCAN_LINES = 3000          # 段の名前は 1 コマ目から出るので、先頭だけ読めば足りる
MEDIA_TICK_KEY = 'enable_media_tick_counters'
RULES_OFF_KEY = 'live_frame_drop_rules_off'      # コマ落ちに強い状態判定で外す箇所 (live_trial_flags.json では frame_drop_rules_off)
RECORD_RULES_KEY = 'frame_drop_rules_off'
RECORD_SITES_KEY = 'frame_drop_robust_sites'      # production_config.LIVE_FRAME_DROP_ROBUST_SITES (採否の単一情報源)
DEFAULT_SITES: tuple[str, ...] = ()                # 2026-10-23 夜の既定 (R06・R10・R15 を外す = 空。R15 は obs21c で +0.086pt)


def read_json(path: Path) -> Any:
    """無い・壊れているときは None (呼び出し側が「未記録」として不合格にする)。"""
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def prefetch_stage_seen(run: Path) -> bool | None:
    """frame_timing の先頭に b3_warning_prefetch の段があるか。ファイルが無ければ None。"""
    path = run/TIMING_FILE
    if not path.is_file():
        return None
    with path.open(encoding='utf-8', errors='replace') as stream:
        return any(PREFETCH_STAGE in line for _, line in zip(range(TIMING_SCAN_LINES), stream))


def entry(name: str, expected: Any, actual: Any) -> dict:
    return dict(name=name, expected=expected, actual=actual, ok=expected == actual)


def flag_checks(run: Path, arm: str, rules_off: tuple[str, ...] = (),
                sites: tuple[str, ...] = DEFAULT_SITES) -> list[dict]:
    """live_trial_flags.json の腕・フラグ・環境変数・外した箇所を期待と突き合わせる。"""
    record = read_json(run/FLAGS_FILE)
    if not isinstance(record, dict):
        return [entry(FLAGS_FILE+' がある', True, False)]
    on = set(ARM_FLAGS[arm])
    flags = record.get('flags', {})
    env = record.get('env', {})
    checks = [entry('記録の腕', RECORD_ARMS[arm], record.get('arm')),
              entry('外した箇所 (frame_drop_rules_off)', sorted(rules_off), sorted(record.get(RECORD_RULES_KEY, []))),
              entry('採否 (frame_drop_robust_sites)', sorted(sites), sorted(record.get(RECORD_SITES_KEY, [])))]
    for key in TRIAL_FLAGS:
        checks.append(entry(f'flags.{key}', key in on, flags.get(key)))
        checks.append(entry(f'env.PUYO_{key.upper()}', '1' if key in on else '0', env.get('PUYO_'+key.upper())))
    return checks


def config_checks(run: Path, arm: str, rules_off: tuple[str, ...] = ()) -> list[dict]:
    """pipeline_config.json (要求) と recognition_config.json (認識 process の実効値)。"""
    on = set(ARM_FLAGS[arm])
    pipeline, recognition = read_json(run/PIPELINE_FILE), read_json(run/RECOGNITION_FILE)
    if not isinstance(pipeline, dict) or not isinstance(recognition, dict):
        return [entry(f'{PIPELINE_FILE}・{RECOGNITION_FILE} がある', True, False)]
    checks = [entry(f'pipeline.{key}', key in on, bool(pipeline.get(key))) for key in TRIAL_FLAGS]
    checks.append(entry(f'recognition.{MEDIA_TICK_KEY}', 'live_frame_drop_robust' in on,
                        bool(recognition.get(MEDIA_TICK_KEY, False))))
    for flag, key in RECOGNITION_FLAGS.items():
        checks.append(entry(f'recognition.{key}', flag in on, bool(recognition.get(key, False))))
    checks.append(entry(f'pipeline.{RULES_OFF_KEY}', sorted(rules_off), sorted(pipeline.get(RULES_OFF_KEY, []))))
    return checks


def trace_checks(run: Path, arm: str) -> list[dict]:
    """X の痕跡 (直接公開・暫定 X の記録) と、先読みの段。"""
    on = set(ARM_FLAGS[arm])
    status = read_json(run/V49F_STATUS_FILE)
    checks = [entry(f'{V49F_STATUS_FILE}.direct_publish', 'live_x_direct_publish' in on,
                    bool(status.get('direct_publish')) if isinstance(status, dict) else None)]
    for name in PREFIRE_FILES:
        checks.append(entry(f'{name} がある', 'live_x_prefire_shadow' in on, (run/name).is_file()))
    checks.append(entry('frame_timing に b3_warning_prefetch の段', 'b3_warning_prefetch' in on,
                        prefetch_stage_seen(run)))
    return checks


def check_run(run: Path, arm: str, rules_off: tuple[str, ...] = (), sites: tuple[str, ...] = DEFAULT_SITES) -> dict:
    """走行の記録 run を腕 arm の期待と突き合わせる。passed は全件一致のときだけ True (母数 = checks の件数)。

    rules_off はコマ落ちに強い状態判定で外した箇所 (両腕。既定は空 = 全部 ON の期待)。
    sites は production_config の採否 (既定は空)。どの腕の記録にも同じ値が残っているはず。
    """
    if arm not in ARM_FLAGS:
        raise ValueError(f'腕は {tuple(ARM_FLAGS)} のどれか: {arm!r}')
    checks = flag_checks(run, arm, rules_off, sites)+config_checks(run, arm, rules_off)+trace_checks(run, arm)
    failed = [c for c in checks if not c['ok']]
    return dict(run=str(run), arm=arm, rules_off=sorted(rules_off), sites=sorted(sites), total=len(checks), failed=len(failed), passed=not failed,
                failures=failed, checks=checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path, help='走行の記録 (output/日時)')
    parser.add_argument('--arm', choices=tuple(ARM_FLAGS), required=True)
    parser.add_argument('--rules-off', default='', help='両腕: 外した箇所 (例 R04,R35)。省略 = 全箇所 ON の期待')
    parser.add_argument('--sites', default=','.join(DEFAULT_SITES), help='production_config の採否の期待 (既定は空)')
    parser.add_argument('--out', type=Path, help='結果の JSON を書く場所 (省略時は標準出力だけ)')
    args = parser.parse_args()
    split = lambda text: tuple(r.strip() for r in text.split(',') if r.strip())  # noqa: E731
    result = check_run(args.run, args.arm, split(args.rules_off), split(args.sites))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(text)
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
