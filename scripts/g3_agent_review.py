"""G3の工程変化だけをClaude CLI Agent SDKへ渡す有限の接続口。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator
import uuid

from scripts.agent_output_budget import preview

VERIFY = Path('D:/puyo_analyzer/verify' if os.name == 'nt' else '/mnt/d/puyo_analyzer/verify')
ROOT = Path(__file__).resolve().parents[1]
MAX_EVIDENCE_CHARS, MAX_PACKET_CHARS, MAX_FILE_BYTES = 6000, 24000, 2_000_000
MAX_SLOTS, TIMEOUT_SECONDS, STOP_SECONDS = 2, 240, 10
DISPLAY_CHARS = 4500
MAX_RESPONSE_TOKENS = 8192  # 3500では思考中断→CLI内部再生成が4回発生した。
RESPONSE_POLL_SECONDS = 0.05
EVENTS = frozenset({'design', 'manufacturing_diff', 'run_failed', 'evidence_changed', 'needs_user', 'first_pass_finished'})
MODELS = frozenset({'claude-fable-5-1', 'claude-opus-5'})
FIELDS = frozenset({'event_id', 'event', 'unit', 'purpose', 'completion', 'scope', 'delta', 'unresolved', 'evidence'})
SYSTEM = ('あなたはぷよ解析G3の独立判断担当。日本語で簡潔に回答する。ReadでWORK_PACKET.mdを実読し、'
          'その固定版コードと原証拠の抜粋だけを判断する。欠測と未読を明記し、自己検収を独立合格にしない。'
          '変更、コマンド実行、再委譲、学習、本番採用、G4以降の開始は禁止。'
          '返答は結論、根拠file:line、未完、次条件。重要指摘は最大3件、残る重大未完も明記する。')


def encoded(value: Any) -> bytes:
    """同じ意味のJSONは同じbyte列にする。時刻は呼出内容へ入れない。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def save(path: Path, value: Any) -> None:
    """原票は一回だけ作成し、実書込を同期する。"""
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def read(path: Path) -> dict[str, Any]:
    """破損票を空の正常状態に置き換えない。"""
    return json.loads(path.read_text(encoding='utf-8'))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def evidence(item: dict[str, Any]) -> dict[str, Any]:
    """許可したテキスト範囲を実読する。省略して成功にはしない。"""
    path = Path(item['path']).resolve()
    if not any(under(path, root) for root in (ROOT, VERIFY)):
        raise ValueError('evidence_outside_project')
    if any(part in str(path).lower() for part in ('formal100', 'hidden_reserve', '.credentials', '.env')):
        raise ValueError('protected_evidence')
    if path.suffix.lower() not in {'.py', '.sh', '.md', '.json', '.jsonl', '.log', '.txt', '.toml'}:
        raise ValueError('evidence_not_text')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('evidence_file_too_large_use_existing_summary')
    raw = path.read_bytes()
    if sha(raw) != item['sha256']:
        raise ValueError('evidence_sha_changed:' + str(path))
    lines = raw.decode('utf-8').splitlines()
    first, last = item['first_line'], item['last_line']
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines):
        raise ValueError('evidence_line_range')
    excerpt = '\n'.join(f'{n}: {lines[n-1]}' for n in range(first, last + 1))
    if len(excerpt) > MAX_EVIDENCE_CHARS:
        raise ValueError('evidence_excerpt_too_large')
    return dict(path=str(path), sha256=sha(raw), first_line=first, last_line=last, excerpt=excerpt)


def prepare(packet: dict[str, Any], model: str) -> tuple[str, str]:
    """呼出契約と証拠を固定する。IDだけを変えた同内容も再利用する。"""
    if set(packet) != FIELDS or packet['event'] not in EVENTS or model not in MODELS:
        raise ValueError('g3_packet_contract')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', packet['event_id']):
        raise ValueError('event_id_invalid')
    if not packet['evidence'] or not packet['unit'] or not packet['scope']:
        raise ValueError('missing_scope_or_evidence')
    content = {key: value for key, value in packet.items() if key != 'event_id'}
    content['evidence'] = [evidence(item) for item in packet['evidence']]
    body = json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2)
    if len(body) > MAX_PACKET_CHARS:
        raise ValueError('packet_too_large')
    key = sha(encoded({'system': SYSTEM, 'content': content, 'model': model, 'effort': 'medium',
                       'max_response_tokens': MAX_RESPONSE_TOKENS}))
    return key, body


@contextmanager
def slot(root: Path) -> Iterator[None]:
    """このG3接続口の同時担当を2以内にする。死slotを勝手に回収しない。"""
    slots = root / 'slots'
    slots.mkdir(exist_ok=True)
    owner = dict(owner_pid=os.getpid(), claim_id=str(uuid.uuid4()), created_epoch=time.time())
    claimed = None
    for index in range(MAX_SLOTS):
        path = slots / f'{index}.json'
        try:
            save(path, owner)
            claimed = path
            break
        except FileExistsError:
            continue
    if claimed is None:
        raise RuntimeError('claude_slots_full_reconcile_actual_processes')
    try:
        yield
    finally:
        if read(claimed) != owner:
            raise RuntimeError('slot_ownership_changed')
        claimed.unlink()


def existing(root: Path, event_id: str, key: str) -> dict[str, Any] | None:
    """未完票も含めて再送を抑止し、同IDの内容差替えを拒否する。"""
    event_path = root / 'events' / f'{event_id}.json'
    if event_path.exists():
        prior = read(event_path)
        if prior['key'] != key:
            raise ValueError('event_content_conflict:existing_key=' + prior['key'])
    request = root / 'requests' / key
    if not request.exists() and not event_path.exists():
        return None
    if not event_path.exists():
        try:
            save(event_path, dict(key=key, state='REUSED'))
        except FileExistsError:
            if read(event_path)['key'] != key:
                raise ValueError('event_content_conflict')
    result = read(request / 'RESULT.json') if (request / 'RESULT.json').is_file() else None
    launch = read(request / 'LAUNCH.json') if (request / 'LAUNCH.json').is_file() else None
    identity = process_identity(launch['pid']) if result is None and launch else None
    return dict(reused=True, status=result['status'] if result else 'UNKNOWN_NO_RESEND',
                request=str(request), result=result, new_cli_invocations=0,
                current_process_identity=identity, saved_launch=launch if result is None else None,
                process_match=reconcile_identity(launch, identity) if result is None else 'WAITED')


def reconcile_identity(launch: dict | None, current: dict | None) -> str:
    """死亡・同じ起動・PID再利用を区別し、不明票から再送しない。"""
    if not launch or not current:
        return 'UNKNOWN'
    if current['state'] == 'absent':
        return 'ABSENT'
    saved = launch.get('process_identity', {})
    if saved.get('state') != 'present' or current['state'] != 'present':
        return 'UNKNOWN'
    return 'SAME_PROCESS' if saved['start_identity'] == current['start_identity'] else 'PID_REUSED'


def process_identity(pid: int) -> dict[str, Any]:
    """PID再利用を識別する実OS開始値を取得する。取得不能を死亡と断定しない。"""
    if os.name != 'nt':
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
            return dict(pid=pid, state='present', start_identity=fields[19])
        except FileNotFoundError:
            return dict(pid=pid, state='absent')
        except (OSError, ValueError, IndexError):
            return dict(pid=pid, state='unknown')
    script = (f'$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; '
              'if ($p) { try { $p.StartTime.ToUniversalTime().Ticks } catch { "unknown" } } else { "absent" }')
    try:
        result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                                capture_output=True, text=True, timeout=STOP_SECONDS)
        value = result.stdout.strip()
        if result.returncode == 0 and value.isdigit():
            return dict(pid=pid, state='present', start_identity=value)
        return dict(pid=pid, state='absent' if result.returncode == 0 and value == 'absent' else 'unknown')
    except (OSError, subprocess.TimeoutExpired):
        return dict(pid=pid, state='unknown')


def command(model: str, session: str, packet: Path) -> list[str]:
    """公式CLI Agent SDKをReadだけで使い、設定探索とMCPを無効化する。"""
    prompt = f'Readで次の絶対pathを一度実読し、指定範囲を検収してください: {packet.resolve()}'
    return ['claude', '-p', prompt,
            '--model', model, '--effort', 'medium', '--system-prompt', SYSTEM,
            '--tools', 'Read', '--allowedTools', 'Read', '--permission-prompts', 'none',
            '--restricted', '--disable-slash-commands', '--setting-sources', '',
            '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
            '--output-format', 'stream-json', '--include-partial-messages', '--verbose', '--no-session-persistence',
            '--session-id', session]


def stop(child: subprocess.Popen[bytes]) -> int:
    """自分が起動したPopenだけを終了し、実waitまで所有する。"""
    if child.poll() is None:
        child.terminate()
    try:
        return child.wait(timeout=STOP_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def execute(request: Path, argv: list[str], timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    """PID保存前の例外でも起動済みchildを解放してから終了票を残す。"""
    started, child, error, code = time.time(), None, None, None
    env = os.environ.copy()
    env.update(CLAUDE_CODE_MAX_RETRIES='1', CLAUDE_CODE_EFFORT_LEVEL='medium',
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS='1',
               CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK='1',
               CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(MAX_RESPONSE_TOKENS))
    try:
        with (request / 'stdout.jsonl').open('xb') as out, (request / 'stderr.log').open('xb') as err:
            child = subprocess.Popen(argv, cwd=request, env=env, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err)
            save(request / 'LAUNCH.json', dict(pid=child.pid, parent_pid=os.getpid(),
                                              started_epoch=time.time(), state='PID_SAVED',
                                              process_identity=process_identity(child.pid)))
            code = wait_response(child, request / 'stdout.jsonl', timeout)
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
    finally:
        if child is not None:
            code = stop(child)
    result = dict(exit_code=code, error=error, seconds=time.time() - started,
                  pid=None if child is None else child.pid, state='WAITED', actual_wait=child is not None,
                  cli_processes_started=int(child is not None), launch_attempts=1)
    save(request / 'WAIT.json', result)
    return result


def wait_response(child: Any, path: Path, timeout: float) -> int:
    """出力上限イベントで所有CLIを停止し、無条件の内部再生成を抑える。"""
    deadline, pending = time.monotonic() + timeout, b''
    with path.open('rb') as stream:
        while True:
            data = pending + stream.read()
            lines = data.split(b'\n')
            pending = lines.pop()
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # 原文は保持し、既存summarizeが破損を報告する。
                if isinstance(event, dict) and 'max_tokens' in completion_reasons([event]):
                    raise RuntimeError('output_limit_detected_stop_owned_cli')
            code = child.poll()
            if code is not None:
                return code
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(child.args, timeout)
            time.sleep(RESPONSE_POLL_SECONDS)


def parse_events(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """JSON破損行は番号を保存し、読めた原イベントを失わない。"""
    events, malformed = [], []
    if not path.is_file():
        return events, malformed
    for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('event_not_object')
            events.append(value)
        except ValueError:
            malformed.append(number)
    return events, malformed


def message(event: dict[str, Any]) -> dict[str, Any]:
    """CLIのsystem通知はmessageが文字列でも正常。会話messageと分離する。"""
    value = event.get('message')
    return value if isinstance(value, dict) else {}


def completion_reasons(events: list[dict[str, Any]]) -> list[str]:
    """partial eventの終端を確認する。古い原票の欠測は推測しない。"""
    reasons = []
    for event in events:
        nested = event.get('event', {}) if event.get('type') == 'stream_event' else {}
        reason = nested.get('delta', {}).get('stop_reason') or message(event).get('stop_reason')
        if reason:
            reasons.append(reason)
    return reasons


def read_receipts(events: list[dict[str, Any]], packet: Path) -> dict[str, Any]:
    """指定packetのRead要求と成功結果をtool idで結合する。"""
    calls, successes = {}, set()
    for event in events:
        for block in message(event).get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') == 'Read':
                calls[block['id']] = block.get('input', {})
            if block.get('type') == 'tool_result' and not block.get('is_error', False):
                successes.add(block.get('tool_use_id'))
    line_count = len(packet.read_text(encoding='utf-8').splitlines())
    packet_ids = set()
    for key, value in calls.items():
        path = Path(value.get('file_path', ''))
        path = path if path.is_absolute() else packet.parent / path
        if (path.resolve() == packet.resolve() and value.get('offset', 1) == 1
                and value.get('limit', 2000) >= line_count):
            packet_ids.add(key)
    return dict(requests=calls, successful_tool_ids=sorted(successes),
                packet_read_completed=bool(packet_ids & successes))


def summarize(request: Path, waited: dict[str, Any]) -> dict[str, Any]:
    """provider生値を保持する。成功終了/読了/レビュー内容の合格は分ける。"""
    events, malformed = parse_events(request / 'stdout.jsonl')
    finals = [event for event in events if event.get('type') == 'result']
    final = finals[0] if len(finals) == 1 else None
    reads = read_receipts(events, request / 'WORK_PACKET.md')
    reasons = completion_reasons(events)
    actual_model = next((e.get('model') for e in events if e.get('subtype') == 'init'), None)
    expected_model = read(request / 'REQUEST.json')['model']
    good = (waited['exit_code'] == 0 and waited['error'] is None and not malformed and final is not None
            and not final.get('is_error', True) and final.get('subtype') == 'success'
            and reads['packet_read_completed'] and actual_model == expected_model
            and 'max_tokens' not in reasons
            and bool(reasons) and reasons[-1] in {'end_turn', 'stop_sequence'})
    usage = final.get('usage') if final else None
    result = dict(status='REVIEW_RETURNED' if good else 'INCOMPLETE_REVIEW', state='FINAL',
                  quality_pass=False, actual_wait=waited, read_receipts=reads, malformed_lines=malformed,
                  actual_model=actual_model, expected_model=expected_model,
                  stop_reasons=reasons, terminal_reason_missing=not reasons,
                  stdout_missing=not (request / 'stdout.jsonl').is_file(),
                  provider_usage=usage, usage_missing=usage is None,
                  provider_model_usage=None if final is None else final.get('modelUsage'),
                  cli_estimated_usd=None if final is None else final.get('total_cost_usd'), actual_billed_usd=None,
                  api_response_count_observed=len({message(e)['id'] for e in events
                      if e.get('type') == 'assistant' and message(e).get('id')}),
                  api_retries_observed=sum(e.get('subtype') == 'api_retry' for e in events),
                  raw_result=final, savings_rate=None, usage_aggregation='provider fields separate; thinking included in output')
    save(request / 'RESULT.json', result)
    return result


def dispatch(packet: dict[str, Any], root: Path, model: str = 'claude-opus-5',
             *, argv_override: list[str] | None = None, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    """G3イベントを一度だけ実行する。テスト用argvはCLIへ公開しない。"""
    if not under(root, VERIFY) or root.resolve() == VERIFY.resolve():
        raise ValueError('g3_output_must_be_under_D_verify')
    key, body = prepare(packet, model)
    root.mkdir(parents=True, exist_ok=True)
    for name in ('events', 'requests'):
        (root / name).mkdir(exist_ok=True)
    prior = existing(root, packet['event_id'], key)
    if prior is not None:
        return prior
    with slot(root):
        event_path = root / 'events' / (packet['event_id'] + '.json')
        try:
            save(event_path, dict(key=key, state='CREATED'))
        except FileExistsError:
            return existing(root, packet['event_id'], key) or {}
        request = root / 'requests' / key
        try:
            request.mkdir()
        except FileExistsError:
            return existing(root, packet['event_id'], key) or {}
        session = str(uuid.uuid4())
        save(request / 'REQUEST.json', dict(key=key, event_id=packet['event_id'], model=model,
            session_id=session, owner_pid=os.getpid(), created_epoch=time.time(), state='CREATED',
            max_response_tokens=MAX_RESPONSE_TOKENS,
            system_sha256=sha(SYSTEM.encode()), packet_sha256=sha(body.encode()), cli_launch_attempts=1))
        with (request / 'WORK_PACKET.md').open('x', encoding='utf-8') as stream:
            stream.write(body)
        argv = argv_override if argv_override is not None else command(model, session, request / 'WORK_PACKET.md')
        waited = execute(request, argv, timeout)
        try:
            result = summarize(request, waited)
        except Exception as error:
            save(request / 'SUMMARY_ERROR.json', dict(type=type(error).__name__, message=str(error)))
            raise
    return dict(reused=False, status=result['status'], request=str(request), result=result,
                new_cli_invocations=waited['cli_processes_started'])


def main() -> int:
    """表示は既存制限器へ委ね、原票と重要な欠測flagは保持する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model', choices=sorted(MODELS), default='claude-opus-5')
    args = parser.parse_args()
    outcome = dispatch(read(args.packet), args.output_root, args.model)
    print(json.dumps({key: value for key, value in outcome.items() if key != 'result'}, ensure_ascii=False))
    result_path = Path(outcome['request']) / 'RESULT.json'
    if result_path.is_file():
        result = outcome['result']
        print(json.dumps({key: result[key] for key in ('status', 'actual_wait', 'usage_missing',
                         'malformed_lines', 'stop_reasons', 'actual_model', 'terminal_reason_missing')}))
        text, total = preview(result_path, DISPLAY_CHARS)
        print(json.dumps(dict(raw_result=str(result_path), decoded_chars=total, omitted=total > DISPLAY_CHARS)))
        print(text)
    return 0 if outcome['status'] == 'REVIEW_RETURNED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
