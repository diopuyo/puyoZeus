"""保存済みG3終了票の変化だけを既存Agent APIとローカル通知へ接続する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable
from scripts import g3_agent_review as A
from scripts import g3_run_outcome as O

NOTIFIER = A.ROOT / 'scripts/g3_notify_event.ps1'
NOTIFY_TIMEOUT = 25
TEXT_LIMIT = 450


def notification(folder: Path, label: str, status: str) -> dict:
    """開始票/表示イベント/実終了を区別し、失敗でも再送しない。"""
    request = folder / 'NOTIFY_REQUEST.json'
    A.save(request, dict(schema='g3-local-notification/v1', title='puyo_analyzer G3 工程終了',
        body=f'{label[:80]}: {status}。G3合格ではありません。Codexで結果をご確認ください。'))
    if os.name != 'nt':
        raise RuntimeError('g3_notification_requires_windows_entry')
    with (folder / 'notify.log').open('xb') as log:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
            str(NOTIFIER), '-RequestPath', str(request)], stdout=log, stderr=subprocess.STDOUT,
            timeout=NOTIFY_TIMEOUT)
    saved = A.read(folder / 'NOTIFY_RESULT.json')
    confirmed = (result.returncode == 0 and (folder / 'NOTIFY_STARTED.json').is_file()
                 and saved.get('shown_event') is True and saved.get('disposed') is True
                 and saved.get('error') is None)
    return dict(confirmed=confirmed, exit_code=result.returncode, receipt=saved)


def packet(folder: Path, parent: Path, value: dict, key: str) -> dict:
    """変動時刻・通知結果を入力へ混ぜず、同じ原票から同じ作業packetを作る。"""
    fields = ('status', 'parent_exit_code', 'wait_verified', 'wait', 'wait_error',
              'artifact_errors', 'audit_error', 'output', 'quality_gate_clear')
    selected = {k: value.get(k) for k in fields}
    for name in ('wait_error', 'artifact_errors', 'audit_error'):
        selected[name] = str(selected[name])[:TEXT_LIMIT]
    entry = value.get('entry') or {}
    selected['entry_status'] = entry.get('status')
    selected['entry_error_excerpt'] = str(entry.get('error'))[:TEXT_LIMIT]
    selected.update(original_parent=str(parent), original_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                    omitted_fields=sorted(set(value) - set(fields)), full_error_in_original=True)
    summary = folder / 'EVIDENCE.json'
    A.save(summary, selected)
    # 索引は制限内、完全な例外・除外項目は同じ原票を参照する。
    evidence = dict(path=str(summary), sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                    first_line=1, last_line=1)
    return dict(event_id='g3-end-' + key, event='run_failed' if value['status'] == O.FAILED else 'evidence_changed',
        unit='G3単一runの終了保存', purpose='新しい終了結果の結論・未完・次条件を整理する。',
        completion='evidenceを実Read。必要な元例外はoriginal_parentを実Read。300字以内。'
                   'コード変更/再走/同一既存レビューの反復は禁止。',
        scope='G3のみ。観測終了を品質合格にしない。GT/学習/本番/G4/追加mergeなし。',
        delta='保存済み親結果の新規終了イベント。全履歴や生動画は送信しない。',
        unresolved='親結果と原票の欠測を明示。ユーザー既読と本線自動再開は未確認。', evidence=[evidence])


def review(work: dict, root: Path) -> dict:
    """実Read契約と既存排他/キャッシュ/使用量保存を再用する。"""
    from scripts.g3_review_contract import configure
    configure(root)
    return A.dispatch(work, root / 'agent_api', 'claude-fable-5-1')


def process(root: Path, label: str, *, notify: Callable = notification,
            judge: Callable = review) -> dict:
    """通知/APIの途中死亡もイベント開始票で再送を抑止する。"""
    parent = O.destination(root, label)
    raw = parent.read_bytes()
    value = json.loads(raw)
    if value.get('status') not in {O.FAILED, O.ENDED} or value.get('quality_gate_clear') is not False:
        raise ValueError('g3_end_parent_contract')
    key = hashlib.sha256(str(parent.resolve()).encode() + b'\0' + raw).hexdigest()
    folder = root / 'end_events' / key
    folder.mkdir(parents=True, exist_ok=True)
    try:
        A.save(folder / 'EVENT_REQUEST.json', dict(parent=str(parent), sha256=hashlib.sha256(raw).hexdigest(),
            owner_pid=os.getpid(), process_identity=A.process_identity(os.getpid())))
    except FileExistsError:
        result = folder / 'EVENT_RESULT.json'
        if result.exists():
            return A.read(result) | dict(reused=True)
        owner = A.read(folder / 'EVENT_REQUEST.json')
        current = A.process_identity(owner['owner_pid'])
        return dict(status='UNKNOWN_NO_RESEND', reused=True, owner=owner,
                    current_process_identity=current, process_match=A.reconcile_identity(owner, current))
    result: dict[str, Any] = dict(status='FAILED', parent_status=value['status'], parent_exit_code=value.get('parent_exit_code'),
                                  quality_gate_clear=False, root_thread_resumed=False)
    try:
        result['notification'] = notify(folder, label, value['status'])
    except Exception as error:
        result['notification_error'] = repr(error)
    try:
        if parent.read_bytes() != raw:
            raise ValueError('g3_end_parent_changed_before_api')
        outcome = judge(packet(folder, parent, value, key), root)
        result['api'] = {k: v for k, v in outcome.items() if k != 'result'}
        result['parent_unchanged'] = parent.read_bytes() == raw
        if (outcome.get('status') == 'REVIEW_RETURNED' and result['parent_unchanged']
                and result.get('notification', {}).get('confirmed') is True):
            result['status'] = 'EVENT_HANDLED_NOT_G3_PASS'
    except Exception as error:
        result['api_error'] = repr(error)
    A.save(folder / 'EVENT_RESULT.json', result)
    return result


def main() -> int:
    """Windows側の有限入口。監視ループや自動再走を起動しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    result = process(args.root, args.label)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'EVENT_HANDLED_NOT_G3_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
