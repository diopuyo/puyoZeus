"""公開用の木を新しいフォルダへ書き出す。commit・push・既存ファイル更新はしない。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize
import xml.etree.ElementTree as ET
from typing import Any

TEXT_EXTENSIONS = {'.md', '.txt', '.rst', '.ps1', '.cmd', '.bat', '.sh'}
EXPLANATION_KEYS = {'description', 'descriptions', 'note', 'notes', 'comment', 'comments', 'reason', 'explanation'}


def git(repo: Path, *args: str) -> bytes:
    """Gitの読み取り専用コマンドを呼ぶ。"""
    return subprocess.check_output(['git', '-C', str(repo), *args])


def read_tree(repo: Path, ref: str, modes: dict | None = None) -> tuple[str, dict[str, bytes]]:
    """指定コミットのblobを一つのcat-fileプロセスで読む。"""
    sha = git(repo, 'rev-parse', '--verify', ref + '^{commit}').decode().strip()
    entries = []
    for row in git(repo, 'ls-tree', '-rz', sha).split(b'\0'):
        if not row:
            continue
        meta, path = row.split(b'\t', 1)
        mode, kind, oid = meta.split()
        if kind != b'blob' or mode == b'120000':
            raise ValueError('通常ファイル以外は個別確認が必要: ' + path.decode())
        entries.append((path.decode(), oid))
        if modes is not None:
            modes[path.decode()] = int(mode, 8)
    request = b''.join(oid + b'\n' for _, oid in entries)
    result = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=request, capture_output=True, check=True)
    stream = io.BytesIO(result.stdout)
    files = {}
    for path, _ in entries:
        header = stream.readline().split()
        size = int(header[2])
        files[path] = stream.read(size)
        assert stream.read(1) == b'\n'
    return sha, files


def patterns(user: str, work: str) -> list[tuple[re.Pattern[str], str]]:
    """対象の個人名は引数で受け取り、公開ソースに固定しない。"""
    separator = r'[/\\]+'
    username = re.escape(user)
    home = rf'(?:/mnt/[a-z]/Users/{username}|[a-z]:{separator}Users{separator}{username}|/home/{username})(?![A-Za-z0-9_.-])'
    workspace = rf'(?:(?:[a-z]:{separator}|/mnt/[a-z]/))?{re.escape(work)}'
    return [(re.compile(home, re.I), '<home>'), (re.compile(workspace, re.I), '<work>')]


def replace_text(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def python_spans(text: str) -> list[tuple[int, int]]:
    """コメントと本当のdocstringだけを対象とする。通常の定数には触らない。"""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            spans.append((offsets[token.start[0]-1]+token.start[1], offsets[token.end[0]-1]+token.end[1]))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
            continue
        value = first.value
        start = len(lines[value.lineno-1].encode()[:value.col_offset].decode())
        end = len(lines[value.end_lineno-1].encode()[:value.end_col_offset].decode())
        spans.append((offsets[value.lineno-1]+start, offsets[value.end_lineno-1]+end))
    return spans


def sanitize_json(value: Any, rules: list, explain: bool = False) -> Any:
    """説明キーの文字列だけを匿名化し、パス・fixture・数値は変更しない。"""
    if isinstance(value, dict):
        return {key: sanitize_json(item, rules, explain or key.lower() in EXPLANATION_KEYS) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item, rules, explain) for item in value]
    if isinstance(value, str) and explain:
        return replace_text(value, rules)
    return value


def sanitize_provenance(value: Any, rules: list, targets: list[str], pointer: str = '') -> Any:
    """個別確認した来歴欄だけを処理する。辞書キーの衝突は拒否する。"""
    selected = any(pointer == target or pointer.startswith(target + '/') for target in targets)
    if isinstance(value, dict):
        changed = {}
        for key, item in value.items():
            clean_key = replace_text(key, rules) if selected else key
            if clean_key in changed:
                raise ValueError('匿名化した来歴のキーが衝突しました')
            changed[clean_key] = sanitize_provenance(item, rules, targets, pointer + '/' + key)
        return changed
    if isinstance(value, list):
        return [sanitize_provenance(item, rules, targets, pointer + '/' + str(i)) for i, item in enumerate(value)]
    return replace_text(value, rules) if isinstance(value, str) and selected else value


def sanitize(path: str, raw: bytes, rules: list, provenance: list[str] | None = None) -> bytes:
    """許可した種類だけを処理する。未分類の残存は後段で拒否する。"""
    if b'\0' in raw:
        return raw
    text = raw.decode('utf-8')
    if not any(pattern.search(text) for pattern, _ in rules):
        return raw
    suffix = Path(path).suffix.lower()
    if suffix == '.py':
        for start, end in sorted(python_spans(text), reverse=True):
            text = text[:start] + replace_text(text[start:end], rules) + text[end:]
        ast.parse(text)
    elif suffix in TEXT_EXTENSIONS or (path.startswith('docs/') and suffix == '.tsv'):
        text = replace_text(text, rules)
    elif suffix == '.json':
        before = json.loads(text)
        after = sanitize_json(before, rules)
        after = sanitize_provenance(after, rules, provenance or [])
        if before != after:
            text = json.dumps(after, ensure_ascii=False, indent=2) + '\n'
    elif path.startswith('docs/') and suffix == '.xml':
        escaped = [(pattern, replacement.replace('<', '&lt;').replace('>', '&gt;')) for pattern, replacement in rules]
        text = replace_text(text, escaped)
        ET.fromstring(text)
    elif suffix == '.toml':
        text = '\n'.join(replace_text(line, rules) if line.lstrip().startswith('#') else line for line in text.split('\n'))
    return text.encode('utf-8')


def exclusion(path: str, policy: dict) -> str | None:
    if path in policy.get('exclude_paths', {}):
        return policy['exclude_paths'][path]
    if set(Path(path).parts) & set(policy['exclude_parts']):
        return '作業出力のディレクトリ'
    if any(path.startswith(prefix) for prefix in policy['artifact_prefixes']):
        if path not in policy['retain_artifacts']:
            return '試験・配布組立・実行の入力として保持指定なし'
    return None


def redact_account(path: str, raw: bytes, user: str) -> bytes:
    """公開テキストの指定アカウント名を消す。バイナリは変更しない。"""
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8'
    if encoding == 'utf-8' and b'\0' in raw:
        return raw
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        return raw
    pattern = re.compile(re.escape(user), re.IGNORECASE)
    if not user or not pattern.search(text):
        return raw
    replacement = '&lt;user&gt;' if Path(path).suffix.lower() == '.xml' else '<user>'
    if Path(path).suffix.lower() == '.sh':
        text = redact_shell_account(text, pattern)
    else:
        text = pattern.sub(lambda _: replacement, text)
    return text.encode(encoding)


def redact_shell_account(text: str, pattern: re.Pattern) -> str:
    """引用外では引用付きプレースホルダーにし、リダイレクト化を防ぐ。"""
    result, quote, escaped, cursor = [], '', False, 0
    for match in pattern.finditer(text):
        prefix = text[cursor:match.start()]
        for char in prefix:
            if escaped:
                escaped = False
            elif char == '\\' and quote != "'":
                escaped = True
            elif char == quote:
                quote = ''
            elif char in "\"'" and not quote:
                quote = char
        result.extend((prefix, '<user>' if quote else '"<user>"'))
        cursor = match.end()
    return ''.join(result) + text[cursor:]


def plan(files: dict[str, bytes], policy: dict, rules: list, private_user: str | None = None) -> tuple[dict[str, bytes], dict]:
    result, removed, changed, exceptions, unknown = {}, {}, [], [], []
    protected_changes = {}
    for path, raw in files.items():
        reason = exclusion(path, policy)
        if reason:
            removed[path] = reason
            continue
        protected = policy['preserve_bytes'].get(path)
        if protected:
            if hashlib.sha256(raw).hexdigest() != protected['sha256']:
                raise ValueError('保持例外の内容が変わったため再確認が必要: ' + path)
            clean = raw
        else:
            clean = sanitize(path, raw, rules, policy.get('json_provenance', {}).get(path, []))
        if private_user:
            clean = redact_account(path, clean, private_user)
        if protected and clean != raw:
            protected_changes[path] = dict(input_sha256=protected['sha256'],
                output_sha256=hashlib.sha256(clean).hexdigest(), reason='公開出力のアカウント名の匿名化')
        text = clean.decode('utf-8', errors='replace')
        count = sum(len(pattern.findall(text)) for pattern, _ in rules)
        if count:
            (exceptions if protected else unknown).append({'path': path, 'matches': count})
        if clean != raw:
            changed.append(path)
        result[path] = clean
    policy_name = 'packaging/public_tree_policy.json'
    if protected_changes and policy_name in result:
        exported_policy = json.loads(result[policy_name])
        for path, change in protected_changes.items():
            exported_policy['preserve_bytes'][path]['sha256'] = change['output_sha256']
        result[policy_name] = (json.dumps(exported_policy, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        if policy_name not in changed:
            changed.append(policy_name)
    return result, dict(excluded=removed, changed=changed, exceptions=exceptions, unreviewed=unknown,
                       protected_redactions=protected_changes)


def export_tree(repo: Path, ref: str, output: Path, report: Path, policy_path: Path, user: str, work: str,
                redact_account_name: bool = False) -> dict:
    """未確認の残存があれば書出し前に止まる。出力先は毎回新しくする。"""
    repo, output, report = repo.resolve(), output.resolve(), report.resolve()
    if output.exists() or output.is_relative_to(repo) or report.exists() or report.is_relative_to(repo):
        raise ValueError('出力・報告はリポジトリ外の未使用パスを指定してください')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    modes = {}
    sha, files = read_tree(repo, ref, modes)
    planned, audit = plan(files, policy, patterns(user, work), private_user=user if redact_account_name else None)
    audit.update(source=sha, source_files=len(files), output_files=len(planned))
    audit['executable_paths'] = [path for path in planned if modes[path] & 0o111]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if audit['unreviewed']:
        raise ValueError(f"未確認の個人パスが {len(audit['unreviewed'])} ファイルにあります。報告を確認してください")
    output.mkdir(parents=True)
    for path, raw in planned.items():
        target = output / path
        if not target.resolve().is_relative_to(output):
            raise ValueError('出力外のパスを拒否: ' + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(modes[path] & 0o777)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('public_tree_policy.json'))
    parser.add_argument('--private-user', required=True)
    parser.add_argument('--private-work-name', required=True)
    parser.add_argument('--redact-account-name', action='store_true', help='全テキストのアカウント名を匿名化する')
    args = parser.parse_args()
    result = export_tree(args.repo, args.ref, args.output, args.report, args.policy, args.private_user,
                         args.private_work_name, redact_account_name=args.redact_account_name)
    print(f"source={result['source']} files={result['output_files']} excluded={len(result['excluded'])} exceptions={len(result['exceptions'])}")


if __name__ == '__main__':
    main()
