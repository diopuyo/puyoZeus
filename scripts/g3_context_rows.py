"""G3 contextの保存済みJSONLを再用し、最新一行だけを保持する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

MAX_ROW_BYTES = 4 * 1024 * 1024


def require(value: bool, reason: str) -> None:
    """欠測や非対応操作を空の履歴へ置き換えない。"""
    if not value:
        raise ValueError('g3_context_rows:' + reason)


def digest(path: Path) -> str:
    """原票全体をメモリへ展開せず同じSHA256を計算する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(row: dict) -> bytes:
    """最新行のconsumerによる変更も検出する。"""
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, allow_nan=False).encode()).digest()


class ContextRows:
    """原endのflush後append専用。通算件数・末尾・全件反復を維持する。"""
    def __init__(self, path: Path) -> None:
        require(path.stat().st_size == 0, 'nonempty_install')
        self.path, self.count, self.offset = path, 0, 0
        self.latest: dict | None = None
        self.latest_signature: bytes | None = None
        self.hash = hashlib.sha256()

    def __len__(self) -> int:
        return self.count

    def unchanged(self) -> None:
        """最新行の変更を履歴再生で黙って消さない。"""
        require(self.latest is None or signature(self.latest) == self.latest_signature,
                'latest_mutated')

    def __getitem__(self, index: int) -> dict:
        if type(index) is not int or index != -1:
            raise IndexError('g3_context_rows:only_latest_supported')
        if self.latest is None:
            raise IndexError('g3_context_rows:empty')
        self.unchanged()
        return self.latest

    def append(self, row: dict) -> None:
        """byte offsetを用い、LF/CRLFと非ASCIIを区別せず原byteを認証する。"""
        self.unchanged()
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            raw = stream.readline(MAX_ROW_BYTES + 1)
            require(raw.endswith(b'\n') and len(raw) <= MAX_ROW_BYTES, 'incomplete_or_large_row')
            require(not stream.read(1), 'extra_unregistered_row')
        require(type(row) is dict and json.loads(raw) == row, 'saved_row_mismatch')
        self.latest, self.latest_signature = row, signature(row)
        self.hash.update(raw)
        self.count, self.offset = self.count + 1, self.offset + len(raw)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """全件検査には原票を逐次返し、読み切り時に件数と原byteを照合する。"""
        self.unchanged()
        count, offset, expected = self.count, self.offset, self.hash.digest()
        actual, seen = hashlib.sha256(), 0
        with self.path.open('rb') as stream:
            while stream.tell() < offset:
                raw = stream.readline(MAX_ROW_BYTES + 1)
                require(raw.endswith(b'\n') and len(raw) <= MAX_ROW_BYTES, 'incomplete_or_large_row')
                require(stream.tell() <= offset, 'saved_prefix_changed')
                actual.update(raw)
                seen += 1
                yield json.loads(raw)
            require(not stream.read(1), 'extra_unregistered_row')
        require((self.count, self.offset) == (count, offset), 'iteration_changed')
        require(seen == count and actual.digest() == expected, 'saved_history_changed')
        self.unchanged()


def saved_rows(path: Path) -> ContextRows:
    """閉鎖原票を一回認証し、全件dictを保持せず反復可能にする。"""
    rows = ContextRows.__new__(ContextRows)
    rows.path, rows.count, rows.offset = path, 0, 0
    rows.latest = rows.latest_signature = None
    rows.hash = hashlib.sha256()
    with path.open('rb') as stream:
        while raw := stream.readline(MAX_ROW_BYTES + 1):
            require(raw.endswith(b'\n') and len(raw) <= MAX_ROW_BYTES, 'incomplete_or_large_row')
            rows.latest = json.loads(raw)
            require(type(rows.latest) is dict, 'saved_row_type')
            rows.hash.update(raw)
            rows.count, rows.offset = rows.count + 1, rows.offset + len(raw)
    if rows.latest is not None:
        rows.latest_signature = signature(rows.latest)
    return rows


def verify(module: Any, output: Path, *, expected_frames: list[int] | None = None) -> dict:
    """元verifyの全条件を維持し、全件list構築だけを逐次読みに置き換える。"""
    status = json.loads((output / module.STATUS).read_text())
    receipt = json.loads((output / module.RECEIPT).read_text())
    module.require(status == dict(closed=True, installed=True, errors=[]), 'context_saved_status')
    module.require(set(receipt['sha256']) == {module.SIDECAR, module.STATUS}, 'context_artifact_set')
    module.require(all(digest(output / n) == h for n, h in receipt['sha256'].items()), 'context_artifact_changed')
    module.require(receipt['guards'] == module.guards(), 'context_saved_guards')
    frames = module.expected() if expected_frames is None else expected_frames
    module.require(receipt['expected_frames'] == frames, 'context_saved_expected_scope')
    rows = saved_rows(output / module.SIDECAR)
    module.require(all(r['source_id'] == receipt['source_id'] and r['run_id'] == receipt['run_id']
                       for r in rows), 'context_saved_provenance')
    result = module.coverage(rows, frames)
    module.require(all(receipt.get(k) == v for k, v in result.items()), 'context_saved_summary')
    module.require(all(receipt[k] is False for k in
        ('current_publication_allowed', 'accounting_permission', 'quality_gate_clear')), 'context_permission')
    return receipt
