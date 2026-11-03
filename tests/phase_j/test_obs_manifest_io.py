"""試験用のI/O並行化は、欠落・破損・失敗を含む全件照合を変えない。"""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from src.phase_j import launcher_manifest as manifest


@pytest.mark.parametrize('failed', [False, True])
def test_parallel_report_matches_serial(monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    names = ['ok', 'missing', 'size', 'hash', 'last']
    expected = dict(files={name: dict(size=3, sha256='abc') for name in names})
    class FakeRoot:
        def __truediv__(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, is_file=lambda: name != 'missing',
                                   stat=lambda: SimpleNamespace(st_size=4 if name == 'size' else 3))

    def digest(path: SimpleNamespace) -> str:
        if failed and path.name == 'last':
            raise OSError('読取り失敗')
        return 'different' if path.name == 'hash' else 'abc'

    monkeypatch.setattr(manifest, 'sha256_file', digest)
    reports = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        report = manifest.ManifestReport()
        if failed:
            with pytest.raises(OSError, match='読取り失敗'):
                manifest.verify_files(FakeRoot(), expected, report)
        else:
            manifest.verify_files(FakeRoot(), expected, report)
        reports.append(asdict(report))
    assert reports[0] == reports[1]
    assert report.checked == 4 and report.checked_bytes == 12
    assert report.missing == ['missing']
    assert report.size_mismatch == ['size']
    assert report.hash_mismatch == ['hash']
