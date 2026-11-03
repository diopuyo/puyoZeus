"""仮の宛先は通信せず、確認した内容を保存する。"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.phase_j.distribution_logs import ReportStore


@pytest.mark.parametrize('endpoint', [
    'https://example.invalid/reports', 'https://EXAMPLE.INVALID/reports',
    'https://a.example.invalid/reports',
])
def test_invalid_destination_never_opens_network(tmp_path: Path, endpoint: str) -> None:
    store = ReportStore(tmp_path)
    report = {'report_id': 'local-test', 'summary': 'reviewed'}
    preview = store.preview(report, endpoint)
    assert preview['local_only'] is True
    with patch('src.phase_j.distribution_logs.build_opener') as opener:
        assert store.send(preview['token'], True) is False
        opener.assert_not_called()
    assert len(list(tmp_path.glob('pending-*.json'))) == 1
    with pytest.raises(ValueError):
        store.send(preview['token'], True)


def test_real_https_keeps_confirmed_send(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    preview = store.preview({'report_id': 'remote-test'}, 'https://reports.example.org/api')
    assert preview['local_only'] is False
    response = MagicMock()
    response.__enter__.return_value.status = 200
    with patch('src.phase_j.distribution_logs.build_opener') as opener:
        opener.return_value.open.return_value = response
        assert store.send(preview['token'], True) is True
        opener.return_value.open.assert_called_once()
    assert not list(tmp_path.glob('pending-*.json'))


def test_powershell_local_guard_precedes_network() -> None:
    script = (Path(__file__).parents[2] / 'packaging/SendLogs.ps1').read_text('ascii')
    assert script.index("if ($localOnly) { throw 'Local storage only' }") < script.index('Invoke-WebRequest')
