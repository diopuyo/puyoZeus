"""0.6.1準備の旗・数値要約・確認後の手動送信を検証する。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
import threading
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from src import production_config
from src.phase_j.distribution_config import DISTRIBUTION_SITE_URL, load_settings, save_settings, validate_settings
from src.phase_j.distribution_logs import MAX_REPORT_BYTES, SITE_USER_AGENT, ReportStore, receipt_counts
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.distribution_summary import MAX_MATCHES, MAX_POINTS, NumericSummary
from tests.phase_j.test_distribution_app import distribution, post


def snapshot(match: str = 'private-name', probability: float = 0.7) -> dict[str, Any]:
    return dict(identity=dict(match_id=match),
                display=dict(match_phase='playing', primary_hold_reason=None),
                timing=dict(capture_to_publish_latency_ms=100),
                evaluations=dict(practical=dict(p1_win_probability=probability),
                                 firepower_x=987654321), path='C:/Users/private/video.mp4',
                pc_name='private-pc', image='private-image')


def test_feedback_flag_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-11-06 user 採用で出荷は ON (試験一式は conftest で OFF から始めるので、出荷の値は原本で確かめる)。
    # 宛先は公開サイトの 1 か所、更新の確認は既定で行わない。
    source = Path(production_config.__file__).read_text(encoding='utf-8')
    assert re.search(r'^DISTRIBUTION_SITE_FEEDBACK = True$', source, re.MULTILINE)
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', True)
    settings = validate_settings({})
    assert settings['relay_url'] == 'https://puyolive-site.pages.dev/api/reports'
    assert settings['releases_url'] == 'https://puyolive-site.pages.dev/releases.json'
    assert settings['check_updates'] is False


def test_feedback_flag_off_keeps_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', False)
    settings = validate_settings({})
    assert settings['relay_url'] == 'https://puyolive-reports.example.invalid/api/reports'
    assert settings['releases_url'] == 'https://puyolive.example.invalid/releases.json'
    assert settings['check_updates'] is False


def test_site_addresses_have_one_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', True)
    monkeypatch.setattr('src.phase_j.distribution_config.DISTRIBUTION_SITE_URL', 'https://example.test/')
    settings = validate_settings({'relay_url': 'https://old.invalid/reports'})
    assert settings['relay_url'] == 'https://example.test/api/reports'
    assert settings['releases_url'] == 'https://example.test/releases.json'
    assert settings['check_updates'] is False


def test_saved_settings_cannot_bypass_disabled_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', True)
    enabled = validate_settings({})
    save_settings(tmp_path, enabled)
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', False)
    assert '.invalid/' in load_settings(tmp_path)['relay_url']
    assert '.invalid/' in validate_settings(enabled)['relay_url']


@pytest.mark.parametrize('enabled', [False, True])
def test_runtime_summary_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', enabled)
    runtime = DistributionRuntime(tmp_path)
    runtime.accept_snapshot(snapshot())
    runtime.accept_snapshot(snapshot())
    report = runtime.report()
    assert ('numeric_summary' in report) is enabled
    if enabled:
        assert report['numeric_summary']['observations'] == 1
        assert len(json.dumps(report, ensure_ascii=False).encode()) <= MAX_REPORT_BYTES


def test_summary_counts_percentiles_and_privacy() -> None:
    summary = NumericSummary()
    for index in range(10):
        data = snapshot('private-name' if index < 5 else 'private-name-2')
        data['timing']['capture_to_publish_latency_ms'] = index * 10
        if index < 2:
            data['display']['primary_hold_reason'] = 'recognition_unreliable'
        summary.observe(data)
    result = summary.snapshot(60)
    assert result['match_count'] == 2
    assert result['delay_median_ms'] == 45
    assert result['delay_p90_ms'] == 80
    assert result['recognition_unreliable_fraction'] == 0.2
    assert result['recognition_observations'] == 10
    encoded = json.dumps(result)
    assert all(word not in encoded for word in ('private', 'video', 'image', 'path', '987654321', 'firepower'))
    assert result['matches'][0]['p1_win_probabilities'] == [0.7]


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '0.8', -1, 2])
def test_summary_rejects_invalid_probability(value: Any) -> None:
    summary = NumericSummary()
    summary.observe(snapshot(probability=value))
    assert summary.snapshot(0)['matches'][0]['p1_win_probabilities'] == []


def test_summary_bounded_history(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = NumericSummary()
    clock = iter(range(10000))
    monkeypatch.setattr('src.phase_j.distribution_summary.time.monotonic', lambda: next(clock))
    for match in range(50):
        for _ in range(100):
            summary.observe(snapshot(str(match)))
    result = summary.snapshot(5000)
    assert result['match_count'] == 50
    assert result['retained_matches'] == MAX_MATCHES
    assert all(len(m['p1_win_probabilities']) <= MAX_POINTS for m in result['matches'])
    assert result['delay_window_samples'] == 2048
    assert len(json.dumps(result).encode()) < 10 * 1024


def test_summary_missing_values() -> None:
    summary = NumericSummary()
    summary.observe({})
    result = summary.snapshot(0)
    assert result['recognition_unreliable_fraction'] is None
    assert result['delay_median_ms'] is None
    assert result['match_count'] == 0


def test_report_size_and_redaction(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    summary = NumericSummary().snapshot(1)
    report = store.report({}, {}, {}, {'huge': 'a' * 100000}, ['あ' * 100000], [], summary)
    assert len(json.dumps(report, ensure_ascii=False).encode()) <= MAX_REPORT_BYTES
    assert report['numeric_summary'] == summary
    assert report['omitted_fields']
    assert report['schema_version'] == 1


def test_compact_report_keeps_version_required_by_site(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    version = {'app': '0.6.1-test'}
    huge = 'a' * 100000
    # 要約だけで上限を超える極端な場合でも version は空にしない
    report = store.report(version, {'s': huge}, {'type': huge}, {'d': huge}, [huge], [huge],
                          {'oversized': [0.5] * 20000})
    assert 'version' not in report['omitted_fields']
    assert report['version']['app'] == '0.6.1-test'


@pytest.mark.parametrize('body,expected', [
    (b'{"accepted":0,"duplicates":1}', {'accepted': 0, 'duplicates': 1}),
    (b'{"accepted":true,"duplicates":-1,"message":"private"}', {}),
    # 配布サイトの受け口の実際の形 (11/6 本番で確認): 受理の件数は inserted
    (b'{"accepted":true,"inserted":1,"duplicates":0}', {'accepted': 1, 'duplicates': 0}),
    (b'{"accepted":true,"inserted":-1,"duplicates":0,"message":"private"}', {'duplicates': 0}),
    (b'not-json', {}),
    (b'[]', {}),
])
def test_receipt_accepts_only_counts(body: bytes, expected: dict) -> None:
    assert receipt_counts(BytesIO(body)) == expected


def test_preview_limit_keeps_unselected_reports(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    store.save([{'report_id': str(index)} for index in range(20)])
    preview = store.preview({'report_id': 'new'}, 'https://example.test/api/reports', max_reports=20)
    assert len(preview['payload']['reports']) == 20
    selected = {report['report_id'] for report in preview['payload']['reports']}
    with patch('src.phase_j.distribution_logs.build_opener') as opener:
        response = opener.return_value.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'{"accepted":20,"duplicates":0}'
        assert store.send(preview['token'], True)
    remaining = list(tmp_path.glob('pending-*.json'))
    assert len(remaining) == 1
    assert json.loads(remaining[0].read_text())['report_id'] not in selected


@pytest.mark.parametrize('status', [200, 429, 413])
def test_local_receipt_and_explicit_confirmation(tmp_path: Path, status: int) -> None:
    received: list[dict[str, Any]] = []
    agents: list[str] = []

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            agents.append(self.headers['User-Agent'])
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"accepted":1,"duplicates":0,"message":"private-pc"}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        store = ReportStore(tmp_path)
        preview = store.preview({'report_id': 'one'}, f'http://127.0.0.1:{server.server_port}/api/reports')
        assert not received
        with pytest.raises(ValueError):
            store.send(preview['token'], False, allow_local=True)
        assert not received
        assert store.send(preview['token'], True, allow_local=True) is (status == 200)
        assert received == [preview['payload']]
        # 配布サイトは Python-urllib の名乗りを 403 で拒む (11/6 の実物の送信の失敗)。製品名で名乗る。
        assert agents == [SITE_USER_AGENT]
        assert store.last_send_result == (dict(accepted=1, duplicates=0) if status == 200 else dict(status=status))
        assert len(list(tmp_path.glob('pending-*.json'))) == int(status != 200)
    finally:
        server.shutdown()
        server.server_close()


def test_open_report_folder_is_fixed(distribution: tuple) -> None:
    runtime, url = distribution
    with patch('src.phase_j.distribution_http.os.startfile') as start:
        assert post(url + '/api/reports/open', {'path': 'C:/ignored'}) == {'opened': True}
        start.assert_called_once_with(str(runtime.store.directory.resolve()))


@pytest.mark.parametrize('enabled', [False, True])
def test_feedback_browser_gate(distribution: tuple, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', enabled)
    _, url = distribution
    with patch('src.phase_j.distribution_http.webbrowser.open', return_value=True) as browser:
        if enabled:
            assert post(url + '/api/feedback/open', {}) == {'opened': True}
            browser.assert_called_once_with(DISTRIBUTION_SITE_URL + '/#feedback')
        else:
            with pytest.raises(HTTPError):
                post(url + '/api/feedback/open', {})
            browser.assert_not_called()


def test_tests_cannot_reach_production_site() -> None:
    """conftest の見張り: 試験から本番の配布サイトへは要求を作れない (送信の差し替え漏れの保険)。"""
    from urllib.request import Request, build_opener
    # 監査の事象 urllib.Request は接続の前 (OpenerDirector.open の入口) で出るので、実際の通信は起きない。
    with pytest.raises(AssertionError, match='本番の配布サイト'):
        build_opener().open(Request(DISTRIBUTION_SITE_URL.rstrip('/') + '/api/reports', b'{}', method='POST'), timeout=1)
