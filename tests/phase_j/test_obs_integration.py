"""実機設定を変更せず、入力選択・OBS v5・マスコット素材を検証する。"""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading
from typing import Iterator

import cv2
import numpy as np
from PIL import Image
import pytest
from websockets.sync.server import serve

from src.phase_j.distribution_config import load_settings, validate_settings
from src.phase_j.distribution_input import discover, has_feedback, inspect_frame, marker_image
from src.phase_j.distribution_obs import authentication, integrate
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.mascot_assets import mascot_asset, retime
from src.phase_j.overlay_config import validate_overlay

BRAND = Path('src/phase_j/brand')


@pytest.mark.parametrize('scale', [1, .8, .7, .5])
def test_feedback_marker(scale: float) -> None:
    frame = np.zeros((1080, 1920, 3), np.uint8)
    marker = cv2.resize(marker_image(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = marker.shape[:2]
    frame[700:700+h, 800:800+w] = marker
    assert has_feedback(frame)
    assert inspect_frame(frame, lambda _: True) == 'feedback'


@pytest.mark.parametrize('level', [0, 80, 255])
def test_plain_screen_never_feedback(level: int) -> None:
    assert not has_feedback(np.full((1080, 1920, 3), level, np.uint8))


@pytest.mark.parametrize('shape,passes,reason', [((720,1280,3),True,'wrong_size'),
    ((1080,1920,3),False,'not_game'), ((1080,1920,3),True,'ready')])
def test_reuses_verifier(shape: tuple, passes: bool, reason: str) -> None:
    assert inspect_frame(np.zeros(shape, np.uint8), lambda _: passes) == reason


@pytest.mark.parametrize('obs_reason,expected', [('ready',4),('not_game',0),('timeout',0),('feedback',0)])
def test_all_inputs_tried_and_obs_preferred(obs_reason: str, expected: int) -> None:
    tried = []
    def probe(index: int) -> dict:
        tried.append(index)
        return dict(reason=obs_reason if index == 4 else 'ready')
    result = discover([dict(index=0,name='Webcam'),dict(index=4,name='OBS Virtual Camera')], probe)
    assert tried == [0, 4]
    assert result['selected']['index'] == expected


@pytest.mark.parametrize('reason', ['not_game','feedback','unavailable','timeout'])
def test_invalid_input_not_selected(reason: str) -> None:
    result = discover([dict(index=4,name='OBS Virtual Camera')], lambda _: dict(reason=reason))
    assert result['selected'] is None
    assert result['reason'] == reason
    assert result['message']


def test_bom_and_secret_not_in_report(tmp_path: Path) -> None:
    (tmp_path/'distribution.json').write_text(json.dumps(dict(obs_password='local-secret')), encoding='utf-8-sig')
    assert load_settings(tmp_path)['obs_password'] == 'local-secret'
    runtime = DistributionRuntime(tmp_path)
    assert 'local-secret' not in json.dumps(runtime.report())


@pytest.mark.parametrize('options', [dict(mascot_dopamine='bad'),dict(mascot_speed='3'),
    dict(mascot_bloodshot=1),dict(mascot_speed=2)])
def test_invalid_mascot_options(options: dict) -> None:
    with pytest.raises(ValueError):
        validate_overlay(options)


@pytest.mark.parametrize('speed', [.5, 1, 1.5, 2])
def test_retime_preserves_frames(speed: float) -> None:
    original = Image.open(BRAND/'mascot_90_high_b3.webp')
    modified = Image.open(io.BytesIO(retime((BRAND/'mascot_90_high_b3.webp').read_bytes(), speed)))
    assert modified.n_frames == original.n_frames
    for index in range(original.n_frames):
        original.seek(index); modified.seek(index)
        assert np.array_equal(np.array(original.convert('RGBA')), np.array(modified.convert('RGBA')))
        assert modified.info['duration'] == round(original.info['duration']/speed)


@pytest.mark.parametrize('name', ['10','30','50','70_low','70_normal','70_high','90_low','90_normal','90_high'])
def test_bloodshot_only_changes_antenna(name: str) -> None:
    original = Image.open(BRAND/f'mascot_{name}.webp')
    for stage in (1,2,3):
        changed = Image.open(BRAND/f'mascot_{name}_b{stage}.webp')
        for index in range(original.n_frames):
            original.seek(index); changed.seek(index)
            a,b = np.array(original.convert('RGBA')),np.array(changed.convert('RGBA'))
            assert np.array_equal(a[175:],b[175:])
            assert np.array_equal(a[:,:,3],b[:,:,3])
            assert not np.array_equal(a,b)


class MockObs:
    """認証と実WebSocketフレームを通すv5模擬サーバー。"""
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.inputs: list[dict] = []
        self.items: list[dict] = []
        self.active = False
        self.settings: dict = {}

    def handle(self, socket: object) -> None:
        socket.send(json.dumps(dict(op=0,d=dict(rpcVersion=1,authentication=dict(salt='salt',challenge='challenge')))))
        identify = json.loads(socket.recv())
        if identify['d'].get('authentication') != authentication('secret','salt','challenge'):
            socket.close(4009, 'Authentication failed')
            return
        socket.send(json.dumps(dict(op=2,d=dict(negotiatedRpcVersion=1))))
        for raw in socket:
            data = json.loads(raw)['d']; kind = data['requestType']; request = data['requestData']
            self.calls.append(kind)
            result = self.respond(kind, request)
            socket.send(json.dumps(dict(op=7,d=dict(requestId=data['requestId'],
                requestStatus=dict(result=True,code=100), responseData=result))))

    def respond(self, kind: str, data: dict) -> dict:
        if kind == 'GetCurrentProgramScene':
            return dict(currentProgramSceneName='Temporary test scene')
        if kind == 'GetSceneItemList':
            return dict(sceneItems=self.items)
        if kind == 'GetInputSettings':
            return dict(inputSettings=self.settings[data['inputName']])
        if kind == 'SetInputSettings':
            self.settings[data['inputName']].update(data['inputSettings'])
        if kind == 'GetInputList':
            return dict(inputs=self.inputs)
        if kind == 'GetVirtualCamStatus':
            return dict(outputActive=self.active)
        if kind == 'StartVirtualCam':
            self.active = True
        if kind == 'CreateInput':
            assert data['inputSettings']['width'] == 1920
            assert data['inputSettings']['height'] == 1080
            self.settings[data['inputName']] = data['inputSettings'].copy()
            self.inputs.append(dict(inputName=data['inputName']))
            self.items.append(dict(sourceName=data['inputName'],inputKind='browser_source'))
        return {}


def test_obs_real_protocol_read_apply_idempotent_and_auth() -> None:
    obs = MockObs()
    with serve(obs.handle, '127.0.0.1', 0, subprotocols=['obswebsocket.json']) as server:
        threading.Thread(target=server.serve_forever,daemon=True).start()
        settings = validate_settings(dict(obs_enabled=True,obs_port=server.socket.getsockname()[1],obs_password='secret'))
        result = integrate(settings, 'http://127.0.0.1:8765/')
        assert not result['applied'] and not obs.active
        assert not any(call.startswith(('Create','Start','Set')) for call in obs.calls)
        integrate(settings, 'http://127.0.0.1:8765/', True)
        integrate(settings, 'http://127.0.0.1:8765/', True)
        assert obs.calls.count('CreateInput') == 1
        assert obs.calls.count('StartVirtualCam') == 1
        assert len(obs.inputs) == 1
        with pytest.raises(ValueError, match='OBS のパスワードが違います.*The OBS password is incorrect'):
            integrate({**settings,'obs_password':'wrong'}, 'http://127.0.0.1:8765/')
        server.shutdown()


def test_obs_default_off() -> None:
    with pytest.raises(ValueError,match='Enable OBS'):
        integrate(validate_settings({}), 'http://127.0.0.1:8765/')


def test_probe_resolves_packaged_models_independent_of_launch_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from src.phase_j.distribution_input import isolated_probe
    calls = []
    def run(command: list, **kwargs: object) -> object:
        calls.append(kwargs['cwd'])
        assert (kwargs['cwd'] / 'src/score_ocr.py').is_file()
        return SimpleNamespace(stdout='{"reason":"not_game"}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.phase_j.distribution_input.subprocess.run',run)
    assert isolated_probe(4)['reason'] == 'not_game'
    assert calls[0] != tmp_path
