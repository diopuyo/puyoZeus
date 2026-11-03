"""OBS WebSocket v5。既定オフ、既存ソース設定を上書きしない。"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any
from uuid import uuid4

TIMEOUT_SECONDS = 5
SOURCE_PREFIX = 'puyoZeus Live'


def authentication(password: str, salt: str, challenge: str) -> str:
    def digest(value: str) -> str:
        return base64.b64encode(hashlib.sha256(value.encode()).digest()).decode()
    return digest(digest(password + salt) + challenge)


class ObsClient:
    def __init__(self, port: int, password: str) -> None:
        from websocket import create_connection
        self.socket = create_connection(f'ws://127.0.0.1:{port}', timeout=TIMEOUT_SECONDS,
                                        subprotocols=['obswebsocket.json'], http_proxy_host=None)
        try:
            hello = json.loads(self.socket.recv())
            if hello['op'] != 0:
                raise ValueError('OBS Hello がありません / Missing OBS Hello')
            identify: dict[str, Any] = dict(rpcVersion=1, eventSubscriptions=0)
            if 'authentication' in hello['d']:
                identify['authentication'] = authentication(password, **hello['d']['authentication'])
            self.socket.send(json.dumps(dict(op=1, d=identify)))
            if json.loads(self.socket.recv())['op'] != 2:
                raise ValueError('OBS 認証に失敗 / OBS authentication failed')
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.socket.close()

    def request(self, kind: str, **data: Any) -> dict[str, Any]:
        identity = uuid4().hex
        self.socket.send(json.dumps(dict(op=6, d=dict(requestType=kind, requestId=identity, requestData=data))))
        while True:
            message = json.loads(self.socket.recv())
            if message['op'] != 7 or message['d']['requestId'] != identity:
                continue
            result = message['d']
            if not result['requestStatus']['result']:
                raise ValueError(f'OBS {kind}: {result["requestStatus"].get("code")}')
            return result.get('responseData', {})


def scene_sources(client: ObsClient, scene: str, seen: set[str] | None = None,
                  group: bool = False) -> list[dict]:
    seen = set() if seen is None else seen
    if scene in seen:
        return []
    seen.add(scene)
    kind = 'GetGroupSceneItemList' if group else 'GetSceneItemList'
    items = client.request(kind, sceneName=scene)['sceneItems']
    result = list(items)
    for item in items:
        if item.get('isGroup') or item.get('sourceType') == 'OBS_SOURCE_TYPE_SCENE':
            result.extend(scene_sources(client, item['sourceName'], seen, item.get('isGroup', False)))
    return result


def integrate(settings: dict, url: str, apply: bool = False, scene: str | None = None) -> dict:
    """適用ボタンだけが変更を許可する。接続確認は読み取り専用。"""
    if not settings['obs_enabled']:
        raise ValueError('OBS 連携をオンにしてください / Enable OBS integration')
    try:
        client = ObsClient(settings['obs_port'], settings['obs_password'])
    except Exception as error:
        raise ValueError('OBS に接続できません。WebSocket の有効化・ポート・パスワードを確認してください / Check OBS WebSocket, port and password') from error
    try:
        target = scene or client.request('GetCurrentProgramScene')['currentProgramSceneName']
        items = scene_sources(client, target)
        overlays = [item['sourceName'] for item in items if item.get('inputKind') == 'browser_source']
        name = SOURCE_PREFIX + ' ' + hashlib.sha256(url.encode()).hexdigest()[:8]
        if apply:
            existing = client.request('GetInputList')['inputs']
            if not any(item['inputName'] == name for item in existing):
                client.request('CreateInput', sceneName=target, inputName=name, inputKind='browser_source',
                    inputSettings=dict(url=url, width=1920, height=1080), sceneItemEnabled=True)
            elif not any(item['sourceName'] == name for item in items):
                client.request('CreateSceneItem', sceneName=target, sourceName=name, sceneItemEnabled=True)
            if not client.request('GetVirtualCamStatus')['outputActive']:
                client.request('StartVirtualCam')
        return dict(scene=target, browser_sources=overlays, applied=apply,
            virtual_camera=client.request('GetVirtualCamStatus')['outputActive'],
            output_mode='manual_check_required', warning='シーン全体を出力すると表示が映り込みます。仮想カメラの歯車 → 出力の種類「ソース」→ ゲーム映像。 / Scene output can include the overlay. Virtual Camera gear → Output type: Source → game capture.')
    except Exception as error:
        raise ValueError('OBS の操作が完了しませんでした。接続と現在のシーンを確認してください / OBS operation failed; check connection and current scene') from error
    finally:
        client.close()
