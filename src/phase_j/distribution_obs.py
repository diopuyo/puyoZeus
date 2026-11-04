"""OBS WebSocket v5。適用操作で表示を設定し、起動時に既存の表示を更新する。"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import threading
from typing import Any
from uuid import uuid4

TIMEOUT_SECONDS = 5
SOURCE_PREFIX = 'puyoZeus Live'
AUTHENTICATION_FAILED = 4009


def refresh_browser(client: ObsClient, name: str) -> None:
    """古いHTML・JSを残さず、ブラウザソースを読み直す。"""
    client.request('PressInputPropertiesButton', inputName=name, propertyName='refreshnocache')


def refresh_saved_sources(settings: dict) -> None:
    """起動時に一度だけ既存の製品ソースを更新し、接続失敗は無視する。"""
    try:
        client = ObsClient(settings['obs_port'], settings['obs_password'])
        try:
            for item in client.request('GetInputList')['inputs']:
                name = item['inputName']
                if (item.get('inputKind') == 'browser_source'
                        and re.fullmatch(re.escape(SOURCE_PREFIX) + r' [0-9a-f]{8}', name)):
                    refresh_browser(client, name)
        finally:
            client.close()
    except Exception:
        pass


def refresh_on_startup(settings: dict) -> None:
    """保存済み連携だけを背景で更新し、アプリ起動を待たせない。"""
    if settings.get('obs_enabled') and settings.get('obs_password'):
        threading.Thread(target=refresh_saved_sources, args=(dict(settings),), daemon=True).start()


class ObsAuthenticationError(ValueError):
    """OBSが認証失敗の終了コードを返した。"""


def receive_identified(connection: Any) -> None:
    """closeフレームのコードを失わずに認証結果を読む。"""
    from websocket import ABNF
    opcode, payload = connection.recv_data(control_frame=True)
    if opcode == ABNF.OPCODE_CLOSE:
        code = int.from_bytes(payload[:2], 'big') if len(payload) >= 2 else None
        if code == AUTHENTICATION_FAILED:
            raise ObsAuthenticationError('OBS authentication failed')
        raise OSError('OBS closed the connection')
    if json.loads(payload)['op'] != 2:
        raise ValueError('OBS Identified がありません / Missing OBS Identified')


def authentication(password: str, salt: str, challenge: str) -> str:
    def digest(value: str) -> str:
        return base64.b64encode(hashlib.sha256(value.encode()).digest()).decode()
    return digest(digest(password + salt) + challenge)


class ObsClient:
    def __init__(self, port: int, password: str) -> None:
        from websocket import create_connection
        direct = socket.create_connection(('127.0.0.1', port), timeout=TIMEOUT_SECONDS)
        try:
            self.socket = create_connection(f'ws://127.0.0.1:{port}', timeout=TIMEOUT_SECONDS,
                                            subprotocols=['obswebsocket.json'], socket=direct)
        except Exception:
            direct.close()
            raise
        try:
            hello = json.loads(self.socket.recv())
            if hello['op'] != 0:
                raise ValueError('OBS Hello がありません / Missing OBS Hello')
            identify: dict[str, Any] = dict(rpcVersion=1, eventSubscriptions=0)
            if 'authentication' in hello['d']:
                identify['authentication'] = authentication(password, **hello['d']['authentication'])
            self.socket.send(json.dumps(dict(op=1, d=identify)))
            receive_identified(self.socket)
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
                status = result['requestStatus']
                raise ValueError(f'OBS {kind}: {status.get("comment") or status.get("code")}')
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


def update_browser_url(client: ObsClient, name: str, url: str) -> bool:
    """既存のURLと比較し、実際に変更したかを返す。"""
    previous = client.request('GetInputSettings', inputName=name)['inputSettings']
    changed = previous.get('url') != url
    client.request('SetInputSettings', inputName=name, inputSettings=dict(url=url), overlay=True)
    return changed


def integrate(settings: dict, url: str, apply: bool = False, scene: str | None = None) -> dict:
    """適用ボタンだけが変更を許可する。接続確認は読み取り専用。"""
    if not settings['obs_enabled']:
        raise ValueError('OBS 連携をオンにしてください / Enable OBS integration')
    try:
        client = ObsClient(settings['obs_port'], settings['obs_password'])
    except ObsAuthenticationError as error:
        raise ValueError('OBS のパスワードが違います。OBS の「ツール → WebSocket サーバー設定」の「接続情報を表示」で確かめてください / '
                         'The OBS password is incorrect. In OBS, open Tools → WebSocket Server Settings → Show Connect Info.') from error
    except Exception as error:
        port = settings['obs_port']
        raise ValueError(f'OBS につながりません。OBS が起動しているか、WebSocket サーバーが有効か、ポート ({port}) が合っているかを確かめてください / '
                         f'Cannot connect to OBS. Check that OBS is running, WebSocket Server is enabled, and the port ({port}) is correct.') from error
    try:
        target = scene or client.request('GetCurrentProgramScene')['currentProgramSceneName']
        items = scene_sources(client, target)
        overlays = [item['sourceName'] for item in items if item.get('inputKind') == 'browser_source']
        name = SOURCE_PREFIX + ' ' + hashlib.sha256(url.encode()).hexdigest()[:8]
        reusable = name if name in overlays else next((item for item in overlays if re.fullmatch(re.escape(SOURCE_PREFIX) + r' [0-9a-f]{8}', item)), None)
        already_added = name in overlays or reusable is not None
        url_updated = False
        if apply:
            existing = client.request('GetInputList')['inputs']
            if reusable:
                name = reusable
                url_updated = update_browser_url(client, name, url)
            elif not any(item['inputName'] == name for item in existing):
                client.request('CreateInput', sceneName=target, inputName=name, inputKind='browser_source',
                    inputSettings=dict(url=url, width=1920, height=1080), sceneItemEnabled=True)
            elif not any(item['sourceName'] == name for item in items):
                client.request('CreateSceneItem', sceneName=target, sourceName=name, sceneItemEnabled=True)
            refresh_browser(client, name)
            if not client.request('GetVirtualCamStatus')['outputActive']:
                client.request('StartVirtualCam')
        return dict(scene=target, browser_sources=overlays, applied=apply, already_added=already_added,
            url_updated=url_updated, url=url,
            virtual_camera=client.request('GetVirtualCamStatus')['outputActive'],
            output_mode='manual_check_required', warning='シーン全体を出力すると表示が映り込みます。仮想カメラの歯車 → 出力の種類「ソース」→ ゲーム映像。 / Scene output can include the overlay. Virtual Camera gear → Output type: Source → game capture.')
    except Exception as error:
        reason = ' '.join(str(error).split()).replace(' / ', ' · ')
        raise ValueError(f'OBS の操作が完了しませんでした（{reason}）。接続と現在のシーンを確認してください / OBS operation failed ({reason}); check connection and current scene') from error
    finally:
        client.close()
