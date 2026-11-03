"""OBS試験の入口をD:へ置く。起動先は未使用の製品ステージ。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json
from pathlib import Path

BUNDLE = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive'))
OUT = Path('D:/PuyoLive_v49f_20261019')
CAMERA_SCRIPT = '''import sys
import cv2
from src.phase_j.launcher import probe_devices, format_devices

if '--check' not in sys.argv:
    print(format_devices(probe_devices()))
    choice = input('Preview index (blank to quit): ').strip()
    if choice:
        capture = cv2.VideoCapture(int(choice), cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    print('No frame. Start OBS Virtual Camera and try again.')
                    break
                cv2.imshow('Camera index '+choice+' - ESC to close', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(30) & 255 == 27:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
'''


def cmd(name: str, lines: list[str]) -> None:
    data = ('\r\n'.join(lines)+'\r\n').encode('ascii')
    (OUT/name).write_bytes(data)
    assert b'\n' not in data.replace(b'\r\n', b'')


def main() -> None:
    OUT.mkdir(exist_ok=False)
    config = json.loads((BUNDLE/'puyo_live.json').read_text(encoding='utf-8'))
    for key, filename in [('b3_model', 'model.joblib'), ('b3_timing_table', 'timing_table.json')]:
        config[key] = str(BUNDLE/'app/models/b3'/filename)
    config['output_dir'] = str(BUNDLE.parent/'obs_user_output')
    (OUT/'puyo_live.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    common = ['@echo off', 'setlocal', 'set PYTHONUTF8=1', f'cd /d "{BUNDLE}"']
    cmd('Start_PuyoLive_v49f.cmd', common+[
        'echo OBS Virtual Camera must be running.', 'echo Overlay: http://127.0.0.1:8765/',
        'echo Press Ctrl+C in this window to stop PuyoLive.',
        '"python\\python.exe" -m src.phase_j.launcher --config "%~dp0puyo_live.json" --require-manifest',
        'if errorlevel 1 pause'])
    cmd('List_Cameras.cmd', common+['"python\\python.exe" "%~dp0camera_indices.py"', 'pause'])
    (OUT/'camera_indices.py').write_text(CAMERA_SCRIPT, encoding='ascii')
    write_guide()
    print(OUT)


def write_guide() -> None:
    guide = _public_dev_text('撃ち合い勝率 v4.9f — OBS 試験版 build_20261019a\n\n1. OBS を起動し、1920×1080のゲーム映像を映します。\n   仮想カメラの出力は「ソース」でゲーム映像だけを選び、分析表示を映り込ませないでください。\n2. OBS の「仮想カメラ開始」を押します。\n3. 初回は List_Cameras.cmd を開きます。番号ごとの映像有無と解像度を表示します。\n   続いて候補の番号（前回は3）を入力するとプレビューが開きます。\n   OBSと同じゲーム映像が映る番号を確認し、ESCキーでプレビューを閉じます。\n   一覧は機器名を表示しません。1920×1080だけで仮想カメラと断定しないでください。\n   同じフォルダの puyo_live.json をメモ帳で開き、device_index（初期値3）をその番号へ直します。\n   device_name は名札です。入力の選択に使われるのは device_index です。\n4. Start_PuyoLive_v49f.cmd をダブルクリックします。黒いウィンドウは開いたままにします。\n   起動・入力確認には少し時間がかかります。ブラウザで http://127.0.0.1:8765/ を開きます。\n   OBSに重ねる場合も同じURLをブラウザソースに指定します。\n5. 「B3 勝率」が従来の判定、その下の黄色い枠「撃ち合い勝率 v4.9f」が試験欄です。\n   1P/2Pそれぞれの勝率を表示します。「—」は計算結果がまだない状態です。\n   「○秒前」と理由は、保持中の入力時刻からの経過と更新待ちの理由です。\n   「死亡断定」は証明された側に対応した0%/100%です。\n   試合開始直後・撃ち合いの入力が揃う前は「入力待ち」でも異常とは限りません。\n6. 止めるときは黒いウィンドウで Ctrl+C を押し、処理が終了するまで待ちます。\n   その後 OBS の「仮想カメラ停止」を押します。OBSの終了は最後です。\n\n以前のPuyoLiveが起動中なら先に終了してください（ポート8765の重複を避けます）。\n本体: <work>\\build_20261019a_stage\\PuyoLive\n設定: D:\\PuyoLive_v49f_20261019\\puyo_live.json\nログ: <work>\\build_20261019a_stage\\obs_user_output\nこの.cmdは上記の本体を使います。本体フォルダを移動・削除しないでください。\nZIPを別PCへ移す場合は展開先のPuyoLive.batとpuyo_live.jsonを使ってください。\nこれはOBS試験用です。製品本線の既定OFFは変更していません。\n今回の低スペック換算では更新の最大619.9ms・500ms超7件で、速度条件は不合格です。\nブラウザの実画面・paint時間は未検証です。詳細はLIVE_V49F_OBS_20261019.mdを参照してください。\n')
    (OUT/'OBS_TEST_README_ja.txt').write_bytes(b'\xef\xbb\xbf'+guide.replace('\n', '\r\n').encode('utf-8'))


if __name__ == '__main__':
    main()
