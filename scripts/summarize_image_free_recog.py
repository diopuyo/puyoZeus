"""画像非同梱試験の現在値を12行以内にまとめる。未測定は合格にしない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

WATCH_SECONDS = 60


def read_report(root: Path, name: str) -> dict:
    path = root / name / 'comparison.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def fraction(report: dict, key: str) -> str:
    values = report.get('metrics', {}).get(key, {})
    if not values:
        return '未測定'
    return f"{values['equal']}/{values['total']} ({values['rate']:.6%})"


def write_summary(root: Path, output: Path, commit: str) -> bool:
    digit = read_report(root, 'c68_stream_v3_batch')
    ui = read_report(root, 'ui_mask_v1')
    wide = read_report(root, 'wide_v4')
    frame_state = '完走' if digit.get('complete') else '未完走'
    lines = [
        '状態: 未完了・公開不可。画像非同梱の全体合格を確認していない。',
        f'コミット: 前回試作1a8cbb11、今回{commit}。pushなし。',
        '候補実装: 得点/式OCRの8桁一括推論、R1広域×の専用回帰器、窒息位置×印の分類器。既定OFF。',
        '既知11場面: 追加学習後の回帰試験で差0。2119.291秒2Pの40×64を復元（学習内・独立合格ではない）。',
        f"c68連続OCR: {digit.get('frames', 0)}/188400フレーム・{frame_state}。得点{fraction(digit, 'scores')}、式{fraction(digit, 'formulas')}、桁{fraction(digit, 'digits')}。",
        f"×印標本: {ui.get('equal', '未測定')}/{ui.get('total', '未測定')}。R1広域×標本: {wide.get('equal', '未測定')}/{wide.get('total', '未測定')}。全フレーム合格ではない。",
        '全体/勝率: 勝敗・中央テロップ・予告は未置換。全消し・窒息・確定盤面・B3特徴・表示勝率は未比較。',
        '差の重要度: c68の121.817秒2P等で旧得点0→新欠測。表示勝率への影響は未確定（軽微とは分類できない）。',
        'zenchi: 既知保存先とDドライブで原動画未発見。取得は通常/Node指定ともHTTP403。local_framesの途中データと保持契約を保存。',
        '新repo: puyozeus-liveは変更なし。未合格のため実行コード移行・zip作成・起動→初期設定→試合待ちは未実施。',
        '検証: 旧方式実資産テスト68件、新候補/監視/R1/完走判定テスト35件成功。画像の追加0件。OBS優先監視20秒間隔、既存配布物は未変更。',
        f'原票: {root.as_posix()}。c68_stream_v3_batch/differences.jsonl.gzに連続差分。残りは全置換・全体比較・合格後移行。',
    ]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return bool(digit.get('complete') and ui and wide)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', default='未コミット')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        finished = write_summary(args.root, args.output, args.commit)
        if finished or not args.watch:
            return
        time.sleep(WATCH_SECONDS)


if __name__ == '__main__':
    main()
