"""Phase 6の原票から日本語報告と再現用の指紋を生成する。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prefire_v6_replay import OUT
from scripts.prefire_v6_measure import save


def diagnostic_summary() -> dict:
    """不一致例だけの価値差分布を、全体の分布と分けて記録する。"""
    rows = [json.loads((OUT/'diagnostics_full'/f'diagnose_{i:03d}.json').read_text()) for i in range(100)]
    choices = [choice for row in rows for choice in row['sides']]
    def distribution(items: list[dict]) -> dict:
        values = np.asarray([c['regret_logit'] for c in items])
        return dict(count=len(values), mean=float(values.mean()),
                    quantiles=dict(zip(('min', 'p25', 'p50', 'p75', 'p90', 'p95', 'max'),
                                      np.percentile(values, (0, 25, 50, 75, 90, 95, 100)).tolist())))
    equal = sum(c['same_move'] for c in choices)
    return dict(positions=100, choices=len(choices), matches=equal, match_rate=equal/len(choices),
                all=distribution(choices), mismatches=distribution([c for c in choices if not c['same_move']]),
                scope='5Bと同じ100局面・両側1手の全候補。通知速度は既知3組の全列挙で別測定。')


def manifest() -> dict:
    """本番設定・原票・コードのSHAと新規関数の規約を検査する。"""
    paths = [*Path('src').glob('*v6*.py'), *Path('scripts').glob('prefire_v6*.py'),
             *Path('tests').glob('test_prefire*v6*.py')]
    violations = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_bytes())):
            if not isinstance(node, ast.FunctionDef):
                continue
            missing = [a.arg for a in node.args.args if a.arg not in ('self','cls') and a.annotation is None]
            if node.end_lineno-node.lineno+1 > 50 or missing or node.returns is None:
                violations.append(dict(file=str(path), function=node.name, missing=missing,
                                       lines=node.end_lineno-node.lineno+1))
    paths.extend([Path('src/production_config.py'), Path('scripts/replay_exchange_event_20260926.py'),
                  Path('src/exchange_display_smoothing.py'), OUT/'PREREGISTRATION.md', OUT/'latency.json', OUT/'strength.json'])
    artifacts = []
    for mode in ('train', 'replay', 'cut'):
        for receipt in (OUT/mode).glob('*/DONE.json'):
            artifacts.extend(receipt.parent/name for name in ('DONE.json', 'events.jsonl',
                'display.npz', 'prefire_trace.npz', 'prefire_v6_features.npz'))
            artifacts.append(Path(json.loads(receipt.read_text())['input']))
    for part in ('s1', 's5'):
        artifacts.extend(OUT/'baseline'/part/name for name in ('DONE.json', 'display.npz', 'events.jsonl'))
    return dict(sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                data_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(artifacts))},
                violations=violations)


def report(result: dict, diagnostic: dict, strength: dict) -> str:
    """条件の変更や未測定値の補完なしで報告する。"""
    latency, regression = result['latency'], result['regression']
    lines = ['# Phase 6 — 軽量比較・選択手だけ本番評価', '',
        f"**総合判定: {'合格' if result['passed'] else '不合格'}**。既定OFF。本番登録なし。`src/production_config.py` は未変更。",
        '作業元: `D:/puyo_analyzer/wt_prefire`、開始HEAD `6e49df1`。', '',
        '## 事前登録の6条件', '']
    lines.extend(f"- 条件{i}: {'合格' if result['gates'][f'condition{i}'] else '不合格'}" for i in range(1,7))
    lines.extend(['', f"初回通知P95: **{latency['p95_ms']:.3f}ms**。固定遅れ **L={latency['latency_sec']}秒**。",
        '5Aと同じindex 0/25/44/71/77の初回5通知、モデルロードを除き、全列挙・遷移・両側評価を含む。',
        '各通知の前後にpgrep原票を保存。他の重いPythonがあれば正式測定を開始しない。5標本なので全通知分布の保証ではない。', '',
        '## セット2（58試合）', ''])
    for name in ('prefire', 'm_time', 'agree', 'placebo'):
        row = result[name]
        lines.append(f"- {name}: 本番 {row['baseline']:.9f} → Phase6/対照 {row['phase6']:.9f}、差 {row['difference']:+.9f}、95%区間 {row['ci95']}、{row['games']}試合。")
    lines.extend([f"- ±3帯反転: {result['flips']}", f"- 誤った負け確定: {result['false_deaths']}（試合・側単位、追加E35根拠も監査）。",
        f"- 既存5記録: 誤確定 {regression['false_deaths']}/{regression['cases']}件、3:00場面 {regression['scene_first_sec']}秒、第14試合 {regression['game14']}。",
        f"- q 300/600秒打切り監査: {result['leakage_passed']}。表示・予測内訳・zを全長再生の同時刻と照合。", '',
        '## 合否外の100局面診断', '', diagnostic['scope'],
        f"選択手一致: {diagnostic['matches']}/{diagnostic['choices']} = {diagnostic['match_rate']:.1%}。",
        f"不一致例の価値差(logit): {diagnostic['mismatches']}。", '',
        '## セット1だけの反映係数学習', '',
        'a(z)=sigmoid(切片+w·z)。目的関数は57試合等重みLL、L2=0.0001。重みの分母は各試合の表示全行数。traceがない行のLLは係数に依存しない定数として省略。セット2ラベルを学習へ使わない。',
        'zはNEXT採用読みとの一致継続・繰上げ確認の証拠割合、欠測、探索打切り、軽量logitと本番logitの差/(1+差)。a(z)専用の4列として保存・表示時刻選択・学習・推論の列順を照合。既存45指標の変換へは追加しない。',
        '既知3組は全列挙するため探索打切り列は0。2組既知の応手は5Cと同じ決定的8標本。',
        f"係数（切片、zの順）: {strength['coefficients']}。非ゼロ補正行 {strength['nonzero_delta']}/{strength['rows']}。",
        f"識別可能: {strength['identifiable']}。識別不能の場合、正則化最小解を学習成功の根拠にはしない。", '',
        '## 実装と検証', '',
        '5Cの候補列挙と遷移表は無変更。比較は送量・相殺残量・窒息・空き容量/同色隣接/中央列余裕の線形値。',
        '各側の選択候補と無発火終端の最良待機だけを本番評価。仮想着弾G_feとS3をlogit合成し、E35は全色を認める保守的証明。',
        '反映は5Aの完了時刻規則。入力置換前に未完了の要求は取消し、通常値への復帰を予測適用と数えない。',
        '本番との短窓照合は3,570行・確率/表示/由来/時刻すべて一致（CHECKS_v2.json）。',
        '本worktreeに欠けていた採用済み表示平滑だけを任意指定で移植。本番CLIへの自動配線や設定登録はしない。',
        '再現: `bash logs/prefire_prediction/v6/workflow.sh`。完了原票は上書きしない。指定WSL venv、PYTHONPATH=.。',
        '詳細: v6/RESULT.json、DIAGNOSTICS.json、strength.json、latency.json、notifications/、resource_history.jsonl、manifest.json。', ''])
    return '\n'.join(lines)


def main() -> None:
    """採点済み原票を必須とし、報告を出す。"""
    result = json.loads((OUT/'RESULT.json').read_text())
    strength = json.loads((OUT/'strength.json').read_text())
    diagnostic = diagnostic_summary()
    save(OUT/'DIAGNOSTICS.json', diagnostic)
    receipt = manifest()
    save(OUT/'manifest.json', receipt)
    if receipt['violations']:
        raise ValueError(receipt['violations'])
    content = report(result, diagnostic, strength) + execution_notes()
    Path('logs/prefire_prediction/PHASE6.md').write_text(content, encoding='utf-8')


def execution_notes() -> str:
    """反映実績・資源・検証証跡を最終報告に付け、無効果の理由を明示する。"""
    lines = ['', '## 実行実績', '']
    for mode in ('train', 'replay', 'cut'):
        totals = dict(submitted=0, cancelled=0, completed=0, pending=0)
        applied, raw_changed = 0, 0
        paths = sorted((OUT/mode).glob('*/DONE.json'))
        for path in paths:
            info = json.loads(path.read_text())
            for key in totals:
                totals[key] += info['deferred'][key]
            with np.load(path.parent/'prefire_v6_features.npz') as data:
                rows = data['values']
                applied += int(np.count_nonzero(rows[:,4]))
                raw_changed += int(np.count_nonzero(rows[:,2] != rows[:,3]))
        lines.append(f'- {mode}: {len(paths)}記録、要求処理 {totals}、a>0の表示行 {applied}、生の増分≠0の行 {raw_changed}。')
    history = [json.loads(line) for line in (OUT/'resource_history.jsonl').read_text().splitlines()]
    minimum = min(row['available_kib'] for row in history)/(1024*1024)
    maximum = max(row['python_count'] for row in history)
    checks = json.loads((OUT/'CHECKS_v2.json').read_text())['light_board_ms']
    lines.extend([f'- 記録されたWSL全体のPython最大数: {maximum}、MemAvailable最小: {minimum:.3f}GiB。',
        f"- 軽量盤面評価は{checks['count']}盤面、P95={checks['p95']:.6f}ms、最大={checks['maximum']:.6f}ms。",
        '- 関連単体テスト114件合格（final_tests.log）。採点器再確認8件合格（scoring_tests.log）。',
        '- 未知NEXT期待値を含む軽量評価28件合格（unknown_sample_test.log）。境界連結・予測増分の限定を含む採点10件合格（score_boundary_test.log）。表示行への整列を含む学習4件合格（alignment_test.log）。重複を除く関連テストは121件。',
        '- セット1のウォームアップ重複を採点前に是正（TRAIN_ALIGNMENT.json）。表示された時刻だけで再集計し、係数が固定済み値と完全一致することを確認。旧原票はstrength_before_alignment.jsonに保管。L・z・目的関数は変更していない。',
        '- L=5.5秒を満たす前の入力更新は5Aの規則で取消し。非ゼロ増分がない場合、a(z)の係数0・a=0.5は正則化解であり、有効性を学習できたという意味ではない。',
        '- 再開は `python -m scripts.prefire_v6_batch`。固定済みlatency.jsonとstrength.jsonを読み、DONE.jsonがある記録を飛ばす。',
        '- 初期のdiagnostics/はG_feのみの予備診断。正式診断はS3/E35を含むdiagnostics_full/。正式速度原票はnotifications/。',
        '- 採点前に全区間の表示を時刻連結する既存METRIC.mdの定義に是正。区間境界の端数フレームを除外した暫定採点はpre_boundary_alignment/に保管し撤回。試合窓・指標式・閾値・L・係数は変更していない。',
        '- 保存済み本番と今回の通常評価にはs1の78確率行・s5の246確率行の差（いずれもS3_landing）がある。同じworktreeのOFF全長再生とは時刻・確率・表示・由来が完全一致（RESULT.json: off_controls）。全区間LL差はPhase 6の改善効果とは扱わない。',
        '- 門の対照は指定セット2の保存済み本番のまま。循環ずらしには通常評価の版差を混ぜず、traceに保存した実際の予測logit増分だけを使う。今回の増分は0なので対照も本番と一致。', ''])
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
