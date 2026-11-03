"""短い納品状況と、小さな再現証跡を保存する。未測定を合格にしない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
import ctypes
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/live_parallel'
EVIDENCE = ROOT/'docs/live_parallel_evidence'


def trial(name: str) -> dict[str, Any]:
    return json.loads((OUT/name/'summary.json').read_text(encoding='utf-8'))


def average(names: list[str], field: str, child: str | None = None) -> float:
    rows = [trial(name)[field] for name in names]
    return statistics.mean([row[child] for row in rows] if child else rows)


def pair(name: str) -> str:
    before = [f'pair_{name}_{index}' for index in (0, 3)]
    after = [f'pair_{name}_{index}' for index in (1, 2)]
    return f"{average(before, 'rec_ms', 'mean'):.2f}→{average(after, 'rec_ms', 'mean'):.2f}"


def evidence() -> list[dict]:
    """原票の場所を残しつつ、モデル・大きなpickleはGitへ入れない。"""
    sys.path.insert(0, str(ROOT/'scripts'))
    from compare_live_parallel import compare
    EVIDENCE.mkdir(exist_ok=True)
    trials, comparisons = [], []
    for directory in sorted(OUT.iterdir()):
        if not (directory/'candidate.json').exists():
            continue
        trials.append(dict(name=directory.name, summary=trial(directory.name),
            candidate=json.loads((directory/'candidate.json').read_text())))
        if directory.name.startswith(('pair_', 'intra', 'inter', 'cv', 'sides', 'both', 'base')):
            comparisons.append(compare(OUT/'base_0', directory))
    (EVIDENCE/'trials.json').write_text(json.dumps(trials, indent=1), encoding='utf-8')
    (EVIDENCE/'comparisons.json').write_text(json.dumps(comparisons, indent=1), encoding='utf-8')
    for name in ('input_inventory.json', 'video_comparison.json', 'prepare_comparison.json',
                 'matrix.json', 'focused.json', 'final_matrix.json', 'color_pixels.json',
                 'build_off.json', 'build_on.json', 'tests.xml', 'tests_final.exit.json',
                 'verified_smoke_off.exit.json', 'verified_smoke_on.exit.json'):
        shutil.copy2(OUT/name, EVIDENCE/name)
    for mode in ('off', 'on'):
        shutil.copy2(OUT/f'verified_smoke_{mode}'/'result.json', EVIDENCE/f'smoke_{mode}.json')
    b3 = [[json.loads(line) for line in (OUT/f'verified_smoke_{mode}'/'b3_updates.jsonl').read_text().splitlines()]
          for mode in ('off', 'on')]
    fields = ('frame', 't_sec', 'trigger', 'raw_probability', 'error')
    projected = [[{name: row[name] for name in fields} for row in rows] for rows in b3]
    assert json.dumps(projected[0]) == json.dumps(projected[1])
    (EVIDENCE/'b3_smoke_parity.json').write_text(json.dumps(dict(rows=len(b3[0]),
        fields=fields, equal=True, scope='生確率とトリガーのみ。公開確率・全評価行の検証ではない'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    return trials


def lines(trials: list[dict]) -> list[str]:
    before, after = ['video_0', 'video_3'], ['video_1', 'video_2']
    camera = OUT/'camera_status.json'
    state = json.loads(camera.read_text())['state'] if camera.exists() else '未起動'
    state = {'waiting_camera': '先行レーン待機中（自動起動ジョブ稼働）',
             'running': '実行中', 'completed': '完了', 'failed': '失敗・停止'}.get(state, state)
    return [
        '結論: 小幅な短縮候補は実装済み。ただし全件品質門未達で未採用、配布pは未作成。',
        '基準m=dc24ee44。作業はC:、既存d〜m変更なし、D読取・pushなし。',
        f"連続900枚ABBA（速度n=839/試行）: 認識平均 {average(before,'rec_ms','mean'):.2f}→{average(after,'rec_ms','mean'):.2f}ms、P95の試行平均 {average(before,'rec_ms','p95'):.2f}→{average(after,'rec_ms','p95'):.2f}ms。",
        f"同台処理fps {average(before,'fps_mean'):.2f}→{average(after,'fps_mean'):.2f}、認識CPU {average(before,'steady_proc_cpu_cores'):.2f}→{average(after,'steady_proc_cpu_cores'):.2f}コア（実機fpsではない）。",
        '以下PNGは210枚・暖機等除外後n=149/試行。初期12条件×往復と、候補ごとのABBAを区別。',
        'ONNX通常intra=2/3/4: 平均30.04〜35.36ms、CPU2.95〜6.87コア。inter=2/3/4並列: 30.51〜40.40ms、2.96〜6.63コア。',
        f'ONNXスピンなしABBA平均ms: intra2 {pair("spin2")}、3 {pair("spin3")}、4 {pair("spin4")}。製品候補は2を選択。',
        f'左右スレッドABBA平均ms: {pair("sides")}、悪化のため不採用。プロセス分割は未測定。共有彩度較正ONは並行不可。',
        f'OpenCV ABBA平均ms: 2thread {pair("cv2")}、4thread {pair("cv4")}。3thread往復は30.57/30.84ms。未採用。',
        f'優先度/割当ABBA平均ms: BelowNormal→Idle {pair("idle")}、性能CPU set12→4 {pair("sets4")}。OBS競合は未検証。',
        '準備段（PNG210枚×往復、n=149）: 周期平均35.37→32.37ms（次画像copy+補正先行）。映像順/時刻・通知一致、実capture未接続。',
        '補正buffer単独: 同周期35.37→34.18ms、208/210回再利用。保持画像/view/非同期alias試験通過、通知・logits全210件一致。',
        '900枚の判定: 全通知900/900、読取盤140,400セル、確定盤140,088セルがバイト一致（両者STABLE438枚）。',
        '同CNN: 2,710呼出・99,200パッチのlogits全件バイト一致、argmax差0、最大絶対差0。評価行・公開確率は未照合。',
        '指定保存記録: g76,800/h82,749/j83,425/k82,903/e48,367更新、計374,244。i不在、全記録completeなし。',
        '問題: 指定inputs.jsonl.gzには画素/logitsがなく、元の実機認識全件を再投入できない。動画900枚の一致で代用合格にはしない。',
        f'55分試験: {state}。内部m→候補、device2/port18843、delta→queueの完了後に開始。状態はlogs/live_parallel/camera_status.json。',
        'capture→評価終了: P50/P95/P99/最大・長時間fps/CPUは未測定。OBS相当人工負荷も未実施。',
        '実装: live_onnx_parallel / live_color_buffer ともproduction_config既定False。配布pのMANIFEST・起動確認は未実施。',
        '関連156テスト合格・書出し依存2skip。拡張範囲は83合格/既存Windows・依存制約8失敗（全体合格とはしない）。',
        '内部版のMANIFEST m889/889・候補893/893。同梱Python各PNG60枚/B3生確率28行一致、終了0、起動監査D読取0。',
        '全試験HOLD/空き6GiBを確認、低優先度。負荷と条件・母数はdocs/live_parallel_evidence、原票logs/live_parallel。',
        '実装commit: 0c698e42（補正）/ e4a35774（ONNX）。検証手順とGIL根拠: docs/LIVE_PAR_VALIDATION.md。',
        '判断依頼: 過去実機画像がないため、「新規の同一画素・固定時刻による全件再投入」を認識品質門の代替としてよいか。合意までは配布pを作成しない。',
    ]


def main() -> None:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    trials = evidence()
    output = lines(trials)
    assert len(output) <= 25
    Path(_public_dev_text('<work>/LIVE_PAR.md')).write_text('\n'.join(output)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
