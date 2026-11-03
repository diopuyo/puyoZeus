"""E29の固定基準、途中予測の例外、場面の初判定を検収記録へまとめる。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.run_e17_ablation_20260928 import ALL_SOURCES

OUT = Path('logs/e29')


def directory(source: str) -> Path:
    """固定検収の出力ディレクトリを返す。"""
    return OUT/'on'/(source if source in ('review', 'zenchi') else f'renders/{source}/on')


def exceptions() -> list[dict]:
    """採用後不一致・未確認の全件を、動画と予測層の種類付きで残す。"""
    rows = []
    for source in ALL_SOURCES:
        for kind, file in (('midchain', 'midchain_audit.json'), ('hidden', 'hidden_final_audit.json')):
            for row in json.loads((directory(source)/file).read_text())['rows']:
                if row['outcome'] == 'accepted' and row['final_mismatch'] is not False:
                    rows.append(dict(source=source, kind=kind, in_three_videos=source in SOURCES, **row))
    save_json(OUT/'PREDICTION_EXCEPTIONS.json', rows)
    return rows


def scene() -> dict:
    """予測の式一致と死亡成立の時刻を分け、表示閾値到達との順序を残す。"""
    events = [json.loads(line) for line in (directory('review')/'events.jsonl').read_text().splitlines()]
    values = sorted([v for event in events for v in event['values'] if 2755 <= v['t_sec'] <= 2771],
                    key=lambda v: v['t_sec'])
    deaths = [v for v in values if v['source'] == 'unavoidable_death' and '2P' in v.get('dead_sides', [])]
    audit = json.loads((directory('review')/'hidden_final_audit.json').read_text())
    candidates = [r for r in audit['rows'] if r['side'] == '2P' and 2755 <= r['candidate_sec'] <= 2771]
    attempts = [v for v in values if v.get('multi_landing', [{}, {}])[1].get('proofs')]
    result = dict(first_death=deaths[0] if deaths else None, hidden_candidates=candidates,
        first_hidden_death_attempt=attempts[0] if attempts else None)
    save_json(OUT/'SCENE_AUDIT.json', result)
    return result


def main() -> None:
    """判定を緩めず、採用候補の維持と条件付き動画生成を明記する。"""
    metrics = json.loads((OUT/'on/METRICS.json').read_text())
    errors, detail = exceptions(), scene()
    baseline = json.loads(Path('logs/e27/on/METRICS.json').read_text())
    summary = dict(metrics=metrics, baseline=baseline, prediction_exceptions=errors,
        scene=detail, video_eligible=metrics['candidate'],
        video_generated=Path('logs/review_zenchi_g41_43_e29/complete.json').exists())
    save_json(OUT/'SUMMARY.json', summary)
    lines = ['# E29 検収', '',
        'E27構成＋--midchain-single-observation。新フラグ既定OFF。production_config.py変更なし。',
        '通常・隠し段とも1観測で候補化し、次段式一致まで確率・死亡へ公開しない。', '',
        '固定条件・母数・全合否：on/METRICS.json。採用後不一致・未確認の全件：PREDICTION_EXCEPTIONS.json。',
        '通常予測と隠し段上限の件数：PREDICTION_COUNTS.json。場面の式一致と死亡判定：SCENE_AUDIT.json。',
        'OFF_*.jsonは5入力のE27出力とのバイト一致。入力・モデル不変はVERIFICATION.json。', '',
        '指定場面では2755.450秒に隠し段候補の次段一致が成立し、最大打ち返し75440点を確認。',
        '2760.117秒には純受け171個で死亡探索へ進むが、20000ノード上限を超えて証明未完了。',
        '表示2P≤5%は2771.250秒のまま。今回は観測数だけを変更し、探索上限は変更していない。', '',
        '3動画の通常予測では、q動画649.133秒の予測77380点に対し最終63620点。',
        '次段式は一致したが後続式で撤回された。撤回済み採用も最終不一致へ含めている。',
        '隠し段は死亡専用の最大上限であり、確率・表示用の予測とは分けて集計する。',
        '最終得点未確定は正解へ含めない。次段一致だけでは完走得点の一致を保証できなかった。', '',
        json.dumps(metrics, ensure_ascii=False, indent=2), '',
        '全条件合格。指定レビュー動画を生成する。' if metrics['candidate'] else
        '不合格。採用候補はE27を維持。条件未達のためE29レビュー動画は生成しない。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({k: metrics[k] for k in ('gates', 'candidate', 'prediction_counts')}, flush=True)


if __name__ == '__main__':
    main()
