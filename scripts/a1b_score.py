"""A1の採点器を同じ定義のまま使い、A1b二構成を別の出力先に採点する。"""
from __future__ import annotations
import argparse
import json
import pandas as pd
from scripts.a1_completion_replay import ROOT, OUT as BASELINE
from scripts import a1_completion_score as scoring
from scripts import a1_completion_errors as errors


def main() -> None:
    """採点器内の互換キーa1を維持し、構成名と実フラグを結果へ明記する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('single', 'combo'))
    args = parser.parse_args()
    out = ROOT/'logs/a1b'/args.mode
    scoring.OUT = errors.OUT = out
    result = scoring.score()
    print(args.mode, '門v2・誤確定・回帰', result['gates'], flush=True)
    chains, dynamic = [], []
    for name in errors.NAMES:
        part, frames = errors.extract('a1', name)
        chains.append(part)
        dynamic.extend(frames)
        print(args.mode, name, len(part), flush=True)
    combined, frames = pd.concat(chains), pd.DataFrame(dynamic)
    combined.to_csv(out/'a1_all_chains.csv', index=False)
    frames.to_csv(out/'a1_dynamic_predictions.csv', index=False)
    baseline = pd.read_csv(BASELINE/'error_summary.csv')
    summary = pd.concat([baseline[baseline['mode'] == 'prod'],
                         pd.DataFrame(errors.errors('a1', combined, frames))])
    summary.to_csv(out/'error_summary.csv', index=False)
    pd.DataFrame(errors.scenes('a1')).to_csv(out/'scenes.csv', index=False)
    score = summary[(summary.group == 'observed_7plus') & (summary.metric == 'score')].set_index('mode')
    result['gates']['score_median'] = bool(score.loc['a1', 'abs_p50'] < score.loc['prod', 'abs_p50'])
    result['configuration'] = args.mode
    result['baseline'] = str(BASELINE/'replay/prod')
    result['passed'] = all(result['gates'].values())
    (out/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result['gates']), flush=True)


if __name__ == '__main__':
    main()
