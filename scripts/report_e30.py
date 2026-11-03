"""E30の指定場面・固定合否・候補件数と所要時間を一つの検収記録へまとめる。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts import run_e30 as run
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e30')
REPORT = Path('docs/E30_PREFIRE_CANDIDATES_20260928.md')


def format_quantile(value: dict, scale: float = 1.) -> str:
    """欠測を0と書かず、比較可能な母数を添える。"""
    if not value['n']:
        return '未確定（n=0）'
    return f"{value['p50']*scale:.3f} / {value['p95']*scale:.3f}（n={value['n']}）"


def gate_rows(metrics: dict) -> list[str]:
    """母数を省略せず、事前登録の四条件だけを表にする。"""
    q, z, d, scene = (metrics[k] for k in ('q', 'zenchi', 'deaths', 'scene'))
    rows = [('q', 'q log loss', f"{q['log_loss']:.9f}、{q['frames']}行・{q['matches']}試合", '≤ .512778'),
        ('zenchi', 'zenchi一致率', f"{z['agreement']*100:.4f}%、{z['hits']}/{z['frames']}", '≥ 84.68%'),
        ('deaths', '誤発火', f"{d['false']}/{d['total']}、未判定{d['unlabelled']}", '≤ 1/28'),
        ('scene', '表示2P≤5%初到達', f"{scene['first_sec']}秒、対象1場面", '≤ 2766.0秒')]
    return ['|項目|実測・母数|基準|合否|', '|---|---|---|---|']+[
        f"|{label}|{value}|{threshold}|{'合格' if metrics['gates'][key] else '不合格'}|"
        for key, label, value, threshold in rows]


def candidate_rows(counts: dict) -> list[str]:
    """三動画とその合計を表示し、レビュー・zenchiは二重加算しない。"""
    rows = ['|動画|列挙発火|残存|0候補|列挙不可|残存時の絶対得点差P50/P95|', '|---|---:|---:|---:|---:|---|']
    for source in (*run.SOURCES, 'three_videos'):
        value = counts['three_videos'] if source == 'three_videos' else counts['sources'][source]
        rows.append(f"|{source}|{value['enumerated']}|{value['remaining']}|{value['zero']}|{value['skipped']}|"
                    +format_quantile(value['remaining_last_absolute_error'])+'|')
    return rows


def main() -> None:
    """数値を固定原票から読み、不合格なら動画は生成しない。"""
    metrics = run.report()
    counts = json.loads((OUT/'CANDIDATE_COUNTS.json').read_text())
    scenes = json.loads((OUT/'SCENE_CANDIDATES.json').read_text())
    summary = counts['three_videos']
    scene = scenes[0]
    status = '合格' if metrics['candidate'] else '不合格、採用候補E27を維持'
    lines = ['# E30: 発火候補の全列挙', '', f'## 最終判定: {status}', '',
        '開始点861ee34、比較基準E27 87b0fca。E27条件＋既定OFFの `--prefire-candidates`。',
        '`production_config.py`・モデル・入力記録は変更していない。', '', *gate_rows(metrics), '',
        '## 指定場面', '', f"2P連鎖ID{scene['chain_id']}、起点{scene['trigger_sec']:.6f}秒。",
        f"列挙{scene['trials']}経路、発火{scene['initial']['candidates']}候補（異なる配置{scene['initial']['unique']}）。",
        f"列挙時間{scene['enumeration_sec']:.6f}秒。実最終得点{scene['final_score']}点。", '',
        '|時刻|観測段|累積得点|残候補（配置の種類）|平均最終得点|', '|---|---:|---:|---:|---:|']
    lines += [f"|{r['t_sec']:.6f}|{r['count']}|{r['observed']}|{r['candidates']}（{r['unique']}）|"
              +('—' if r['mean_score'] is None else f"{r['mean_score']:.3f}")+'|' for r in scene['observations']]
    lines += ['', '4段目の式で全候補が棄却され、13連鎖打ち返しの復元には至らなかった。', '',
        '## 固定3動画の候補と精度', '', *candidate_rows(counts), '',
        '残存は最後の式観測で候補が1個以上の発火、0候補は列挙対象になった発火のうち最後に0個のもの。',
        f"初期には候補があり後に0となった発火は{summary['became_zero']}件。残存の最終得点未確認は{summary['final_unresolved']}件。",
        '途中棄却を含む最初の式一致時の絶対得点差P50/P95: '+format_quantile(summary['first_absolute_error'])+'点。',
        '途中棄却を含む最後の残存時の絶対得点差P50/P95: '+format_quantile(summary['last_absolute_error'])+'点。', '',
        '## 所要時間', '', '列挙P50/P95: '+format_quantile(summary['enumeration_seconds'])+'秒。',
        '絞り込みP50/P95: '+format_quantile(summary['filtering_seconds'], 1000)+'ms。',
        '発火時の同期列挙は30Hzの33.3ms枠を超え、現在のまま認識ループ内のライブ利用には間に合わない。', '',
        *design_notes(), '', '## 検証と再現', '',
        '`bash scripts/_launch_e30.sh`、`bash scripts/_launch_e30_tests.sh`、',
        '`python -m scripts.report_e30`、`python -m scripts.verify_e30`（既存WSL venv、PYTHONPATH=.）。',
        'OFF対照はreviewとqのE27保存出力をNPZ全列・イベントJSONL・診断JSONで照合。',
        '関連回帰628成功・1スキップ。表示接続の追加検証はpanel_tests.log。',
        '原票: `logs/e30/on/METRICS.json`、`CANDIDATE_COUNTS.json`、`SCENE_CANDIDATES.json`、',
        '各再生の `prefire_final_audit.json`。最終得点との照合は監査だけに限定し、実行時の候補選択へ戻さない。', '',
        '## レビュー動画', '', '全条件合格の場合のみ `python -m scripts.render_e30_review`。',
        '今回不合格のため `logs/review_zenchi_g41_43_e30/` と',
        '`D:/puyo_analyzer/videos/review/zenchi_g41-43_e30_mobile.mp4` は生成しない。']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(OUT/'SUMMARY.json', dict(metrics=metrics, counts=counts, scene=scenes, report=str(REPORT)))


def design_notes() -> list[str]:
    """平均と最大の境界、および候補母数の定義を明記する。"""
    return ['## 実装の境界', '',
        '保存起点が0連鎖の場合、試合で確認した4色の10組と読取NEXT/NEXT2の和集合を全22配置で試す。',
        '1手目が発火せず窒息していない場合だけ2手目を全列挙する。UNKNOWN・未確認4色は補完しない。',
        '同じ配置の計算を共用し、経路の多重度を保存する。全経路を等重みにした平均と最頻を使う。',
        '確率のS3には平均得点を既存の小数送り量換算へ渡し、landingには各候補の整数送り量の平均を渡す。',
        '完走盤面は最頻が唯一の場合だけ使用し、同数なら使わない。死亡証明には最大打ち返しの全同点終端を渡す。',
        '候補0は従来予測へ撤回。段ごとの得点照合を行い、消去色が入力にあれば照合する（固定記録には色観測なし）。',
        '現在層の盤面・履歴は不変。イベント原票にprefire_prediction、画面とCSVに「発火候補予測込み」を記録する。',
        '新しい学習列は追加せず、既存のS3側交換・差分反転・正規化とcount特徴経路を維持した。']


if __name__ == '__main__':
    main()
