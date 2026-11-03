"""video_30 (レビュー 01) を配信ソフトへ流し直す起動口 (F=全コマ / L=配信と同じ時間制約)。合言葉の門つき。

scripts/run_live_x_replay.py (21c の OBS 録画に固定) を壊さず、入力だけ video_30 に差し替えた別の入口。
門番・起動の子・差し替えの受領票・ABBA の腕は run_live_x_replay のものをそのまま使う。

plan: 何も作らず計画 (腕ごとの設定・費用の単位・区間) を表示する。状態確認はこれだけで行う。
run:  --token と環境変数の確認語が両方そろったときだけ、出力先を作って順に走らせる (誤起動防止)。
入力: 1280x720・60fps のゲーム画面そのもの。切り抜き・色補正・時刻のずらしは無し (scripts/live_x_video30_spec.py)。
区間: 先頭は取り込み位置 175.0 秒 (試合 1 の最初の手の 12.9 秒前)。終端は 284.0 秒 (ファイルの終端)。
腕:   ON は既定で既存の台と同じ live_x_direct_publish だけ。--on integrated で段 3 (暫定 X の記録・表示) も入れる。
採点: 各腕の出力は scripts/compare_live_video30_review.py でレビュー動画の値と 1 コマずつ突き合わせる。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import live_x_video30_spec as spec  # noqa: E402
from scripts import run_live_x_replay as replay  # noqa: E402

DEFAULT_ROOT = Path('D:/puyo_analyzer/live_x_video30_20261022/replay')
from src.phase_j.live_speed import SPEED_TRIAL_KEYS, X_TRIAL_KEYS  # noqa: E402

X_FLAGS = X_TRIAL_KEYS   # puyo_live.json の X の各フラグ (製品側 live_speed が単一情報源)
# all = 23y の ALL (X + コマ落ちに強い状態判定 + 予告 CNN の試作 2 つ)。OFF の腕には何も足さない。
ON_PRESETS = {'direct': ('live_x_direct_publish',), 'integrated': X_FLAGS, 'all': X_FLAGS+SPEED_TRIAL_KEYS}
# 先頭から最初の手までに最低限あけたい秒数。検証 (1 秒周期) と較正 (数コマ) と worker の初回起動を吸収する。
MIN_LEAD_SEC = 5.0
MIN_F_COVERAGE = 0.99   # F 条件: 認識するはずのレビュー区間のコマのうち、評価行が出た割合の下限 (母数つきで受領票へ)
EVALUATIONS = 'output/spool/evaluations.pickle'


def user_config(arm: str, out: Path, first_sec: float, end_sec: float, on_flags: tuple[str, ...],
                saved: Path = replay.SAVED) -> dict[str, Any]:
    """21c の puyo_live.json に、入力 (video_30)・区間・出力先・X のフラグだけを上書きする。"""
    values = json.loads((saved/'puyo_live.json').read_text(encoding='utf-8-sig'))
    values.update(source='video', video_path=str(spec.VIDEO), start_sec=spec.live_start_sec(first_sec),
                  end_sec=end_sec, port=replay.PORT, output_dir=str(out/'output'),
                  b3_model=str(replay.BUNDLE/'app/models/b3/model.joblib'),
                  b3_timing_table=str(replay.BUNDLE/'app/models/b3/timing_table.json'))
    for key in X_FLAGS:
        values[key] = arm == 'ON' and key in on_flags
    if arm == 'ON':
        values.update({key: True for key in on_flags if key not in X_FLAGS})   # X 以外の試作フラグ (preset all)
    for key in ('device_name', 'device_index', 'dshow_color_correction'):
        values.pop(key, None)   # 実機入力の項目。601→709 も video_30 には掛けない
    return values


def child_plan(condition: str, out: Path, modules: list[str], saved: Path = replay.SAVED) -> dict[str, Any]:
    """21c の差し替え計画から、入力の差し替え (切り抜き・色補正・時刻のずらし) だけを無効にする。"""
    return dict(replay.child_plan(condition, out, modules, saved), crop=None, offset_sec=0., color_correction=None)


def check_range(first_sec: float, end_sec: float) -> None:
    """区間が 2 試合とその前後を含むこと (検査の対象が台の被覆に入っていること。原則 2)。"""
    first_move, last_scoring = spec.MATCHES[0].first_move_sec, spec.MATCHES[-1].last_scoring_sec
    if not 0. <= first_sec <= first_move-MIN_LEAD_SEC:
        raise SystemExit(f'--first-sec は 0〜{first_move-MIN_LEAD_SEC:.1f} 秒 (試合 1 の最初の手 {first_move:.2f} 秒の'
                         f' {MIN_LEAD_SEC:.0f} 秒前まで)')
    if not last_scoring < end_sec <= spec.END_SEC:
        raise SystemExit(f'--end-sec は {last_scoring:.2f} 秒より後〜{spec.END_SEC:.0f} 秒 (試合 2 の最後の得点を含める)')


def plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """副作用なしで各腕の出力先・設定・差し替え対象・コマンドを組む。"""
    check_range(args.first_sec, args.end_sec)
    modules = replay.overlay_modules()
    drift = replay.bundle_drift(modules)
    if drift:
        raise SystemExit(f'配布 app と起点 {replay.BASE_COMMIT} のソースが違う: {drift[:10]}')
    runs = []
    for order, arm in enumerate(args.arms, 1):
        out = args.root/f'{args.condition}{order}_{arm}'
        runs.append(dict(name=out.name, arm=arm, out=str(out),
                         config=user_config(arm, out, args.first_sec, args.end_sec, ON_PRESETS[args.on]),
                         child=child_plan(args.condition, out, modules),
                         command=[str(replay.RUNTIME/'python.exe'), '-B', '-u', str(replay.CHILD), '--config',
                                  str(out/'puyo_live.json'), '--skip-manifest', '--dev-allow-video']))
    return runs


def plan_document(args: argparse.Namespace) -> dict[str, Any]:
    """plan の表示用。費用の単位 (stride 後の処理コマ数) と区間・突き合わせの前提を先頭に置く。"""
    return dict(condition=args.condition, on_flags=list(ON_PRESETS[args.on]), video=str(spec.VIDEO),
                video_sha256=spec.VIDEO_SHA256, workload=spec.workload(args.first_sec, args.end_sec),
                matches=[dict(index=m.index, frames=list(m.frames())) for m in spec.MATCHES],
                gate=dict(token=replay.RUN_TOKEN, environment=f'{replay.CONFIRM_ENV}={replay.CONFIRM_WORD}'),
                runs=plan(args))


def evaluated_frames(output: Path) -> set[int]:
    """配信ソフトの評価行 (spool/evaluations.pickle) に出たコマ番号。無ければ空。"""
    from scripts.measure_live_x_timing import pickle_stream
    path = output/EVALUATIONS
    return {row['frame'] for row in pickle_stream(path)} if path.exists() else set()


def output_check(out: Path, first_sec: float, end_sec: float) -> dict[str, Any]:
    """レビュー区間で、配信ソフトが認識するはずのコマに評価行が出た割合 (母数つき)。"""
    work = spec.workload(first_sec, end_sec)
    review_end = spec.REVIEW_FIRST_FRAME+spec.REVIEW_FRAME_COUNT
    expected = {f for f in spec.processed_frames(work['first_frame'], work['end_frame'], work['stride'])
                if spec.REVIEW_FIRST_FRAME <= f < review_end}
    seen = evaluated_frames(out) & expected
    return dict(workload=work, expected_review_frames=len(expected), evaluated_review_frames=len(seen),
                coverage=len(seen)/len(expected) if expected else None)


def accept(receipt: dict[str, Any], condition: str) -> str | None:
    """受領票の合否。不合格なら理由を返す。0 件を成功と読み替えない (母数も見る)。"""
    if receipt['exit_code'] or receipt['patches']['video_source'] < replay.MIN_SOURCE_PATCHED_PROCESSES:
        return '終了コードまたは差し替え件数'
    if not receipt['recognition_matches_21c']:
        return '認識設定が 21c と違う'
    if not receipt['expected_review_frames'] or not receipt['evaluated_review_frames']:
        return '評価行が出ていない (レビュー区間の母数または出た件数が 0)'
    if condition == 'F' and receipt['coverage'] < MIN_F_COVERAGE:
        return f"F 条件の評価行の割合 {receipt['coverage']:.3f} が下限 {MIN_F_COVERAGE} 未満"
    return None


def execute(args: argparse.Namespace) -> None:
    replay.authorize(args.token, os.environ)
    if (reason := replay.blocked()) is not None:
        raise SystemExit(f'起動しない: {reason}')
    if replay.digest(spec.VIDEO) != spec.VIDEO_SHA256:
        raise SystemExit('video_30 の SHA256 が固定値と違う')
    runs = plan(args)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/f'plan_{args.condition}.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2),
                                                        encoding='utf-8')
    for run in runs:
        receipt = dict(replay.run_one(run), **output_check(Path(run['out']), args.first_sec, args.end_sec))
        (Path(run['out'])/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        if (failure := accept(receipt, args.condition)) is not None:
            raise SystemExit(f"{run['name']} が不合格 ({failure}): 原票を残して止める")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--condition', choices=tuple(replay.CONDITIONS), default='F')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arms', type=lambda s: tuple(s.split(',')), default=replay.ARMS)
    parser.add_argument('--on', choices=tuple(ON_PRESETS), default='direct',
                        help='ON の腕で有効にするフラグ。direct=既存の台と同じ、integrated=段 3 も、all=23y の ALL (X + FDR + 予告 CNN の試作)')
    parser.add_argument('--first-sec', type=float, default=spec.FIRST_PROCESSED_SEC,
                        help='認識を始める映像秒 (既定はレビューの取り込みと同じ 175.0)')
    parser.add_argument('--end-sec', type=float, default=spec.END_SEC)
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    if any(arm not in ('OFF', 'ON') for arm in args.arms):
        parser.error('--arms は OFF/ON のカンマ区切り')
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    if args.command == 'plan':
        print(json.dumps(plan_document(args), ensure_ascii=False, indent=2))
        return
    execute(args)


if __name__ == '__main__':
    main()
