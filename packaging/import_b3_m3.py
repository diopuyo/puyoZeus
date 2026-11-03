"""M1と同じ列・観測着地の差分を、既存v4.9f会計を保持して取り込む。"""
from pathlib import Path
import ast
import subprocess

BRANCH = 'claude/b3-ledger-inflight-20261012'


def source(name: str) -> str:
    return subprocess.check_output(['git', 'show', f'{BRANCH}:{name}']).decode('utf-8')


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main() -> None:
    files = ['src/b3_column_spec.py', 'src/b3_features.py']
    patch = subprocess.check_output(['git', 'diff', 'bfdb67c0', BRANCH, '--', *files])
    if 'def _inflight_cols(' not in Path(files[0]).read_text(encoding='utf-8'):
        subprocess.run(['git', 'apply'], input=patch, check=True)
    for name in [*files, 'src/b3_state.py']:
        path = Path(name)
        text = path.read_text(encoding='utf-8').replace(
            'from src import production_config', 'from src import b3_trial_config as production_config')
        text = text.replace('if production_config.B3_LEDGER_MODE_DEFAULT:',
                            'if production_config.b3_ledger_mode_enabled():')
        path.write_text(text, encoding='utf-8')
    Path('src/b3_landing_observation.py').write_text(source('src/b3_landing_observation.py'), encoding='utf-8')
    path = Path('src/b3_score_ledger.py')
    text = path.read_text(encoding='utf-8')
    text = text.replace('from src.b3_profile import timed',
        'from src.b3_profile import timed\nfrom src import b3_trial_config as production_config')
    text = text.replace('BOARD_CELLS: int = 78', 'SMALL_BOARD_INCREASE_MAX: int = 2\nBOARD_CELLS: int = 78')
    text = replace_once(text, '    def __init__(\n',
        '    enable_inflight: bool = False\n    observed_landing_rule: bool = False\n\n    def __init__(\n')
    text = text.replace('        all_clear_native: object = None,',
        '        all_clear_native: object = None,\n        enable_inflight: Optional[bool] = None,\n'
        '        observed_landing_rule: Optional[bool] = None,')
    text = replace_once(text, '        self.mode = mode',
        '        if (production_config.B3_LEDGER_INFLIGHT_DEFAULT if enable_inflight is None else enable_inflight):\n'
        '            self.enable_inflight = True\n'
        '        if (production_config.B3_LEDGER_OBSERVED_RULE_DEFAULT if observed_landing_rule is None else observed_landing_rule):\n'
        '            self.observed_landing_rule = True\n'
        '        self.mode = mode')
    text = replace_once(text, '        self.all_clear = [AllClearCredit(), AllClearCredit()]',
        '        self.all_clear = [AllClearCredit(), AllClearCredit()]\n'
        '        if self.observed_landing_rule:\n'
        '            from src.b3_landing_observation import LandingObservation\n'
        '            self.landing_observations = [LandingObservation() for _ in range(SIDE_COUNT)]')
    research = source(path.as_posix())
    tree, lines = ast.parse(research), research.splitlines(keepends=True)
    current_tree, current_lines = ast.parse(text), text.splitlines(keepends=True)
    names = {'_observe_landing', '_on_observed_landing', '_drain_observed_rule', 'inflight'}
    # 関数名は枝の実装から取得し、対象外の全消し・相殺順の処理を保持する。
    names.add('_observe_ojama_board')
    functions = {n.name: ''.join(lines[n.lineno-1:n.end_lineno])+'\n'
                 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for node in sorted((n for n in ast.walk(current_tree) if isinstance(n, ast.FunctionDef)
                        and (n.name in names or 'cur = so.ojama_cells' in ''.join(current_lines[n.lineno-1:n.end_lineno]))),
                       key=lambda n: n.lineno, reverse=True):
        current_lines[node.lineno-1:node.end_lineno] = [functions[node.name]]
    text = ''.join(current_lines)
    text = text.replace('    def _flag_interrupt_landing(', functions['_drain_observed_rule']+'\n    def _flag_interrupt_landing(')
    text = text.replace('    def _side_ledger(', functions['inflight']+'\n    def _side_ledger(')
    text = replace_once(text, '        return snapshot_to_dict(snap)',
        '        out = snapshot_to_dict(snap)\n'
        '        if self.core.enable_inflight:\n'
        '            out.update(self.core.inflight(obs.t_sec))\n'
        '            if out["history_incomplete"] or out["history_waiting"]:\n'
        '                for side in ("p1", "p2"):\n'
        '                    out[f"inflight_{side}"] = out[f"inflight_send_{side}"] = float("nan")\n'
        '        return out')
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
