"""原whole終了検査後のgoalだけ正常Bへ限定。Aのreset/公開成功を捏造しない。"""
from pathlib import Path
from typing import Any
import normal_settings as K

ROOT = Path(__file__).resolve().parent


def install(stack: Any, main: Any, replace: Any) -> None:
    common, target = main.__globals__['K'], main.__globals__['Q'].FINAL
    def evaluate(goals: Any, rows: Any, legal: Any, output: Path, state: dict) -> dict:
        require = common.require
        driver, session = state['whole_session_driver'], state['belief_m1_session']
        require(output == state['output'] and driver.session is session and driver.closed and driver.stopped,
                'normal_final_owner')
        require(common.FRAMES == driver.bridge.frames == tuple(range(K.FIRST, K.LAST + K.STRIDE, K.STRIDE))
                and driver.last == K.LAST and session.owner.closed, 'normal_final_bounds')
        verify = session.saved_verifier
        require(Path(verify.__code__.co_filename).resolve() == ROOT / 'normal_saved.py', 'normal_final_verifier')
        checked = verify(output)
        common.write(output / 'NORMAL_M1_WINDOW_REVIEW.json', checked)
        return dict(scope='normal_m1_window', normal_m1_window_verified=True,
            current_event_observed=False, outer_publication_observed=False,
            input_bounds=[K.FIRST, K.LAST], physical_certified=False,
            production_permission=False, quality_gate_clear=False)
    replace(stack, target, 'evaluate', evaluate)
