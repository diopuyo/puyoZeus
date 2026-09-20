"""子の終了票欠測も親の失敗結果として保存するG3限定接続。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any
from scripts import g3_first_pass as F

ENDED = 'OBSERVATION_ENDED_NOT_G3_PASS'
FAILED = 'FAILED'


def destination(root: Path, label: str) -> Path:
    """保存先をDの単一run名へ限定し、既存原票の出力先と分離する。"""
    root = root.resolve()
    if not root.is_relative_to(F.VERIFY.resolve()) or not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('g3_outcome_destination')
    return root / (label + '_PARENT_RESULT.json')


def finish(root: Path, label: str, waited: Any, *, wait_error: str | None = None) -> dict:
    """不明waitを成功へ補完せず、原票監査には既存集計器を使う。"""
    target = destination(root, label)
    result: dict[str, Any] = dict(status=FAILED, quality_gate_clear=False, wait_verified=False,
        wait=waited, wait_error=wait_error, output=str(root / label), parent_exit_code=1)
    try:
        F.validate_wait(waited)
        result.update(F.arm_artifacts(root, label, root / label, waited))
        good = (wait_error is None and not F.shared_failure(result)
                and not result['artifact_errors'] and waited['child_exit_code'] == 0
                and result['entry'].get('status') == ENDED and not result['entry'].get('error'))
        if good:
            result.update(status=ENDED, parent_exit_code=0)
    except Exception as error:
        result['audit_error'] = repr(error)
    # 排他保存失敗は呼出元へ伝播。保存できなかったことを成功にしない。
    F.save(target, result)
    return result


def run(root: Path, label: str, command: list[str], runner: Path, *, api: Any = None) -> dict:
    """既存supervisorの終了から親結果へ接続する。失敗時の再送は行わない。"""
    destination(root, label)
    F.save(root / (label + '_REQUEST.json'), dict(command=command, automatic_retry=False,
                                                stop_boundary='G3_ONLY_NO_G4'))
    waited, failure = None, None
    try:
        with F.log_output(root / (label + '.log')):
            waited = (api or F.supervisor()).run_pair(command, str(F.GUARD), str(runner),
                                                    root / (label + '.resources.jsonl'))
        F.save(root / (label + '_ACTUAL_WAIT.json'), waited)
    except Exception as error:
        failure = repr(error)
    return finish(root, label, waited, wait_error=failure)


def main() -> int:
    """実wait保存後のrunnerが呼ぶ終了処理。GPUや再走は起動しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    destination(args.root, args.label)
    try:
        waited, failure = F.read(args.root / (args.label + '_ACTUAL_WAIT.json')), None
    except Exception as error:
        waited, failure = None, repr(error)
    result = finish(args.root, args.label, waited, wait_error=failure)
    print(result['status'])
    return result['parent_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
