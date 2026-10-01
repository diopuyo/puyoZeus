"""表示時刻への整列を是正し、固定済み係数が変わらないことを採点前に確認する。"""
import hashlib
import json

from scripts.prefire_v6_train import samples, fit, OUT, EVALSET


def main() -> None:
    """係数が変わる場合は保存せず中断する。旧原票は消さない。"""
    assert not (OUT/'RESULT.json').exists(), '採点前のデータ整列確認に限定する'
    path = OUT/'strength.json'
    original = json.loads(path.read_text())
    table, labels, weights, files = samples()
    corrected = fit(table, labels, weights)
    assert corrected['coefficients'] == original['coefficients']
    assert corrected['nonzero_delta'] == original['nonzero_delta'] == 0
    files.extend([EVALSET/'labels.json', OUT/'latency.json'])
    corrected['sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    temporary = OUT/'strength_aligned.json'
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(corrected, stream, ensure_ascii=False, indent=2)
    archive = OUT/'strength_before_alignment.json'
    assert not archive.exists()
    path.rename(archive)
    temporary.rename(path)
    receipt = dict(before_rows=original['rows'], after_rows=corrected['rows'],
                   coefficients_identical=True, coefficients=corrected['coefficients'],
                   nonzero_delta=corrected['nonzero_delta'], scoring_not_started=True)
    with (OUT/'TRAIN_ALIGNMENT.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(receipt)


if __name__ == '__main__':
    main()
