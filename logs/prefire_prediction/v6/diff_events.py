"""補正外の通常評価差をイベントの入力・出力単位で照合する。"""
import json
from pathlib import Path


def value_at(path: Path, stamp: float) -> dict:
    """一致する時刻の採用値を読む。"""
    for line in path.read_text().splitlines():
        for value in json.loads(line)['values']:
            if abs(value['t_sec']-stamp) < 1e-6:
                return value
    raise ValueError(stamp)


def differences(left: object, right: object, prefix: str = '') -> list:
    """計算所要時間を含めて差を列挙し、出力を選別しない。"""
    if isinstance(left, dict) and isinstance(right, dict):
        return [row for key in left.keys() | right.keys()
                for row in differences(left.get(key), right.get(key), prefix+'/'+key)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [row for index,(a,b) in enumerate(zip(left,right))
                for row in differences(a,b,prefix+'/'+str(index))]
    return [] if left == right else [(prefix,left,right)]


def main() -> None:
    """最初の確率差を生んだ通常イベントを調べる。"""
    out = Path('logs/prefire_prediction/v6')
    base = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2/replay/prod')
    parity = json.loads((out/'FULL_PARITY.json').read_text())
    result = {}
    for part in ('s1','s5'):
        stamp = parity[part]['display_p1']['examples'][0]['t']
        result[part] = differences(value_at(base/part/'events.jsonl', stamp),
                                  value_at(out/'replay'/part/'events.jsonl', stamp))
    (out/'EVENT_DIFFERENCES.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
