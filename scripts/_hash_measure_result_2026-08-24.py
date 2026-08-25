"""物差し結果 JSON の「再現性のあるハッシュ」を出す (2026-08-24)。

## なぜ必要か

`scripts/measure_stable_cell_acc.py` の結果 JSON をそのまま md5 で比較すると、
**同一コード・同一入力でもハッシュが変わることがある**。

原因は 2 つ。

1. `disagreement_cells` — 不一致セルの**診断用サンプル配列**。
   `--workers 4` の並列実行で、どの動画のワーカーが先に終わるかによって
   積まれる順序が変わる。集計値には一切影響しない。
2. `meta.output` — 出力パス。条件ごとに変えるので必ず違う。

実際に 2026-08-24 の Gate 0 検証で次が起きた。

| 比較 | 生 md5 | 全指標 |
|---|---|---|
| OFF基準 vs OFF最終 (逐次実行) | 一致 | 一致 |
| OFF基準 vs F1 親ガードのみ | 一致 | 一致 |
| 式読取2要素 vs F2 完全構成 | **不一致** | 一致 |
| OFF基準 vs Q-01修正後のOFF | **不一致** | 一致 |

**md5 が一致した 2 件はワーカーの完了順がたまたま揃っただけ**で、
再現性の保証になっていない。生 md5 を bit-identical の判定に使うと、
「悪化した」と誤読する事故が起きる。

## 何を出すか

上記 2 つを除外し、キーを正規化 (ソート) した JSON の sha256 を出す。
これなら同一コード・同一入力なら並列度や実行順に関係なく一致する。

使い方:
    python scripts/_hash_measure_result_2026-08-24.py <result.json> [...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# 結果の同一性判定から外すキー。値そのものではなく「実行ごとに変わるが
# 集計値に影響しないもの」だけを列挙する。
VOLATILE_TOP_KEYS: tuple[str, ...] = ("disagreement_cells",)
VOLATILE_META_KEYS: tuple[str, ...] = ("output",)


def canonical_hash(path: Path) -> str:
    """揮発キーを除いた正規化 JSON の sha256 を返す。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in VOLATILE_TOP_KEYS:
        data.pop(key, None)
    meta = data.get("meta")
    if isinstance(meta, dict):
        for key in VOLATILE_META_KEYS:
            meta.pop(key, None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"{'(見つかりません)':>64}  {arg}")
            continue
        print(f"{canonical_hash(path)}  {arg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
