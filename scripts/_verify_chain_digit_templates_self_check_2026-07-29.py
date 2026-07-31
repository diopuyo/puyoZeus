"""digit_0〜digit_9 テンプレの自己分類チェック (使い捨て・軽量)。

各テンプレ画像自身を ChainCountOcr._classify() に通し、最良一致クラスが
自分自身のラベルと一致するか (クラス間の取り違えがないか) を確認する。
2026-07-29 に追加した digit_5〜digit_9・digit_0 が既存 digit_1〜4 と
混同されないかの最低限の健全性チェック。

nice -n 19 で実行。画像読み込みのみで動画I/Oなし、瞬時に終わる。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain_count_ocr import ChainCountOcr  # noqa: E402


def main() -> None:
    ocr = ChainCountOcr.load_default()
    print(f"登録テンプレ数: {len(ocr._templates_gray)}")  # noqa: SLF001 (診断用)
    all_ok = True
    for label, tpl in sorted(ocr._templates_gray.items()):  # noqa: SLF001
        result = ocr._classify(tpl)  # noqa: SLF001
        # "0" は単体では None を返す仕様 (2桁の一の位専用) のため別判定にする。
        expected = None if label == 0 else label
        ok = result.chain_count == expected
        all_ok &= ok
        print(f"digit_{label}: 自己分類={result.chain_count} (期待値={expected}) "
              f"confidence={result.confidence:.3f} {'OK' if ok else 'NG'}")
    print("\n全件OK" if all_ok else "\n一部NG (要確認)")


if __name__ == "__main__":
    main()
