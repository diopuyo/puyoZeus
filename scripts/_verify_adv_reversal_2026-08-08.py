"""大連鎖中の有利不利逆転が解消したかを数値で検証する (2026-08-08).

user 指示「9 の反転の部分は完全にクリアしてから動画出したい」への検証器。

## 何を判定するか
1P が t=53.97 に 9 連鎖、 2P が t=56.47 に 7 連鎖を撃っている
(診断 _diag_2p_osc_transitions_afterfix より)。 大連鎖を撃った側が
**その直後に不利と表示されていないか** を、 生成ログの adv 値の時系列で見る。

visualize_advantage_overlay は進捗ログに
`... 1500 frames (t=50.0s adv=-24)` の形式で adv を出す (adv>0 = 1P有利)。
これを拾って発火前後の推移を比較する。

判定基準 (シーンに合わせて緩めない):
- **1P の 9 連鎖発火後、 adv が発火前より上向く** (1P 有利方向へ動く) こと
- 逆に下がっている場合は逆転が残っているとみなす

使い方:
    python scripts/_verify_adv_reversal_2026-08-08.py \\
        logs/demo_adv_A_2026-08-08.log logs/demo_adv_A_efire_2026-08-08.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 1P / 2P の大連鎖の発火時刻 (診断で確定済み)
P1_FIRE_SEC: float = 53.97
P2_FIRE_SEC: float = 56.47
# 発火前後で比較する窓 (秒)。 発火直前の値と、 発火から数秒後の値を比べる。
PRE_WINDOW_SEC: float = 3.0
POST_WINDOW_SEC: float = 4.0

_LINE_RE = re.compile(r"\(t=([0-9.]+)s adv=(-?\d+)\)")


def _load_series(log_path: Path) -> list[tuple[float, int]]:
    """進捗ログから (t_sec, adv) の時系列を読む。"""
    out: list[tuple[float, int]] = []
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LINE_RE.search(line)
        if m:
            out.append((float(m.group(1)), int(m.group(2))))
    return out


def _value_at(series: list[tuple[float, int]], t: float) -> int | None:
    """指定時刻に最も近いサンプルの adv を返す。"""
    if not series:
        return None
    best = min(series, key=lambda kv: abs(kv[0] - t))
    return best[1]


def _report(name: str, series: list[tuple[float, int]]) -> bool:
    """1 本分の判定を出力し、 逆転が解消していれば True を返す。"""
    if not series:
        print(f"[{name}] adv の時系列が取れない (ログ形式を確認)")
        return False
    pre = _value_at(series, P1_FIRE_SEC - PRE_WINDOW_SEC)
    post = _value_at(series, P1_FIRE_SEC + POST_WINDOW_SEC)
    pre2 = _value_at(series, P2_FIRE_SEC - PRE_WINDOW_SEC)
    post2 = _value_at(series, P2_FIRE_SEC + POST_WINDOW_SEC)
    print(f"[{name}] サンプル数 {len(series)}")
    print(f"  1P 9連鎖 (t={P1_FIRE_SEC}): 発火前 adv={pre} -> 発火後 adv={post}")
    print(f"  2P 7連鎖 (t={P2_FIRE_SEC}): 発火前 adv={pre2} -> 発火後 adv={post2}")
    ok = pre is not None and post is not None and post > pre
    verdict = "解消 (1P発火後に1P有利方向へ動いた)" if ok else "**逆転が残っている**"
    print(f"  判定: {verdict}")
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    all_ok = True
    for arg in sys.argv[1:]:
        p = Path(arg)
        ok = _report(p.stem, _load_series(p))
        all_ok = all_ok and ok
        print()
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
