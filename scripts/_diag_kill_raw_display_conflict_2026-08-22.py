"""kill_override が生モデル (adv_raw/p1_raw) と逆方向に最終表示 (adv_ema/p1) を
決めているフレームを、30先2セット動画の全8区間 dump から数え上げる (2026-08-22)。

コードは変更しない (dump 済み npz を読むだけの計装スクリプト)。

判定基準:
  - kill_override 関数 (scripts/visualize_advantage_overlay.py:260) をそのまま
    再現し、pending_p1/p2, room1/room2 から g (上書き強度, 0=無効, 1=完全上書き)
    を計算する。
  - g >= 0.9 (ほぼ完全に安全弁が支配) かつ adv_raw と adv_ema の符号が逆
    (raw が有利と言う側と、最終表示が有利とする側が逆転) の行を
    「生モデル対最終表示 符号衝突」としてフラグする。
  - 連続するフラグ行を1エピソードとしてまとめ、持続時間を測る。

[2026-08-22 追加] --dir で読み込み先ディレクトリを差し替え可能にした
(既定は従来通り data/verify/zenchi_render_2026-08-21、旧本番dump=修正前の
基準値 112エピソードを再現する経路は不変)。修正① 適用後の dump
(例: data/verify/zenchi_render_kill_override_fix_2026-08-22) を指定すれば
同一ロジックで「修正後は何件に減ったか」を測れる。

[2026-08-23 根治①対応] 上記コメントの「g は生の pending/room から再計算する」
という設計そのものが盲点だった。生の pending/room は
enable_kill_override_chain_completion の有無に関わらず常に是正前の値であり、
実際に kill_override へ渡された値 (kpending_p1/p2・kroom1/kroom2、
是正が効いていなければ生値と同一) は dump に記録されていなかった。
是正が「一時的な踊り場」を挿入すると、生値ベースの g は踊り場の間ずっと
>=0.9 のままなのに実際の adv_ema は踊り場内で符号衝突しない
(=衝突は踊り場の入口と出口の遷移フレームだけで起きる) ため、
1つの連続現象が2つの「新規エピソード」に分裂して数えられていた
(2026-08-22夜32件の内実)。

本改修: dump に kpending_p1/p2・kroom1/kroom2 があればそちらで g を
再計算する (=実際に安全弁へ渡された値と完全一致、盲点を根治)。
無ければ (旧 dump) 従来通り pending_p1/p2・room1/room2 にフォールバックし、
**修正前の基準値 (112エピソード) が既定ディレクトリで再現すること**を
テストで確認済み (tests/test_diag_kill_raw_display_conflict.py 相当の
自己検証、または本スクリプトの --dir 未指定実行)。
--compare-raw を付けると「生値ベースの g (旧ロジック)」と「是正後の g
(新ロジック)」を両方計算し、両者の差分件数 (=盲点でカウントされていた分)
を表示する。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

DEFAULT_DUMP_DIR = PROJECT_ROOT / "data/verify/zenchi_render_2026-08-21"

KILL_ROOM_FLOOR = 4
KILL_RATIO_MIN = 0.6
KILL_RATIO_FULL = 1.5
KILL_MIN_PENDING = 40


def kill_g(inc1: float, inc2: float, room1: float, room2: float) -> float:
    l1 = inc1 / max(KILL_ROOM_FLOOR, room1) if inc1 >= KILL_MIN_PENDING else 0.0
    l2 = inc2 / max(KILL_ROOM_FLOOR, room2) if inc2 >= KILL_MIN_PENDING else 0.0
    lead = l1 - l2
    mag = abs(lead)
    if mag < KILL_RATIO_MIN:
        return 0.0
    return min(1.0, (mag - KILL_RATIO_MIN) / (KILL_RATIO_FULL - KILL_RATIO_MIN))


def _pending_room_arrays(
    d: "np.lib.npyio.NpzFile", use_corrected: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """pending/room 配列を取り出す (2026-08-23 根治①対応)。

    use_corrected=True かつ dump に kpending_p1 等があれば「kill_override へ
    実際に渡された是正後の値」を返す (盲点の根治本体)。無ければ (旧dump、
    または use_corrected=False 明示指定) 生値 pending_p1/p2・room1/room2に
    フォールバックする。戻り値末尾の bool は「実際に補正値を使えたか」。
    """
    has_k = use_corrected and "kpending_p1" in d.files
    if has_k:
        return (
            d["kpending_p1"].astype(float), d["kpending_p2"].astype(float),
            d["kroom1"].astype(float), d["kroom2"].astype(float), True,
        )
    return (
        d["pending_p1"].astype(float), d["pending_p2"].astype(float),
        d["room1"].astype(float), d["room2"].astype(float), False,
    )


def _episodes_for_file(
    fname: str, t: np.ndarray, adv_raw: np.ndarray, adv_ema: np.ndarray,
    pend1: np.ndarray, pend2: np.ndarray, room1: np.ndarray, room2: np.ndarray,
) -> list[dict]:
    """1区間分の符号衝突エピソードを検出する (kill_g 判定+隣接行の連結)。"""
    n = len(t)
    g = np.array([kill_g(pend1[i], pend2[i], room1[i], room2[i]) for i in range(n)])
    # raw と ema の符号衝突 (どちらも無視できない大きさのときのみ、微小値のノイズ除外)
    sign_conflict = (
        (g >= 0.9)
        & (np.abs(adv_raw) >= 5.0)
        & (np.abs(adv_ema) >= 50.0)
        & (np.sign(adv_raw) != np.sign(adv_ema))
    )
    idx = np.where(sign_conflict)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    episodes = []
    for grp in groups:
        t0, t1 = t[grp[0]], t[grp[-1]]
        episodes.append(dict(
            file=fname, t0=float(t0), t1=float(t1),
            dur=float(t1 - t0), n_rows=len(grp),
            adv_raw_range=(float(adv_raw[grp].min()), float(adv_raw[grp].max())),
            adv_ema_at_start=float(adv_ema[grp[0]]),
            pend1_max=float(pend1[grp].max()), pend2_max=float(pend2[grp].max()),
        ))
    return episodes


def _print_summary(label: str, all_episodes: list[dict], total_rows: int) -> None:
    """エピソード集合1本分のサマリを表示する。"""
    print(f"\n=== {label} ===")
    print(f"総フレーム数: {total_rows}")
    print(f"符号衝突エピソード総数: {len(all_episodes)}")
    durations = [e["dur"] for e in all_episodes]
    if not durations:
        return
    durations_sorted = sorted(durations)
    mid = durations_sorted[len(durations_sorted) // 2]
    print(f"持続時間: 合計={sum(durations):.1f}s 中央値={mid:.2f}s "
          f"最大={max(durations):.2f}s 最小={min(durations):.2f}s")


def _scan_all_files(
    files: list[str], compare_raw: bool,
) -> tuple[list[dict], list[dict], int, bool]:
    """全区間ファイルを走査し (是正後エピソード, 生値エピソード, 総行数,
    是正フィールドが1件でも見つかったか) を返す。compare_raw=False なら
    生値エピソードは常に空リスト (計算自体を省く)。"""
    total_rows = 0
    corrected_episodes: list[dict] = []
    raw_episodes: list[dict] = []
    any_corrected_field_found = False
    for f in files:
        d = np.load(f, allow_pickle=True)
        t, adv_raw, adv_ema = d["t_sec"], d["adv_raw"], d["adv_ema"]
        total_rows += len(t)
        pend1, pend2, room1, room2, used_k = _pending_room_arrays(d, True)
        any_corrected_field_found = any_corrected_field_found or used_k
        eps = _episodes_for_file(Path(f).name, t, adv_raw, adv_ema, pend1, pend2, room1, room2)
        corrected_episodes.extend(eps)
        if compare_raw:
            rp1, rp2, rr1, rr2, _ = _pending_room_arrays(d, False)
            raw_episodes.extend(
                _episodes_for_file(Path(f).name, t, adv_raw, adv_ema, rp1, rp2, rr1, rr2))
        print(f"{Path(f).name}: rows={len(t)}, episodes={len(eps)} "
              f"(是正値使用={used_k})")
    return corrected_episodes, raw_episodes, total_rows, any_corrected_field_found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", type=Path, default=DEFAULT_DUMP_DIR,
        help="seg*.npz を読み込むディレクトリ (既定=修正前の本番dump、"
             "data/verify/zenchi_render_2026-08-21)",
    )
    ap.add_argument(
        "--compare-raw", action="store_true",
        help="是正後(kpending/kroom)ベースの結果に加え、旧ロジック(生値のみ)"
             "の結果も計算し、両者の差分件数(=盲点で二重計上されていた分)"
             "を表示する",
    )
    a = ap.parse_args()
    files = sorted(glob.glob(str(a.dir / "seg*.npz")))
    if not files:
        raise SystemExit(f"seg*.npz が見つかりません: {a.dir}")

    corrected_episodes, raw_episodes, total_rows, any_k = _scan_all_files(
        files, a.compare_raw)

    label = "是正後の値 (kpending/kroom)" if any_k else \
        "生値 (旧dump、kpending/kroom 無し=後方互換フォールバック)"
    _print_summary(label, corrected_episodes, total_rows)
    if a.compare_raw:
        _print_summary("生値のみ (旧ロジック再現、比較用)", raw_episodes, total_rows)
        print(f"\n差分 (盲点で二重計上されていた分の目安): "
              f"{len(raw_episodes)} - {len(corrected_episodes)} = "
              f"{len(raw_episodes) - len(corrected_episodes)} 件")
    print("\n--- エピソード一覧 (先頭30件、是正後ベース) ---")
    for e in corrected_episodes[:30]:
        print(e)


if __name__ == "__main__":
    main()
