"""外部正解の物差し: 盤面単位の人手ラベル用シートを生成する (2026-07-31)。

## なぜ盤面単位か

看板指標「セル正解率95.4%」は raw_cnn / raw_hsv / confirmed の3者多数決を
正解とする**自己無矛盾性チェック**で、CNN と HSV が同じ誤りに合意すると
原理的に検出できない。真の精度には**外部の正解 (人手ラベル)** が必要。

しかし 99.9% と 99.99% を区別するにはセル単位で1万件以上のラベルが必要で、
人手では現実的でない。

**そこで単位を盤面にする。1盤面 = 72セル。**人がぷよ盤面を1枚見て
「全部合っている」と確認するのは数秒で済むので、**100盤面 = 7,200セル**の
ラベルが現実的な工数で得られる。大半の盤面は全て正しいはずなので多くは
1クリックで終わる。

物理則 (浮きぷよ) で盲点を狙い撃つ案は**当てが外れた**: 全フレーム基準データ
での浮きぷよは 0.008% しかなく、確定盤面は物理的にはほぼ整合していた。
= 誤りがあるなら**物理的にあり得る誤った色**なので物理則では捕まえられない。

## 出力

各サンプルについて 1 枚の PNG:
  - 左: 実ゲーム画面の盤面領域を**そのまま大きく切り出したもの** (判断の一次資料)
  - 右: 認識結果を色付きグリッドで描画したもの (比較対象)
**文字は焼き込まない** (スマホで読めない失敗を過去にしているため)。
行・列の番号だけは対応付けに必要なので枠外に小さく描く。

併せて `labels.tsv` の雛形を出す (user が誤セルだけ記入する)。

## サンプリング

`--strategy random` (既定): 層別ランダム。動画・side・ぷよ数帯で均す。
`--strategy floating`: 浮きぷよを含む行のみ (物理違反の確定例、少数)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._build_board_label_sheets_2026-07-31 \
        --n 40 --out-dir data/verify/board_labels_2026-07-31
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

BOARD_ROWS: int = 13
BOARD_COLS: int = 6
HIDDEN_ROWS: int = 1
COLOR_EMPTY: int = 0
COLOR_OJAMA: int = 9
COLOR_UNKNOWN: int = 10
VALID_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})

# 認識結果の描画色 (BGR)。ぷよの実際の見た目に近い色にする。
COLOR_BGR: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),        # 空 = 暗いグレー
    1: (60, 60, 220),       # 赤
    2: (220, 120, 60),      # 青
    3: (80, 200, 80),       # 緑
    4: (60, 210, 230),      # 黄
    5: (200, 80, 200),      # 紫
    9: (170, 170, 170),     # おじゃま = 明るいグレー
    10: (0, 0, 0),          # UNKNOWN = 黒
}
# ============================
# サンプリングの試合文脈フィルタ (2026-07-31 v3)
# ============================
# 初回40枚のNG分析で、誤判定サンプルが全て「決着・遷移・エフェクト」局面に
# 集中していたことが判明した (13=連鎖煙直後+試合末尾4.4s / 18=連鎖後の
# 反映凍結 / 24=試合終了の崩壊 / 30=満杯の×印)。
# 「密な盤面を優先する」サンプリングがクライマックスの遷移フレームを
# 過剰に拾っていたのが原因。
#
# 対処: 測定対象外のフレームをサンプリング段階で除外する。
#   1. 試合区間外 (勝利数パネル基準) → 24型の崩壊シーンを排除
#   2. 試合末尾 MATCH_END_EXCLUDE_SEC 秒以内 → 13型の決着局面を排除
#   3. 連鎖検知 (chain_trigger_sec) から CHAIN_COOLDOWN_SEC 秒以内
#      → 13/007型のエフェクト煙を排除
# **本物の認識バグ (×印誤認・背景誤検出・反映凍結) は除外しない** —
# それを見つけるのが物差しの目的。
MATCH_END_EXCLUDE_SEC: float = 8.0
CHAIN_COOLDOWN_SEC: float = 3.0
WINNERS_PANEL_DIR: Path = Path("data/verify/winners_panel_diff_2026-07-26")

# 出力する 1 セルの描画サイズ [px]
CELL_PX: int = 56
# 盤面切り出しの拡大率 (スマホで見て判断できる大きさにする)
CROP_SCALE: float = 2.0


_MATCH_WINDOWS_CACHE: dict[str, list[tuple[float, float]]] = {}


def _stable_match_windows(video_id: str) -> list[tuple[float, float]]:
    """勝利数パネルから「安定サンプリングして良い区間」を返す (キャッシュ付き)。

    各試合の [start_sec, end_sec - MATCH_END_EXCLUDE_SEC) を許可区間とする。
    パネルデータが無い動画は空リスト = その動画からはサンプルしない
    (試合区間が分からないと24型の崩壊シーンを排除できないため、安全側に倒す)。
    """
    if video_id in _MATCH_WINDOWS_CACHE:
        return _MATCH_WINDOWS_CACHE[video_id]
    path = WINNERS_PANEL_DIR / f"{video_id}.json"
    windows: list[tuple[float, float]] = []
    if path.exists():
        try:
            games = json.loads(path.read_text(encoding="utf-8")).get("games", [])
            for g in games:
                s0 = float(g["start_sec"])
                e0 = float(g["end_sec"]) - MATCH_END_EXCLUDE_SEC
                if e0 > s0:
                    windows.append((s0, e0))
        except Exception:
            windows = []
    _MATCH_WINDOWS_CACHE[video_id] = windows
    return windows


def _in_stable_window(video_id: str, t_sec: float) -> bool:
    """t_sec が安定サンプリング許可区間内か。"""
    return any(s0 <= t_sec < e0 for s0, e0 in _stable_match_windows(video_id))


def _render_grid(grid: np.ndarray) -> np.ndarray:
    """認識結果グリッドを色付き画像として描画する (隠し段も含む)。"""
    h = (BOARD_ROWS - HIDDEN_ROWS) * CELL_PX
    w = BOARD_COLS * CELL_PX
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(grid[r, c])
            col = COLOR_BGR.get(v, (0, 0, 255))
            y1 = (r - HIDDEN_ROWS) * CELL_PX
            x1 = c * CELL_PX
            cv2.rectangle(
                img, (x1 + 2, y1 + 2), (x1 + CELL_PX - 2, y1 + CELL_PX - 2),
                col, -1,
            )
            # 空セルは枠だけにして「何も無い」ことを見て分かるようにする
            if v == COLOR_EMPTY:
                cv2.rectangle(
                    img, (x1 + 2, y1 + 2),
                    (x1 + CELL_PX - 2, y1 + CELL_PX - 2), (90, 90, 90), 1,
                )
    # 背景誤検出の稜線ハイライト (2026-07-31)。各列で最も高い位置にある
    # ぷよ = 盤面輪郭。背景誤検出はこの輪郭より上に1-2セルはみ出す傾向がある
    # (004 で確認: 実盤面の最上ぷよの1段上に偽の赤)。そのセルを黄枠で囲み、
    # 「ここが偽ぷよの出やすい場所」として目視確認を誘導する。
    for c in range(BOARD_COLS):
        top_r = None
        for r in range(HIDDEN_ROWS, BOARD_ROWS):
            if int(grid[r, c]) != COLOR_EMPTY:
                top_r = r
                break
        if top_r is not None:
            # 最上ぷよ自身を黄枠で囲む (これが本当にそこにあるか確認する対象)
            y1 = (top_r - HIDDEN_ROWS) * CELL_PX
            x1 = c * CELL_PX
            cv2.rectangle(
                img, (x1 + 1, y1 + 1),
                (x1 + CELL_PX - 1, y1 + CELL_PX - 1), (0, 220, 220), 2,
            )
    return img


def _crop_board(frame: np.ndarray, side: str) -> np.ndarray:
    """実ゲーム画面から盤面領域を切り出して拡大する。"""
    reg = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    y1, y2 = max(0, reg.y), min(frame.shape[0], reg.y + reg.height)
    x1, x2 = max(0, reg.x), min(frame.shape[1], reg.x + reg.width)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return cv2.resize(
        crop, None, fx=CROP_SCALE, fy=CROP_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )


def _compose(crop: np.ndarray, grid_img: np.ndarray) -> np.ndarray:
    """実画面と認識結果を左右に並べる。

    **行が横一列に並ぶように認識結果を実画面の高さへ拡縮する。**
    高さが揃っていないとセル単位の突き合わせができず、ラベル付けの
    負担が跳ね上がる (自分で1枚見て気づいた)。
    実画面の盤面領域も認識グリッドも同じ 12 行を表すので、
    高さを揃えれば行がそのまま対応する。
    """
    h = crop.shape[0]
    grid_scaled = cv2.resize(
        grid_img, (grid_img.shape[1], h), interpolation=cv2.INTER_NEAREST,
    )
    # 行の対応を目で追えるように水平の区切り線を両方へ引く
    out_crop = crop.copy()
    row_h = h / (BOARD_ROWS - HIDDEN_ROWS)
    for k in range(1, BOARD_ROWS - HIDDEN_ROWS):
        y = int(round(k * row_h))
        cv2.line(out_crop, (0, y), (out_crop.shape[1], y), (70, 70, 70), 1)
        cv2.line(grid_scaled, (0, y), (grid_scaled.shape[1], y), (70, 70, 70), 1)
    gap = np.zeros((h, 24, 3), dtype=np.uint8)
    return np.hstack([out_crop, gap, grid_scaled])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz-dir", type=Path,
        default=Path("data/indicators_v2/boards_lean_allframes_ref_2026-07-30"),
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("data/verify/board_labels_2026-07-31"),
    )
    ap.add_argument("--n", type=int, default=40, help="生成するサンプル数")
    ap.add_argument(
        "--strategy", choices=("random", "floating"), default="random",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--npz-limit", type=int, default=60)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(args.npz_dir.glob("*.npz"))[: args.npz_limit or None]
    if not files:
        print(f"npz が無い: {args.npz_dir}")
        return

    # 候補を集める: (npz名, 行index, video_id, side, frame_idx, grid)
    cands: list[tuple[str, int, str, str, int, np.ndarray]] = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        grids = np.asarray(d["grids"])
        vids = np.asarray(d["video_id"]).astype(str)
        sides = np.asarray(d["side"]).astype(str)
        fidx = np.asarray(d["frame_idx"])
        tsecs = np.asarray(d["t_sec"]).astype(float)
        cts = (
            np.asarray(d["chain_trigger_sec"]).astype(float)
            if "chain_trigger_sec" in d.files
            else np.full(len(grids), np.nan)
        )
        for i in range(len(grids)):
            g = grids[i]
            # 試合文脈フィルタ (2026-07-31 v3): 決着・遷移・エフェクト局面を除外
            t = float(tsecs[i])
            if not _in_stable_window(str(vids[i]), t):
                continue  # 試合外 or 試合末尾 (24型・13型を排除)
            ct = float(cts[i])
            if not np.isnan(ct) and 0.0 <= t - ct < CHAIN_COOLDOWN_SEC:
                continue  # 連鎖検知直後 (エフェクト煙、13/007型を排除)
            if args.strategy == "floating":
                ok = False
                for c in range(BOARD_COLS):
                    for r in range(HIDDEN_ROWS, BOARD_ROWS - 1):
                        v, below = int(g[r, c]), int(g[r + 1, c])
                        if (v in VALID_COLORS or v == COLOR_OJAMA) and below == COLOR_EMPTY:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    continue
            else:
                # ランダム戦略: 空盤面に近い行は情報量が低いので除く
                filled = int((g[HIDDEN_ROWS:, :] != COLOR_EMPTY).sum())
                if filled < 6:
                    continue
            cands.append(
                (f.stem, i, str(vids[i]), str(sides[i]), int(fidx[i]), g.copy()),
            )
    if not cands:
        print("候補なし")
        return
    print(f"候補 {len(cands)} 件 (strategy={args.strategy})")

    # 動画・side で均すため、キーごとにまとめてラウンドロビンで取る
    buckets: dict[tuple[str, str], list] = {}
    for c in cands:
        buckets.setdefault((c[2], c[3]), []).append(c)
    for v in buckets.values():
        rng.shuffle(v)
    picked: list = []
    keys = sorted(buckets.keys())
    while len(picked) < args.n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < args.n:
                picked.append(buckets[k].pop())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.out_dir / "labels.tsv"
    rows_out = [
        "# 誤っているセルだけ wrong_cells に記入してください "
        "(例: r3c2=1,r5c0=0)。全部正しければ ok と書いてください。",
        "# r は画面内の行 (r1=最上段, r12=最下段)、c は列 (c0=左端, c5=右端)。",
        "# 色コード: 0=空 1=赤 2=青 3=緑 4=黄 5=紫 9=おじゃま",
        "sheet\tvideo\tside\tframe\twrong_cells",
    ]
    made = 0
    cache: dict[str, cv2.VideoCapture] = {}
    for (stem, i, vid, side, frame_idx, grid) in picked:
        # video_id は既に "video_c10" 形式 (二重に video_ を付けない)
        stem_name = vid if vid.startswith("video_") else f"video_{vid}"
        vpath = args.video_dir / f"{stem_name}.mp4"
        if not vpath.exists():
            continue
        cap = cache.get(str(vpath))
        if cap is None:
            cap = cv2.VideoCapture(str(vpath))
            cache[str(vpath)] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        sheet = _compose(_crop_board(frame, side), _render_grid(grid))
        name = f"{made:03d}_{vid}_{side}_f{frame_idx}.png"
        cv2.imwrite(str(args.out_dir / name), sheet)
        rows_out.append(f"{name}\t{vid}\t{side}\t{frame_idx}\t")
        made += 1
    for cap in cache.values():
        cap.release()
    tsv.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    print(f"生成 {made} 枚 → {args.out_dir}")
    print(f"ラベル記入用: {tsv}")
    print(
        f"\n→ 1盤面 = {(BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS} セル。"
        f"{made} 枚で {made * (BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS} セル分のラベルになる。"
    )


if __name__ == "__main__":
    main()
