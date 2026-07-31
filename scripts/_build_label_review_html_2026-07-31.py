"""ラベルシートをweb閲覧できる自己完結HTMLに焼き込む (2026-07-31)。

## なぜ

盤面ラベルの確定は user のドメイン知見が必須 (coordinator は 12枚中2枚を誤判定)。
user はリモートなので、40枚のシートを**スマホで見られる1枚のHTML**にまとめる。
Artifact は外部ホストへの参照を CSP でブロックするため、画像は data URI で埋め込む。

## 出力

各シートについて:
  - 番号 / 動画 / side / フレーム のラベル
  - 実画面 (左) と認識結果 (右) を並べたシート画像 (リサイズして JPEG 埋め込み)
  - 既知の誤り (labels.tsv の wrong_cells) があれば赤で明示

スマホ縦持ちで1枚ずつ縦スクロールできるレイアウト。
"""

from __future__ import annotations

import argparse
import base64
import csv
from pathlib import Path

import cv2


def _encode_jpeg(img_path: Path, max_w: int, quality: int) -> str:
    """画像を読み込み、横幅 max_w 以下に縮小して JPEG data URI にする。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(
            img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _load_known_errors(tsv: Path) -> dict[str, str]:
    """labels.tsv から sheet -> wrong_cells を読む (記入済みのみ)。"""
    out: dict[str, str] = {}
    if not tsv.exists():
        return out
    with tsv.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("sheet\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[4].strip():
                out[parts[0]] = parts[4].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", type=Path,
        default=Path("data/verify/board_labels_2026-07-31_v2"),
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("data/verify/board_labels_2026-07-31_v2/review.html"),
    )
    ap.add_argument("--max-width", type=int, default=760)
    ap.add_argument("--quality", type=int, default=78)
    args = ap.parse_args()

    sheets = sorted(
        p for p in args.dir.glob("*.png") if not p.name.startswith("candidate")
    )
    if not sheets:
        print(f"シートが無い: {args.dir}")
        return
    known = _load_known_errors(args.dir / "labels.tsv")

    cards: list[str] = []
    for p in sheets:
        # 000_video_c10_1P_f107305.png → 番号/動画/side/frame
        stem = p.stem
        parts = stem.split("_")
        num = parts[0]
        vid = "_".join(parts[1:3]) if len(parts) >= 3 else "?"
        side = parts[3] if len(parts) >= 4 else "?"
        frame = parts[4] if len(parts) >= 5 else "?"
        uri = _encode_jpeg(p, args.max_width, args.quality)
        err = known.get(p.name, "")
        err_html = (
            f'<div class="err">既知の誤り: {err}</div>' if err else ""
        )
        cards.append(
            f'<figure class="card">'
            f'<figcaption><span class="num">{num}</span>'
            f'<span class="meta">{vid} / {side} / frame {frame}</span>'
            f"{err_html}</figcaption>"
            f'<img loading="lazy" src="{uri}" alt="{stem}"></figure>'
        )
        print(f"埋め込み: {p.name}")

    n = len(cards)
    n_err = len(known)
    html = f"""<title>ぷよ認識ラベルレビュー {args.dir.name}</title>
<style>
  :root {{
    --bg: #f4f5f7; --card: #ffffff; --ink: #1a1c1f; --sub: #5c636e;
    --line: #d9dce1; --accent: #2b6cb0; --err: #c0392b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14161a; --card:#1e2127; --ink:#e6e8eb; --sub:#9aa2ad;
             --line:#2c313a; --accent:#5fa8e0; --err:#ff6b5a; }}
  }}
  :root[data-theme="dark"] {{ --bg:#14161a; --card:#1e2127; --ink:#e6e8eb;
    --sub:#9aa2ad; --line:#2c313a; --accent:#5fa8e0; --err:#ff6b5a; }}
  :root[data-theme="light"] {{ --bg:#f4f5f7; --card:#fff; --ink:#1a1c1f;
    --sub:#5c636e; --line:#d9dce1; --accent:#2b6cb0; --err:#c0392b; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height:1.5; }}
  header {{ padding: 20px 16px 8px; max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 1.15rem; margin: 0 0 4px; }}
  .lead {{ color: var(--sub); font-size: 0.85rem; margin: 0; }}
  .legend {{ max-width:820px; margin: 10px auto 0; padding: 10px 16px;
    background:var(--card); border:1px solid var(--line); border-radius:10px;
    font-size:0.8rem; color:var(--sub); }}
  .legend b {{ color: var(--ink); }}
  main {{ max-width: 820px; margin: 0 auto; padding: 12px; }}
  .card {{ background:var(--card); border:1px solid var(--line);
    border-radius:12px; margin: 0 0 16px; padding: 10px; overflow:hidden; }}
  figcaption {{ display:flex; align-items:baseline; gap:10px;
    flex-wrap:wrap; margin-bottom:8px; }}
  .num {{ font-weight:700; color:var(--accent); font-variant-numeric:tabular-nums; }}
  .meta {{ color:var(--sub); font-size:0.85rem;
    font-variant-numeric:tabular-nums; }}
  .err {{ color:var(--err); font-weight:600; font-size:0.85rem;
    width:100%; }}
  img {{ width:100%; height:auto; display:block; border-radius:6px; }}
</style>
<header>
  <h1>ぷよ認識ラベルレビュー</h1>
  <p class="lead">{n} 盤面 ({n * 72} セル) / 既知の誤り {n_err} 枚</p>
</header>
<div class="legend">
  各カード: <b>左=実ゲーム画面</b>、<b>右=認識結果</b>。右の
  <b>黄枠</b>は各列の最上ぷよ (背景誤検出が出やすい位置)。
  実画面と認識結果でぷよの色・位置がずれていないか確認してください。
  色: 赤=1 青=2 緑=3 黄=4 紫=5 灰=おじゃま。
  <b>操作中/落下中のツモ</b>が盤面領域に写り込んでいる場合、認識が空でも正解です。
</div>
<main>
{chr(10).join(cards)}
</main>
"""
    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"\n生成: {args.out} ({size_mb:.1f} MB, {n} 盤面)")


if __name__ == "__main__":
    main()
