"""W9-B レビュー結果適用: ユーザーが char-coded で送ったラベルを labels.csv に反映。

各シートは 10 行 × 20 列 (= 200 cells)。char マッピング:
    E=EM, R=RED, B=BLUE, G=GRN, Y=YEL, P=PUR, O=OJM
    ?=エフェクト被り等で判別不能 → your_answer 空 (訓練除外)

v18_m03 は 9 行 (= 180 cells) のみ提供されたので最後の 20 cells は空のまま。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

CHAR_TO_LABEL: dict[str, str] = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM",
}

REVIEWS: dict[str, list[str]] = {
    "v09_m02_full": [
        "EEGEEEEEEBPPEEGBBEGB",
        "PPEGPGPERRRGBRPBRRPB",
        "GBBGRRBEEEGEEEEEEEPE",
        "EEGEEEPGGPEPPGBBBGPP",
        "RGPPBPBGRPBGGBBBGRBE",
        "EEEEEEEEEEEEEEEEEEEE",
        "REBEERGEEPGPEEEGPEBB",
        "PBBEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEEEEEEEEE",
        "EEEEEEPEEBBEEEPBGERB",
    ],
    "v13_m02_full": [
        "EEEEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEBEEEEEYEEBE",
        "EYBEEPEPEEPBEEEYEEYE",
        "EEYEEEPPEEEPBEEEEEEE",
        "PPEEEPEERBEEERBYYYBP",
        "EEE??PYERPRBBOYPYYBY",  # ? はエフェクト被り
        "OYYOOOOROYOOOOPYORPB",
        "OOPEEEEBEEEEYEEEEOEE",
        "EEEOEEEEEOEEEYOEEEEE",
        "EEOYEEOREOORREPYRRRP",
    ],
    "v17_m11_full": [
        "EEEEEEEEPBEEEREEERRE",
        "PRPPBRPBBPBBRBRRRRPP",
        "GPPGPPPEEEEEEEEEEEEE",
        "EERRPEBRBPEGGBPBRGGG",
        "BBBPPPPRRRRGGPPPPGPE",
        "EREEREEEEEEEPEEEEEER",
        "ERREBPBGGRBREBRGBBPP",
        "BRGEEEEEEEEEEEEEEEEE",
        "EEE?EEEEEEEEEEEEEEEE",  # ? はエフェクト
        "EEEEEEPEEERERRRREPPG",
    ],
    "v18_m03_full": [
        "EEEEEEEGEGPEEEEYYPGE",
        "GPEGGRPGYYYGPYPGYRPG",
        "RGGEEYEEEEEEEEEEEOPE",
        "RGGGEGPPGEPPYRRGPPGG",
        "YYYGYRGRYRGPPRGGPYRE",
        "YOOYYOREPGREYPPEROER",
        "RRYEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEEYEEEEEE",
        "EEYPEEYEEEEPPEEPGPEP",
        # 10 行目はユーザー未提供 → 残り 20 cells は空
    ],
    "v19_m06_full": [
        "EEEPEEEOEYEEEEYEGEPG",
        "EYPYYGPYRRRYRRYGRRYG",
        "PGGPYYYEEEEEEEEEEEER",
        "EEERYEEYPYEYPRGPGRRY",
        "RRRYGYGYRYGPPPGGPYYE",
        "EEEEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEREREEEEEEE",
        "EEREEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEEEEEEEEE",
        "EERREGYEEPRYEEPPREGG",
    ],
    "v18_m08_full": [
        "EEEEEEEYYYEEEEGEEEGE",
        "EYEEGPEGGBBEBEPBEPPB",
        "YBBYBPPEEEEEEEEEEEEE",
        "EEYEEEBYEEEBBEYEEEYG",
        "GGGEGPBGPPBYYYBBYBPE",
        "EEEEEEEEEEEEEYPBGPBY",
        "BPOYBGGBYBGYBOYBYYPG",
        "YBBEEEEEPEEEEGEEEEYE",
        "EEEEYEEEEEYOGBBOPBGG",
        "OBBPYPGPYBPYPPYPBBYY",
    ],
    "v18_m15_full": [
        "EEEEEEEGEEEEEEEEEEEE",
        "BYGGBGEBGBGEYYGYPYBG",
        "GYYYBGGEEEBEEEEEEEYE",
        "EEYEEEGBEEEGGEEEGEBG",
        "BBPEPGYYGGYPYPYYPPOE",
        "EEEEEEEEEEYEEEEEEEEE",
        "EEEEEEEGBEEEEEEEEEEE",
        "OPEEEEEEEEEYEEEEYBEE",
        "EGGBEEGYBYGEYBGEEEEE",
        "GGGBEPPGYOYPGGYYPYPO",
    ],
    "v04_m07_full": [
        "EEEEEEEEEEEEEEBBEPYB",
        "EPBRBBPPYYYPBBPBYBPB",
        "RBBPYYYREEEPEEEEEEEE",
        "EEYBBEPPBPRYYRBRBRPB",
        "YYPBYPBYBPBRRRBBPYYE",
        "EEEEEEEEEEEEEEEEEEEE",
        "EEEPEEEEEEEEBEEEEEEP",
        "REREEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEERREEEEE",
        "ERPBEPYEERPBEBBRRBEY",
        # 注: OJM 判定された cell はすべてエフェクト (実際は EM)
    ],
    "v06_m06_full": [
        "EEREEEEEEEEEEEPPREGY",
        "GPPGPRGYYYYGYPGPYPGP",
        "RPPRRYREEEYEEEEEEEGY",
        "YYYPGGPPYGPRPRPPRRRE",
        "EEEEEEEEEYEEYEEEYEEE",
        "EEE?EEEREEEEPYGEEGGY",  # T(typo)→?
        "RPYEEEEEEEEEPEEEEPEE",
        "EEEYEEEEEYEEEEPEEEEE",
        "EYEGEEPGEEPGYEYEEYGG",
        # 10 行目未提供
    ],
    "v17_m37_full": [
        "EEREEEEEEEEEEEPPREGY",
        "GPPGPRGYYYYGYPGPYPGP",
        "RPPRRYREEEYEEEEEEEGY",
        "YYYPGGPPYGPRPRPPRRRE",
        "EEEEEEEEEYEEYEEEYEEE",
        "EEEYEEEREEEEPYGEEGGY",
        "RPYEEEEEEEEEPEEEEPEE",
        "EEEYEEEEEYEEEEPEEEEE",
        "EYEGEEPGEEPGYEYEEYGG",
        # 10 行目未提供
    ],
    "v19_m07_full": [
        "EEEEEEEEEEEEEEGEEEGE",
        "EEEEREEEGGPEEEYRYYYR",
        "PRRPYYREEEEEEEEEEEEE",
        "EEEEEEEEEEGEEGRPGRRP",
        "EPRGGEYRRRRYYYPPPRRE",
        "EEEEEEEEEEEEEEEEEEEE",
        "EEPEERPPRPRPYPPPYRRY",
        "RPYEEEEEEEEEE?EGEEYE",  # ` → ?
        "EEEEPEEPEEPEEEEPEEYY",
        "YGYRRGGPPYRRRRRYYPPR",
    ],
}


def apply_one(name: str, chars_rows: list[str], base_dir: Path) -> int:
    """char-coded review を labels.csv に反映、更新数を返す。"""
    csv_path = base_dir / name / "labels.csv"
    if not csv_path.exists():
        print(f"skip (missing): {csv_path}")
        return 0

    # 1 次元化: id (1-indexed) -> char
    id_to_char: dict[int, str] = {}
    cell_id = 1
    for row in chars_rows:
        for ch in row:
            id_to_char[cell_id] = ch
            cell_id += 1

    rows: list[list[str]] = []
    n_filled = 0
    n_skipped_q = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            try:
                rid = int(r[0])
            except ValueError:
                rows.append(r)
                continue
            ch = id_to_char.get(rid, "")
            if ch == "?":
                # エフェクト被り → empty (訓練除外)
                r[6] = ""
                n_skipped_q += 1
            elif ch in CHAR_TO_LABEL:
                r[6] = CHAR_TO_LABEL[ch]
                n_filled += 1
            else:
                # ユーザー未提供 (例: v18 の 10 行目)
                r[6] = ""
            rows.append(r)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {n_filled} labels applied, {n_skipped_q} ? skipped "
          f"-> {to_windows_path(csv_path)}")
    return n_filled


def main() -> int:
    base = Path("data/verify/phase_w_review")
    total = 0
    for name, rows in REVIEWS.items():
        total += apply_one(name, rows, base)
    print(f"\ntotal labels applied: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
