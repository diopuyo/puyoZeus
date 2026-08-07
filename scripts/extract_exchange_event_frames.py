"""#24 打ち合い計測器 Step5「乖離上位イベントの実画面フレーム抽出」(2026-08-02)。

scripts/select_exchange_divergence_events.py が選んだイベント (主系列12件+副系列4件)
について、実ゲーム画面のフルフレーム (1920x1080) を4枚ずつ抽出し、
user が直接「案Dとシミュのどちらが現実に近かったか」を判断できる review_sheet.md
を生成する (採否確定権は user にあり数値だけで決めない規律、
feedback_viz_eval_required)。

## t_sec の定義 (scripts/collect_boards_lean.py で確認済み、2026-08-02 main検品で訂正)
t_sec = フレームインデックス ÷ fps (対象 mp4 ファイル自身の再生時刻、0起点)。
つまり選定イベントの t_sec は、boards_lean 生成に使った**その mp4ファイルそのもの**
の再生時刻と完全に一致する。そのため実画面抽出も同じ mp4 ファイルから行う必要がある
(別途DLし直すと再エンコード等でフレーム位置がズレるリスクがある)。

⚠️ **t_sec の意味づけ訂正 (2026-08-02 main検品で確定)**: exchange ラベルの t_sec は
「連鎖開始 (撃った瞬間)」ではなく「**連鎖終了 (スコア確定・おじゃま送付確定) 時点**」
だった (video_c27_01 の実フレームで確認: t_sec−1秒は9連鎖目の最終盤、t_secは連鎖
完了・盤面ほぼ空)。抽出タイミング自体 (t_sec を基準にした4オフセット) は正しいが、
表示文言を実態に合わせて訂正し、「連鎖開始前 (推定)」フレームを新たに追加する。

## 動画ソースの選び方 (2026-08-02、実行時に判明した事実を反映)
選定イベントの動画 (video_c27/c40/c45 等の c系動画) は、boards_lean 生成時に
使った mp4 が **既にローカルにキャッシュ済み** (data/frames/ 配下、WSL環境では
/home/ryouj/frames/ 配下にも同一サイズのコピーが存在、2026-08-02実行時に66動画
全件で確認済み)。これは「削除済みなので再DLが必要」という前提 (ストレージ管理
ルール的には動画は処理後削除が原則) と食い違うが、**t_sec整合性の観点で
このキャッシュ済みファイルを使う方が正しい** (再DLだと同一動画でも再エンコードで
フレーム位置がズレ得るため)。よって本スクリプトはローカルキャッシュを最優先で
使い、無い場合のみ yt-dlp フォールバックDL (一時領域、処理後削除) を試みる。
既存のキャッシュ済みファイルは他の分析スクリプトが参照する共有資産のため
削除しない (削除するのは本スクリプトが新規にDLしたファイルのみ)。

## 出力
    data/verify/exchange_divergence_review_2026-08-02/frames/<video>_<連番>_<タグ>.png
    data/verify/exchange_divergence_review_2026-08-02/review_sheet.md

## 使い方
    PYTHONPATH=. python -m scripts.extract_exchange_event_frames \\
        --selected-csv data/verify/exchange_divergence_review_2026-08-02/selected_events.csv \\
        --out-dir data/verify/exchange_divergence_review_2026-08-02
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.chain_count_ocr import _ensure_1080p
from src.indicators_v2 import SEC_PER_HAND

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

DEFAULT_SELECTED_CSV = Path("data/verify/exchange_divergence_review_2026-08-02/selected_events.csv")
DEFAULT_OUT_DIR = Path("data/verify/exchange_divergence_review_2026-08-02")
FRAMES_SUBDIR_NAME: str = "frames"

# 連鎖終了直前フレームのオフセット秒 (t_sec より何秒前を見るか)
PRE_FIRE_OFFSET_SEC: float = 1.0

# 着弾確認フレームの余裕秒 (t_sec + sim_k_hands×1手時間 に足す猶予)
LANDING_BUFFER_SEC: float = 2.0

# 連鎖開始前(推定)フレームの1連鎖あたり実測秒
# (memory project_exchange_measurement_foundation_2026-07-22: 8連鎖実測14.5秒の実測値、
# 約1.8秒/連鎖)。approx_fire_chains × この値 を遡って「連鎖開始前」の推定時刻とする。
CHAIN_DURATION_SEC_PER_CHAIN: float = 1.8

# 連鎖開始前(推定)フレームのマージン秒 (連鎖開始よりさらに手前=盤面が崩れる前を見るため)
PRE_CHAIN_START_MARGIN_SEC: float = 1.0

# 4タグ (ファイル名サフィックス、時系列順) + 表示ラベル (review_sheet 用、平易な日本語併記)
FRAME_TAGS: tuple[str, ...] = ("pre_chain_start", "pre_fire", "fire", "landing")
FRAME_TAG_LABELS: dict[str, str] = {
    "pre_chain_start": "連鎖開始前 (推定、両者の盤面が崩れる前)",
    "pre_fire": "連鎖終了直前 (発火側の連鎖の最終盤)",
    "fire": "連鎖終了時点 (おじゃま送付量が確定した瞬間、ラベルのt_sec)",
    "landing": "着弾確認 (相手にお邪魔が降り終わった頃)",
}

# 「連鎖開始前(推定)」フレームの注記文言 (review_sheet に必ず併記する)
PRE_CHAIN_START_ESTIMATE_NOTE: str = "推定時刻のため多少ずれます"
PRE_CHAIN_START_CLAMPED_NOTE: str = "動画先頭のためクランプ済み (実際の連鎖開始より新しい時刻)"

# ローカルキャッシュ探索先 (WSL環境の高速ネイティブパスを優先、無ければ
# リポジトリ相対の data/frames/ にフォールバック)。
WSL_NATIVE_FRAMES_DIR = Path("/home/ryouj/frames")
REPO_FRAMES_DIR = Path("data/frames")

# yt-dlp フォールバックDL先 (一時領域、処理後に必ず削除する)
DEFAULT_DL_TEMP_DIR = Path("data/tmp_exchange_video_dl")

# WSL上のマウントパス接頭辞 ("/mnt/c/" -> "C:\") 変換用
# (feedback_review_image_links: レビュー画像はWindowsパスで)
WSL_MOUNT_PREFIX: str = "/mnt/c/"
WINDOWS_DRIVE_PREFIX: str = "C:\\"

# video_id ("video_c27") から mp4 ファイル名 stem ("c27") への接頭辞
VIDEO_ID_PREFIX: str = "video_"


# =============================================================================
# 1. 動画ソース解決 (ローカルキャッシュ優先、無ければ yt-dlp フォールバック)
# =============================================================================

def _video_id_to_stem(video_id: str) -> str:
    """CSV の video_id ("video_c27") を mp4 ファイル名 stem ("c27") に変換する。"""
    if video_id.startswith(VIDEO_ID_PREFIX):
        return video_id[len(VIDEO_ID_PREFIX):]
    return video_id


def resolve_cached_video_path(video_id: str) -> "Path | None":
    """ローカルキャッシュ済み mp4 のパスを返す (WSLネイティブ優先、無ければNone)。"""
    stem = _video_id_to_stem(video_id)
    for base_dir in (WSL_NATIVE_FRAMES_DIR, REPO_FRAMES_DIR):
        candidate = base_dir / f"video_{stem}.mp4"
        if candidate.exists():
            return candidate
    return None


def download_video_via_ytdlp(
    video_id: str, dest_dir: Path, url_map: "dict[str, str] | None",
) -> "Path | None":
    """yt-dlp フォールバックDL (url_map に video_id のURLが無ければ諦めてNoneを返す)。

    本関数はローカルキャッシュに無い動画のみ呼ばれる想定 (2026-08-02実行では
    選定3動画とも全てキャッシュ済みのため実行されなかった、テスト用の
    防御的実装)。
    """
    if not url_map or video_id not in url_map:
        print(f"  [WARN] {video_id}: 既知のソースURLが無くDL不可 (--video-url-map で指定可)")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = _video_id_to_stem(video_id)
    out_path = dest_dir / f"video_{stem}.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--remux-video", "mp4", "--no-playlist", "-o", str(out_path), url_map[video_id],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        print(f"  [ERROR] {video_id}: yt-dlp DL失敗 ({result.stderr[-300:]})")
        return None
    return out_path


def ensure_video_available(
    video_id: str, dl_temp_dir: Path, url_map: "dict[str, str] | None",
) -> tuple["Path | None", bool]:
    """動画ファイルを用意する (キャッシュ優先)。戻り値2つ目は「本スクリプトが

    新規にDLしたか」(True なら処理後に削除する責任がある)。
    """
    cached = resolve_cached_video_path(video_id)
    if cached is not None:
        print(f"  {video_id}: ローカルキャッシュ使用 ({cached})")
        return cached, False
    downloaded = download_video_via_ytdlp(video_id, dl_temp_dir, url_map)
    return downloaded, downloaded is not None


# =============================================================================
# 2. フレーム時刻計算・抽出
# =============================================================================

def compute_event_timestamps(
    t_sec: float, sim_k_hands: float, approx_fire_chains: float,
) -> tuple[dict[str, float], bool]:
    """イベント1件分の4時刻 (連鎖開始前(推定)/連鎖終了直前/連鎖終了時点/着弾確認) を計算する。

    負値は0にクランプする。戻り値2つ目は「連鎖開始前(推定)がクランプされたか」
    (True なら動画先頭のため推定時刻より新しい時刻になっている、review_sheetの注記用)。
    """
    chain_duration_est_sec = float(approx_fire_chains) * CHAIN_DURATION_SEC_PER_CHAIN + PRE_CHAIN_START_MARGIN_SEC
    raw_pre_chain_start = t_sec - chain_duration_est_sec
    pre_chain_start_clamped = raw_pre_chain_start < 0.0
    landing_delay_sec = float(sim_k_hands) * SEC_PER_HAND + LANDING_BUFFER_SEC
    timestamps = {
        "pre_chain_start": max(0.0, raw_pre_chain_start),
        "pre_fire": max(0.0, t_sec - PRE_FIRE_OFFSET_SEC),
        "fire": max(0.0, t_sec),
        "landing": max(0.0, t_sec + landing_delay_sec),
    }
    return timestamps, pre_chain_start_clamped


def grab_frame(video_path: Path, t_sec: float) -> "np.ndarray | None":
    """指定時刻のフレームを取得し 1080p に正規化する (既存 _grab_frame パターン踏襲)。"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _ensure_1080p(frame)


def extract_frames_for_event(
    video_path: Path, video_id: str, event_no: int, timestamps: dict[str, float], frames_dir: Path,
) -> dict[str, "Path | None"]:
    """1イベント分の4枚のフレームを保存する (タグ->保存パス、失敗はNone+ログ)。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, "Path | None"] = {}
    for tag in FRAME_TAGS:
        frame = grab_frame(video_path, timestamps[tag])
        if frame is None:
            print(f"  [WARN] {video_id} event{event_no:02d} tag={tag}: フレーム取得失敗")
            saved[tag] = None
            continue
        out_path = frames_dir / f"{video_id}_{event_no:02d}_{tag}.png"
        cv2.imwrite(str(out_path), frame)
        saved[tag] = out_path
    return saved


# =============================================================================
# 3. review_sheet.md 生成
# =============================================================================

def to_windows_path(path: Path) -> str:
    """パスを Windows 形式の絶対パス文字列に変換する。

    WSL のマウントパス (/mnt/c/...) を C:\\... に変換する (feedback_review_image_links:
    レビュー画像はWindowsパスで)。Windowsネイティブ環境で実行した場合は
    path.resolve() がそのまま Windows パスを返すため変換不要。
    """
    abs_posix = path.resolve().as_posix()
    if abs_posix.startswith(WSL_MOUNT_PREFIX):
        return WINDOWS_DRIVE_PREFIX + abs_posix[len(WSL_MOUNT_PREFIX):].replace("/", "\\")
    return str(path.resolve())


def _format_frame_line(tag: str, path: "Path | None", pre_chain_start_clamped: bool) -> str:
    """review_sheet.md のフレーム1行分を組み立てる (連鎖開始前(推定)には注記を付す)。"""
    label = FRAME_TAG_LABELS[tag]
    if path is None:
        return f"- {label}: (フレーム取得失敗)"
    line = f"- {label}: {to_windows_path(path)}"
    if tag == "pre_chain_start":
        note = PRE_CHAIN_START_ESTIMATE_NOTE
        if pre_chain_start_clamped:
            note += f"、{PRE_CHAIN_START_CLAMPED_NOTE}"
        line += f" ({note})"
    return line


def _format_event_section(
    row: pd.Series, event_no: int, frame_paths: dict[str, "Path | None"], pre_chain_start_clamped: bool,
) -> str:
    """review_sheet.md の1イベント分のセクションを組み立てる (50行制約対応の分割)。"""
    lines = [
        f"## イベント{event_no:02d}: {row['video_id']} game{row['game_idx']} "
        f"{row['fire_side']} 発火 (位相:{row['phase']}, t={row['t_sec']:.1f}秒, "
        f"approx_fire_chains={row['approx_fire_chains']:.0f})",
        f"選定理由: {row['selection_reason']} ({row['selection_series']})",
        "",
    ]
    lines += [_format_frame_line(tag, frame_paths.get(tag), pre_chain_start_clamped) for tag in FRAME_TAGS]
    lines += [
        "",
        "| 予測器 | 予測値 | 補足 |",
        "| --- | --- | --- |",
        f"| 案D (実データ学習) | {row['net_ojama_after_oof_pred']:.1f} "
        f"(お邪魔換算、実測と同じ単位) | - |",
        f"| 修正シミュ | {row['sim_damage_score']:.3f} "
        f"(0〜1のダメージ度合いスコア、生の個数ではない) | "
        f"k_hands={row['sim_k_hands']:.0f}手, 期待反撃={row['sim_expected_counter_ojama']:.1f}個 |",
        f"| 併用(スタッキング) | {row['stack_pred_net_ojama_after']:.1f} (お邪魔換算) | - |",
        f"| **実測** | {row['net_ojama_after']:.1f} (お邪魔換算) | "
        f"対応成功={'成功' if row['taiou_success'] else '失敗'}, "
        f"生存={'生存' if row['survived'] else '窒息'} |",
        "",
        f"機械判定 (参考、値の単位が違う2予測を順位で公平に比較): "
        f"**{row['closer_to_actual_rank_based']}** の予測が実測の順位に近かった",
        "",
    ]
    return "\n".join(lines)


def render_review_sheet(
    df: pd.DataFrame,
    frame_paths_by_event: dict[int, dict[str, "Path | None"]],
    out_dir: Path,
    pre_chain_start_clamped_by_event: "dict[int, bool] | None" = None,
) -> Path:
    """全イベント分の review_sheet.md を生成する。

    pre_chain_start_clamped_by_event: イベント番号->「連鎖開始前(推定)がクランプ
    されたか」(未指定=全て False 扱い、後方互換用の optional 引数)。
    """
    clamped_map = pre_chain_start_clamped_by_event or {}
    header = [
        "# #24 打ち合い計測器 乖離上位イベント レビューシート",
        "",
        "案D (実データ学習) と修正シミュの判断が最も割れたイベントを実ゲーム画面で示します。",
        "「機械判定」はあくまで参考値です。**どちらの見立てが実際のゲーム展開に合っていたか**を",
        "画像を見て判断してください (数値だけで採否を決めない規律)。",
        "",
        "用語メモ: 「案D」=過去の実データから学習した予測モデル、"
        "「修正シミュ」=物理法則ベースのシミュレーション計算、"
        "「併用」=両方を組み合わせたモデル、"
        "「net_ojama_after」=攻撃側が相手に与えた正味のお邪魔ぷよの個数 (大きいほど攻撃側有利)、"
        "「approx_fire_chains」=発火した連鎖の推定連鎖数 (連鎖開始前フレームの推定時刻算出に使用)。",
        "",
        "⚠️ ラベルの t_sec は「連鎖開始 (撃った瞬間)」ではなく「連鎖終了 "
        "(スコア確定・おじゃま送付確定) 時点」です (2026-08-02 main検品で確定)。",
        "",
    ]
    sections = [
        _format_event_section(
            row, idx + 1, frame_paths_by_event[idx + 1], clamped_map.get(idx + 1, False),
        )
        for idx, row in df.reset_index(drop=True).iterrows()
    ]
    out_path = out_dir / "review_sheet.md"
    out_path.write_text("\n".join(header + sections), encoding="utf-8")
    return out_path


# =============================================================================
# メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="#24 打ち合い計測器 Step5 実画面フレーム抽出")
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dl-temp-dir", type=Path, default=DEFAULT_DL_TEMP_DIR)
    parser.add_argument("--video-url-map", type=Path, default=None,
                         help="video_id,url の2列CSV (ローカルキャッシュ無し動画のDLフォールバック用)")
    return parser.parse_args()


def _load_url_map(path: "Path | None") -> "dict[str, str] | None":
    """--video-url-map の任意CSVを読み込む (無指定ならNone)。"""
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    return dict(zip(df["video_id"], df["url"]))


def main() -> None:
    args = _parse_args()
    print(f"[extract_exchange_event_frames] selected_csv={args.selected_csv}")
    df = pd.read_csv(args.selected_csv)
    url_map = _load_url_map(args.video_url_map)
    frames_dir = args.out_dir / FRAMES_SUBDIR_NAME

    print(f"\n=== 1. 動画ソース解決 ({df['video_id'].nunique()}本) ===")
    downloaded_paths: list[Path] = []
    video_paths: dict[str, "Path | None"] = {}
    for video_id in df["video_id"].unique():
        path, was_downloaded = ensure_video_available(video_id, args.dl_temp_dir, url_map)
        video_paths[video_id] = path
        if was_downloaded and path is not None:
            downloaded_paths.append(path)

    print(f"\n=== 2. フレーム抽出 ({len(df)}イベント×{len(FRAME_TAGS)}枚) ===")
    frame_paths_by_event: dict[int, dict[str, "Path | None"]] = {}
    clamped_by_event: dict[int, bool] = {}
    for idx, row in df.reset_index(drop=True).iterrows():
        event_no = idx + 1
        video_path = video_paths.get(row["video_id"])
        if video_path is None:
            print(f"  [SKIP] event{event_no:02d} ({row['video_id']}): 動画が用意できず抽出不可")
            frame_paths_by_event[event_no] = {tag: None for tag in FRAME_TAGS}
            clamped_by_event[event_no] = False
            continue
        timestamps, clamped = compute_event_timestamps(
            row["t_sec"], row["sim_k_hands"], row["approx_fire_chains"],
        )
        clamped_by_event[event_no] = clamped
        frame_paths_by_event[event_no] = extract_frames_for_event(
            video_path, row["video_id"], event_no, timestamps, frames_dir,
        )
        clamp_note = " (連鎖開始前(推定)はクランプ済み)" if clamped else ""
        print(f"  event{event_no:02d} ({row['video_id']} t={row['t_sec']:.1f}s) 抽出完了{clamp_note}")

    print("\n=== 3. review_sheet.md 生成 ===")
    sheet_path = render_review_sheet(df, frame_paths_by_event, args.out_dir, clamped_by_event)
    print(f"  出力: {sheet_path}")

    print(f"\n=== 4. 新規DLファイルの削除 ({len(downloaded_paths)}件、ローカルキャッシュは削除しない) ===")
    for path in downloaded_paths:
        path.unlink(missing_ok=True)
        print(f"  削除: {path}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
