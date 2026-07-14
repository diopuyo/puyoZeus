"""有利不利オーバーレイ動画 (試作) — tier1 軽量モデルで局面の有利不利を表示。

方針:
  - 学習データ (data/indicators_v2/study/labeled_win.csv) の差分特徴 (自−相手) で
    HistGBC を学習し、WinProb(1P) を出す。有利不利スコア = (p-0.5)*200。
  - 推論では重い火力系 (reach/immediate 等 484手探索) は使わず、
    collect_indicators_v2._fill_indicator_columns と同一の安価な指標関数のみ算出。
  - 認識は visualize_indicators_v2.generate と同じ load_default 経路。
  - 対象動画 (video_124_4min) は学習の study 動画 (v29-38) 外 → リークなし。

使い方:
    python -m scripts.visualize_advantage_overlay \
        --video data/frames/video_124_4min.mp4 \
        --out data/indicators_v2/overlay/advantage_v124.mp4 --max-sec 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker, OjamaAccountSnapshot  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, load_labeled_csv, pair_sides_for_win, build_features,
)

OUT_W, OUT_H = 1280, 720
GRAPH_H = 150          # 下部に足すグラフ専用の黒帯高さ
CANVAS_H = OUT_H + GRAPH_H
DEFAULT_FPS = 30.0
EVEN_THRESHOLD = 5.0  # |有利不利| がこれ未満は「互角」
EMA_ALPHA = 0.25      # 有利不利の時間平滑
# (B) 持続圧力信号: board_ojama の増加を減衰累積 (着弾ダメージの記憶)
PRESSURE_DECAY = 0.985    # 毎フレーム減衰 (半減期 ~1.5s @30fps)
PRESSURE_SCALE = 6.0      # 圧力 → 有利不利[-100,100] 換算
PRESSURE_BLEND_W = 0.6    # (旧2成分) 有利不利 = W×圧力 + (1-W)×現モデル
# (M3) お邪魔予告(incoming)信号: 相手に降る予告が多い=相手が埋まる=有利。
#   得点リード(結果)を廃し、予告(位置=これから相手が埋まる)へ置換(2026-07-14 user方針)。
FORECAST_SCALE = 1.4      # pending(まだ降る)お邪魔差 → 有利不利(72個≒満杯で±100)
FORECAST_DROP_PER_TURN = 30  # =OJAMA_MAX_DROP_PER_TURN(5段×6列)。ツモ1回で降る上限
SCORE_LEAD_SCALE = 0.4    # (旧・未使用) 得点リード換算
# 4成分ブレンド: 圧力(着弾) + 予告(incoming) + 現モデル(位置) + threat(仕込み火力)
W_PRESSURE = 0.35
W_FORECAST = 0.30
W_MODEL = 0.20
W_THREAT = 0.15
THREAT_SCALE = 0.22       # 到達火力差(お邪魔個) → 有利不利換算

# 学習・推論で共通の安価な差分特徴 (重い火力系は除外)
FEATURES: tuple[str, ...] = (
    "board_color_puyo_total", "max_column_height", "column_bumpiness",
    "death_margin", "death_margin_neighbor", "current_max_chain",
    "conn_pair_count", "conn_triple_count",
    "ojama_net_balance", "ojama_forecast", "board_ojama_count", "dig_resistance",
)
# 主要ドライバ表示用の日本語ラベル
JP_LABEL: dict[str, str] = {
    "board_ojama_count": "盤面お邪魔数", "death_margin": "窒息余裕",
    "max_column_height": "最大列高", "current_max_chain": "現在最大連鎖",
    "board_color_puyo_total": "色ぷよ総数", "ojama_forecast": "お邪魔予告",
}
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc",
)


def _font(size: int) -> ImageFont.ImageFont:
    """meiryo を取得 (無ければ default)。"""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _train_model(exclude_video: str | None = None):
    """study データの差分特徴で HistGBC を学習して返す。

    exclude_video: 指定動画IDの行を学習から除外 (対象動画のリーク防止)。
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = load_labeled_csv("data/indicators_v2/study/labeled_win.csv")
    if exclude_video is not None:
        before = len(df)
        df = df[df["video_id"].astype(str) != exclude_video].reset_index(drop=True)
        print(f"[train] {exclude_video} を学習除外: {before} -> {len(df)} 行")
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, list(FEATURES))
    cols = [f"{c}_diff" for c in FEATURES]
    X = feat[cols].fillna(0.0).values
    y = paired["won_1p"].astype(int).values
    # 対称化: 差分を反転しラベルも反転したミラー標本を追加。
    # 有利不利は「側を入れ替えると符号反転」する反対称関数であるべきで、
    # これにより互角(差=0)の予測が厳密に 50% になり、勝ち数の偏りバイアスを除去。
    X_sym = np.vstack([X, -X])
    y_sym = np.concatenate([y, 1 - y])
    model = HistGradientBoostingClassifier(**GBC_PARAMS)
    model.fit(X_sym, y_sym)
    print(f"[train] 元n={len(y)} (1P勝ち{int(y.sum())}) -> 対称化後 {len(y_sym)}")
    return model


class PressureTracker:
    """(B) 持続圧力: 相手盤面お邪魔の増加(=攻撃着弾)を減衰累積した 1P視点の圧力。"""

    def __init__(self) -> None:
        self.pressure = 0.0
        self._prev1 = 0.0
        self._prev2 = 0.0

    def update(self, ojama_1p: float, ojama_2p: float) -> float:
        """毎フレーム呼ぶ。1P視点の圧力を [-100,100] で返す (正=1P攻勢)。"""
        self.pressure *= PRESSURE_DECAY
        self.pressure += max(0.0, ojama_2p - self._prev2) - max(0.0, ojama_1p - self._prev1)
        self._prev1, self._prev2 = ojama_1p, ojama_2p
        return float(max(-100.0, min(100.0, self.pressure * PRESSURE_SCALE)))


class ScoreLeadTracker:
    """(M2) 得点リード: 各side の現在スコア差(=どちらが多く攻撃を通したか)を追う。

    連鎖を撃った側のスコアが伸びる。累積差なので「今実行中の側」だけでなく
    「既により多く撃った側」を正しく評価する。velocity版は2番目に撃つ小連鎖側を
    誤って有利にしたため、累積リードへ変更(2026-07-14 M2改)。
    スコアは試合開始で0にリセットされるため試合内相対で機能する。
    """

    def __init__(self) -> None:
        self._s1 = 0
        self._s2 = 0

    def update(self, score1: int | None, score2: int | None,
               rate: float = 70.0) -> float:
        """毎フレーム。1P視点の得点リードを [-100,100] で返す(正=1Pが多く攻撃)。"""
        if score1 is not None:
            self._s1 = score1
        if score2 is not None:
            self._s2 = score2
        lead = (self._s1 - self._s2) / rate  # お邪魔換算のスコア差
        return float(max(-100.0, min(100.0, lead * SCORE_LEAD_SCALE)))


def _side_feats(board: Board, net: int, forecast: int) -> dict[str, float]:
    """1 side の安価な指標 (collect と同一関数)。"""
    co, _ = iv.connectivity_observation(board)
    return {
        "board_color_puyo_total": iv.board_color_puyo_total(board).score,
        "max_column_height": iv.max_column_height(board).score,
        "column_bumpiness": iv.column_bumpiness(board).score,
        "death_margin": iv.death_margin(board).score,
        "death_margin_neighbor": iv.death_margin_neighbor(board).score,
        "current_max_chain": iv.current_max_chain(board).score,
        "conn_pair_count": float(co.pair_count),
        "conn_triple_count": float(co.triple_count),
        "ojama_net_balance": iv.ojama_net_balance(net).score,
        "ojama_forecast": iv.ojama_forecast(forecast).score,
        "board_ojama_count": iv.board_ojama_count(board).score,
        "dig_resistance": iv.dig_resistance(board).score,
    }


def _score_advantage(
    model, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
) -> tuple[float, float, list[tuple[str, float]]]:
    """両盤面 → (有利不利[-100..100], 1P勝率, 主要ドライバ)。"""
    f1 = _side_feats(b1, snap.net_balance_capped, snap.forecast_p1)
    f2 = _side_feats(b2, -snap.net_balance_capped, snap.forecast_p2)
    diff = {c: f1[c] - f2[c] for c in FEATURES}
    x = np.array([[diff[c] for c in FEATURES]], dtype=float)
    p1 = float(model.predict_proba(x)[0, 1])
    adv = (p1 - 0.5) * 200.0
    drivers = sorted(
        ((c, diff[c]) for c in JP_LABEL if c in diff),
        key=lambda kv: -abs(kv[1]))[:3]
    return adv, p1, drivers


def _threat(b1: Board, b2: Board, sp1, sp2, elapsed: float) -> float:
    """(3/M1) 火力threat = 到達火力差 1P−2P を [-100,100] で返す。

    reach_fire_power(実 next/dnext ペアで2手先読み)を使う。潜在火力
    (potential_fire_power)は greedy 探索が大連鎖を過小評価するバグがあり
    (2026-07-14 Phase1: あん実816に対し potential=360 だが reach=956 と的中)、
    「実際に撃てる火力」を測る reach の方が有利不利に正確なため置換。
    """
    r1 = iv.reach_fire_power(b1, sp1.next_pair, sp1.dnext_pair, elapsed).value.raw
    r2 = iv.reach_fire_power(b2, sp2.next_pair, sp2.dnext_pair, elapsed).value.raw
    return float(max(-100.0, min(100.0, (r1 - r2) * THREAT_SCALE)))


def _forecast_signal(snap: OjamaAccountSnapshot) -> float:
    """(旧M3) 会計の確定予告(連鎖終了時生成)差。ラグがあるため RealtimeForecastTracker を使う。"""
    diff = float(snap.forecast_p2 - snap.forecast_p1)  # 2P incoming - 1P incoming
    return float(max(-100.0, min(100.0, diff * FORECAST_SCALE)))


class RealtimeForecastTracker:
    """(M3改B) pending お邪魔を「相殺 + 30個/ターン配送」でリアルタイム管理する位置信号。

    お邪魔会計(OjamaAccountingTracker)と同じ配送モデルを、連鎖終了を待たず実行中に:
      - 相手が発火(score増)→ 生成量を算出。まず自分の pending を相殺、余剰を相手 pending へ。
      - 自分がツモを置く(tsumo増)→ 自分の pending から最大30個/ターンを盤面へ配送(pending減)。
    「累積スコア(結果)」でも「時間減衰(粗い)」でもなく、実際の降り方(5段=30個ずつ)を
    反映した「まだ降る残りお邪魔=位置」。大連鎖でも一気に埋まらず、返し(相殺)で相殺される。
    配送された分は盤面お邪魔(=圧力)が引き継ぐ。
    """

    def __init__(self) -> None:
        self.inc1 = 0.0  # 1P にこれから降る pending
        self.inc2 = 0.0  # 2P にこれから降る pending
        self._s1: int | None = None
        self._s2: int | None = None
        self._t1: int | None = None
        self._t2: int | None = None

    def _fire(self, gen: float, own_pending: float) -> tuple[float, float]:
        """発火生成 gen で自分の pending を相殺し (残own_pending, 相手へ余剰) を返す。"""
        canceled = min(gen, own_pending)
        return own_pending - canceled, gen - canceled

    def update(self, score1: int | None, score2: int | None,
               tsumo1: int, tsumo2: int, rate: float = 70.0) -> float:
        """1P視点の予告信号 [-100,100](正=2Pに多く降る=1P有利)。"""
        # 試合境界(スコア大幅減)で pending クリア
        if ((score1 is not None and self._s1 is not None and self._s1 - score1 >= 1000)
                or (score2 is not None and self._s2 is not None and self._s2 - score2 >= 1000)):
            self.inc1 = self.inc2 = 0.0
        # 発火(score増)→ 相殺 + 余剰を相手 pending へ
        if score1 is not None:
            if self._s1 is not None and score1 > self._s1:
                self.inc1, surplus = self._fire((score1 - self._s1) / rate, self.inc1)
                self.inc2 += surplus
            self._s1 = score1
        if score2 is not None:
            if self._s2 is not None and score2 > self._s2:
                self.inc2, surplus = self._fire((score2 - self._s2) / rate, self.inc2)
                self.inc1 += surplus
            self._s2 = score2
        # 配送(ツモ増)→ 自分の pending から 30個/ターン 盤面へ(=pending減、圧力が引継ぐ)
        if self._t1 is not None and tsumo1 > self._t1:
            self.inc1 = max(0.0, self.inc1 - FORECAST_DROP_PER_TURN * (tsumo1 - self._t1))
        if self._t2 is not None and tsumo2 > self._t2:
            self.inc2 = max(0.0, self.inc2 - FORECAST_DROP_PER_TURN * (tsumo2 - self._t2))
        self._t1, self._t2 = tsumo1, tsumo2
        return float(max(-100.0, min(100.0, (self.inc2 - self.inc1) * FORECAST_SCALE)))


class ThreatTracker:
    """threat(reach火力=重い)の計算を間引くキャッシュ。

    reach_fire_power は満杯盤面で高コスト。threat は連鎖ビルドに従い緩やかに
    変化するため、every フレームに1回だけ再計算し間は前回値を再利用する。
    毎フレーム呼んでも reach 実計算は 1/every に削減。
    """

    def __init__(self, every: int = 9) -> None:  # ~0.3s @30fps
        self._every = max(1, every)
        self._last = 0.0
        self._n = 0

    def update(self, b1: Board, b2: Board, sp1, sp2, elapsed: float) -> float:
        if self._n % self._every == 0:
            self._last = _threat(b1, b2, sp1, sp2, elapsed)
        self._n += 1
        return self._last


class HeavyAdvCache:
    """重い盤面由来計算(_score_advantage の current_max_chain 等 + threat の reach)を
    ~0.3s 毎に計算してキャッシュ。盤面は STABLE 時しか変わらないため精度低下は最小。

    毎フレーム呼んでも重い simulate 群は 1/every に削減(オーバーレイ実用速度の要)。
    圧力(board_ojama)・得点リード(score)・会計は安価なので呼出側で毎フレーム更新のまま。
    """

    def __init__(self, model, every: int = 9) -> None:  # ~0.3s @30fps
        self._model = model
        self._every = max(1, every)
        self._n = 0
        self._adv = 0.0
        self._threat = 0.0
        self._drivers: list[tuple[str, float]] = []

    def update(self, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
               sp1, sp2, elapsed: float) -> tuple[float, float, list[tuple[str, float]]]:
        """(モデル有利不利, threat, 主要ドライバ) を返す(間引きキャッシュ)。"""
        if self._n % self._every == 0:
            self._adv, _, self._drivers = _score_advantage(self._model, b1, b2, snap)
            self._threat = _threat(b1, b2, sp1, sp2, elapsed)
        self._n += 1
        return self._adv, self._threat, self._drivers


def _draw_graph(
    d: "ImageDraw.ImageDraw", history: list[tuple[float, float]],
    t_rel: float, total: float,
) -> None:
    """リアルタイム評価値グラフ (将棋風) を下部の黒帯に描画。進行に合わせ伸びる。"""
    gx0, gx1, gy0, gy1 = 40, OUT_W - 40, OUT_H + 26, CANVAS_H - 12
    gyc = (gy0 + gy1) // 2
    gw, gh = gx1 - gx0, gy1 - gy0
    d.rectangle([gx0 - 4, gy0 - 20, gx1 + 4, gy1 + 4], fill=(0, 0, 0, 150))
    d.text((gx0, gy0 - 20), "有利不利グラフ (0=互角 上1P/下2P)", font=_font(15),
           fill=(255, 255, 255))
    total = max(total, 1.0)

    def _px(t: float) -> int:
        return int(gx0 + (t / total) * gw)

    def _py(a: float) -> int:
        return int(gyc - (max(-100, min(100, a)) / 100.0) * (gh / 2))
    for t, a in history:  # 各点で中央線から値まで縦線 (塗り面風)
        col = (90, 140, 220) if a >= 0 else (210, 90, 90)
        d.line([(_px(t), gyc), (_px(t), _py(a))], fill=col, width=2)
    d.line([(gx0, gyc), (gx1, gyc)], fill=(255, 255, 255), width=1)
    ph = _px(t_rel)  # 再生ヘッド
    d.line([(ph, gy0), (ph, gy1)], fill=(255, 255, 0), width=2)
    d.rectangle([gx0, gy0, gx1, gy1], outline=(255, 255, 255), width=1)


def _draw_overlay(
    frame: np.ndarray, adv: float, p1: float,
    drivers: list[tuple[str, float]], waiting: bool,
    history: list[tuple[float, float]], t_rel: float, total: float,
) -> np.ndarray:
    """有利不利バー + リアルタイムグラフを描画。下部にグラフ専用黒帯を足して返す。"""
    canvas = Image.new("RGB", (OUT_W, CANVAS_H), (12, 12, 16))
    canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
    img = canvas
    d = ImageDraw.Draw(img, "RGBA")
    bar_w, bar_h, cx, top = 720, 34, OUT_W // 2, 54
    x0 = cx - bar_w // 2
    d.rectangle([x0, top, cx, top + bar_h], fill=(70, 110, 200, 180))   # 1P側(青)
    d.rectangle([cx, top, x0 + bar_w, top + bar_h], fill=(200, 80, 80, 180))  # 2P側(赤)
    d.rectangle([x0, top, x0 + bar_w, top + bar_h], outline=(255, 255, 255), width=2)
    d.text((x0, top - 30), "有利不利オーバーレイ (試作・tier1軽量モデル)",
           font=_font(20), fill=(255, 255, 0))
    if waiting:
        d.text((cx - 90, top + 4), "STABLE 待ち", font=_font(22), fill=(255, 255, 255))
    else:
        mx = int(cx - (max(-100, min(100, adv)) / 100.0) * (bar_w // 2))  # adv>0=1P=左
        d.rectangle([mx - 3, top - 6, mx + 3, top + bar_h + 6], fill=(255, 255, 255))
        verdict = ("互角" if abs(adv) < EVEN_THRESHOLD
                   else f"{'1P' if adv > 0 else '2P'} 有利  {abs(adv):.0f}")
        d.text((cx - 70, top + 4), verdict, font=_font(24), fill=(0, 0, 0))
        d.text((x0 - 34, top + 4), "1P", font=_font(22), fill=(150, 200, 255))
        d.text((x0 + bar_w + 6, top + 4), "2P", font=_font(22), fill=(255, 180, 180))
        d.text((x0, top + bar_h + 8),
               f"勝率  1P {p1 * 100:.0f}%   /   2P {(1 - p1) * 100:.0f}%",
               font=_font(20), fill=(255, 255, 255))
        dl = "  ".join(f"{JP_LABEL[c]}差 {v:+.2f}" for c, v in drivers)
        d.text((x0, top + bar_h + 34), f"主因: {dl}", font=_font(16), fill=(230, 230, 180))
    if history:
        _draw_graph(d, history, t_rel, total)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate(video: Path, out: Path, max_sec: float, sample_interval: float,
             start_sec: float = 0.0, end_sec: float = 0.0,
             exclude_video: str | None = None, warmup_sec: float = 0.0) -> int:
    """有利不利オーバーレイ動画を生成。書き出しフレーム数を返す。

    start_sec: 書き出し開始秒 (ゲームの真の開始=スコア0の瞬間)。
    warmup_sec: start_sec の何秒前から「処理だけ」始めるか (状態機械/会計の初期化用。
        この区間は認識を通すが動画には書き出さない)。
    end_sec: 書き出し終了秒。
    """
    model = _train_model(exclude_video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] open失敗: {video}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    proc_frame = int(max(0.0, start_sec - warmup_sec) * fps)  # 処理開始 (ウォームアップ込み)
    write_frame = int(start_sec * fps)                        # 書き出し開始 (ゲーム頭)
    if end_sec > 0:
        n = min(n, int(end_sec * fps))
    elif max_sec > 0:
        n = min(n, write_frame + int(max_sec * fps))
    start_frame = proc_frame
    if proc_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, proc_frame)
        print(f"[seek] 処理開始 {proc_frame / fps:.1f}s / 書き出し開始 "
              f"{write_frame / fps:.1f}s (ウォームアップ {warmup_sec:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (OUT_W, CANVAS_H))
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    import re
    m = re.search(r"(v\d+|video_\d+)", video.name)
    if m and hasattr(pipe, "set_video_id"):
        pipe.set_video_id(m.group(1))
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0
    p1_last = 0.5
    drivers: list[tuple[str, float]] = []
    ptracker = PressureTracker()
    fctracker = RealtimeForecastTracker()
    hcache = HeavyAdvCache(model)
    history: list[tuple[float, float]] = []  # (ゲーム開始からの秒, 有利不利) 累積
    total_dur = max(1.0, (n / fps) - start_sec)  # グラフ横軸の総尺
    step = max(1, int(round(sample_interval * fps)))
    written = 0
    for fi in range(start_frame, n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        # お邪魔会計は密な駆動が必須のため pipe.update / _drive_ojama は毎フレーム。
        r = pipe.update(fi, t, frame)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if b1 is not None and b2 is not None:
            # 重い盤面由来(モデルadv/threat)はキャッシュ間引き、安価な圧力/リードは毎フレーム
            model_adv, threat, drivers = hcache.update(
                b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
            pres = ptracker.update(iv.board_ojama_count(b1).raw,
                                   iv.board_ojama_count(b2).raw)
            fc = fctracker.update(r.p1.score, r.p2.score,
                                  pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))  # (M3改B)配送予告
            adv = (W_PRESSURE * pres + W_FORECAST * fc
                   + W_MODEL * model_adv + W_THREAT * threat)
            p1 = 0.5 + adv / 200.0  # 表示用勝率もブレンド後に整合
            adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
            p1_last = EMA_ALPHA * p1 + (1 - EMA_ALPHA) * p1_last
            if fi >= write_frame and fi % step == 0:
                history.append((t - start_sec, adv_ema))
        if fi < write_frame:
            continue  # ウォームアップ区間は書き出さない
        waiting = b1 is None or b2 is None
        writer.write(_draw_overlay(frame, adv_ema, p1_last, drivers, waiting,
                                   history, t - start_sec, total_dur))
        written += 1
        if written % 300 == 0:
            print(f"  ... {written} frames (t={t:.1f}s adv={adv_ema:+.0f})")
    cap.release(); writer.release()
    print(f"[done] {written} frames -> {out}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/frames/video_124_4min.mp4")
    ap.add_argument("--out", default="data/indicators_v2/overlay/advantage_v124.mp4")
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--warmup-sec", type=float, default=0.0,
                    help="start_sec の何秒前から処理だけ始めるか (状態機械初期化用)")
    ap.add_argument("--exclude-video", default=None,
                    help="学習から除外する動画ID (対象動画のリーク防止)")
    ap.add_argument("--sample-interval", type=float, default=0.15)
    a = ap.parse_args()
    generate(Path(a.video), Path(a.out), a.max_sec, a.sample_interval,
             start_sec=a.start_sec, end_sec=a.end_sec,
             exclude_video=a.exclude_video, warmup_sec=a.warmup_sec)


if __name__ == "__main__":
    main()
