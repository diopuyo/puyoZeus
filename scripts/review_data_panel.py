"""E11: 評価器の状態を読み取るレビュー専用表示とフレームCSV。"""
from __future__ import annotations

import csv
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.board import BOARD_COLS
from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
from src.exchange_event_landing import logit_mean
from src.scoring import score_to_ojama

REFERENCE_WIDTH = 1280
PANEL_HEIGHT = 600
FONT_SIZE = 18
LINE_HEIGHT = 28
PADDING = 16
PERCENT = 100
TEXT_CACHE_SIZE = 512
PANEL_BACKGROUND = (15, 21, 32)
SIDE_LABELS = ("1P", "2P")
PROBABILITY_SOURCES = ("G_fe", "S1", "S3_provisional", "S3")
PANEL_D_COLUMNS = (
    "diff_board_ojama_count", "board_color_puyo_total", "ukeyasusa",
    "saturation_chain_upper", "current_max_chain", "diff_death_margin",
    "ignition_point_count", "sub_chain_count", "diff_conn_pair_count",
    "diff_column_bumpiness", "board_ojama_count", "diff_max_column_height",
)
PROJECTION_FIELDS = (
    "incoming", "hands", "near_future_send", "required_cancel", "resolving_send",
    "overflow_rows", "verified_attack", "optimistic_send",
)
SIDE_FIELDS = (
    "state", "confirmed_sec", "chain_id", "exchange_step", "chain_count",
    "formula_total", "formula_send", "confirmed_send", "net_send", "forecast",
    "confirmed_incoming", "confirmed_incoming_rows", "incoming_rows",
    "unavoidable_death", "death_reason", *PROJECTION_FIELDS,
    *(f"k{k}" for k in range(1, 6)), *(f"d_{name}" for name in D_COLUMNS),
)
COMMON_FIELDS = (
    "frame_index", "t_sec", "game_idx", "source", "exchange_id", "exchange_step",
    *(f"p1_{name}" for name in PROBABILITY_SOURCES), "p1_landing_gfe",
    "p1_combined", "p1_selected", "p1_display", "display_adv",
    "projection_sec", "death_evidence_sec", "prefire_sec",
)
CSV_FIELDS = (*COMMON_FIELDS, *(f"{side}_{name}" for side in SIDE_LABELS for name in SIDE_FIELDS))


def panel_size(width: int, height: int) -> tuple[int, int]:
    """幅1280で18pxを確保し、元のキャンバスの下へ追加する。"""
    return width, height + round(PANEL_HEIGHT * width / REFERENCE_WIDTH)


def _send(score: float | None, elapsed: float) -> int | None:
    """評価器と同じ換算を使い、未観測をゼロで埋めない。"""
    return None if score is None else math.floor(score_to_ojama(score, elapsed_sec=elapsed).ojama_count)


def _death_reason(projection: dict, death: dict | None, side: str, idx: int) -> str:
    """保存済みの判定根拠に短いコードを付ける。判定は行わない。"""
    if death and side in death.get("dead_sides", ()):
        return "HELD_SHORTFALL" if death is not projection else "RESPONSE_SHORTFALL"
    if side in projection.get("dead_sides", ()):
        return "RESPONSE_SHORTFALL"
    if not projection:
        return "NOT_EVALUATED"
    if projection["incoming"][idx] <= 0:
        return "NO_INCOMING"
    if projection["near_future_send"][idx] is None:
        return "LANDING_SURVIVES"
    return "NOT_CONFIRMED"


def _chain_data(record: Any, tracker: Any, idx: int) -> dict[str, Any]:
    """最新連鎖の式と、撃ち合い全体の確定送り量を区別する。"""
    chains = [] if record is None else [c for c in record.chains if c.side == SIDE_LABELS[idx]]
    chain = chains[-1] if chains else None
    ready = [c for c in chains if c.score_ready_sec is not None and c.score_delta is not None]
    total = sum(c.score_delta for c in ready) if record and len(ready) == len(chains) else None
    resolved = next((c for c in (*tracker.resolver.active(), *tracker.resolver.resolved())
                     if chain is not None and c.chain_id == chain.chain_id), None)
    return dict(chain_id=chain.chain_id if chain else None, exchange_step=len(chains),
                chain_count=getattr(resolved, "step_count", None),
                formula_total=chain.formula_total if chain else None,
                formula_send=_send(chain.formula_total, tracker._score_elapsed) if chain else None,
                confirmed_send=_send(total, tracker._score_elapsed))


def _side_data(overlay: Any, result: Any, snapshot: Any, record: Any,
               projection: dict, idx: int) -> dict[str, Any]:
    """既存の応手計算結果と発火前K特徴をそのまま取り出す。"""
    tracker, landing = overlay.tracker, overlay._landing_projection
    label, side = SIDE_LABELS[idx], (result.p1, result.p2)[idx]
    values = _chain_data(record, tracker, idx)
    history = overlay._history[idx]
    values.update(state=side.state.name, confirmed_sec=history[-1].t_sec if history else None,
                  forecast=getattr(snapshot, f"forecast_p{idx + 1}"),
                  unavoidable_death=label in (landing.death or projection).get("dead_sides", ()),
                  death_reason=_death_reason(projection, landing.death, label, idx))
    for name in PROJECTION_FIELDS:
        values[name] = projection.get(name, (None, None))[idx]
    incoming = values["incoming"]
    values["incoming_rows"] = None if incoming is None else incoming / BOARD_COLS
    if record is not None and tracker.firing is not None:
        for k in range(1, 6):
            values[f"k{k}"] = float(tracker.firing.prefire_sides[idx, SIDE_COLUMNS.index(f"k{k}")])
    return values


def _confirmed_amounts(sides: list[dict]) -> None:
    """両側の確定済み送り量が揃った場合だけ、確定純量を表示する。"""
    for idx, side in enumerate(sides):
        own, opp = side["confirmed_send"], sides[1 - idx]["confirmed_send"]
        net = None if own is None or opp is None else own - opp
        side["net_send"] = net
        incoming = None if net is None else max(0, -net)
        side["confirmed_incoming"] = incoming
        side["confirmed_incoming_rows"] = None if incoming is None else incoming / BOARD_COLS


def build_review_row(overlay: Any, result: Any, snapshot: Any, frame_index: int,
                     t_sec: float, game_idx: int, display_p1: float, display_adv: float,
                     board_features: Callable, side_features: Callable) -> dict[str, Any]:
    """評価・平滑化完了後の一時点を、CSVと描画で共有する。"""
    tracker, landing = overlay.tracker, overlay._landing_projection
    record = tracker.current or landing.death_record
    projection = landing.last if record and landing.identity == (game_idx, record.exchange_id) else None
    projection = projection or {}
    row = dict.fromkeys(CSV_FIELDS)
    row.update(frame_index=frame_index, t_sec=t_sec, game_idx=game_idx, source=tracker.source,
               exchange_id=record.exchange_id if record else None,
               exchange_step=len(record.chains) if record else 0,
               p1_G_fe=tracker._static_probability, p1_selected=tracker.probability,
               p1_display=display_p1, display_adv=display_adv,
               projection_sec=projection.get("t_sec"),
               death_evidence_sec=(landing.death or {}).get("t_sec"),
               prefire_sec=record.trigger_sec if record else None)
    for value in record.values if record else ():
        if value["source"] in PROBABILITY_SOURCES:
            row[f"p1_{value['source']}"] = value["p1"]
    row["p1_landing_gfe"] = projection.get("gfe_p1")
    row["p1_combined"] = (logit_mean(projection["base_p1"], projection["gfe_p1"])
                          if projection else tracker.probability)
    sides = [_side_data(overlay, result, snapshot, record, projection, idx) for idx in range(2)]
    _confirmed_amounts(sides)
    _add_design_features(sides, overlay, snapshot, board_features, side_features)
    for label, values in zip(SIDE_LABELS, sides):
        row.update({f"{label}_{key}": value for key, value in values.items()})
    return row


def _add_design_features(sides: list[dict], overlay: Any, snapshot: Any,
                         board_features: Callable, side_features: Callable) -> None:
    """STABLE履歴の両側D列を既存変換で作る。絶対量を符号反転しない。"""
    if not all(overlay._history):
        return
    grids = [history[-1].board._grid for history in overlay._history]
    bases = [board_features(g.tobytes(), g.shape, g.dtype.str) for g in grids]
    for idx, side in enumerate(sides):
        net = snapshot.net_balance_capped * (1 if idx == 0 else -1)
        design = side_features(bases[idx], bases[1 - idx], net, side["forecast"])
        side.update({f"d_{name}": design[name] for name in D_COLUMNS})


class ReviewCsv:
    """暖機を除く出力フレームを逐次保存し、メモリ増加を防ぐ。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8-sig", newline="", buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        """画面と同じ未丸め値を1行保存する。"""
        self.writer.writerow(row)

    def close(self) -> None:
        """最後の行まで確実に閉じる。"""
        self.stream.close()


def _number(value: Any) -> str:
    """未観測値は空白やゼロと区別する。"""
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "--"
    return f"{value:.3g}" if isinstance(value, (float, int)) else str(value)


def panel_lines(row: dict, labels: dict[str, str]) -> tuple[list[str], list[list[str]]]:
    """共通行と左右同じ構成の行を作り、CSVの値だけから描く。"""
    def probability(key: str) -> str:
        value = row.get(key)
        return "--" if value is None else f"{PERCENT * value:.1f}%"
    common = [f"REVIEW  {row['t_sec']:.2f}s   source: {row['source']}   "
              f"撃ち合いID: {_number(row['exchange_id'])}  段: {row['exchange_step']}",
              "1P勝率  " + "   ".join(f"{label} {probability(key)}" for label, key in (
                  ("G_fe", "p1_G_fe"), ("S1", "p1_S1"), ("暫定S3", "p1_S3_provisional"), ("確定S3", "p1_S3"))),
              "1P勝率  " + "   ".join(f"{label} {probability(key)}" for label, key in (
                  ("仮想着弾G_fe", "p1_landing_gfe"), ("合成後", "p1_combined"),
                  ("採用値", "p1_selected"), ("表示値", "p1_display")))]
    columns = [_side_lines(row, side, labels) for side in SIDE_LABELS]
    return common, columns


def _side_lines(row: dict, side: str, labels: dict[str, str]) -> list[str]:
    """長い項目名も左右幅内へ収まるよう、物理量を複数行に分ける。"""
    def value(name: str) -> str:
        return _number(row.get(f"{side}_{name}"))
    lines = [f"{side}  {value('state')}   撃ち合い内の段 {value('exchange_step')}",
             f"予告 {value('forecast')}個  確定受け {value('confirmed_incoming')}個 / {value('confirmed_incoming_rows')}段",
             f"相殺後純送り {value('net_send')}個  未着弾 {value('incoming')}個 / {value('incoming_rows')}段",
             f"連鎖 {value('chain_count')}段  式累積 {value('formula_total')}点  換算 {value('formula_send')}個",
             f"応手 n={value('hands')}手  近未来火力 {value('near_future_send')}個  必要相殺 {value('required_cancel')}個",
             f"消去中火力 {value('resolving_send')}個  楽観火力 {value('optimistic_send')}個",
             f"回避不能死 {value('unavoidable_death')}  {value('death_reason')}",
             "近未来火力(発火前・0〜1)  " + " / ".join(f"k{k}:{value(f'k{k}')}" for k in range(1, 4)),
             "近未来火力(発火前・0〜1)  " + " / ".join(f"k{k}:{value(f'k{k}')}" for k in range(4, 6)),
             "D指標（最新STABLE・学習入力値）"]
    for idx in range(0, len(PANEL_D_COLUMNS), 2):
        lines.append("   ".join(f"{labels.get(name, name)} {value('d_' + name)}"
                                for name in PANEL_D_COLUMNS[idx:idx + 2]))
    return lines


def draw_review_panel(frame: np.ndarray, row: dict, labels: dict[str, str],
                      font_loader: Callable[[int], ImageFont.ImageFont]) -> np.ndarray:
    """元画像には触れず下帯だけを描画し、同じ幅の画像を連結する。"""
    height, width = frame.shape[:2]
    scale = width / REFERENCE_WIDTH
    band = Image.new("RGB", (width, panel_size(width, height)[1] - height), PANEL_BACKGROUND)
    draw, font_size = ImageDraw.Draw(band), round(FONT_SIZE * scale)
    common, columns = panel_lines(row, labels)
    pad, step = round(PADDING * scale), round(LINE_HEIGHT * scale)
    for index, line in enumerate(common):
        tile = _text_tile(line, font_size, font_loader, (242, 244, 248), width - 2 * pad, step)
        band.paste(tile, (pad, pad + index * step))
    top = pad + len(common) * step
    for side, lines in enumerate(columns):
        x = pad + side * width // 2
        color = (123, 198, 255) if side == 0 else (255, 176, 143)
        for index, line in enumerate(lines):
            tile = _text_tile(line, font_size, font_loader, color, width // 2 - 2 * pad, step)
            band.paste(tile, (x, top + index * step))
    draw.line((width // 2, top, width // 2, band.height - pad), fill=(70, 80, 96))
    return np.concatenate((frame, cv2.cvtColor(np.asarray(band), cv2.COLOR_RGB2BGR)), axis=0)


@lru_cache(maxsize=TEXT_CACHE_SIZE)
def _text_tile(text: str, size: int, font_loader: Callable, color: tuple[int, int, int],
               width: int, height: int) -> Image.Image:
    """同じ文字列のラスタライズだけを再用し、評価データを保持しない。"""
    tile = Image.new("RGB", (width, height), PANEL_BACKGROUND)
    ImageDraw.Draw(tile).text((0, 0), text, font=font_loader(size), fill=color)
    return tile
