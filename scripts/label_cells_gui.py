"""ぷよぷよ cell ラベリング GUI (= 日本語 UI 重視版).

操作:
    ① 赤枠の cell をクリックして選択
    ② 色キー (R/Y/P/B/G/O/E) または色パレットで色を設定
    ③ 全 cell の赤枠がなくなったら Space キーで次 frame
    ④ 全 frame 完了で Q キーで終了 (= 自動保存済)

ラベルデータ保存先: data/pseudo_labels/<video_id>/cell.jsonl
CNN 再訓練: phase_i_fine_tune.py --component cell_color
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN,
    COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.image_reader import (
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, ColorClassifier,
)
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL, PseudoLabelSample,
)
from scripts.label_cells import (
    extract_cell_patches, extract_cell_patches_for_display,
    predict_cells, sample_frames,
    save_frame_labels, FrameLabels,
    VISIBLE_ROW_START, VISIBLE_ROW_END, VISIBLE_ROW_COUNT,
    COLOR_TO_BGR, COLOR_TO_NAME, KEY_TO_COLOR,
    KEY_TO_FRAME_STATE, KEY_EFFECT_TOGGLE,
    FRAME_STATE_STABLE, FRAME_STATE_CHAIN,
    FRAME_STATE_OJAMA_FALL, FRAME_STATE_TSUMO_FALL,
    FRAME_STATE_MENU, FRAME_STATE_SKIP,
)


# 動作状態 → 日本語名 + 表示色 (= 排他選択)
FRAME_STATE_INFO: dict[str, tuple[str, str]] = {
    FRAME_STATE_STABLE: ("STABLE (通常)", "#44dd44"),
    FRAME_STATE_CHAIN: ("CHAIN (連鎖中)", "#ff8844"),
    FRAME_STATE_OJAMA_FALL: ("OJAMA_FALL (おじゃま落下)", "#bbbbbb"),
    FRAME_STATE_TSUMO_FALL: ("TSUMO_FALL (ツモ落下)", "#88ccff"),
    FRAME_STATE_MENU: ("MENU (試合外)", "#888888"),
    FRAME_STATE_SKIP: ("SKIP (学習から除外)", "#ff4444"),
}


# ============================
# UI 定数
# ============================

CELL_PX: int = 64           # 見切れ防止のため 96 → 64 に縮小
GRID_W: int = CELL_PX * BOARD_COLS  # 384
GRID_H: int = CELL_PX * VISIBLE_ROW_COUNT  # 768
PREVIEW_PX: int = 200
PALETTE_BTN_W: int = 90
PALETTE_BTN_H: int = 50
WINDOW_BG: str = "#1e1e2e"
PANEL_BG: str = "#2a2a3a"
TEXT_FG: str = "#e0e0e0"
ACCENT_FG: str = "#88bbff"
WARN_FG: str = "#ff8888"
HELP_FG: str = "#ffee88"

# 日本語色名
COLOR_TO_JP: dict[int, str] = {
    COLOR_EMPTY: "空白",
    COLOR_RED: "赤",
    COLOR_BLUE: "青",
    COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄",
    COLOR_PURPLE: "紫",
    COLOR_OJAMA: "おじゃま",
    COLOR_UNKNOWN: "不明",
}

# 色ボタン定義 (= 色, 日本語名, 英文字キー, hex 表示色)
COLOR_BUTTONS: list[tuple[int, str, str, str]] = [
    (COLOR_RED, "赤", "R", "#ff4444"),
    (COLOR_YELLOW, "黄", "Y", "#ffee44"),
    (COLOR_GREEN, "緑", "G", "#44dd44"),
    (COLOR_BLUE, "青", "B", "#4488ff"),
    (COLOR_PURPLE, "紫", "P", "#cc66ff"),
    (COLOR_OJAMA, "おじゃま", "O", "#bbbbbb"),
    (COLOR_EMPTY, "空白", "E", "#444444"),
    # UNKNOWN: 落下中のぷよ・演出で見えない cell 用. 学習から除外される.
    (COLOR_UNKNOWN, "不明", "X", "#e0c060"),
]


# ============================
# 画像変換ヘルパー
# ============================


def bgr_to_photo(
    bgr: np.ndarray, target_size: tuple[int, int] | None = None,
) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ============================
# メイン GUI
# ============================


class CellLabelerGUI:
    def __init__(
        self,
        root: tk.Tk,
        frame_labels: list[FrameLabels],
        store: LabelStore,
        video_id: str,
    ) -> None:
        self.root = root
        self.frames = frame_labels
        self.store = store
        self.video_id = video_id
        self.cur_idx = 0
        self.selected: tuple[str, int, int] = ("1P", VISIBLE_ROW_END - 1, 0)
        self.saved_frames = 0
        self.undo_stack: list[tuple[int, str, int, int, int, bool]] = []
        self._image_refs: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self._bind_keys()
        self._render_all()

    # --------------------------------------------------------------
    # UI 構築
    # --------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("ぷよ cell ラベリング tool")
        self.root.configure(bg=WINDOW_BG)
        # 画面解像度に応じて自動サイズ調整 (= 見切れ防止)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win_w = min(1500, sw - 100)
        win_h = min(1000, sh - 80)
        self.root.geometry(f"{win_w}x{win_h}+50+30")

        # ① ステータスバー (= 最下部、 最初に pack で底に固定)
        self.status_var = tk.StringVar()
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bg=PANEL_BG, fg=TEXT_FG, anchor="w",
            padx=12, pady=6, font=("Yu Gothic UI", 10),
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ② 使い方ガイド (= 最上部、 常時表示)
        guide_frame = tk.Frame(self.root, bg="#3a2a3a")
        guide_frame.pack(side=tk.TOP, fill=tk.X)
        guide_text = (
            "【使い方】 "
            "① 動作状態を 1 キー (n通常/c連鎖/t落下/j おじゃま/m メニュー/k skip) "
            "+ 演出があれば f キー → 2P は Shift+  "
            "→ ② cell の赤枠を色キー (R/Y/P/B/G/O/E/X)  "
            "→ ③ Space で次フレーム"
        )
        tk.Label(
            guide_frame, text=guide_text, bg="#3a2a3a", fg=HELP_FG,
            font=("Yu Gothic UI", 10, "bold"), padx=10, pady=6, anchor="w",
        ).pack(fill=tk.X)

        # ②.5 フレーム状態表示 (= X-2、 1P/2P 独立、 常時)
        self.state_frame = tk.Frame(self.root, bg="#2a3a3a")
        self.state_frame.pack(side=tk.TOP, fill=tk.X)
        # 1P state バー
        self.lbl_frame_state_1p = tk.Label(
            self.state_frame, text="", bg="#2a3a3a",
            fg=HELP_FG, font=("Yu Gothic UI", 11, "bold"),
            padx=10, pady=4, anchor="w",
        )
        self.lbl_frame_state_1p.pack(side=tk.TOP, fill=tk.X)
        # 2P state バー
        self.lbl_frame_state_2p = tk.Label(
            self.state_frame, text="", bg="#2a3a3a",
            fg=HELP_FG, font=("Yu Gothic UI", 11, "bold"),
            padx=10, pady=4, anchor="w",
        )
        self.lbl_frame_state_2p.pack(side=tk.TOP, fill=tk.X)
        # 旧 API 互換 (描画で使うかも)
        self.lbl_frame_state = self.lbl_frame_state_1p
        # frame state 切替ボタン (= 1P 行 + 2P 行)
        state_btn_outer = tk.Frame(self.state_frame, bg="#2a3a3a")
        state_btn_outer.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Label(
            state_btn_outer, text="1P", bg="#2a3a3a", fg=ACCENT_FG,
            font=("Yu Gothic UI", 10, "bold"), width=4,
        ).grid(row=0, column=0)
        tk.Label(
            state_btn_outer, text="2P", bg="#2a3a3a", fg=ACCENT_FG,
            font=("Yu Gothic UI", 10, "bold"), width=4,
        ).grid(row=1, column=0)
        state_shortcuts: list[tuple[str, str]] = [
            (FRAME_STATE_STABLE, "N\n通常"),
            (FRAME_STATE_CHAIN, "C\n連鎖"),
            (FRAME_STATE_OJAMA_FALL, "J\nおじゃま"),
            (FRAME_STATE_TSUMO_FALL, "T\nツモ落下"),
            (FRAME_STATE_MENU, "M\nメニュー"),
            (FRAME_STATE_SKIP, "K\nSkip"),
        ]
        for col_idx, (st, lbl) in enumerate(state_shortcuts):
            _name, color = FRAME_STATE_INFO[st]
            # 1P ボタン (= 1 行目)
            btn1 = tk.Button(
                state_btn_outer, text=lbl, bg=color, fg="#000000",
                font=("Yu Gothic UI", 9, "bold"),
                width=8, height=2,
                command=lambda s=st: self._set_frame_state_1p(s),
            )
            btn1.grid(row=0, column=col_idx + 1, padx=2)
            # 2P ボタン (= 2 行目、 Shift+キー)
            btn2 = tk.Button(
                state_btn_outer, text=f"Sh+{lbl[0]}\n{lbl.split(chr(10))[1]}",
                bg=color, fg="#000000",
                font=("Yu Gothic UI", 8, "bold"),
                width=8, height=2,
                command=lambda s=st: self._set_frame_state_2p(s),
            )
            btn2.grid(row=1, column=col_idx + 1, padx=2)
        # 演出 toggle ボタン (= 動作状態と独立、 2 軸目)
        effect_col = len(state_shortcuts) + 1
        self.btn_effect_1p = tk.Button(
            state_btn_outer, text="F\n演出",
            bg="#ffee44", fg="#000000",
            font=("Yu Gothic UI", 9, "bold"),
            width=8, height=2,
            command=self._toggle_effect_1p,
        )
        self.btn_effect_1p.grid(row=0, column=effect_col, padx=2)
        self.btn_effect_2p = tk.Button(
            state_btn_outer, text="Sh+F\n演出",
            bg="#ffee44", fg="#000000",
            font=("Yu Gothic UI", 8, "bold"),
            width=8, height=2,
            command=self._toggle_effect_2p,
        )
        self.btn_effect_2p.grid(row=1, column=effect_col, padx=2)

        # ③ ツールバー (= ガイドの下、 大きいボタン)
        toolbar = tk.Frame(self.root, bg=PANEL_BG)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_prev = tk.Button(
            toolbar, text="◀ 前フレーム", bg=PANEL_BG, fg=TEXT_FG,
            font=("Yu Gothic UI", 10),
            command=self._go_prev, padx=10, pady=6,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=4, pady=6)

        self.btn_save_next = tk.Button(
            toolbar, text="保存して次へ ▶ (Space)", bg="#4488ff", fg="#ffffff",
            font=("Yu Gothic UI", 11, "bold"),
            command=self._save_and_next, padx=14, pady=6,
        )
        self.btn_save_next.pack(side=tk.LEFT, padx=4, pady=6)

        self.btn_save = tk.Button(
            toolbar, text="保存のみ (Ctrl+S)", bg=PANEL_BG, fg=TEXT_FG,
            font=("Yu Gothic UI", 10),
            command=self._save_only, padx=10, pady=6,
        )
        self.btn_save.pack(side=tk.LEFT, padx=4, pady=6)

        self.btn_undo = tk.Button(
            toolbar, text="元に戻す (Ctrl+Z)", bg=PANEL_BG, fg=TEXT_FG,
            font=("Yu Gothic UI", 10),
            command=self._undo, padx=10, pady=6,
        )
        self.btn_undo.pack(side=tk.LEFT, padx=4, pady=6)

        self.btn_quit = tk.Button(
            toolbar, text="終了 (Q)", bg=PANEL_BG, fg=WARN_FG,
            font=("Yu Gothic UI", 10),
            command=self._quit, padx=10, pady=6,
        )
        self.btn_quit.pack(side=tk.RIGHT, padx=4, pady=6)

        self.lbl_progress = tk.Label(
            toolbar, text="", bg=PANEL_BG, fg=ACCENT_FG,
            font=("Yu Gothic UI", 11, "bold"),
        )
        self.lbl_progress.pack(side=tk.RIGHT, padx=12)

        # ④ メイン領域 (= 残りスペース全部)
        main = tk.Frame(self.root, bg=WINDOW_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 1P canvas
        frm_1p = tk.LabelFrame(
            main, text="1P 盤面", bg=WINDOW_BG, fg=ACCENT_FG,
            font=("Yu Gothic UI", 11, "bold"),
        )
        frm_1p.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.Y)
        self.canvas_1p = tk.Canvas(
            frm_1p, width=GRID_W, height=GRID_H,
            bg="#000000", highlightthickness=0,
        )
        self.canvas_1p.pack(padx=4, pady=4)
        self.canvas_1p.bind("<Button-1>", lambda e: self._on_cell_click("1P", e))

        # 2P canvas
        frm_2p = tk.LabelFrame(
            main, text="2P 盤面", bg=WINDOW_BG, fg=ACCENT_FG,
            font=("Yu Gothic UI", 11, "bold"),
        )
        frm_2p.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.Y)
        self.canvas_2p = tk.Canvas(
            frm_2p, width=GRID_W, height=GRID_H,
            bg="#000000", highlightthickness=0,
        )
        self.canvas_2p.pack(padx=4, pady=4)
        self.canvas_2p.bind("<Button-1>", lambda e: self._on_cell_click("2P", e))

        # 右ペイン
        right = tk.Frame(main, bg=WINDOW_BG)
        right.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.BOTH, expand=True)

        # 拡大プレビュー
        prev_frm = tk.LabelFrame(
            right, text="選択中の cell (拡大)", bg=WINDOW_BG, fg=ACCENT_FG,
            font=("Yu Gothic UI", 10, "bold"),
        )
        prev_frm.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.canvas_preview = tk.Canvas(
            prev_frm, width=PREVIEW_PX, height=PREVIEW_PX,
            bg="#000000", highlightthickness=0,
        )
        self.canvas_preview.pack(padx=4, pady=4)
        self.lbl_preview_info = tk.Label(
            prev_frm, text="cell を選択してください", bg=WINDOW_BG, fg=TEXT_FG,
            justify=tk.LEFT, font=("Yu Gothic UI", 9),
        )
        self.lbl_preview_info.pack(padx=4, pady=2, anchor="w")

        # 色パレット
        pal_frm = tk.LabelFrame(
            right, text="色パレット (クリック または キー)",
            bg=WINDOW_BG, fg=ACCENT_FG, font=("Yu Gothic UI", 10, "bold"),
        )
        pal_frm.pack(side=tk.TOP, fill=tk.X, pady=4)
        pal_grid = tk.Frame(pal_frm, bg=WINDOW_BG)
        pal_grid.pack(padx=4, pady=4)
        for i, (color, jp, key, hex_bg) in enumerate(COLOR_BUTTONS):
            r_, c_ = divmod(i, 2)
            btn = tk.Button(
                pal_grid, text=f"{jp}\n[{key}]", bg=hex_bg, fg="#000000",
                width=8, height=2, font=("Yu Gothic UI", 10, "bold"),
                command=lambda c=color: self._set_selected_color(c),
            )
            btn.grid(row=r_, column=c_, padx=3, pady=3)

        # ヘルプ
        help_frm = tk.LabelFrame(
            right, text="キーボードショートカット",
            bg=WINDOW_BG, fg=ACCENT_FG,
            font=("Yu Gothic UI", 10, "bold"),
        )
        help_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)
        help_txt = (
            "■ 色を設定:\n"
            "  R=赤  Y=黄  P=紫\n"
            "  B=青  G=緑  O=おじゃま\n"
            "  E=空白  X=不明(落下中)\n"
            "  数字 1-5 / 9 / 0 も可\n\n"
            "■ 動作状態 1P (排他):\n"
            "  n=STABLE c=CHAIN\n"
            "  j=おじゃま落下\n"
            "  t=ツモ落下 m=MENU\n"
            "  k=Skip\n\n"
            "■ 動作状態 2P (Shift+):\n"
            "  N C J T M K\n"
            "  (= 大文字で 2P)\n\n"
            "■ 演出フラグ (= 動作と独立):\n"
            "  f = 1P 演出 toggle\n"
            "  Shift+F = 2P 演出 toggle\n"
            "  (全消し/連鎖カウント/相殺波動)\n\n"
            "■ カーソル/操作:\n"
            "  矢印キー: cell 移動\n"
            "  Tab: 1P ↔ 2P\n"
            "  Space: 保存+次へ\n"
            "  Ctrl+S: 保存のみ\n"
            "  Ctrl+Z: 元に戻す\n"
            "  Q / Esc: 終了\n\n"
            "■ 枠の意味:\n"
            "  🔵 青枠 = 選択中\n"
            "  🟢 緑枠 = 修正済\n"
            "  🔴 赤枠 = 要確認"
        )
        tk.Label(
            help_frm, text=help_txt, bg=WINDOW_BG, fg=TEXT_FG,
            justify=tk.LEFT, font=("Yu Gothic UI", 9),
        ).pack(padx=6, pady=6, anchor="nw")

    # --------------------------------------------------------------
    # キーバインド
    # --------------------------------------------------------------

    def _bind_keys(self) -> None:
        for k_int, color in KEY_TO_COLOR.items():
            try:
                key_char = chr(k_int)
                self.root.bind(
                    f"<KeyPress-{key_char}>",
                    lambda e, c=color: self._set_selected_color(c),
                )
            except ValueError:
                pass
        # X-2: 動作状態 keys
        # 小文字 (= 1P) と Shift+ (= 2P) で 1P/2P 独立に設定
        for key_char, state in KEY_TO_FRAME_STATE.items():
            # 小文字 = 1P
            self.root.bind(
                f"<KeyPress-{key_char}>",
                lambda e, s=state: self._set_frame_state_1p(s),
            )
            # 大文字 (Shift+) = 2P
            self.root.bind(
                f"<KeyPress-{key_char.upper()}>",
                lambda e, s=state: self._set_frame_state_2p(s),
            )
        # 演出 toggle: F = 1P, Shift+F = 2P (動作状態と独立)
        self.root.bind(
            f"<KeyPress-{KEY_EFFECT_TOGGLE}>",
            lambda e: self._toggle_effect_1p(),
        )
        self.root.bind(
            f"<KeyPress-{KEY_EFFECT_TOGGLE.upper()}>",
            lambda e: self._toggle_effect_2p(),
        )
        self.root.bind("<Up>", lambda e: self._move_cursor(0, -1))
        self.root.bind("<Down>", lambda e: self._move_cursor(0, 1))
        self.root.bind("<Left>", lambda e: self._move_cursor(-1, 0))
        self.root.bind("<Right>", lambda e: self._move_cursor(1, 0))
        self.root.bind("<Tab>", lambda e: self._swap_side())
        self.root.bind("<space>", lambda e: self._save_and_next())
        self.root.bind("<Shift-space>", lambda e: self._go_prev())
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Control-s>", lambda e: self._save_only())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<KeyPress-q>", lambda e: self._quit())
        self.root.bind("<KeyPress-Q>", lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())

    # --------------------------------------------------------------
    # クリック
    # --------------------------------------------------------------

    def _on_cell_click(self, side: str, event: tk.Event) -> None:
        col = event.x // CELL_PX
        visible_row = event.y // CELL_PX
        row = visible_row + VISIBLE_ROW_START
        if 0 <= col < BOARD_COLS and VISIBLE_ROW_START <= row < VISIBLE_ROW_END:
            self.selected = (side, int(row), int(col))
            self._render_all()

    # --------------------------------------------------------------
    # 色設定 + カーソル移動
    # --------------------------------------------------------------

    def _set_selected_color(self, color: int) -> None:
        side, r, c = self.selected
        fl = self.frames[self.cur_idx]
        labels = fl.labels_1p if side == "1P" else fl.labels_2p
        modified = fl.modified_1p if side == "1P" else fl.modified_2p
        prev_color = labels.get((r, c), COLOR_UNKNOWN)
        prev_mod = (r, c) in modified
        self.undo_stack.append(
            (fl.frame_idx, side, r, c, prev_color, prev_mod)
        )
        labels[(r, c)] = color
        modified.add((r, c))
        self._move_cursor(1, 0)

    def _move_cursor(self, dx: int, dy: int) -> None:
        side, r, c = self.selected
        new_c = c + dx
        new_r = r + dy
        if new_c >= BOARD_COLS:
            new_c = 0
            new_r += 1
        elif new_c < 0:
            new_c = BOARD_COLS - 1
            new_r -= 1
        new_r = max(VISIBLE_ROW_START, min(VISIBLE_ROW_END - 1, new_r))
        self.selected = (side, new_r, new_c)
        self._render_all()

    def _swap_side(self) -> None:
        side, r, c = self.selected
        new_side = "2P" if side == "1P" else "1P"
        self.selected = (new_side, r, c)
        self._render_all()

    def _set_frame_state(self, state: str) -> None:
        """旧 API 互換: 1P/2P 両方を同じ state に設定."""
        self._set_frame_state_1p(state)
        self._set_frame_state_2p(state)

    def _set_frame_state_1p(self, state: str) -> None:
        """X-2: 1P 側 frame state を設定."""
        fl = self.frames[self.cur_idx]
        fl.frame_state_1p = state
        name, _ = FRAME_STATE_INFO.get(state, (state, "#ffffff"))
        print(f"[frame_state] frame {fl.frame_idx} 1P → {name}")
        self._render_all()

    def _set_frame_state_2p(self, state: str) -> None:
        """X-2: 2P 側 frame state を設定."""
        fl = self.frames[self.cur_idx]
        fl.frame_state_2p = state
        name, _ = FRAME_STATE_INFO.get(state, (state, "#ffffff"))
        print(f"[frame_state] frame {fl.frame_idx} 2P → {name}")
        self._render_all()

    def _toggle_effect_1p(self) -> None:
        """1P 演出フラグを toggle."""
        fl = self.frames[self.cur_idx]
        fl.effect_1p = not fl.effect_1p
        print(f"[effect] frame {fl.frame_idx} 1P → {'ON' if fl.effect_1p else 'OFF'}")
        self._render_all()

    def _toggle_effect_2p(self) -> None:
        """2P 演出フラグを toggle."""
        fl = self.frames[self.cur_idx]
        fl.effect_2p = not fl.effect_2p
        print(f"[effect] frame {fl.frame_idx} 2P → {'ON' if fl.effect_2p else 'OFF'}")
        self._render_all()

    # --------------------------------------------------------------
    # フレーム操作
    # --------------------------------------------------------------

    def _save_and_next(self) -> None:
        fl = self.frames[self.cur_idx]
        n = save_frame_labels(fl, self.store, self.video_id)
        self.saved_frames += 1
        print(
            f"[保存] frame {fl.frame_idx} t={fl.time_sec:.2f}s : "
            f"{n} cells (累計 {self.saved_frames})"
        )
        if self.cur_idx < len(self.frames) - 1:
            self.cur_idx += 1
        else:
            print("[完了] 最終フレームに到達しました")
        self._render_all()

    def _go_prev(self) -> None:
        if self.cur_idx > 0:
            self.cur_idx -= 1
            self._render_all()

    def _save_only(self) -> None:
        fl = self.frames[self.cur_idx]
        n = save_frame_labels(fl, self.store, self.video_id)
        self.saved_frames += 1
        print(f"[保存] {n} cells (手動)")
        self._render_all()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        fi, side, r, c, prev_color, prev_mod = self.undo_stack.pop()
        fl = self.frames[self.cur_idx]
        if fl.frame_idx != fi:
            return
        labels = fl.labels_1p if side == "1P" else fl.labels_2p
        modified = fl.modified_1p if side == "1P" else fl.modified_2p
        labels[(r, c)] = prev_color
        if not prev_mod:
            modified.discard((r, c))
        self._render_all()

    def _quit(self) -> None:
        self.root.destroy()

    # --------------------------------------------------------------
    # 描画
    # --------------------------------------------------------------

    def _render_all(self) -> None:
        self._image_refs.clear()
        fl = self.frames[self.cur_idx]
        self._render_grid("1P", self.canvas_1p, fl)
        self._render_grid("2P", self.canvas_2p, fl)
        self._render_preview(fl)
        self._render_status(fl)
        # X-2: 動作状態 + 演出フラグ バー (= 1P/2P 独立表示)
        name1, color1 = FRAME_STATE_INFO.get(
            fl.frame_state_1p, (fl.frame_state_1p, "#ffffff"),
        )
        name2, color2 = FRAME_STATE_INFO.get(
            fl.frame_state_2p, (fl.frame_state_2p, "#ffffff"),
        )
        eff1 = " + 演出ON" if fl.effect_1p else ""
        eff2 = " + 演出ON" if fl.effect_2p else ""
        self.lbl_frame_state_1p.configure(
            text=f"1P 状態: {name1}{eff1}", fg=color1,
        )
        self.lbl_frame_state_2p.configure(
            text=f"2P 状態: {name2}{eff2}", fg=color2,
        )
        # 演出ボタン背景を toggle 状態に応じて変える
        self.btn_effect_1p.configure(
            relief=tk.SUNKEN if fl.effect_1p else tk.RAISED,
            bg="#ffaa44" if fl.effect_1p else "#ffee44",
        )
        self.btn_effect_2p.configure(
            relief=tk.SUNKEN if fl.effect_2p else tk.RAISED,
            bg="#ffaa44" if fl.effect_2p else "#ffee44",
        )

    def _render_grid(
        self, side: str, canvas: tk.Canvas, fl: FrameLabels,
    ) -> None:
        canvas.delete("all")
        for row in range(VISIBLE_ROW_START, VISIBLE_ROW_END):
            for col in range(BOARD_COLS):
                visible_row = row - VISIBLE_ROW_START
                x1 = col * CELL_PX
                y1 = visible_row * CELL_PX
                # 表示用 patch (= cell 全体、 上部 row も上下見える)
                patch = fl.get_display_patch(side, row, col)
                if patch is not None:
                    photo = bgr_to_photo(patch, (CELL_PX, CELL_PX))
                    self._image_refs.append(photo)
                    canvas.create_image(x1, y1, anchor="nw", image=photo)
                lbl_color = fl.get_label(side, row, col)
                pred_color = (
                    fl.predictions_1p.get((row, col), COLOR_UNKNOWN)
                    if side == "1P"
                    else fl.predictions_2p.get((row, col), COLOR_UNKNOWN)
                )
                # ラベル色バー
                bar_h = max(16, int(CELL_PX * 0.28))
                bgr = COLOR_TO_BGR.get(lbl_color, (200, 200, 200))
                hex_bgr = "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])
                canvas.create_rectangle(
                    x1, y1 + CELL_PX - bar_h,
                    x1 + CELL_PX, y1 + CELL_PX,
                    fill=hex_bgr, outline="",
                )
                canvas.create_text(
                    x1 + CELL_PX // 2, y1 + CELL_PX - bar_h // 2,
                    text=COLOR_TO_JP[lbl_color], fill="#ffffff",
                    font=("Yu Gothic UI", 8, "bold"),
                )
                # 修正済 = 緑枠
                modified = fl.modified_1p if side == "1P" else fl.modified_2p
                if (row, col) in modified:
                    canvas.create_rectangle(
                        x1 + 1, y1 + 1,
                        x1 + CELL_PX - 1, y1 + CELL_PX - 1,
                        outline="#44ff44", width=3,
                    )
                # CNN 不一致 = 赤枠 (修正候補)
                if pred_color != lbl_color and (row, col) not in modified:
                    canvas.create_rectangle(
                        x1 + 2, y1 + 2,
                        x1 + CELL_PX - 2, y1 + CELL_PX - 2,
                        outline="#ff4444", width=2,
                    )
                # 選択中 = 青太枠
                if self.selected == (side, row, col):
                    canvas.create_rectangle(
                        x1 + 3, y1 + 3,
                        x1 + CELL_PX - 3, y1 + CELL_PX - 3,
                        outline="#4488ff", width=4,
                    )

    def _render_preview(self, fl: FrameLabels) -> None:
        canvas = self.canvas_preview
        canvas.delete("all")
        side, r, c = self.selected
        # 拡大プレビューも表示用 patch (= cell 全体) を使う
        patch = fl.get_display_patch(side, r, c)
        if patch is None:
            canvas.create_text(
                PREVIEW_PX // 2, PREVIEW_PX // 2,
                text="(選択なし)", fill="#888888",
                font=("Yu Gothic UI", 10),
            )
            self.lbl_preview_info.configure(text="cell を選択してください")
            return
        photo = bgr_to_photo(patch, (PREVIEW_PX, PREVIEW_PX))
        self._image_refs.append(photo)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        lbl = fl.get_label(side, r, c)
        pred = (
            fl.predictions_1p.get((r, c), COLOR_UNKNOWN)
            if side == "1P"
            else fl.predictions_2p.get((r, c), COLOR_UNKNOWN)
        )
        mod = (
            (r, c) in (fl.modified_1p if side == "1P" else fl.modified_2p)
        )
        info = (
            f"プレイヤー: {side}   "
            f"行: {r}  列: {c}\n"
            f"CNN 予測 : {COLOR_TO_JP[pred]}\n"
            f"現在の色 : {COLOR_TO_JP[lbl]}\n"
            f"状態       : {'修正済' if mod else 'CNN 予測のまま'}"
        )
        self.lbl_preview_info.configure(text=info)

    def _render_status(self, fl: FrameLabels) -> None:
        total = len(self.frames)
        mod_1p = len(fl.modified_1p)
        mod_2p = len(fl.modified_2p)
        red_1p = sum(
            1 for k, v in fl.labels_1p.items()
            if fl.predictions_1p.get(k) != v and k not in fl.modified_1p
        )
        red_2p = sum(
            1 for k, v in fl.labels_2p.items()
            if fl.predictions_2p.get(k) != v and k not in fl.modified_2p
        )
        prog = (
            f"フレーム {self.cur_idx + 1}/{total}  "
            f"(frame={fl.frame_idx} t={fl.time_sec:.2f}s)   "
            f"保存済={self.saved_frames}"
        )
        self.lbl_progress.configure(text=prog)
        self.status_var.set(
            f"修正済: 1P={mod_1p} 2P={mod_2p}  |  "
            f"要確認 (赤枠): 1P={red_1p} 2P={red_2p}  |  "
            f"取り消し可能={len(self.undo_stack)}"
        )


# ============================
# メイン
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=75.0)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--store-root", type=Path, default=Path("data/pseudo_labels"),
    )
    args = parser.parse_args()

    print(
        f"[sampling] {args.video} sec=[{args.start_sec}, {args.end_sec}] "
        f"interval={args.interval_sec}"
    )
    sampled = sample_frames(
        args.video, args.start_sec, args.end_sec, args.interval_sec,
    )
    print(f"[sampling] got {len(sampled)} frames")
    if not sampled:
        print("[error] no frames sampled")
        return 1

    classifier = ColorClassifier()
    print("[init] ColorClassifier loaded; predicting cells ...")
    frame_labels: list[FrameLabels] = []
    p1, p2 = DEFAULT_P1_REGION, DEFAULT_P2_REGION
    for fi, t_sec, frame in sampled:
        patches_1p = extract_cell_patches(frame, p1)
        patches_2p = extract_cell_patches(frame, p2)
        pred_1p = predict_cells(patches_1p, classifier)
        pred_2p = predict_cells(patches_2p, classifier)
        frame_labels.append(FrameLabels(
            frame_idx=fi, time_sec=t_sec,
            patches_1p=patches_1p, patches_2p=patches_2p,
            predictions_1p=pred_1p, predictions_2p=pred_2p,
        ))
    print(f"[init] {len(frame_labels)} frames ready. GUI を起動します ...")

    store = LabelStore(video_id=args.video_id, root=args.store_root)
    root = tk.Tk()
    app = CellLabelerGUI(root, frame_labels, store, args.video_id)
    root.mainloop()
    print(f"[done] saved {app.saved_frames} frames to "
          f"{args.store_root}/{args.video_id}/cell.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
