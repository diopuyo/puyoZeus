"""観測で否定できない物理候補を残し、総得点の下限だけを返す。"""
from __future__ import annotations
from typing import Any
import numpy as np
from src import puyo_core_bridge as native
from src.board import Board
from src.chain import ChainSimulator

VISIBLE_ROWS, COLS, CLASSES = 12, 6, 7
MIN_FRAMES, MIN_WINDOW_SECONDS = 6, .25
PROB_FLOOR, LOG_RATIO = 1e-12, float(np.log(9.))
# この候補経路だけ、消去群・順序が同一の既存の高速探索を使う。
SIMULATOR = ChainSimulator(cache_enabled=False, exclude_hidden_row_from_pop=True, fast_groups=True)


def advance(board: np.ndarray) -> np.ndarray:
    """隠し段を補わず、既存の消去・落下で次の段間へ進める。"""
    if (native.NATIVE_AVAILABLE and (board >= 0).all()
            and ((board <= 5) | (board == 9)).all()):
        trace = native._native.simulate_chain_trace_py(board.ravel().tolist(), True)
        if trace:
            return np.asarray(trace[0][1], dtype=np.uint8).reshape(board.shape)
        # 消去が無い場合にも従来どおり落下させる。
        value = Board.from_list(board.tolist())
        SIMULATOR._gravity_fast(value)
        return value._grid.copy()
    value = Board.from_list(board.tolist())
    groups = SIMULATOR.find_erasable_groups(value)
    SIMULATOR._erase_fast(value, groups)
    SIMULATOR._gravity_fast(value)
    return value._grid.copy()


def advance_matrix(matrix: np.ndarray, cache: dict[bytes, np.ndarray]) -> np.ndarray:
    """同じ更新内で同じ盤へ合流した候補は、一段の計算結果だけを共有する。"""
    result = []
    for board in matrix:
        key = board.tobytes()
        if key not in cache:
            cache[key] = advance(board)
        result.append(cache[key])
    return np.stack(result)


def probabilities(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (result.shape != (VISIBLE_ROWS, COLS, CLASSES) or not np.isfinite(result).all()
            or np.any(result < 0) or not np.allclose(result.sum(axis=-1), 1., atol=1e-5)):
        return None
    return result


class CandidateEvidence:
    """観測は段をまたいで足し、式と尤度比の両方を満たす集合を維持する。"""
    def __init__(self, candidates: list) -> None:
        self.candidates = candidates
        self.matrix = np.stack([c.board for c in candidates])
        self.scores = np.zeros(len(candidates), dtype=np.float64)
        self.frames, self.stage = 0, 1
        self.first_t: float | None = None
        self.last_t: float | None = None
        self.retained: list[int] = []
        self.viable = np.ones(len(candidates), dtype=bool)
        self.steps: tuple[int, ...] | None = None
        self.final: np.ndarray | None = None

    def update(self, t: float, observed: tuple, value: Any = None) -> None:
        cache: dict[bytes, np.ndarray] = {}
        while self.stage < len(observed):
            self.matrix = advance_matrix(self.matrix, cache)
            self.stage += 1
        probs = probabilities(value)
        if probs is not None and (self.last_t is None or t > self.last_t):
            matrix = self.matrix[:, 1:]
            variable = np.any(matrix != matrix[0], axis=0)
            codes = np.where(matrix[:, variable] == 9, CLASSES-1, matrix[:, variable])
            logs = np.log(np.maximum(probs[variable], PROB_FLOOR))
            self.scores += logs[np.arange(variable.sum())[None, :], codes].sum(axis=1)
            self.first_t = t if self.first_t is None else self.first_t
            self.last_t = t
            self.frames += 1
        self.select(observed)

    def select(self, observed: tuple) -> None:
        self.retained, self.steps, self.final = [], None, None
        if (self.frames < MIN_FRAMES or self.last_t-self.first_t+1e-7 < MIN_WINDOW_SECONDS):
            return
        eligible = np.array([i for i,c in enumerate(self.candidates)
                             if self.viable[i] and c.steps[:len(observed)] == observed], dtype=int)
        self.viable[:] = False
        if not len(eligible):
            return
        means = self.scores[eligible] / self.frames
        self.retained = eligible[means.max()-means <= LOG_RATIO].tolist()
        # 同じ発火の過去の観測で除いた盤を、後の平均の変化だけでは戻さない。
        self.viable[self.retained] = True
        kept = [self.candidates[i] for i in self.retained]
        self.steps = min((c.steps for c in kept), key=lambda s:(sum(s), len(s), s))
        if all(np.array_equal(c.final, kept[0].final) for c in kept):
            self.final = kept[0].final.copy()


def read_probabilities(reader: Any, frame: Any, side: str) -> list | None:
    """明示ON時だけ既存CNNの生確率を取り、現在層の盤には書き戻さない。"""
    from src.light_consumption import enabled
    if not enabled() or frame is None:
        return None
    classifier = getattr(reader, '_classifier', None)
    cnn = getattr(classifier, '_cnn', classifier)
    region = getattr(reader, '_p1_region' if side == '1P' else '_p2_region', None)
    if cnn is None or region is None or not hasattr(cnn, 'predict_proba_batch'):
        return None
    from src.phase_j.live_color_roi import full_image
    image = full_image(frame)
    height, width = image.shape[:2]
    patches = []
    for row in range(1, VISIBLE_ROWS+1):
        for col in range(COLS):
            x1,y1,x2,y2 = region.cell_sample_rect(row, col)
            x1,y1 = max(0,min(int(x1),width-1)),max(0,min(int(y1),height-1))
            x2,y2 = max(x1+1,min(int(x2),width)),max(y1+1,min(int(y2),height))
            patches.append(image[y1:y2,x1:x2])
    return cnn.predict_proba_batch(patches).reshape(VISIBLE_ROWS,COLS,CLASSES).tolist()
