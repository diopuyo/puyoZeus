"""固定テンプレートのFFTを再利用する単一スレッドNCC照合。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

# float32 FFTとOpenCVの丸め差が判定境界に近い場合は元の照合へ戻す。
NCC_DECISION_MARGIN = 1e-4
NCC_VARIANCE_EPS_FACTOR = 10.0
NCC_VARIANCE_FLOOR = 0.5


@dataclass
class PreparedImage:
    """同じフレーム内の同一ROIだけで再利用し、次フレームへ持ち越さない。"""
    image: np.ndarray

    @cached_property
    def shape(self) -> tuple[int, int]:
        return tuple(cv2.getOptimalDFTSize(size) for size in self.image.shape)

    @cached_property
    def spectrum(self) -> np.ndarray:
        source = np.zeros(self.shape, dtype=np.float32)
        height, width = self.image.shape
        source[:height, :width] = self.image
        return cv2.dft(source)

    @cached_property
    def integrals(self) -> tuple[np.ndarray, np.ndarray]:
        return cv2.integral2(self.image)


@dataclass
class PreparedTemplate:
    template: np.ndarray
    spectra: dict = field(default_factory=dict, repr=False)

    def correlate(self, image: np.ndarray, prepared: PreparedImage | None = None) -> np.ndarray:
        """循環相関のうち回り込みのないvalid領域だけを取り出す。"""
        height, width = image.shape
        th, tw = self.template.shape
        prepared = prepared or PreparedImage(image)
        shape = prepared.shape
        if shape not in self.spectra:
            self.spectra.clear()
            kernel = np.zeros(shape, dtype=np.float32)
            kernel[:th, :tw] = self.template
            self.spectra[shape] = cv2.dft(kernel)
        product = cv2.mulSpectrums(prepared.spectrum, self.spectra[shape], 0, conjB=True)
        cross = cv2.idft(product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return cross[:height-th+1, :width-tw+1]

    @cached_property
    def statistics(self) -> tuple[float, float]:
        mean, std = cv2.meanStdDev(self.template)
        return mean.item(), std.item()

    def scores(self, image: np.ndarray, prepared: PreparedImage | None = None) -> np.ndarray:
        """OpenCVと同じ平均除去・局所分散で正規化する。"""
        th, tw = self.template.shape
        count = self.template.size
        prepared = prepared or PreparedImage(image)
        sums, squares = prepared.integrals
        total = (sums[th:, tw:]-sums[:-th, tw:]-sums[th:, :-tw]+sums[:-th, :-tw]).astype(np.float64)
        total_sq = squares[th:, tw:]-squares[:-th, tw:]-squares[th:, :-tw]+squares[:-th, :-tw]
        mean, std = self.statistics
        if std == 0:
            return np.ones(total.shape, dtype=np.float32)
        numerator = self.correlate(image, prepared).astype(np.float64)-total*mean
        variance = np.maximum(total_sq-total*total/count, 0)
        floor = np.minimum(NCC_VARIANCE_FLOOR,
                           NCC_VARIANCE_EPS_FACTOR*np.finfo(np.float32).eps*total_sq)
        variance[variance <= floor] = 0
        denominator = np.sqrt(variance)*std*np.sqrt(count)
        result = np.zeros(total.shape, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
        return np.clip(result, -1, 1).astype(np.float32)

    def peak(self, image: np.ndarray, threshold: float,
             prepared: PreparedImage | None = None) -> tuple[float, tuple[int, int]]:
        """閾値近傍では元の実装を用い、丸め差による採否の変更を避ける。"""
        scores = self.scores(image, prepared)
        _, maximum, _, position = cv2.minMaxLoc(scores)
        if maximum >= threshold-NCC_DECISION_MARGIN:
            scores = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, position = cv2.minMaxLoc(scores)
        return float(maximum), position
