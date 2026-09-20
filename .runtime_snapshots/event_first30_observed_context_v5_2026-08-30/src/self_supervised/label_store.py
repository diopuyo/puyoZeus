"""LabelStore: 擬似ラベルを動画別 / component 別にディスク永続化.

保存先:
    data/pseudo_labels/{video_id}/{component}.jsonl

各行は PseudoLabelSample.to_jsonable() の JSON。
画像 (numpy 配列) は base64 埋込み。

スレッドセーフ性:
    append() は threading.Lock で同期。複数 worker 同時書き込み対応。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

from src.self_supervised.pseudo_label import PseudoLabelSample

# デフォルト保存ルート
DEFAULT_STORE_ROOT: Path = Path("data/pseudo_labels")

# JSONL 1 ファイルあたり想定レコード数 (大きく超えそうなら shard 検討)
_RECORDS_HINT_PER_FILE: int = 100_000


class LabelStore:
    """擬似ラベルの append-only JSONL ストア.

    Usage:
        store = LabelStore(video_id="video_02")
        store.append(samples)
        for s in store.load("score"):
            ...
    """

    def __init__(
        self,
        video_id: str,
        root: Path | str | None = None,
    ) -> None:
        if not video_id:
            raise ValueError("video_id is required")
        self._video_id = str(video_id)
        if root is None:
            self._root = DEFAULT_STORE_ROOT
        else:
            self._root = Path(root)
        self._lock = threading.Lock()
        self._video_dir = self._root / self._video_id
        self._video_dir.mkdir(parents=True, exist_ok=True)

    @property
    def video_id(self) -> str:
        return self._video_id

    @property
    def video_dir(self) -> Path:
        return self._video_dir

    def _path_for(self, component: str) -> Path:
        if not component:
            raise ValueError("component must be non-empty")
        return self._video_dir / f"{component}.jsonl"

    def append(self, samples: list[PseudoLabelSample]) -> None:
        """擬似ラベルを追加保存 (component 別に分離書込み).

        Args:
            samples: PseudoLabelSample のリスト
        """
        if not samples:
            return
        # component 別に分類
        by_comp: dict[str, list[PseudoLabelSample]] = {}
        for s in samples:
            by_comp.setdefault(s.component, []).append(s)
        with self._lock:
            for comp, sample_list in by_comp.items():
                path = self._path_for(comp)
                with path.open("a", encoding="utf-8") as f:
                    for s in sample_list:
                        f.write(
                            json.dumps(s.to_jsonable(), ensure_ascii=False)
                            + "\n",
                        )

    def load(
        self,
        component: str,
        video_id: str | None = None,
    ) -> Iterator[PseudoLabelSample]:
        """指定 component の擬似ラベルを順次 yield する.

        video_id を指定すれば別動画のラベルも読める (None なら自身)。
        ファイルが無ければ空 iterator。
        """
        target_dir = (
            self._root / video_id if video_id is not None else self._video_dir
        )
        path = target_dir / f"{component}.jsonl"
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield PseudoLabelSample.from_jsonable(obj)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # 破損行は skip (error handling: silent skip 方針)
                    continue

    def stats(self) -> dict[str, int]:
        """component 別の累計件数を返す."""
        out: dict[str, int] = {}
        for path in self._video_dir.glob("*.jsonl"):
            comp = path.stem
            count = 0
            with path.open("r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            out[comp] = count
        return out

    @classmethod
    def list_videos(
        cls, root: Path | str | None = None,
    ) -> list[str]:
        """ストア内の video_id 一覧."""
        r = Path(root) if root is not None else DEFAULT_STORE_ROOT
        if not r.is_dir():
            return []
        return sorted([p.name for p in r.iterdir() if p.is_dir()])

    @classmethod
    def aggregate_stats(
        cls, root: Path | str | None = None,
    ) -> dict[str, dict[str, int]]:
        """全 video の component 別件数を返す."""
        r = Path(root) if root is not None else DEFAULT_STORE_ROOT
        out: dict[str, dict[str, int]] = {}
        if not r.is_dir():
            return out
        for vd in r.iterdir():
            if not vd.is_dir():
                continue
            store = cls(video_id=vd.name, root=r)
            out[vd.name] = store.stats()
        return out


__all__ = ["DEFAULT_STORE_ROOT", "LabelStore"]
