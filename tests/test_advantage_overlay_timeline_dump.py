"""タイムラインdump工事 (2026-08-11) のテスト。

- TimelineDumpRow の npz 保存/読み込みの往復整合 (純粋なデータ層、動画不要)
- generate() の render=False / dump_timeline_path 配線
  (RecognitionPipeline.load_default・cv2.VideoCapture/VideoWriter・
  _train_model を全て軽量スタブに差し替え、実動画・実モデル無しで検証する)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402


# ============================
# save_timeline_dump / load_timeline_dump の往復整合
# ============================

def _sample_rows() -> list[vao.TimelineDumpRow]:
    return [
        vao.TimelineDumpRow(
            t_sec=1.5, game_idx=0, adv_raw=12.3, adv_ema=10.1, p1=0.6,
            p1_raw=0.55,
            pending_p1=5, pending_p2=0, room1=60, room2=72,
            is_dead1=False, is_dead2=False,
            drivers_top1_name="board_color_puyo_total", drivers_top1_val=0.4,
            drivers_top3_names=("board_color_puyo_total", "max_column_height", ""),
            drivers_top3_vals=(0.4, -0.2, 0.0),
            score1=1200, score2=800, b1_hash=111, b2_hash=222,
            state1="STABLE", state2="STABLE",
        ),
        vao.TimelineDumpRow(
            t_sec=3.25, game_idx=1, adv_raw=-40.0, adv_ema=-38.2, p1=0.1,
            p1_raw=0.15,
            pending_p1=0, pending_p2=80, room1=72, room2=4,
            is_dead1=False, is_dead2=True,
            drivers_top1_name="", drivers_top1_val=0.0,
            drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
            score1=vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL, score2=5000,
            b1_hash=333, b2_hash=444, state1="CHAIN", state2="STABLE",
        ),
    ]


class TestTimelineDumpRoundTrip:
    def test_save_load_preserves_all_fields(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        path = tmp_path / "v_test.npz"
        vao.save_timeline_dump(path, "v_test", rows)
        video_id, loaded = vao.load_timeline_dump(path)
        assert video_id == "v_test"
        assert len(loaded) == len(rows)
        for original, restored in zip(rows, loaded):
            assert restored == original

    def test_empty_rows_round_trip(self, tmp_path: Path) -> None:
        """0レコード (settled が一度も起きなかった動画) でも例外にならない。"""
        path = tmp_path / "v_empty.npz"
        vao.save_timeline_dump(path, "v_empty", [])
        video_id, loaded = vao.load_timeline_dump(path)
        assert video_id == "v_empty"
        assert loaded == []

    def test_none_score_sentinel_round_trips(self, tmp_path: Path) -> None:
        """score=None (OCR失敗) は TIMELINE_DUMP_SCORE_NONE_SENTINEL に変換される
        (npz は int 列のため None をそのまま格納できない)。
        """
        board = _empty_board()
        row = vao._build_timeline_dump_row(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=0, pending_p2=0, room1=72, room2=72,
            b1=board, b2=board,
            drivers=[], score1=None, score2=None, state1="MENU", state2="MENU",
        )
        assert row.score1 == vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL
        assert row.score2 == vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL


class TestLoadTimelineDumpKeyAccessCount:
    """根治③の回帰ガード: npz の各キーが行数に関わらず1回しかアクセスされない
    ことを直接確認する (出力一致テストだけでは「たまたま速い実装に戻す」
    再発を検知できないため、アクセス回数そのものを計装して固定する)。
    """

    def test_each_npz_key_accessed_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _sample_rows() * 50  # 行数を増やしても回数が変わらないことを見る
        path = tmp_path / "v_access_count.npz"
        vao.save_timeline_dump(path, "v_access_count", rows)

        access_counts: dict[str, int] = {}
        real_getitem = np.lib.npyio.NpzFile.__getitem__

        def _counting_getitem(self: object, key: str) -> object:
            access_counts[key] = access_counts.get(key, 0) + 1
            return real_getitem(self, key)

        monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", _counting_getitem)
        _, loaded = vao.load_timeline_dump(path)
        assert len(loaded) == len(rows)
        # video_id を含め、どのキーも1回しかアクセスされていない
        # (0回ではない=そもそも読めていることも同時に保証する)。
        assert access_counts, "npz へのアクセスが一度も記録されなかった"
        assert all(count == 1 for count in access_counts.values()), access_counts


class TestLoadTimelineDumpPerfRegression:
    """根治③: 全フィールド1回ロード化がループ内毎回参照の旧実装と
    完全に同じ出力を返すことを確認する (出力不変の回帰テスト)。"""

    def _load_old_style(self, path: Path) -> list[vao.TimelineDumpRow]:
        """修正前の実装 (ループ内で d["field"][i] を毎回引く版) を再現する。"""
        d = np.load(str(path), allow_pickle=True)
        n = int(d["t_sec"].shape[0])
        rows: list[vao.TimelineDumpRow] = []
        for i in range(n):
            rows.append(vao.TimelineDumpRow(
                t_sec=float(d["t_sec"][i]), game_idx=int(d["game_idx"][i]),
                adv_raw=float(d["adv_raw"][i]), adv_ema=float(d["adv_ema"][i]),
                p1=float(d["p1"][i]), p1_raw=float(d["p1_raw"][i]),
                pending_p1=int(d["pending_p1"][i]),
                pending_p2=int(d["pending_p2"][i]),
                room1=int(d["room1"][i]), room2=int(d["room2"][i]),
                is_dead1=bool(d["is_dead1"][i]), is_dead2=bool(d["is_dead2"][i]),
                drivers_top1_name=str(d["drivers_top1_name"][i]),
                drivers_top1_val=float(d["drivers_top1_val"][i]),
                drivers_top3_names=tuple(
                    str(x) for x in d["drivers_top3_names"][i]),
                drivers_top3_vals=tuple(
                    float(x) for x in d["drivers_top3_vals"][i]),
                score1=int(d["score1"][i]), score2=int(d["score2"][i]),
                b1_hash=int(d["b1_hash"][i]), b2_hash=int(d["b2_hash"][i]),
                state1=str(d["state1"][i]), state2=str(d["state2"][i]),
                kpending_p1=float(d["kpending_p1"][i]),
                kpending_p2=float(d["kpending_p2"][i]),
                kroom1=int(d["kroom1"][i]), kroom2=int(d["kroom2"][i]),
            ))
        return rows

    def test_new_loader_matches_old_per_index_loader(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        path = tmp_path / "v_perf.npz"
        vao.save_timeline_dump(path, "v_perf", rows)
        _, new_rows = vao.load_timeline_dump(path)
        old_rows = self._load_old_style(path)
        assert new_rows == old_rows == rows


def _empty_board() -> Any:
    from src.board import Board
    return Board.from_list(np.zeros((13, 6), dtype=np.int8).tolist())


# ============================
# 根治① (2026-08-23): kpending_p1/p2・kroom1/kroom2 の後方互換+往復整合
# ============================

class TestKillOverrideCorrectedFieldsBackwardCompat:
    def test_omitted_k_fields_default_to_raw_values(self) -> None:
        """新4フィールドを省略すると生値 (pending_p1/p2・room1/room2) と
        同じ値が自動で入る (__post_init__、旧呼出元との bit-identical 保証)。
        """
        row = vao.TimelineDumpRow(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=12, pending_p2=34, room1=56, room2=78,
            is_dead1=False, is_dead2=False,
            drivers_top1_name="", drivers_top1_val=0.0,
            drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
            score1=0, score2=0, b1_hash=0, b2_hash=0,
            state1="STABLE", state2="STABLE",
        )
        assert row.kpending_p1 == 12.0
        assert row.kpending_p2 == 34.0
        assert row.kroom1 == 56
        assert row.kroom2 == 78

    def test_explicit_k_fields_preserved_when_correction_differs(self) -> None:
        """是正が働いたフレーム (k* が生値と異なる) はそのまま保持される。"""
        row = vao.TimelineDumpRow(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=216, pending_p2=0, room1=5, room2=72,
            is_dead1=False, is_dead2=False,
            drivers_top1_name="", drivers_top1_val=0.0,
            drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
            score1=0, score2=0, b1_hash=0, b2_hash=0,
            state1="CHAIN", state2="STABLE",
            kpending_p1=0.0, kpending_p2=594.0, kroom1=62, kroom2=72,
        )
        assert row.kpending_p1 == 0.0
        assert row.kpending_p2 == 594.0
        assert row.kroom1 == 62
        assert row.kroom2 == 72
        # 生値側は一切変更されない (根治①は表示への入力を正すだけ)
        assert row.pending_p1 == 216 and row.room1 == 5

    def test_build_timeline_dump_row_defaults_k_fields_to_raw(self) -> None:
        """_build_timeline_dump_row も kpending/kroom 省略時は生値と同じ
        (是正未配線の呼出元 = 従来の全呼出元との bit-identical)。"""
        board = _empty_board()
        row = vao._build_timeline_dump_row(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=7, pending_p2=9, room1=60, room2=61,
            b1=board, b2=board,
            drivers=[], score1=None, score2=None, state1="MENU", state2="MENU",
        )
        assert (row.kpending_p1, row.kpending_p2) == (7.0, 9.0)
        assert (row.kroom1, row.kroom2) == (60, 61)

    def test_round_trip_preserves_corrected_fields(self, tmp_path: Path) -> None:
        """是正値ありの行を保存/読込しても値が保たれる。"""
        rows = _sample_rows() + [vao.TimelineDumpRow(
            t_sec=886.5, game_idx=2, adv_raw=-80.0, adv_ema=-80.0, p1=0.05,
            p1_raw=0.05, pending_p1=216, pending_p2=0, room1=5, room2=72,
            is_dead1=False, is_dead2=False,
            drivers_top1_name="", drivers_top1_val=0.0,
            drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
            score1=1000, score2=900, b1_hash=1, b2_hash=2,
            state1="CHAIN", state2="STABLE",
            kpending_p1=0.0, kpending_p2=594.0, kroom1=62, kroom2=72,
        )]
        path = tmp_path / "v_kfields.npz"
        vao.save_timeline_dump(path, "v_kfields", rows)
        _, loaded = vao.load_timeline_dump(path)
        assert loaded == rows

    def test_old_dump_without_k_fields_loads_with_raw_fallback(
        self, tmp_path: Path,
    ) -> None:
        """新4キーを含まない npz (根治①以前のdump相当) でも読み込める
        (後方互換)。値は生値と同じになる。"""
        rows = _sample_rows()
        path = tmp_path / "v_old.npz"
        n = len(rows)
        # save_timeline_dump 相当だが新4キーを意図的に書かない (旧dump再現)。
        np.savez_compressed(
            str(path), video_id=np.array("v_old"),
            t_sec=np.array([r.t_sec for r in rows], dtype=np.float64),
            game_idx=np.array([r.game_idx for r in rows], dtype=np.int32),
            adv_raw=np.array([r.adv_raw for r in rows], dtype=np.float64),
            adv_ema=np.array([r.adv_ema for r in rows], dtype=np.float64),
            p1=np.array([r.p1 for r in rows], dtype=np.float64),
            p1_raw=np.array([r.p1_raw for r in rows], dtype=np.float64),
            pending_p1=np.array([r.pending_p1 for r in rows], dtype=np.int32),
            pending_p2=np.array([r.pending_p2 for r in rows], dtype=np.int32),
            room1=np.array([r.room1 for r in rows], dtype=np.int32),
            room2=np.array([r.room2 for r in rows], dtype=np.int32),
            is_dead1=np.array([r.is_dead1 for r in rows], dtype=bool),
            is_dead2=np.array([r.is_dead2 for r in rows], dtype=bool),
            drivers_top1_name=np.array(
                [r.drivers_top1_name for r in rows], dtype=object),
            drivers_top1_val=np.array(
                [r.drivers_top1_val for r in rows], dtype=np.float64),
            drivers_top3_names=np.array(
                [r.drivers_top3_names for r in rows], dtype=object),
            drivers_top3_vals=np.array(
                [r.drivers_top3_vals for r in rows], dtype=np.float64),
            score1=np.array([r.score1 for r in rows], dtype=np.int32),
            score2=np.array([r.score2 for r in rows], dtype=np.int32),
            b1_hash=np.array([r.b1_hash for r in rows], dtype=np.int64),
            b2_hash=np.array([r.b2_hash for r in rows], dtype=np.int64),
            state1=np.array([r.state1 for r in rows], dtype=object),
            state2=np.array([r.state2 for r in rows], dtype=object),
        )
        assert n == 2  # _sample_rows() の件数を前提にした自己確認
        _, loaded = vao.load_timeline_dump(path)
        assert len(loaded) == n
        for original, restored in zip(rows, loaded):
            assert restored.kpending_p1 == float(original.pending_p1)
            assert restored.kpending_p2 == float(original.pending_p2)
            assert restored.kroom1 == original.room1
            assert restored.kroom2 == original.room2


# ============================
# generate() render / dump 配線 (実動画・実モデル不要の軽量スタブ)
# ============================

class _FakeSide:
    def __init__(self) -> None:
        self.state = BoardState.MENU
        self.confirmed_board = None
        self.score: int | None = None
        self.chain_event = None
        self.next_pair = None


class _FakeResult:
    def __init__(self) -> None:
        self.p1 = _FakeSide()
        self.p2 = _FakeSide()


class _FakePipeline:
    """RecognitionPipeline.load_default の軽量スタブ。常に MENU 状態を返す
    (b1/b2 が STABLE にならないため有利不利の再計算経路には入らないが、
    render/dump の配線=描画有無・npz保存有無だけを見るには十分)。"""

    def update(self, fi: int, t: float, frame: object) -> _FakeResult:
        return _FakeResult()

    def tsumo_count(self, side: str) -> int:
        return 0


class _FakeCapture:
    """cv2.VideoCapture の軽量スタブ (n_frames 枚のダミーフレームを返す)。"""

    def __init__(self, n_frames: int = 4) -> None:
        self._n = n_frames
        self._i = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n)
        return 0.0

    def set(self, *_a: object) -> bool:
        return True

    def read(self) -> tuple[bool, "np.ndarray | None"]:
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, np.zeros((vao.OUT_H, vao.OUT_W, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _SpyVideoWriter:
    """cv2.VideoWriter の軽量スタブ。生成された回数を記録する。"""

    instances: list["_SpyVideoWriter"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        _SpyVideoWriter.instances.append(self)
        self.write_count = 0

    def write(self, _frame: object) -> None:
        self.write_count += 1

    def release(self) -> None:
        pass


@pytest.fixture()
def _stub_heavy_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate() から実動画・実CNN・実モデル学習を排除する共通スタブ。"""
    monkeypatch.setattr(vao, "_train_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        vao.RecognitionPipeline, "load_default",
        staticmethod(lambda **_k: _FakePipeline()),
    )
    monkeypatch.setattr(vao.cv2, "VideoCapture", lambda *_a: _FakeCapture())
    _SpyVideoWriter.instances = []
    monkeypatch.setattr(vao.cv2, "VideoWriter", _SpyVideoWriter)


class TestGenerateRenderAndDumpWiring:
    def test_default_behavior_unchanged_creates_video_writer(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """render/dump_timeline_path 省略時 (既存呼出元と同じ呼び方) は従来通り
        VideoWriter が生成される (backwards compat)。"""
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15,
        )
        assert len(_SpyVideoWriter.instances) == 1
        assert _SpyVideoWriter.instances[0].write_count == written
        assert written > 0

    def test_no_render_skips_video_writer(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """render=False では VideoWriter が一切生成されない (動画が作られない)。"""
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
        )
        assert _SpyVideoWriter.instances == []
        assert written > 0  # 計算自体は最後まで走る

    def test_dump_timeline_path_writes_npz(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """dump_timeline_path 指定時は npz が書き出される (中身は0件でも良い、
        本テストは MENU 状態のみのスタブで settled に到達しないため空)。"""
        dump_path = tmp_path / "dump.npz"
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
            dump_timeline_path=dump_path,
        )
        assert dump_path.exists()
        video_id, rows = vao.load_timeline_dump(dump_path)
        assert video_id == "dummy_never_opened"
        assert rows == []  # スタブは MENU 固定 = settled 更新が一度も起きない

    def test_no_dump_path_means_no_file_written(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """dump_timeline_path=None (既定) では npz を一切書かない (backwards compat)。"""
        out_dir_before = set(tmp_path.iterdir())
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
        )
        new_files = set(tmp_path.iterdir()) - out_dir_before
        assert not any(p.suffix == ".npz" for p in new_files)


# ============================
# is_dead 凍結盤面誤判定の遡及訂正 (2026-08-24 根治、enable_stable_confirmed_is_dead)
# ============================

def _row(t_sec: float, game_idx: int, is_dead1: bool, is_dead2: bool,
         state1: str, state2: str) -> vao.TimelineDumpRow:
    """テスト用に is_dead/state/game_idx/t_sec だけ変えた最小 TimelineDumpRow。"""
    return vao.TimelineDumpRow(
        t_sec=t_sec, game_idx=game_idx, adv_raw=0.0, adv_ema=0.0, p1=0.5,
        p1_raw=0.5, pending_p1=0, pending_p2=0, room1=72, room2=72,
        is_dead1=is_dead1, is_dead2=is_dead2,
        drivers_top1_name="", drivers_top1_val=0.0,
        drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
        score1=0, score2=0, b1_hash=0, b2_hash=0, state1=state1, state2=state2,
    )


class TestRetroactiveDeadCorrection:
    def test_false_positive_during_chain_gets_corrected(self) -> None:
        """STABLE(凍結直前=True、実測 t=6701.667 相当) -> CHAIN(凍結、
        旧is_dead1=True、実測 t=6702.5-6717.0 相当) -> STABLE(連鎖解決、
        真の値=False、実測 t=6717.4 相当) という実データ実測パターン
        (logs/is_dead_persist_2026-08-23/) を、遡及訂正で「CHAIN中も
        一貫してFalse」に直せることを確認する。

        凍結直前の STABLE 行自体 (index 0) は「その瞬間の実観測」であり
        訂正対象ではない (本関数は非STABLE区間だけを直後のSTABLE値で
        遡って上書きする、docstring 参照)。受け入れ条件のt窓
        (t=6701.67-6717.03) は実データでもこの STABLE 行 (t=6701.667)
        を含まないように選ばれている。
        """
        rows = [
            _row(0.0, 0, True, False, "STABLE", "STABLE"),  # 凍結直前 (訂正対象外)
            _row(0.1, 0, True, False, "CHAIN", "STABLE"),   # 凍結 (誤、訂正対象)
            _row(0.2, 0, True, False, "CHAIN", "STABLE"),   # 凍結 (誤、訂正対象)
            _row(0.3, 0, False, False, "STABLE", "STABLE"),  # 連鎖解決、真の値
        ]
        corrected = vao._retroactively_correct_dead_dump_rows(rows)
        assert [r.is_dead1 for r in corrected] == [True, False, False, False]
        # 2P 側 (常に STABLE・False) は無変化
        assert [r.is_dead2 for r in corrected] == [False, False, False, False]

    def test_terminal_run_without_stable_recovery_is_unchanged(self) -> None:
        """試合終了 (game_idx 変化) まで STABLE に一度も復帰しない区間は、
        受け入れ条件「死亡見逃しゼロ」を最優先し **一切変更しない**
        (最後の STABLE 時点で真に死亡していた可能性を握りつぶさない)。
        """
        rows = [
            _row(0.0, 0, True, False, "STABLE", "STABLE"),
            _row(0.1, 0, True, False, "CHAIN", "STABLE"),
            _row(0.2, 0, True, False, "CHAIN", "STABLE"),  # 試合終了まで復帰なし
            _row(0.3, 1, False, False, "STABLE", "STABLE"),  # 次の試合 (別 game_idx)
        ]
        corrected = vao._retroactively_correct_dead_dump_rows(rows)
        assert [r.is_dead1 for r in corrected] == [True, True, True, False]

    def test_correction_does_not_leak_across_game_idx_boundary(self) -> None:
        """訂正は同一 game_idx 内に閉じる (次の試合の STABLE 値を前の試合の
        非STABLE区間に誤って伝播させない)。"""
        rows = [
            _row(0.0, 0, True, False, "CHAIN", "STABLE"),  # game0、末尾まで非STABLE
            _row(0.1, 1, False, False, "STABLE", "STABLE"),  # game1 (別試合、True ではない)
        ]
        corrected = vao._retroactively_correct_dead_dump_rows(rows)
        # game0 側は復帰なしのため無変化 (True のまま)。game1 の False が
        # game0 側に遡って伝播していないことを確認する。
        assert corrected[0].is_dead1 is True
        assert corrected[1].is_dead1 is False

    def test_multiple_excursions_each_corrected_independently(self) -> None:
        """1試合内に「凍結->復帰」が複数回あっても、各区間が直後の
        STABLE 値でそれぞれ独立に訂正される。"""
        rows = [
            _row(0.0, 0, True, False, "STABLE", "STABLE"),
            _row(0.1, 0, True, False, "CHAIN", "STABLE"),
            _row(0.2, 0, False, False, "STABLE", "STABLE"),  # 1回目: 解決してFalse
            _row(0.3, 0, False, False, "OJAMA_FALL", "STABLE"),  # 凍結 (旧False)
            _row(0.4, 0, True, False, "STABLE", "STABLE"),  # 2回目: 再び危険 True
        ]
        corrected = vao._retroactively_correct_dead_dump_rows(rows)
        assert [r.is_dead1 for r in corrected] == [True, False, False, True, True]

    def test_bit_identical_when_no_excursions(self) -> None:
        """凍結区間が一切無い (常に STABLE) dump は完全に無変化。"""
        rows = [
            _row(0.0, 0, False, True, "STABLE", "STABLE"),
            _row(0.1, 0, True, False, "STABLE", "STABLE"),
        ]
        corrected = vao._retroactively_correct_dead_dump_rows(rows)
        assert corrected == rows

    def test_default_flag_off_never_calls_correction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_pipeline: None,
    ) -> None:
        """enable_stable_confirmed_is_dead 省略時 (既定 False) は
        _retroactively_correct_dead_dump_rows が一切呼ばれない
        (bit-identical、backwards compat)。"""
        calls: list[int] = []
        monkeypatch.setattr(
            vao, "_retroactively_correct_dead_dump_rows",
            lambda rows: (calls.append(len(rows)), rows)[1],
        )
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
            dump_timeline_path=tmp_path / "dump.npz",
        )
        assert calls == []

    def test_flag_on_calls_correction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_pipeline: None,
    ) -> None:
        """enable_stable_confirmed_is_dead=True では dump 保存前に
        _retroactively_correct_dead_dump_rows が呼ばれる (配線確認)。"""
        calls: list[int] = []
        monkeypatch.setattr(
            vao, "_retroactively_correct_dead_dump_rows",
            lambda rows: (calls.append(len(rows)), rows)[1],
        )
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
            dump_timeline_path=tmp_path / "dump.npz",
            enable_stable_confirmed_is_dead=True,
        )
        assert calls == [0]  # MENU固定スタブのため rows=0件だが呼ばれたことは確認できる


# ============================
# Gate 3R-6 案A (2026-08-25): 非STABLE中の is_dead 判定保留 (リアルタイム版)
# enable_nonstable_hold_is_dead — 未来参照なし・現在の own state のみで保留を決める
# ============================

def _dead_board() -> Any:
    """窒息盤面 (3列目 index:2 の可視最上段 row=1 = DEATH_ROW にぷよ) を作る。"""
    from src.board import Board, DEATH_ROW, DEATH_COL
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[DEATH_ROW, DEATH_COL] = 1  # 赤
    return Board.from_list(grid.tolist())


class TestNonStableHoldIsDead:
    """`_resolve_nonstable_hold_is_dead` (純関数、stateless) の判定保留仕様。

    user 伝授の絶対律 (memory `reference_full_board_is_not_death_2026-08-22`):
    設置前 / 積み上げ中 / 連鎖直前 / 連鎖中は窒息としない。案A は
    「own state が STABLE でない = 落ち着いていない」の現在情報だけで
    保留を決める (未来参照なし = リアルタイム可能)。
    保留の表現は「False + 保留フラグ」(= unknown の2値エンコード。dump の
    state 列と合わせれば保留行を完全に復元できる)。
    """

    @pytest.mark.parametrize("state_name", [
        "CHAIN", "GRAVITY_SETTLE", "TSUMO_FALL", "OJAMA_FALL",
    ])
    def test_nonstable_holds_judgment_as_false(self, state_name: str) -> None:
        """非STABLE中は生判定 True でも窒息を主張しない (保留=held True)。"""
        recorded, held = vao._resolve_nonstable_hold_is_dead(True, state_name)
        assert recorded is False
        assert held is True

    def test_stable_passes_raw_true_through(self) -> None:
        """STABLE では生判定をそのまま確定する (真の窒息 t=223 の検出を維持)。"""
        recorded, held = vao._resolve_nonstable_hold_is_dead(True, "STABLE")
        assert recorded is True
        assert held is False

    def test_stable_passes_raw_false_through(self) -> None:
        recorded, held = vao._resolve_nonstable_hold_is_dead(False, "STABLE")
        assert recorded is False
        assert held is False

    def test_nonstable_raw_false_still_counts_as_held(self) -> None:
        """生判定 False の非STABLE行も「保留」(判定していない) として数える。
        0 が「起きていない」のか「測っていない」のかを区別するため。"""
        recorded, held = vao._resolve_nonstable_hold_is_dead(False, "CHAIN")
        assert recorded is False
        assert held is True

    def test_true_death_sequence_stable_rows_kept(self) -> None:
        """真の窒息 run (2P 実試合2 t=221.9-223.4 の実測パターン) を模した列で、
        STABLE 行の True が全て維持されることを確認する (最重要基準)。"""
        seq = [  # (raw_dead, state) — 実測 dump の並びを縮約
            (True, "STABLE"), (True, "STABLE"),      # 窒息確定 (維持必須)
            (True, "OJAMA_FALL"), (True, "OJAMA_FALL"),  # 保留区間
            (True, "STABLE"), (True, "STABLE"),      # 再確定 (維持必須)
        ]
        out = [vao._resolve_nonstable_hold_is_dead(d, s) for d, s in seq]
        assert [r for r, _ in out] == [True, True, False, False, True, True]


class TestIsDeadHoldStats:
    """保留カウンタ (母数付き)。「3/29」の形で可視化できること。"""

    def test_counts_held_and_suppressed_with_denominator(self) -> None:
        stats = vao._IsDeadHoldStats()
        # 1P: STABLE生False / CHAIN生True(抑制) / 2P: 全行STABLE
        stats.record(held1=False, suppressed1=False, held2=False, suppressed2=False)
        stats.record(held1=True, suppressed1=True, held2=False, suppressed2=False)
        stats.record(held1=True, suppressed1=False, held2=False, suppressed2=False)
        assert stats.total == 3
        assert stats.held1 == 2
        assert stats.suppressed1 == 1
        assert stats.held2 == 0
        s = stats.summary()
        assert "2/3" in s  # 母数付き (held1/total)
        assert "0/3" in s  # 2P 側も母数付き (0 と「未測定」を区別できる)

    def test_zero_rows_summary_shows_zero_denominator(self) -> None:
        """母数0 (dump行なし) は「0/0」と表示され「保留0件」と区別できる。"""
        assert "0/0" in vao._IsDeadHoldStats().summary()


class TestBuildRowIsDeadOverride:
    def test_default_none_uses_board_judgment(self) -> None:
        """override 省略 (既定 None) は従来通り Board.is_dead() を使う
        (フラグOFF既定経路の bit-identical 保証)。"""
        dead = _dead_board()
        row = vao._build_timeline_dump_row(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=0, pending_p2=0, room1=72, room2=72,
            b1=dead, b2=_empty_board(),
            drivers=[], score1=0, score2=0, state1="STABLE", state2="STABLE",
        )
        assert row.is_dead1 is True
        assert row.is_dead2 is False

    def test_override_replaces_board_judgment(self) -> None:
        """is_dead1/is_dead2 override 指定時は盤面判定より優先される
        (保留適用済みの値を dump に記録する経路)。"""
        dead = _dead_board()
        row = vao._build_timeline_dump_row(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
            pending_p1=0, pending_p2=0, room1=72, room2=72,
            b1=dead, b2=dead,
            drivers=[], score1=0, score2=0, state1="CHAIN", state2="STABLE",
            is_dead1=False, is_dead2=True,
        )
        assert row.is_dead1 is False  # 盤面は窒息形だが保留 (非STABLE)
        assert row.is_dead2 is True

    def test_generate_accepts_new_flag_smoke(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """generate() が enable_nonstable_hold_is_dead を受け付ける
        (optional 引数追加のみ = backwards compat)。"""
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
            dump_timeline_path=tmp_path / "dump.npz",
            enable_nonstable_hold_is_dead=True,
        )
        assert written > 0
        assert (tmp_path / "dump.npz").exists()
