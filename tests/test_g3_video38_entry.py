"""G3実入口の範囲束縛・実入力・終了解放を限定CPUで反証する。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
import unittest

from scripts import g3_video38_entry as G


def common() -> N:
    """元関数globalsとnamespaceが異なる実構造を再現する。"""
    space = dict(FIRST=29052, END=36902, FPS=60, STRIDE=2,
                 FRAMES=tuple(range(29052, 36902, 2)), __file__=str(G.COMMON_FILE))
    exec(compile('def bounds():\n return dict(first_frame=FIRST,end_exclusive=END)\n',
                 str(G.COMMON_FILE), 'exec'), space)
    return N(**space)


class Camera:
    def __init__(self, end: int = G.END, fail_release: bool = False) -> None:
        self.position, self.end = 0, end
        self.closed, self.fail_release = False, fail_release

    def get(self, key: int) -> float:
        return float(self.position)

    def set(self, key: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, object]:
        if self.position == self.end:
            return False, None
        self.position += 1
        return True, object()

    def release(self) -> None:
        if self.fail_release:
            raise OSError('fixture_release')
        self.closed = True


class EntryTests(unittest.TestCase):
    def fixture(self, camera: Camera) -> tuple[N, dict]:
        cv = N(VideoCapture=lambda *a, **k: camera, CAP_PROP_FPS=5, CAP_PROP_POS_FRAMES=1)
        return N(cv2=cv), dict(decoded_frames=0)

    def test_real_function_globals_changed_and_restored(self) -> None:
        k, secondary = common(), common()
        ns = dict(K=k, S=N(K=secondary), Q=N(K=secondary), R=N(K=secondary))
        exec('def entry(): pass', ns)
        h = dict(FIRST_FRAME=29052, END_FRAME=36902)
        exec('def begin_frame(): pass', h)
        history = N(HistoryRecorder=N(begin_frame=h['begin_frame']))
        with ExitStack() as stack:
            receipt = G.bind_bounds(stack, ns['entry'], history)
            self.assertEqual(k.bounds()['first_frame'], 0)
            self.assertEqual(secondary.bounds()['end_exclusive'], 36300)
            self.assertEqual(k.FIRST, 0)
            self.assertEqual(h['FIRST_FRAME'], 0)
            self.assertEqual(len(receipt), 5)
        self.assertEqual(k.FIRST, 29052)
        self.assertEqual(k.bounds()['end_exclusive'], 36902)
        self.assertEqual(h['FIRST_FRAME'], 29052)
        self.assertEqual(h['END_FRAME'], 36902)

    def test_exact_exclusive_end_without_reading_end(self) -> None:
        camera = Camera()
        collector, state = self.fixture(camera)
        original = collector.cv2.VideoCapture
        with G.capture_scope(collector, state):
            cap = collector.cv2.VideoCapture(G.SOURCE)
            self.assertEqual(cap.get(5), 60)
            for frame in range(G.END):
                self.assertTrue(cap.read()[0])
            with self.assertRaisesRegex(ValueError, 'decode_continuity'):
                cap.read()
        self.assertEqual(state['decoded_frames'], 36300)
        self.assertTrue(camera.closed)
        self.assertIs(collector.cv2.VideoCapture, original)

    def test_wrong_video_and_seek_rejected(self) -> None:
        collector, state = self.fixture(Camera())
        with G.capture_scope(collector, state):
            with self.assertRaisesRegex(ValueError, 'actual_capture_source'):
                collector.cv2.VideoCapture(G.SOURCE.with_name('video_39.mp4'))
            cap = collector.cv2.VideoCapture(G.SOURCE)
            with self.assertRaisesRegex(ValueError, 'seek_forbidden'):
                cap.set(1, 29052)

    def test_early_eof_preserves_count_and_closes(self) -> None:
        camera = Camera(end=1)
        collector, state = self.fixture(camera)
        with self.assertRaisesRegex(ValueError, 'early_eof'):
            with G.capture_scope(collector, state):
                cap = collector.cv2.VideoCapture(G.SOURCE)
                cap.read()
                cap.read()
        self.assertEqual(state['decoded_frames'], 1)
        self.assertTrue(state['captures_closed'])

    def test_release_error_preserved_with_original_exception(self) -> None:
        collector, state = self.fixture(Camera(fail_release=True))
        with self.assertRaisesRegex(RuntimeError, 'original_failure'):
            with G.capture_scope(collector, state):
                collector.cv2.VideoCapture(G.SOURCE)
                raise RuntimeError('original_failure')
        self.assertFalse(state['captures_closed'])
        self.assertEqual(len(state['capture_release_errors']), 1)


if __name__ == '__main__':
    unittest.main()
