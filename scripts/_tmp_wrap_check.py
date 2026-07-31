import cv2
import scripts._collect_1t  # noqa: F401  (import 時に setNumThreads(1) が走る)

print("wrapper import OK, cv2 threads =", cv2.getNumThreads())
