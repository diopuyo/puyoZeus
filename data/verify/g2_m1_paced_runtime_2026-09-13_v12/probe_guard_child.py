"""guard検査専有の無害な子。GPUを使用しない。"""
import sys
import time

if __name__ == '__main__':
    time.sleep(float(sys.argv[1]))
