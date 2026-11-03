#!/bin/bash
# 旧glogのMSVC専用移植層を避け、MinGW対応版を専用領域へ静的ビルドする。
set -euo pipefail
root=<home>/obs_20261020_toolchain
cd "$root/debs"
apt-get download cmake cmake-data libjsoncpp25 librhash0 libarchive13t64 libuv1t64
for package in cmake*.deb libjsoncpp*.deb librhash*.deb libarchive*.deb libuv*.deb; do dpkg-deb -x "$package" "$root"; done
cd "$root"
curl -L --fail https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz -o glog-0.6.0.tar.gz
tar xf glog-0.6.0.tar.gz
export PATH="$root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
cmake -S glog-0.6.0 -B glog-build -G Ninja -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_C_COMPILER="$root/usr/bin/x86_64-w64-mingw32-gcc-posix" \
  -DCMAKE_CXX_COMPILER="$root/usr/bin/x86_64-w64-mingw32-g++-posix" \
  -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DWITH_UNWIND=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/glog-install"
cmake --build glog-build -j 2
cmake --install glog-build
