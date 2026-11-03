#!/bin/bash
# Linux原票と同じlibstdc++の順位処理を保つため、ローカル展開だけでMinGWを用意する。
set -euo pipefail
root=<home>/obs_20261020_toolchain
mkdir -p "$root/debs"
cd "$root/debs"
apt-get download g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix-runtime mingw-w64-x86-64-dev mingw-w64-common binutils-mingw-w64-x86-64
for package in ./*.deb; do dpkg-deb -x "$package" "$root"; done
"$root/usr/bin/x86_64-w64-mingw32-g++-posix" --version
