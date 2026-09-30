ぷよぷよ有利不利オーバーレイ (配布版) セットアップ
================================================

1. 準備
   - このフォルダ (PuyoLive) を書き込み可能な場所 (例: D:\PuyoLive) に展開する。
     Program Files など書き込み制限のある場所には置かない (機器別の色較正を app\config に保存するため)。
   - puyo_live.example.json をコピーして puyo_live.json を作る。
     機器を選ぶのは device_index (0,1,2...) で、device_name は色較正の保存名 (名札) にすぎない。
     index が分からないときは、OBS の仮想カメラを開始した状態で
         PuyoLive.bat --list-devices
     を実行し、「映像あり 1920x1080」と出る index を device_index に書く。
     port を変える場合は OBS 側の URL も同じ番号にする。
     機器別の色較正の保存先は "calibration_location" で選ぶ:
         "app" (既定)   ... このフォルダの app\config\device_calibration
         "localappdata" ... %LOCALAPPDATA%\PuyoLive\device_calibration
     Program Files など書き込めない場所に置くときは "localappdata" にする。

2. OBS 側
   - ボードのキャプチャ映像を「ソース出力」で仮想カメラに流す
     (配信用の重ね表示を含めない。ぷよぷよ画面のみ、1920x1080 / 16:9)。
   - 仮想カメラを開始する。
   - 「ブラウザ」ソースを追加し、URL に http://127.0.0.1:8765/ を入れる (幅 1920 / 高さ 1080 を推奨)。

3. 起動
   - PuyoLive.bat をダブルクリックする。
   - 初回は「入力確認中」→「色を較正中」の表示が出て、判定は較正が終わるまで出ない。
   - 終了はコマンドウィンドウを閉じるか Ctrl+C。

4. うまくいかないとき
   - [設定エラー]     : puyo_live.json の書き方。メッセージの項目名を直す。
   - [配布物エラー]   : ファイルが欠けている/書き換わっている。展開し直す。
   - 画面に判定が出ない: OBS の仮想カメラが開始済みか、映像がぷよぷよ画面 (1920x1080) か確認する。
   - ログと記録は output フォルダに出る。

動作に必要なもの: Windows 10/11 (64bit)。GPU は不要 (CPU のみで動作)。
Visual C++ ランタイム (vcruntime140 / msvcp140 等) は python フォルダに同梱済み。
不足の場合は起動時に [実行環境エラー] で案内が出る (公式: https://aka.ms/vs/17/release/vc_redist.x64.exe)。

ライセンス・帰属表記
  同梱ライブラリのライセンスは LICENSES フォルダを参照。
  - 動画ファイル入力 (source=video) は OpenCV 同梱の FFmpeg 4.4.6 (LGPL-2.1-or-later、cv2\opencv_videoio_ffmpeg*.dll) を使う。
    この DLL は差し替え可能な別ファイルのまま同梱している。対応するソース:
      FFmpeg 4.4.6   https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz
      OpenCV         https://github.com/opencv/opencv-python
  - Portions of this software are copyright (c) The FreeType Project (www.freetype.org). All rights reserved.
  - Microsoft Visual C++ ランタイム DLL は Microsoft の再頒布可能コードとして同梱している。
