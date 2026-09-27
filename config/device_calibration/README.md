機器別の色較正JSONの保存先。ファイル名は `sha256(機器名:index).json`。
DirectShowSourceのDeviceConfig.calibration_pathがパスを返す。
較正の生成・適用は次段で実装する。機器名だけで自動選択しない。
