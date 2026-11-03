"""B20: 検証用 device 'B8-candidate' の較正 profile を zenchi-video の種で毎回初期化する (持ち越し防止)。"""
import json

from scripts.measure_live_b6 import save
from src.phase_j.live_device import DeviceConfig

seed = DeviceConfig('zenchi-video', 0, True).calibration_path
device = DeviceConfig('B8-candidate', 0, True)
profile = json.loads(seed.read_text(encoding='utf-8'))
profile['device'] = dict(name=device.name, index=0)
save(device.calibration_path, profile)
print('seeded', device.calibration_path)
