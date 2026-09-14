"""前段の成功driverを専用aliasで読み、共有sourceは変更しない。"""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parent
SETTLED=ROOT.parent/'g2_conditional_firing_settlement_2026-09-10_v1'
sys.path.insert(0,str(SETTLED))
spec=importlib.util.spec_from_file_location('_postsettlement_existing_full',SETTLED/'run_full.py')
R=importlib.util.module_from_spec(spec); sys.modules[spec.name]=R
spec.loader.exec_module(R)
E,C=R.E,R.C
