"""
Baghewala Digital Twin - Well Database Root Link
Oil India Limited | Smart India Hackathon 2026
"""

import sys
import os
import importlib.util

_inner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_digital_twin", "well_data.py")
_spec = importlib.util.spec_from_file_location("baghewala_well_data_inner", _inner_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

WELL_DATABASE = _mod.WELL_DATABASE
CUSTOM_SCENARIO_DEFAULTS = _mod.CUSTOM_SCENARIO_DEFAULTS
