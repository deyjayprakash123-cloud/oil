"""
Baghewala Digital Twin - Well Database Root Link
Oil India Limited | Smart India Hackathon 2026
"""

import sys
import os

pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_digital_twin")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from well_data import WELL_DATABASE, CUSTOM_SCENARIO_DEFAULTS
