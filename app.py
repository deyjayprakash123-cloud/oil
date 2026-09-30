"""
Baghewala Digital Twin - Root Entry Point
Oil India Limited | Smart India Hackathon 2026
"""

import sys
import os
import runpy

# Append digital twin package directory to sys.path
pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_digital_twin")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# Execute digital twin application in main context
target_file = os.path.join(pkg_dir, "app.py")
runpy.run_path(target_file, run_name="__main__")
