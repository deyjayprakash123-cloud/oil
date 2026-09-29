"""
Baghewala Digital Twin - Root Entry Point
Oil India Limited | Smart India Hackathon 2026
"""

import sys
import os

# Append project directory to path
pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_digital_twin")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# Execute main application
import app
