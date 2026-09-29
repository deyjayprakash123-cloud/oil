"""
Baghewala Digital Twin - Well Database (well_data.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

Static reservoir, formation, and geological property profiles for
Baghewala heavy oil wells in the Bikaner-Nagaur Basin, Rajasthan.
"""

WELL_DATABASE = {
    "BG-01": {
        "name": "Baghewala-01 (Main Pad)",
        "depth_m": 1020.0,
        "api_gravity": 15.8,
        "viscosity_cp": 7800.0,
        "reservoir_temp_c": 47.0,
        "porosity_pct": 18.0,
        "permeability_md": 500.0,
        "reservoir_pressure_psi": 1200.0,
        "formation": "Jodhpur Sandstone"
    },
    "BG-04": {
        "name": "Baghewala-04 (North Pad)",
        "depth_m": 1050.0,
        "api_gravity": 17.2,
        "viscosity_cp": 6200.0,
        "reservoir_temp_c": 48.5,
        "porosity_pct": 20.0,
        "permeability_md": 650.0,
        "reservoir_pressure_psi": 1350.0,
        "formation": "Jodhpur Sandstone"
    },
    "BG-07": {
        "name": "Baghewala-07 (East Flank)",
        "depth_m": 980.0,
        "api_gravity": 14.5,
        "viscosity_cp": 9500.0,
        "reservoir_temp_c": 46.0,
        "porosity_pct": 15.0,
        "permeability_md": 350.0,
        "reservoir_pressure_psi": 1050.0,
        "formation": "Jodhpur Sandstone"
    },
    "BG-12": {
        "name": "Baghewala-12 (Central Core)",
        "depth_m": 1015.0,
        "api_gravity": 16.5,
        "viscosity_cp": 7100.0,
        "reservoir_temp_c": 47.5,
        "porosity_pct": 19.0,
        "permeability_md": 580.0,
        "reservoir_pressure_psi": 1250.0,
        "formation": "Jodhpur Sandstone"
    },
    "BG-15": {
        "name": "Baghewala-15 (South Extension)",
        "depth_m": 1080.0,
        "api_gravity": 18.0,
        "viscosity_cp": 5500.0,
        "reservoir_temp_c": 49.0,
        "porosity_pct": 22.0,
        "permeability_md": 720.0,
        "reservoir_pressure_psi": 1400.0,
        "formation": "Jodhpur Sandstone"
    }
}

CUSTOM_SCENARIO_DEFAULTS = {
    "name": "Custom Field Scenario",
    "depth_m": 1000.0,
    "api_gravity": 16.0,
    "viscosity_cp": 8000.0,
    "reservoir_temp_c": 48.0,
    "porosity_pct": 18.5,
    "permeability_md": 520.0,
    "reservoir_pressure_psi": 1250.0,
    "formation": "Jodhpur Sandstone (Custom)"
}
