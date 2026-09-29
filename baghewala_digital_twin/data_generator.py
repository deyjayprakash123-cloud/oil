"""
Baghewala Digital Twin - Synthetic Data Generator (data_generator.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

Generates synthetic CSS cycles and SRP dynamics with realistic field noise (±5%)
based on the physics engine for training ML models and offline experimentation.
"""

import numpy as np
import pandas as pd
import os
import sys

# Support direct and relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from physics_engine import (
    calculate_reservoir_temperature,
    calculate_oil_viscosity,
    calculate_rod_load,
    calculate_steam_oil_ratio,
    calculate_pump_efficiency
)


def generate_css_cycle_data(num_cycles: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic CSS cycles by sampling operational parameters and computing
    resulting reservoir temperature, viscosity, oil production, and rod load
    with realistic field sensor noise (±5%).

    Parameters:
    -----------
    num_cycles : int
        Number of synthetic cycles to simulate (e.g., 1000 to 10000)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Columns: Steam_Volume, Injection_Pressure, Soak_Days, Production_Days,
                 SPM, Stroke_Length, Reservoir_Temp, Oil_Viscosity,
                 Oil_Production, Rod_Load, MPRL, SOR, Pump_Efficiency, Rod_Floating_Flag
    """
    np.random.seed(random_seed)    # 1. Parameter distributions according to specifications
    steam_volume = np.random.uniform(5000.0, 20000.0, num_cycles)       # barrels
    injection_pressure = np.random.uniform(1000.0, 3000.0, num_cycles) # psi
    soak_days = np.random.randint(3, 11, num_cycles)                    # days
    production_days = np.random.randint(30, 91, num_cycles)             # days
    spm = np.random.uniform(4.0, 12.0, num_cycles)                      # strokes/min
    stroke_length = np.random.uniform(40.0, 120.0, num_cycles)          # inches
    pump_depth = np.random.uniform(700.0, 1300.0, num_cycles)           # meters
    initial_res_temp = np.random.uniform(44.0, 52.0, num_cycles)        # native reservoir temp °C
    api_gravity = np.random.uniform(14.0, 18.5, num_cycles)             # crude API gravity (14.5 - 18.0)

    # Steam temperature derived from saturation curve vs injection pressure
    steam_temp = 115.0 * ((injection_pressure / 14.5038) ** 0.22)

    # 2. Physics calculations for each cycle
    oil_production_list = []
    avg_temp_list = []
    avg_visc_list = []
    rod_load_list = []
    mprl_list = []
    sor_list = []
    pump_eff_list = []
    rod_floating_list = []

    for i in range(num_cycles):
        # Peak temperature post-soak
        soak_retention = np.exp(-0.015 * soak_days[i])
        peak_t = initial_res_temp[i] + (steam_temp[i] - initial_res_temp[i]) * soak_retention

        # Average reservoir temperature during production
        temps = calculate_reservoir_temperature(
            initial_temp=initial_res_temp[i],
            steam_temp=steam_temp[i],
            soak_days=int(soak_days[i]),
            production_days=int(production_days[i]),
            pump_depth=pump_depth[i]
        )
        avg_temp = float(np.mean(temps))
        avg_temp_list.append(avg_temp)

        # Average oil viscosity at average temperature
        avg_visc = calculate_oil_viscosity(avg_temp, api_gravity=api_gravity[i])
        avg_visc_list.append(avg_visc)

        # Theoretical pump displacement in bpd (D_plunger = 2.25")
        theo_disp_bpd = 0.1166 * (2.25 ** 2) * stroke_length[i] * spm[i]

        # Heavy oil production rate calibrated to steam energy, API gravity & viscosity thinning
        steam_factor = (steam_volume[i] / 12000.0) ** 0.45
        api_factor = (api_gravity[i] / 15.5) ** 0.8
        visc_factor = (100.0 / max(avg_visc, 10.0)) ** 0.35
        # Total cycle oil production in barrels
        avg_daily_oil = theo_disp_bpd * 0.65 * steam_factor * visc_factor * api_factor
        total_oil = avg_daily_oil * production_days[i]

        # Add ±5% sensor and reservoir heterogeneity noise
        noise_factor = np.random.uniform(0.95, 1.05)
        total_oil_with_noise = max(total_oil * noise_factor, 150.0)
        oil_production_list.append(float(np.round(total_oil_with_noise, 2)))

        # Sucker rod load dynamics
        rod_info = calculate_rod_load(
            spm=spm[i],
            stroke_length=stroke_length[i],
            fluid_viscosity=avg_visc,
            pump_depth=pump_depth[i],
            api_gravity=api_gravity[i]
        )
        # Apply ±5% noise to load cells
        rod_load_noisy = rod_info["PPRL"] * np.random.uniform(0.95, 1.05)
        mprl_noisy = rod_info["MPRL"] * np.random.uniform(0.95, 1.05)
        rod_load_list.append(float(np.round(rod_load_noisy, 2)))
        mprl_list.append(float(np.round(mprl_noisy, 2)))
        rod_floating_list.append(bool(mprl_noisy <= 0.0))

        # SOR & Pump Efficiency
        sor = calculate_steam_oil_ratio(total_oil_with_noise, steam_volume[i])
        sor_list.append(sor)

        actual_daily = total_oil_with_noise / production_days[i]
        eff = calculate_pump_efficiency(actual_daily, theo_disp_bpd)
        pump_eff_list.append(eff)

    df = pd.DataFrame({
        "Steam_Volume": np.round(steam_volume, 1),
        "Injection_Pressure": np.round(injection_pressure, 1),
        "Soak_Days": soak_days,
        "Production_Days": production_days,
        "SPM": np.round(spm, 2),
        "Stroke_Length": np.round(stroke_length, 1),
        "Reservoir_Temp": np.round(avg_temp_list, 2),
        "API_Gravity": np.round(api_gravity, 2),
        "Oil_Viscosity": np.round(avg_visc_list, 2),
        "Oil_Production": oil_production_list,
        "Rod_Load": rod_load_list,
        "MPRL": mprl_list,
        "SOR": sor_list,
        "Pump_Efficiency": pump_eff_list,
        "Rod_Floating_Flag": rod_floating_list
    })

    return df


def generate_rod_failure_data(num_samples: int = 5000, random_seed: int = 101) -> pd.DataFrame:
    """
    Generate synthetic sucker rod operational records with varying SPM,
    viscosity, stroke length, rod load, pump depth, and API gravity.
    """
    np.random.seed(random_seed)

    spm = np.random.uniform(4.0, 14.0, num_samples)
    stroke_length = np.random.uniform(50.0, 140.0, num_samples)
    api_gravity = np.random.uniform(14.0, 18.5, num_samples)
    viscosity = np.where(
        np.random.rand(num_samples) < 0.25,
        np.random.uniform(2500.0, 15000.0, num_samples),  # cold/slugging heavy oil
        np.random.uniform(15.0, 800.0, num_samples)       # warm/steam thinned oil
    )
    pump_depth = np.random.uniform(600.0, 1400.0, num_samples)

    pprl_list = []
    mprl_list = []
    failure_risk_list = []

    safe_pprl_limit = 24000.0  # API Grade D sucker rod working tensile limit (lbs)

    for i in range(num_samples):
        rod_info = calculate_rod_load(
            spm=spm[i],
            stroke_length=stroke_length[i],
            fluid_viscosity=viscosity[i],
            pump_depth=pump_depth[i],
            api_gravity=api_gravity[i]
        )

        noise_pprl = np.random.normal(1.0, 0.02)
        noise_mprl = np.random.normal(1.0, 0.02)

        pprl_val = float(rod_info["PPRL"] * noise_pprl)
        mprl_val = float(rod_info["MPRL"] * noise_mprl)

        is_failure = int(mprl_val <= 0.0 or pprl_val > safe_pprl_limit or rod_info["Rod_Floating_Flag"])

        pprl_list.append(np.round(pprl_val, 2))
        mprl_list.append(np.round(mprl_val, 2))
        failure_risk_list.append(is_failure)

    df = pd.DataFrame({
        "SPM": np.round(spm, 2),
        "Stroke_Length": np.round(stroke_length, 1),
        "Viscosity": np.round(viscosity, 2),
        "API_Gravity": np.round(api_gravity, 2),
        "Rod_Load": pprl_list,
        "Pump_Depth": np.round(pump_depth, 1),
        "MPRL": mprl_list,
        "Rod_Failure_Risk": failure_risk_list
    })

    return df


if __name__ == "__main__":
    print("Generating sample CSS cycle data...")
    css_df = generate_css_cycle_data(100)
    print(css_df.head(3))
    print("\nGenerating sample rod failure data...")
    rod_df = generate_rod_failure_data(100)
    print(rod_df.head(3))
    print(f"Rod Failure Rate in synthetic sample: {rod_df['Rod_Failure_Risk'].mean() * 100:.1f}%")
