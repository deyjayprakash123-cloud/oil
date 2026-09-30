"""
Baghewala Digital Twin - Physics Engine (physics_engine.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

Physics & Petroleum Engineering Models:
1. Marx-Langenheim heat loss & thermal distribution for heavy oil CSS
2. Arrhenius equation for non-Newtonian heavy oil viscosity reduction
3. API RP 11L polished rod load dynamics (PPRL, MPRL, Rod Floating)
4. Steam-to-Oil Ratio (SOR) and Pump Volumetric Efficiency
5. Comprehensive CSS Huff-and-Puff cycle simulator
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


# ============================================================================
# 1. RESERVOIR TEMPERATURE (Marx-Langenheim Heat Loss Model)
# ============================================================================

def calculate_reservoir_temperature(
    initial_temp: float = 48.0,
    steam_temp: float = 290.0,
    soak_days: int = 5,
    production_days: int = 60,
    thermal_conductivity: float = 1.73,
    pump_depth: float = 1000.0
) -> np.ndarray:
    """
    Use a simplified Marx-Langenheim model for heat loss.
    Returns a time-series array of reservoir temperature over the production cycle.
    Deeper wells (e.g. BG-15 at 1080m) experience greater conductive heat loss through the overburden.

    Parameters:
    -----------
    initial_temp : float
        Native Baghewala reservoir temperature in °C (~46-49°C)
    steam_temp : float
        Injected steam saturation temperature in °C (~260-310°C)
    soak_days : int
        Duration of steam soaking phase in days (3-10 days)
    production_days : int
        Duration of production phase in days (30-90 days)
    thermal_conductivity : float
        Overburden & underburden thermal conductivity in W/(m·K) (~1.5-2.2)
    pump_depth : float
        Well depth in meters (deeper wells lose more heat to surrounding rock)

    Returns:
    --------
    np.ndarray
        Array of reservoir temperatures for each production day in °C.
    """
    days = np.arange(1, production_days + 1)

    # Deeper wells exhibit larger surface area for heat conduction into overburden/underburden
    depth_cooling_factor = (max(pump_depth, 500.0) / 1000.0) ** 0.65

    # Heat loss coefficient calibrated to Marx-Langenheim decay
    decay_rate = (0.012 + 0.004 * (thermal_conductivity / 1.73)) * depth_cooling_factor

    # Temperature at start of production after soaking period
    soak_retention = np.exp(-0.015 * max(soak_days, 0))
    peak_temp = initial_temp + (steam_temp - initial_temp) * soak_retention

    # Marx-Langenheim conductive dissipation over production days
    temp_profile = initial_temp + (peak_temp - initial_temp) * np.exp(-decay_rate * days)
    temp_profile = np.maximum(temp_profile, initial_temp)

    return np.round(temp_profile, 2)


# ============================================================================
# 2. HEAVY OIL VISCOSITY (Arrhenius Equation)
# ============================================================================

def calculate_oil_viscosity(
    temperature: float,
    reference_viscosity: float = 8500.0,
    reference_temp: float = 48.0,
    activation_energy: float = 45000.0,
    api_gravity: float = 15.5
) -> float:
    """
    Use the Arrhenius equation:
        mu = mu_ref * exp( (Ea / R) * (1/T - 1/T_ref) )

    Handles heavy oil behavior (Baghewala crude: 14.5-18.0° API).
    Lower API gravity (e.g. BG-07 at 14.5°) contains higher asphaltic content,
    yielding higher flow activation energy and significantly higher viscosity at all temperatures.

    Parameters:
    -----------
    temperature : float
        Current temperature in °C
    reference_viscosity : float
        Dead oil viscosity at reference reservoir temp in cP (~5500-9500 cP)
    reference_temp : float
        Reference reservoir temperature in °C (typically 46-49°C)
    activation_energy : float
        Flow activation energy Ea in J/mol (~42,000 - 52,000 J/mol)
    api_gravity : float
        API gravity (e.g., 14.5 for BG-07 up to 18.0 for BG-15)

    Returns:
    --------
    float
        Viscosity in cP
    """
    r_gas_constant = 8.314  # Universal Gas Constant, J/(mol·K)

    # Convert Celsius to Kelvin
    t_kelvin = np.clip(temperature + 273.15, 273.15 + 10.0, 273.15 + 350.0)
    t_ref_kelvin = reference_temp + 273.15

    # Scale activation energy based on API gravity (lower API = thicker oil, higher Ea)
    api_factor = (16.0 / max(api_gravity, 10.0)) ** 0.70
    effective_ea = activation_energy * api_factor

    # Arrhenius exponent
    exponent = (effective_ea / r_gas_constant) * (1.0 / t_kelvin - 1.0 / t_ref_kelvin)

    # Calculate dynamic viscosity
    visc = reference_viscosity * np.exp(exponent)

    # Physical heavy oil bounds in reservoir under steam (1.5 cP minimum, 60,000 cP max)
    return float(np.clip(visc, 1.5, 60000.0))


# ============================================================================
# 3. SUCKER ROD PUMP POLISHED ROD LOAD (API RP 11L)
# ============================================================================

def calculate_rod_load(
    spm: float = 5.0,
    stroke_length: float = 100.0,
    fluid_viscosity: float = 85.0,
    pump_depth: float = 1000.0,
    rod_diameter: float = 1.0,
    fluid_density: float = None,
    api_gravity: float = 15.5
) -> Dict[str, Any]:
    """
    Implement API RP 11L standard equations for polished rod load in heavy oil wells.

    Calculates:
    - Peak Polished Rod Load (PPRL)
    - Minimum Polished Rod Load (MPRL)
    - Rod Floating Index & Status (if MPRL < 0, rod floating occurs)

    Parameters:
    -----------
    spm : float
        Pumping speed in strokes per minute (4-12 SPM)
    stroke_length : float
        Surface polished rod stroke length in inches (40-120 in)
    fluid_viscosity : float
        Produced crude viscosity at pump intake in cP
    pump_depth : float
        Pump setting depth in meters (500-1500 m)
    rod_diameter : float
        Average sucker rod diameter in inches (typically 0.875" - 1.125")
    fluid_density : float, optional
        Produced fluid density in kg/m3 (~950-1020 kg/m3)
    api_gravity : float
        Crude API gravity (lower API = denser, more viscous crude)

    Returns:
    --------
    Dict[str, Any]
        {
            'PPRL': float (lbs),
            'MPRL': float (lbs),
            'Rod_Floating_Flag': bool,
            'Rod_Floating_Index': float,
            'Viscous_Drag': float (lbs)
        }
    """
    # Unit conversions
    depth_ft = pump_depth * 3.28084
    if fluid_density is None:
        oil_sg = 141.5 / (api_gravity + 131.5)
        # Formation fluid mix (water + oil)
        mixed_sg = 0.65 * oil_sg + 0.35 * 1.02
        fluid_density = mixed_sg * 1000.0
    else:
        mixed_sg = fluid_density / 1000.0

    fluid_gradient_psi_ft = 0.433 * mixed_sg

    # Rod cross-sectional area and weight in air
    rod_area_sq_in = (np.pi / 4.0) * (rod_diameter ** 2)
    # Unit weight of steel rod ~ 3.4 lb/(sq in * ft)
    unit_rod_wt_lbs_ft = 3.4 * rod_area_sq_in
    w_r = unit_rod_wt_lbs_ft * depth_ft

    # Buoyant rod weight Wrf
    buoyancy_factor = 1.0 - (fluid_density / 7850.0)  # Steel density ~7850 kg/m3
    w_rf = w_r * buoyancy_factor

    # Plunger area (typical heavy oil bore Dp = 2.25 inches)
    d_plunger_in = 2.25
    a_plunger_sq_in = (np.pi / 4.0) * (d_plunger_in ** 2)
    f_o = fluid_gradient_psi_ft * depth_ft * a_plunger_sq_in  # Fluid Load Fo in lbs

    # API RP 11L acceleration factor (alpha = S * N^2 / 70500)
    alpha = (stroke_length * (spm ** 2)) / 70500.0

    # Downhole viscous friction drag in heavy oil (Mills & API RP 11L heavy crude standard)
    # Drag opposes downward rod fall on downstroke, and increases upward load on upstroke.
    # Lower API gravity (thicker crude) and higher initial viscosity trigger earlier rod floating.
    avg_velocity_fps = (2.0 * (stroke_length / 12.0) * spm) / 60.0
    annular_clearance_factor = 1.0 / max(2.875 - rod_diameter, 0.5)  # 2-7/8" tubing
    api_visc_multiplier = (16.0 / max(api_gravity, 10.0)) ** 0.5
    visc_shear_factor = ((fluid_viscosity * api_visc_multiplier) / 100.0) ** 0.65
    viscous_drag_lbs = 0.18 * visc_shear_factor * (depth_ft / 1000.0) * (avg_velocity_fps ** 1.35) * 550.0 * annular_clearance_factor

    # Peak Polished Rod Load (Upstroke: buoyant rod weight + dynamic acceleration + fluid load + viscous drag)
    pprl = w_rf * (1.0 + 0.95 * alpha) + f_o * (1.05 + 0.1 * alpha) + viscous_drag_lbs

    # Minimum Polished Rod Load (Downstroke: buoyant rod weight * (1 - alpha) - viscous drag)
    # If viscous drag + downward inertial deceleration exceeds buoyant weight, MPRL becomes negative!
    mprl = w_rf * (1.0 - 0.90 * alpha) - viscous_drag_lbs

    # Rod floating condition:
    rod_floating_flag = bool(mprl <= 0.0)

    # Rod floating index: ratio of retarding forces to downward gravitational force
    downward_gravity = max(w_rf * (1.0 - 0.90 * alpha), 1.0)
    rod_floating_index = float(viscous_drag_lbs / downward_gravity)

    return {
        "PPRL": float(np.round(pprl, 2)),
        "MPRL": float(np.round(mprl, 2)),
        "Rod_Floating_Flag": rod_floating_flag,
        "Rod_Floating_Index": float(np.round(rod_floating_index, 3)),
        "Viscous_Drag": float(np.round(viscous_drag_lbs, 2)),
        "Acceleration_Factor": float(np.round(alpha, 4))
    }


# ============================================================================
# 4. STEAM-TO-OIL RATIO (SOR)
# ============================================================================

def calculate_steam_oil_ratio(oil_produced: float, steam_injected: float) -> float:
    """
    Calculate Steam-to-Oil Ratio (SOR):
        SOR = steam_injected / oil_produced
    Returns infinity gracefully if oil_produced is zero and steam is non-zero.
    """
    if oil_produced <= 0:
        return float('inf') if steam_injected > 0 else 0.0
    return float(np.round(steam_injected / oil_produced, 2))


# ============================================================================
# 5. PUMP EFFICIENCY
# ============================================================================

def calculate_pump_efficiency(actual_production: float, theoretical_production: float) -> float:
    """
    Returns volumetric pump efficiency percentage:
        Efficiency = (actual_production / theoretical_production) * 100
    """
    if theoretical_production <= 0:
        return 0.0
    eff = (actual_production / theoretical_production) * 100.0
    return float(np.round(np.clip(eff, 0.0, 100.0), 2))


# ============================================================================
# 6. MASTER CSS CYCLE SIMULATOR
# ============================================================================

def simulate_css_cycle(
    steam_volume: float = 12000.0,
    spm: float = 6.0,
    stroke_length: float = 80.0,
    pump_depth: float = 1000.0,
    reservoir_temp: float = None,
    oil_viscosity: float = None,
    api_gravity: float = 15.5,
    injection_pressure: float = 1800.0,
    soak_days: int = 5,
    production_days: int = 60,
    rod_diameter: float = 1.0,
    initial_res_temp: float = None,
    steam_temp: float = 290.0,
    ref_viscosity: float = None,
    activation_energy: float = 45000.0,
    fluid_density: float = None
) -> pd.DataFrame:
    """
    Simulate CSS cycle for a specific well.

    Parameters:
    -----------
    steam_volume : float
        Steam injection volume (bbl)
    spm : float
        Pumping speed in strokes/min
    stroke_length : float
        Stroke length in inches
    pump_depth : float
        Well/pump depth in meters (e.g., BG-15 at 1080m loses more heat)
    reservoir_temp : float, optional
        Initial reservoir temperature in °C
    oil_viscosity : float, optional
        Initial oil viscosity at reservoir temperature (cP)
    api_gravity : float
        API gravity (e.g. 14.5 for BG-07, 18.0 for BG-15)
    """
    # Resolve aliases
    if reservoir_temp is not None:
        init_t = float(reservoir_temp)
    elif initial_res_temp is not None:
        init_t = float(initial_res_temp)
    else:
        init_t = 48.0

    if oil_viscosity is not None:
        ref_visc = float(oil_viscosity)
    elif ref_viscosity is not None:
        ref_visc = float(ref_viscosity)
    else:
        ref_visc = 8500.0

    # 1. Injection duration and effective steam heating
    if steam_volume <= 0:
        injection_days = 1
        daily_steam_inj = 0.0
        effective_steam_temp = init_t
    else:
        inj_rate_bpd = 1000.0
        injection_days = int(np.clip(np.ceil(steam_volume / inj_rate_bpd), 5, 20))
        daily_steam_inj = steam_volume / injection_days
        effective_steam_temp = steam_temp

    soak_days = max(int(soak_days), 0)
    production_days = max(int(production_days), 0)
    total_days = injection_days + soak_days + production_days

    # Pre-calculate production phase temperatures using depth-adjusted Marx-Langenheim model
    if production_days > 0:
        if steam_volume > 0:
            prod_temps = calculate_reservoir_temperature(
                initial_temp=init_t,
                steam_temp=effective_steam_temp,
                soak_days=soak_days,
                production_days=production_days,
                pump_depth=pump_depth
            )
        else:
            prod_temps = np.full(production_days, init_t)
    else:
        prod_temps = np.array([])

    # Theoretical pump displacement capacity in bpd (API formula)
    d_plunger_in = 2.25
    theoretical_disp_bpd = 0.1166 * (d_plunger_in ** 2) * stroke_length * max(spm, 0.0)

    records = []
    cum_oil = 0.0
    cum_steam = 0.0
    current_day = 1

    # --- PHASE 1: INJECTION ---
    for d in range(1, injection_days + 1):
        cum_steam += daily_steam_inj
        frac = d / injection_days
        t_curr = init_t + (effective_steam_temp - init_t) * (0.3 + 0.7 * frac)
        visc_curr = calculate_oil_viscosity(
            t_curr, ref_visc, init_t, activation_energy, api_gravity=api_gravity
        )

        records.append({
            "Day": current_day,
            "Phase": "Injection",
            "Reservoir_Temp": float(np.round(t_curr, 2)),
            "Oil_Viscosity": float(np.round(visc_curr, 2)),
            "Oil_Production": 0.0,
            "Steam_Injected": float(np.round(cum_steam, 2)),
            "SOR": 0.0,
            "Rod_Load": 0.0,
            "MPRL": 0.0,
            "Pump_Efficiency": 0.0,
            "Rod_Floating_Flag": False,
            "Rod_Floating_Index": 0.0,
            "Viscous_Drag": 0.0
        })
        current_day += 1

    # --- PHASE 2: SOAK ---
    for d in range(1, soak_days + 1):
        t_curr = effective_steam_temp if steam_volume <= 0 else (effective_steam_temp - (d * 1.8))
        visc_curr = calculate_oil_viscosity(
            t_curr, ref_visc, init_t, activation_energy, api_gravity=api_gravity
        )

        records.append({
            "Day": current_day,
            "Phase": "Soak",
            "Reservoir_Temp": float(np.round(t_curr, 2)),
            "Oil_Viscosity": float(np.round(visc_curr, 2)),
            "Oil_Production": 0.0,
            "Steam_Injected": float(np.round(cum_steam, 2)),
            "SOR": 0.0,
            "Rod_Load": 0.0,
            "MPRL": 0.0,
            "Pump_Efficiency": 0.0,
            "Rod_Floating_Flag": False,
            "Rod_Floating_Index": 0.0,
            "Viscous_Drag": 0.0
        })
        current_day += 1

    # --- PHASE 3: PRODUCTION ---
    # Inflow potential scaled by API gravity and steam energy
    api_multiplier = (api_gravity / 15.5) ** 0.8
    steam_energy_factor = (steam_volume / 12000.0) ** 0.4
    peak_production_bopd = theoretical_disp_bpd * 0.78 * steam_energy_factor * api_multiplier

    for d in range(1, production_days + 1):
        t_curr = prod_temps[d - 1]
        visc_curr = calculate_oil_viscosity(
            t_curr, ref_visc, init_t, activation_energy, api_gravity=api_gravity
        )

        rod_info = calculate_rod_load(
            spm=spm,
            stroke_length=stroke_length,
            fluid_viscosity=visc_curr,
            pump_depth=pump_depth,
            rod_diameter=rod_diameter,
            fluid_density=fluid_density,
            api_gravity=api_gravity
        )

        thermal_visc_factor = (100.0 / max(visc_curr, 10.0)) ** 0.38
        time_decline = (1.0 + 0.5 * 0.02 * d) ** (-2.0)

        if theoretical_disp_bpd <= 0.0:
            daily_oil = 0.0
        else:
            max_oil = theoretical_disp_bpd * 0.95
            oil_calc = peak_production_bopd * thermal_visc_factor * time_decline
            daily_oil = float(np.clip(oil_calc, 0.0, max_oil))

        cum_oil += daily_oil
        sor = calculate_steam_oil_ratio(cum_oil, cum_steam)
        pump_eff = calculate_pump_efficiency(daily_oil, theoretical_disp_bpd)

        records.append({
            "Day": current_day,
            "Phase": "Production",
            "Reservoir_Temp": float(np.round(t_curr, 2)),
            "Oil_Viscosity": float(np.round(visc_curr, 2)),
            "Oil_Production": float(np.round(daily_oil, 2)),
            "Steam_Injected": float(np.round(cum_steam, 2)),
            "SOR": sor,
            "Rod_Load": rod_info["PPRL"],
            "MPRL": rod_info["MPRL"],
            "Pump_Efficiency": pump_eff,
            "Rod_Floating_Flag": rod_info["Rod_Floating_Flag"],
            "Rod_Floating_Index": rod_info["Rod_Floating_Index"],
            "Viscous_Drag": rod_info["Viscous_Drag"]
        })
        current_day += 1

    df = pd.DataFrame(records)
    return df


# ============================================================================
# 7. SUCKER ROD PUMP DYNACARD GENERATOR (API RP 11L)
# ============================================================================

def generate_dynacard(
    stroke_length: float = 80.0,
    spm: float = 6.0,
    pump_depth: float = 1000.0,
    rod_diameter: float = 1.0,
    fluid_viscosity: float = 100.0,
    api_gravity: float = 15.5,
    num_points: int = 100
) -> Dict[str, Any]:
    """
    Generate realistic API RP 11L Surface and Downhole Dynamometer Cards (Dynacards)
    for Sucker Rod Pump operations in heavy oil.

    Models:
    - Kinematic polished rod displacement x(theta) = (S/2) * (1 - cos(theta))
    - Dynamic load pickup on upstroke and load release on downstroke
    - Viscous drag distortion and rod floating compressive collapse
    - Downhole pump card (effective pump stroke Sp, fluid load Fo)

    Returns:
    --------
    Dict[str, Any]
        Coordinates and diagnostics for plotting.
    """
    rod_info = calculate_rod_load(
        spm=spm,
        stroke_length=stroke_length,
        fluid_viscosity=fluid_viscosity,
        pump_depth=pump_depth,
        rod_diameter=rod_diameter,
        api_gravity=api_gravity
    )

    pprl = rod_info["PPRL"]
    mprl = rod_info["MPRL"]
    is_floating = rod_info["Rod_Floating_Flag"]
    visc_drag = rod_info["Viscous_Drag"]
    alpha = rod_info["Acceleration_Factor"]

    depth_ft = pump_depth * 3.28084
    oil_sg = 141.5 / (api_gravity + 131.5)
    mixed_sg = 0.65 * oil_sg + 0.35 * 1.02
    fluid_gradient_psi_ft = 0.433 * mixed_sg
    d_plunger_in = 2.25
    a_plunger_sq_in = (np.pi / 4.0) * (d_plunger_in ** 2)
    fluid_load = fluid_gradient_psi_ft * depth_ft * a_plunger_sq_in

    # Angles around cycle [0, 2*pi]
    thetas = np.linspace(0, 2 * np.pi, num_points)
    positions = []
    surface_loads = []
    pump_positions = []
    pump_loads = []

    # Downhole rod stretch (elastic elongation under fluid load)
    # Rod area: A_r = pi/4 * d^2, Steel modulus E = 30e6 psi
    rod_area = (np.pi / 4.0) * (rod_diameter ** 2)
    rod_stretch_in = (fluid_load * depth_ft * 12.0) / (rod_area * 30e6)
    downhole_stroke = max(stroke_length - rod_stretch_in, 10.0)

    half_stroke = stroke_length / 2.0
    half_dh_stroke = downhole_stroke / 2.0
    stroke_offset = (stroke_length - downhole_stroke) / 2.0

    for theta in thetas:
        # Polished rod kinematics
        x_surf = half_stroke * (1.0 - np.cos(theta))
        positions.append(float(np.round(x_surf, 2)))

        if theta <= np.pi:
            # UPSTROKE (theta from 0 to pi)
            t_ratio = min(1.0, theta / 0.55)
            # Dynamic elastic pickup
            load_baseline = mprl + (pprl - mprl) * (t_ratio ** 0.85)
            # Sucker rod natural frequency harmonic wave
            harmonic = 0.04 * (pprl - mprl) * np.sin(4.0 * theta) * np.exp(-0.8 * theta)
            s_load = load_baseline + harmonic
            surface_loads.append(float(np.round(s_load, 1)))

            # Downhole pump card (upstroke picks up fluid load)
            p_ratio = min(1.0, theta / 0.35)
            p_pos = stroke_offset + half_dh_stroke * (1.0 - np.cos(theta))
            p_load = fluid_load * (p_ratio ** 1.5)
            pump_positions.append(float(np.round(p_pos, 2)))
            pump_loads.append(float(np.round(p_load, 1)))
        else:
            # DOWNSTROKE (theta from pi to 2*pi)
            phi = theta - np.pi
            t_ratio = min(1.0, phi / 0.55)
            load_baseline = pprl - (pprl - mprl) * (t_ratio ** 0.85)
            harmonic = -0.03 * (pprl - mprl) * np.sin(4.0 * phi) * np.exp(-0.8 * phi)
            s_load = load_baseline + harmonic

            # If rod is floating, compressive viscous drag causes lower boundary collapse
            if is_floating or mprl <= 0:
                s_load = max(s_load * 0.45, 0.0)

            surface_loads.append(float(np.round(s_load, 1)))

            # Downhole pump card (downstroke dumps fluid load to standing valve)
            p_ratio = min(1.0, phi / 0.35)
            p_pos = stroke_offset + half_dh_stroke * (1.0 - np.cos(theta))
            p_load = fluid_load * (1.0 - (p_ratio ** 1.5))
            pump_positions.append(float(np.round(p_pos, 2)))
            pump_loads.append(float(np.round(p_load, 1)))

    # Compute card area via trapezoidal numerical integration (in-lbs)
    dx = np.diff(positions)
    avg_load = 0.5 * (np.array(surface_loads[:-1]) + np.array(surface_loads[1:]))
    card_area_in_lbs = float(np.abs(np.sum(avg_load * dx)))
    indicated_hp = float(np.round((card_area_in_lbs * max(spm, 0.0)) / (12.0 * 33000.0), 2))

    # Diagnostic categorization
    if is_floating or mprl <= 0.0:
        diagnostic = "Severe Rod Floating (Viscous Retardation on Downstroke)"
    elif pprl > 22000.0:
        diagnostic = "Excessive Rod Tensile Stress (Approaching Grade D Fatigue Limit)"
    elif visc_drag > 2500.0:
        diagnostic = "High Viscous Drag (Heavy Crude Damping)"
    else:
        diagnostic = "Normal Full Pump Card (Optimal API RP 11L Compliance)"

    return {
        "surface_position": positions,
        "surface_load": surface_loads,
        "pump_position": pump_positions,
        "pump_load": pump_loads,
        "pprl": pprl,
        "mprl": mprl,
        "fluid_load": float(np.round(fluid_load, 1)),
        "card_area_in_lbs": float(np.round(card_area_in_lbs, 1)),
        "indicated_hp": indicated_hp,
        "diagnostic": diagnostic,
        "is_rod_floating": is_floating
    }

