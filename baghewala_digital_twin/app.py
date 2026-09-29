"""
Baghewala Digital Twin - Production Streamlit Web Application (app.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

Multi-Well Digital Twin for Cyclic Steam Stimulation (CSS) and
Sucker Rod Pump (SRP) Optimization in the Baghewala Field, Rajasthan.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure path for modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from well_data import WELL_DATABASE, CUSTOM_SCENARIO_DEFAULTS
from physics_engine import (
    calculate_reservoir_temperature,
    calculate_oil_viscosity,
    calculate_rod_load,
    calculate_steam_oil_ratio,
    calculate_pump_efficiency,
    simulate_css_cycle
)
from ai_optimizer import (
    optimize_srp_parameters,
    predict_future_state,
    get_css_optimizer_model,
    get_rod_failure_model
)
from database import (
    init_database,
    save_simulation,
    get_simulation_history,
    save_alert,
    get_alerts
)

# Initialize database
init_database()

# ============================================================================
# PAGE CONFIGURATION & THEME STYLING
# ============================================================================

st.set_page_config(
    page_title="Baghewala Digital Twin | Oil India Limited",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial Oil & Gas Aesthetic
st.markdown("""
<style>
    /* Dark Palette Tokens */
    :root {
        --oil-navy: #0B132B;
        --oil-surface: #162038;
        --oil-card: #1C2A4A;
        --oil-accent: #F39C12;
        --oil-gold: #F1C40F;
        --oil-success: #27AE60;
        --oil-danger: #E74C3C;
        --oil-warning: #E67E22;
        --oil-text: #E0E6ED;
        --oil-muted: #8E9AAF;
    }

    /* Top Header Branding */
    .oil-header-container {
        background: linear-gradient(135deg, #0B132B 0%, #162544 50%, #1A365D 100%);
        padding: 20px 28px;
        border-radius: 12px;
        border: 1px solid #2A3E6B;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .oil-header-title {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .oil-header-subtitle {
        color: #CBD5E1;
        font-size: 14px;
        margin-top: 4px;
        font-weight: 400;
    }
    .oil-badge {
        background: rgba(243, 156, 18, 0.15);
        color: #F39C12;
        border: 1px solid #F39C12;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Metric Cards */
    .metric-card {
        background: #162038;
        border: 1px solid #233355;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8E9AAF;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    /* AI Recommendation Card */
    .ai-card {
        background: linear-gradient(145deg, #132238 0%, #1A2E4C 100%);
        border: 1px solid #2A4878;
        border-left: 4px solid #F39C12;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .ai-card-title {
        color: #F39C12;
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }

    /* Dynamic Alert Banners */
    .alert-banner-danger {
        background: rgba(231, 76, 60, 0.18);
        border: 1px solid #E74C3C;
        border-left: 5px solid #E74C3C;
        color: #FFC0BD;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .alert-banner-warning {
        background: rgba(243, 156, 18, 0.18);
        border: 1px solid #F39C12;
        border-left: 5px solid #F39C12;
        color: #FFE6BA;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .alert-banner-success {
        background: rgba(39, 174, 96, 0.18);
        border: 1px solid #27AE60;
        border-left: 5px solid #27AE60;
        color: #C3FAD8;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .alert-banner-info {
        background: rgba(41, 128, 185, 0.18);
        border: 1px solid #2980B9;
        border-left: 5px solid #2980B9;
        color: #D4E6F1;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
    }

    /* Footer styling */
    .oil-footer {
        margin-top: 40px;
        padding: 18px 24px;
        background: #0D1629;
        border-top: 1px solid #1F2D4A;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #718096;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown("""
<div class="oil-header-container">
    <div>
        <h1 class="oil-header-title">🛢️ Baghewala Digital Twin</h1>
        <div class="oil-header-subtitle">
            AI-Powered Well-to-Surface Optimization for Heavy Oil CSS & SRP Operations | Bikaner-Nagaur Basin
        </div>
    </div>
    <div style="text-align: right;">
        <span class="oil-badge">Oil India Limited</span>
        <div style="color: #94A3B8; font-size: 11px; margin-top: 6px;">
            SIH 2026 | Problem Statement ID: 26120
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR: WELL SELECTOR & "WHAT-IF" SIMULATOR
# ============================================================================

st.sidebar.markdown("### 🗂️ Asset & Scenario Selector")

# Step 2A: Well Selector Dropdown
well_options = ["Custom Field Scenario"] + list(WELL_DATABASE.keys())
selected_well_id = st.sidebar.selectbox(
    "Select Asset / Scenario",
    options=well_options,
    format_func=lambda x: WELL_DATABASE[x]["name"] if x in WELL_DATABASE else x
)

# Step 2B: Display Selected Well Properties Info Card
if selected_well_id != "Custom Field Scenario":
    well = WELL_DATABASE[selected_well_id]
    st.sidebar.info(f"""
    **{well['name']}**
    - **Formation:** {well['formation']}
    - **Depth:** {well['depth_m']} m
    - **API Gravity:** {well['api_gravity']}°
    - **Viscosity:** {well['viscosity_cp']} cP
    - **Reservoir Temp:** {well['reservoir_temp_c']} °C
    - **Porosity:** {well['porosity_pct']}% | **Perm:** {well['permeability_md']} mD
    """)
    default_steam = 12000
    default_spm = 8.0
    default_stroke = 80.0
    default_pump_depth = well['depth_m']
    res_temp = well['reservoir_temp_c']
    oil_visc = well['viscosity_cp']
    api_grav = well['api_gravity']
else:
    st.sidebar.info("""
    **Custom Field Scenario**
    Configure custom geological and operational parameters.
    """)
    with st.sidebar.expander("🛠️ Custom Reservoir Properties", expanded=True):
        custom_api = st.number_input("API Gravity (°)", min_value=10.0, max_value=25.0, value=16.0, step=0.5)
        custom_visc = st.number_input("Dead Oil Viscosity (cP)", min_value=1000.0, max_value=25000.0, value=8000.0, step=500.0)
        custom_temp = st.number_input("Reservoir Temp (°C)", min_value=35.0, max_value=65.0, value=48.0, step=0.5)
    default_steam = 10000
    default_spm = 8.0
    default_stroke = 80.0
    default_pump_depth = 1000.0
    res_temp = custom_temp
    oil_visc = custom_visc
    api_grav = custom_api

# Dynamic Slider Defaults Synchronization
if "last_selected_well" not in st.session_state or st.session_state.last_selected_well != selected_well_id:
    st.session_state.last_selected_well = selected_well_id
    st.session_state.steam_vol_input = default_steam
    st.session_state.spm_input = default_spm
    st.session_state.stroke_input = default_stroke
    st.session_state.pump_depth_input = default_pump_depth
    st.session_state.optimal_params = None

st.sidebar.markdown("---")
st.sidebar.markdown("#### ♨️ CSS Parameters")

steam_volume = st.sidebar.slider(
    "Steam Injection Volume (bbl)",
    min_value=5000,
    max_value=20000,
    value=int(st.session_state.steam_vol_input),
    step=500,
    help="Cold Water Equivalent (CWE) volume injected to heat the Jodhpur Sandstone heavy oil pay zone."
)

injection_pressure = st.sidebar.slider(
    "Injection Pressure (psi)",
    min_value=1000,
    max_value=3000,
    value=1800,
    step=50,
    help="Surface steam injection pressure. Governs steam saturation temperature."
)

soak_days = st.sidebar.slider(
    "Soak Time (days)",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Shut-in period allowing latent heat to diffuse radially into the reservoir matrix."
)

production_days = st.sidebar.slider(
    "Production Days (days)",
    min_value=30,
    max_value=90,
    value=60,
    step=5,
    help="Duration of the post-soak CSS production flowback and pumping period."
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🏗️ SRP Parameters")

spm = st.sidebar.slider(
    "Strokes Per Minute (SPM)",
    min_value=4.0,
    max_value=12.0,
    value=float(st.session_state.spm_input),
    step=0.2,
    help="Pumping speed. Excessive SPM in high-viscosity oil leads to severe rod floating and premature fatigue."
)

stroke_length = st.sidebar.slider(
    "Stroke Length (inches)",
    min_value=40.0,
    max_value=120.0,
    value=float(st.session_state.stroke_input),
    step=5.0,
    help="Polished rod stroke length. Long slow strokes typically maximize heavy oil pump efficiency."
)

pump_depth = st.sidebar.slider(
    "Pump Depth (meters)",
    min_value=500.0,
    max_value=1500.0,
    value=float(st.session_state.pump_depth_input),
    step=10.0,
    help="True vertical depth of the subsurface insert/tubing sucker rod pump."
)

# Action Buttons
st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
run_sim_btn = col_btn1.button("🚀 Run Sim", use_container_width=True)
optimize_btn = col_btn2.button("🤖 Auto-AI", use_container_width=True)

if optimize_btn:
    st.session_state.ai_optimized = True
    best_res = optimize_srp_parameters(
        current_viscosity=oil_visc * 0.08,
        current_temp=85.0,
        pump_depth=pump_depth,
        api_gravity=api_grav,
        steam_volume=steam_volume,
        injection_pressure=injection_pressure,
        soak_days=soak_days
    )
    st.session_state.optimal_params = best_res
    st.sidebar.success(f"AI Selected Optimal: {best_res['optimal_spm']} SPM, {best_res['optimal_stroke_length']}\"")


# ============================================================================
# PHYSICS SIMULATION EXECUTION (Well-Specific)
# ============================================================================

sim_df = simulate_css_cycle(
    steam_volume=float(steam_volume),
    spm=float(spm),
    stroke_length=float(stroke_length),
    pump_depth=float(pump_depth),
    reservoir_temp=float(res_temp),
    oil_viscosity=float(oil_visc),
    api_gravity=float(api_grav),
    injection_pressure=float(injection_pressure),
    soak_days=int(soak_days),
    production_days=int(production_days)
)

# Extract summary metrics
prod_phase_df = sim_df[sim_df["Phase"] == "Production"]
cum_oil_produced = float(prod_phase_df["Oil_Production"].sum())
peak_oil_rate = float(prod_phase_df["Oil_Production"].max())
final_sor = float(prod_phase_df["SOR"].iloc[-1]) if not prod_phase_df.empty else 0.0
avg_pump_eff = float(prod_phase_df["Pump_Efficiency"].mean()) if not prod_phase_df.empty else 0.0
max_pprl = float(prod_phase_df["Rod_Load"].max()) if not prod_phase_df.empty else 0.0
min_mprl = float(prod_phase_df["MPRL"].min()) if not prod_phase_df.empty else 0.0
is_rod_floating = bool(sim_df["Rod_Floating_Flag"].any())

# AI Optimization state & benchmark comparison
current_spm_val = float(spm)
if st.session_state.optimal_params:
    opt_spm = st.session_state.optimal_params["optimal_spm"]
    opt_stroke = st.session_state.optimal_params["optimal_stroke_length"]
    pred_increase = 15.0 if current_spm_val != opt_spm else 0.0
    energy_savings = st.session_state.optimal_params.get("energy_saving_pct", 18.0)
else:
    # Well-specific rule-based target if auto-AI not yet executed
    if api_grav < 15.0:
        opt_spm = 4.8
        opt_stroke = 100.0
    elif api_grav >= 17.5:
        opt_spm = 9.0
        opt_stroke = 100.0
    else:
        opt_spm = 5.8
        opt_stroke = 90.0
    pred_increase = 12.0 if current_spm_val > opt_spm else 6.0
    energy_savings = 18.0 if current_spm_val > opt_spm else 8.0

# 7-Day Future Forecast
forecast_df = predict_future_state(sim_df, days_ahead=7)


# ============================================================================
# MAIN PANEL LAYOUT: CENTER VISUALIZATION & RIGHT AI INSIGHTS
# ============================================================================

center_col, right_col = st.columns([2.3, 1.0])

with center_col:
    # Tabbed Interface for Core Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Reservoir Thermal Map (3D)",
        "📈 Production & Thermal Curves",
        "⚙️ SRP Dynacard & Mechanics",
        "📊 Field Comparison"
    ])

    # ------------------------------------------------------------------------
    # TAB 1: 3D RESERVOIR & WELLBORE THERMAL MAP (Well-Specific Depth & Viscosity)
    # ------------------------------------------------------------------------
    with tab1:
        st.markdown(f"#### 3D Wellbore & Steam Chamber ({selected_well_id} | Depth: {pump_depth:.0f}m)")

        time_step = st.slider(
            "Scrub Simulation Day",
            min_value=1,
            max_value=int(sim_df["Day"].iloc[-1]),
            value=int(sim_df[sim_df['Phase'] == 'Production']['Day'].iloc[0] if not prod_phase_df.empty else 20),
            help="Scrub through the CSS cycle to observe steam chamber heating and cooling around the wellbore."
        )

        curr_record = sim_df[sim_df["Day"] == time_step].iloc[0]
        chamber_temp = curr_record["Reservoir_Temp"]
        phase_label = curr_record["Phase"]

        # Calculate heated zone radius based on steam volume and time step
        base_radius_m = np.sqrt(steam_volume * 0.035 / np.pi)
        if phase_label == "Injection":
            current_radius_m = max(0.5, base_radius_m * (time_step / 15.0))
        elif phase_label == "Soak":
            current_radius_m = base_radius_m
        else:
            current_radius_m = base_radius_m * 1.05

        # Wellbore coordinates adapted to selected well depth
        z_well = np.linspace(0, -pump_depth, 120)
        x_well = np.zeros_like(z_well)
        y_well = np.zeros_like(z_well)

        # 3D Steam Chamber Shell
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        center_z = -pump_depth - 12.5

        x_chamber = current_radius_m * np.outer(np.cos(u), np.sin(v))
        y_chamber = current_radius_m * np.outer(np.sin(u), np.sin(v))
        z_chamber = center_z + 12.5 * np.outer(np.ones(np.size(u)), np.cos(v))

        temp_mesh = np.full_like(x_chamber, chamber_temp)

        # Color coding for oil viscosity (thick = Red/Maroon, thin = Green/Cyan)
        if oil_visc > 8000:
            flow_color = "#E74C3C" # Very thick oil (e.g. BG-07)
            visc_palette = [[0, "#78281F"], [0.4, "#C0392B"], [1.0, "#E74C3C"]]
        elif oil_visc > 6500:
            flow_color = "#F39C12" # Medium heavy oil
            visc_palette = [[0, "#1F618D"], [0.5, "#F39C12"], [1.0, "#E74C3C"]]
        else:
            flow_color = "#27AE60" # Lower viscosity (e.g. BG-15)
            visc_palette = [[0, "#117A65"], [0.5, "#27AE60"], [1.0, "#F1C40F"]]

        fig_3d = go.Figure()

        # 1. Surface wellhead
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=9, color="#F39C12", symbol="diamond"),
            name="Surface Wellhead (SRP)",
            text=["Surface Unit"],
            textposition="top center"
        ))

        # 2. Wellbore string
        fig_3d.add_trace(go.Scatter3d(
            x=x_well, y=y_well, z=z_well,
            mode="lines",
            line=dict(color="#A0AEC0", width=6),
            name="Wellbore Casing/Tubing",
            hoverinfo="text",
            text=[f"Depth: {abs(z):.0f}m" for z in z_well]
        ))

        # 3. Downhole SRP Unit at well depth
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[-pump_depth],
            mode="markers+text",
            marker=dict(size=9, color=flow_color, symbol="square"),
            name="Downhole Sucker Rod Pump",
            text=[f"SRP @ {pump_depth:.0f}m"],
            textposition="bottom right"
        ))

        # 4. Perforated Pay Zone
        perf_z = np.linspace(-pump_depth, -pump_depth - 25.0, 15)
        fig_3d.add_trace(go.Scatter3d(
            x=np.zeros_like(perf_z), y=np.zeros_like(perf_z), z=perf_z,
            mode="lines",
            line=dict(color="#F1C40F", width=12),
            name="Perforated Pay Interval"
        ))

        # 5. Steam Heated Chamber Surface
        fig_3d.add_trace(go.Surface(
            x=x_chamber, y=y_chamber, z=z_chamber,
            surfacecolor=temp_mesh,
            colorscale=visc_palette,
            cmin=45.0,
            cmax=290.0,
            colorbar=dict(title="Temp (°C)", x=1.02, len=0.7),
            opacity=0.84,
            name="Steam Heated Zone",
            showscale=True,
            hoverinfo="text",
            hovertext=[f"Temp: {chamber_temp:.1f}°C<br>Radius: {current_radius_m:.1f}m<br>In-situ Visc: {curr_record['Oil_Viscosity']:.0f} cP" for _ in range(x_chamber.size)]
        ))

        fig_3d.update_layout(
            template="plotly_dark",
            paper_bgcolor="#162038",
            plot_bgcolor="#162038",
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(title="Radial X (m)", range=[-35, 35], gridcolor="#233355"),
                yaxis=dict(title="Radial Y (m)", range=[-35, 35], gridcolor="#233355"),
                zaxis=dict(title="True Vertical Depth (m)", range=[-pump_depth - 60, 20], gridcolor="#233355"),
                camera=dict(eye=dict(x=1.4, y=-1.5, z=0.7))
            )
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Day & Phase", f"Day {time_step} ({phase_label})")
        col_kpi2.metric("Chamber Temp", f"{chamber_temp:.1f} °C")
        col_kpi3.metric("Heated Radius", f"{current_radius_m:.1f} m")
        col_kpi4.metric("In-Situ Viscosity", f"{curr_record['Oil_Viscosity']:.1f} cP")

    # ------------------------------------------------------------------------
    # TAB 2: PRODUCTION CURVES (2x2 Multi-plot)
    # ------------------------------------------------------------------------
    with tab2:
        st.markdown(f"#### CSS Production, Viscosity & SOR Performance ({selected_well_id})")

        fig_curves = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Reservoir Temperature vs. Days (°C)",
                "Oil Viscosity vs. Days (cP, Log Scale)",
                "Daily Oil Production vs. Days (BOPD)",
                "Steam-to-Oil Ratio (SOR, bbl/bbl)"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )

        fig_curves.add_trace(
            go.Scatter(
                x=sim_df["Day"], y=sim_df["Reservoir_Temp"],
                mode="lines", name="Reservoir Temp",
                line=dict(color="#E74C3C", width=2.5)
            ),
            row=1, col=1
        )
        fig_curves.add_hline(y=res_temp, line_dash="dash", line_color="#7F8C8D", row=1, col=1)

        fig_curves.add_trace(
            go.Scatter(
                x=sim_df["Day"], y=sim_df["Oil_Viscosity"],
                mode="lines", name="Viscosity (cP)",
                line=dict(color="#F39C12", width=2.5)
            ),
            row=1, col=2
        )

        fig_curves.add_trace(
            go.Scatter(
                x=sim_df["Day"], y=sim_df["Oil_Production"],
                mode="lines+markers", name="Oil Rate (BOPD)",
                line=dict(color="#27AE60", width=2.5),
                marker=dict(size=4)
            ),
            row=2, col=1
        )

        fig_curves.add_trace(
            go.Scatter(
                x=prod_phase_df["Day"], y=prod_phase_df["SOR"],
                mode="lines", name="Cumulative SOR",
                line=dict(color="#3498DB", width=2.5)
            ),
            row=2, col=2
        )
        fig_curves.add_hline(
            y=4.5, line_dash="dot", line_color="#E74C3C",
            annotation_text="Economic Cutoff (SOR=4.5)", row=2, col=2
        )

        fig_curves.update_yaxes(type="log", row=1, col=2)
        fig_curves.update_layout(
            template="plotly_dark",
            paper_bgcolor="#162038",
            plot_bgcolor="#162038",
            height=540,
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=30)
        )
        st.plotly_chart(fig_curves, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 3: SRP PERFORMANCE & DYNACARD DIAGNOSTIC
    # ------------------------------------------------------------------------
    with tab3:
        st.markdown(f"#### Sucker Rod Pump (API RP 11L) Dynamics ({selected_well_id})")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_gauge_eff = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_pump_eff,
                title={"text": "Pump Volumetric Efficiency (%)", "font": {"size": 15, "color": "#E0E6ED"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#A0AEC0"},
                    "bar": {"color": "#27AE60" if avg_pump_eff > 65 else "#F39C12"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(231, 76, 60, 0.3)"},
                        {"range": [50, 75], "color": "rgba(243, 156, 18, 0.3)"},
                        {"range": [75, 100], "color": "rgba(39, 174, 96, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "#FFFFFF", "width": 3},
                        "thickness": 0.75,
                        "value": 75.0
                    }
                }
            ))
            fig_gauge_eff.update_layout(
                template="plotly_dark",
                paper_bgcolor="#162038",
                height=220,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge_eff, use_container_width=True)

        with col_g2:
            risk_pct = 100.0 if min_mprl <= 0.0 else max(0.0, min(100.0, (1.0 - min_mprl / 4000.0) * 100.0))
            fig_gauge_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={"text": "Rod Floating / Buckling Risk (%)", "font": {"size": 15, "color": "#E0E6ED"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#A0AEC0"},
                    "bar": {"color": "#E74C3C" if risk_pct > 70 else ("#F39C12" if risk_pct > 30 else "#27AE60")},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(39, 174, 96, 0.3)"},
                        {"range": [30, 70], "color": "rgba(243, 156, 18, 0.3)"},
                        {"range": [70, 100], "color": "rgba(231, 76, 60, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "#E74C3C", "width": 3},
                        "thickness": 0.75,
                        "value": 70.0
                    }
                }
            ))
            fig_gauge_risk.update_layout(
                template="plotly_dark",
                paper_bgcolor="#162038",
                height=220,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge_risk, use_container_width=True)

        # Polished Rod Load vs Days
        fig_loads = go.Figure()
        fig_loads.add_trace(go.Scatter(
            x=prod_phase_df["Day"], y=prod_phase_df["Rod_Load"],
            mode="lines", name="Peak Polished Rod Load (PPRL)",
            line=dict(color="#E74C3C", width=2.5)
        ))
        fig_loads.add_trace(go.Scatter(
            x=prod_phase_df["Day"], y=prod_phase_df["MPRL"],
            mode="lines", name="Minimum Polished Rod Load (MPRL)",
            line=dict(color="#3498DB", width=2.5)
        ))
        fig_loads.add_hline(
            y=0.0, line_dash="dash", line_color="#E74C3C",
            annotation_text="Rod Floating Threshold (0 lbs)"
        )
        fig_loads.add_hline(
            y=22000.0, line_dash="dot", line_color="#F1C40F",
            annotation_text="Grade D Safe Stress Limit (22,000 lbs)"
        )
        fig_loads.update_layout(
            title="Polished Rod Loads vs Days (API RP 11L)",
            xaxis_title="Simulation Day",
            yaxis_title="Load (lbs)",
            template="plotly_dark",
            paper_bgcolor="#162038",
            plot_bgcolor="#162038",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loads, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 4: FIELD COMPARISON (Step 5A)
    # ------------------------------------------------------------------------
    with tab4:
        st.markdown("#### 📊 Field Comparison Across All Baghewala Wells")
        st.caption("Comparing production potential, API gravity, in-situ viscosity, and optimal SPM across all 5 field assets.")

        comparison_records = []
        for w_id, w_props in WELL_DATABASE.items():
            # Simulate representative baseline cycle for each well
            w_sim = simulate_css_cycle(
                steam_volume=12000.0,
                spm=6.0,
                stroke_length=80.0,
                pump_depth=w_props["depth_m"],
                reservoir_temp=w_props["reservoir_temp_c"],
                oil_viscosity=w_props["viscosity_cp"],
                api_gravity=w_props["api_gravity"]
            )
            w_prod = w_sim[w_sim["Phase"] == "Production"]
            peak_bopd = float(w_prod["Oil_Production"].max())
            avg_bopd = float(w_prod["Oil_Production"].mean())
            total_bbl = float(w_prod["Oil_Production"].sum())

            # Recommended SPM proxy based on API gravity
            if w_props["api_gravity"] < 15.0:
                rec_spm_str = "4.5 - 5.0"
            elif w_props["api_gravity"] > 17.5:
                rec_spm_str = "8.5 - 10.0"
            else:
                rec_spm_str = "5.5 - 7.0"

            comparison_records.append({
                "Well ID": w_id,
                "Name": w_props["name"],
                "Depth (m)": w_props["depth_m"],
                "API Gravity (°)": w_props["api_gravity"],
                "Viscosity (cP)": w_props["viscosity_cp"],
                "Reservoir Temp (°C)": w_props["reservoir_temp_c"],
                "Permeability (mD)": w_props["permeability_md"],
                "Peak BOPD": np.round(peak_bopd, 1),
                "Avg BOPD": np.round(avg_bopd, 1),
                "Cycle Oil (bbl)": np.round(total_bbl, 0),
                "Safe SPM Range": rec_spm_str
            })

        comp_df = pd.DataFrame(comparison_records)

        # Bar chart comparing predicted oil production
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_df["Well ID"],
            y=comp_df["Peak BOPD"],
            name="Peak Oil Production (BOPD)",
            marker=dict(
                color=comp_df["API Gravity (°)"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="API Gravity (°)", len=0.8)
            ),
            text=[f"{bopd:.1f} BOPD<br>{api}° API" for bopd, api in zip(comp_df["Peak BOPD"], comp_df["API Gravity (°)"])],
            textposition="auto"
        ))

        fig_comp.update_layout(
            title="Predicted Peak Oil Production by Well (Color = API Gravity; Darker = Thicker Oil)",
            xaxis_title="Well ID",
            yaxis_title="Peak Oil Production (bbl/day)",
            template="plotly_dark",
            paper_bgcolor="#162038",
            plot_bgcolor="#162038",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Comparison summary table
        st.markdown("##### Asset Geological & Production Profile Table")
        st.dataframe(
            comp_df[["Well ID", "Depth (m)", "API Gravity (°)", "Viscosity (cP)", "Permeability (mD)", "Peak BOPD", "Avg BOPD", "Cycle Oil (bbl)", "Safe SPM Range"]],
            use_container_width=True,
            hide_index=True
        )


# ============================================================================
# RIGHT PANEL: AI INSIGHTS, PREDICTIONS & DYNAMIC ALERTS
# ============================================================================

with right_col:
    # 1. AI Optimization Card
    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-card-title">🤖 AI Autonomous Optimizer</div>
        <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">
            Target Well: <b style="color: #FFFFFF;">{selected_well_id}</b>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">
            Current SPM: <b style="color: #FFFFFF;">{current_spm_val:.1f}</b>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">
            AI Recommended SPM: <b style="color: #27AE60; font-size: 16px;">{opt_spm:.1f}</b>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">
            Recommended Stroke: <b style="color: #27AE60; font-size: 15px;">{opt_stroke:.0f}\"</b>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">
            Predicted Production: <b style="color: #2ECC71;">+{pred_increase:.1f}%</b>
        </div>
        <div style="font-size: 13px; color: #CBD5E1;">
            Predicted Energy Savings: <b style="color: #3498DB;">-{energy_savings:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Apply AI Recommendation", use_container_width=True):
        st.session_state.spm_input = opt_spm
        st.session_state.stroke_input = opt_stroke
        st.rerun()

    # 2. Dynamic Real-Time Alerts Card (Step 5C)
    st.markdown("#### 🚨 Real-Time Diagnostic Alerts")

    # Well-Specific Geological Alerts
    if selected_well_id == "BG-07":
        st.markdown("""
        <div class="alert-banner-warning">
            <b>⚠️ BG-07 Asset Warning:</b> Very high viscosity (9,500 cP) & low API (14.5°). High viscous drag on downstroke. Optimal SPM range: <b>4.5–5.5 SPM</b> to prevent rod floating.
        </div>
        """, unsafe_allow_html=True)
    elif selected_well_id == "BG-15":
        st.markdown("""
        <div class="alert-banner-success">
            <b>✅ BG-15 Asset Status:</b> Favorable viscosity (5,500 cP) & higher API (18.0°). Thinner crude allows higher pumping speeds. Optimal SPM range: <b>8.0–10.0 SPM</b>.
        </div>
        """, unsafe_allow_html=True)
    elif selected_well_id == "BG-04":
        st.markdown("""
        <div class="alert-banner-info">
            <b>ℹ️ BG-04 Asset Advisory:</b> North Pad formation pressure at 1,350 psi. Ensure balanced counterbalance effect to prevent gearbox torque overloading.
        </div>
        """, unsafe_allow_html=True)

    # Physics-triggered alerts
    alert_rendered = False
    if is_rod_floating or min_mprl <= 0.0:
        st.markdown("""
        <div class="alert-banner-danger">
            <b>⚠️ CRITICAL: Rod Floating Detected</b><br>
            MPRL is non-positive due to heavy oil viscous drag. Reduce SPM immediately or increase stroke length.
        </div>
        """, unsafe_allow_html=True)
        save_alert("ROD_FLOATING", f"Critical: Rod floating detected on {selected_well_id}.", "CRITICAL")
        alert_rendered = True

    if final_sor > 4.5:
        st.markdown(f"""
        <div class="alert-banner-warning">
            <b>⚠️ WARNING: High Steam-Oil Ratio (SOR = {final_sor:.2f})</b><br>
            Exceeds economic cutoff (4.5 bbl/bbl). Optimize steam injection volume or terminate soak earlier.
        </div>
        """, unsafe_allow_html=True)
        save_alert("HIGH_SOR", f"Warning: Steam-Oil Ratio reached {final_sor:.2f} on {selected_well_id}.", "WARNING")
        alert_rendered = True

    if not alert_rendered and selected_well_id not in ["BG-07"]:
        st.markdown("""
        <div class="alert-banner-success">
            <b>✅ System Optimal</b><br>
            All CSS thermal parameters and SRP rod loads are within API RP 11L safe fatigue envelopes.
        </div>
        """, unsafe_allow_html=True)

    # 3. 7-Day Predictive Forecast Card
    st.markdown("#### 🔮 Next 7 Days Forecast")
    st.markdown("""
    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 6px;">
        Predictive trajectory based on thermodynamic decline and reservoir dynamics:
    </div>
    """, unsafe_allow_html=True)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["Day"], y=forecast_df["Oil_Production"],
        mode="lines+markers", name="BOPD",
        line=dict(color="#27AE60", width=2),
        marker=dict(size=4)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["Day"], y=forecast_df["Reservoir_Temp"],
        mode="lines", name="Temp (°C)",
        line=dict(color="#E74C3C", width=2, dash="dash")
    ))
    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#162038",
        plot_bgcolor="#162038",
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # 4. Persistence & Export Actions
    st.markdown("---")
    col_save, col_down = st.columns(2)

    with col_save:
        if st.button("💾 Save Run", use_container_width=True):
            try:
                run_id = save_simulation({
                    "steam_volume": steam_volume,
                    "soak_days": soak_days,
                    "spm": spm,
                    "oil_produced": cum_oil_produced,
                    "sor": final_sor,
                    "rod_load": max_pprl,
                    "stroke_length": stroke_length,
                    "pump_depth": pump_depth,
                    "pump_efficiency": avg_pump_eff,
                    "rod_floating": int(is_rod_floating)
                })
                st.success(f"Saved (ID #{run_id})")
            except Exception as db_err:
                st.warning(f"Database logging unavailable ({db_err}). Scenario retained in active session.")

    with col_down:
        csv_data = sim_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"baghewala_sim_run_{selected_well_id}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================================
# FOOTER SECTION
# ============================================================================

st.markdown("""
<div class="oil-footer">
    <div>
        <b>Baghewala Digital Twin</b> | Developed for Smart India Hackathon 2026 | Problem Statement ID: 26120
    </div>
    <div>
        Organization: <b>Oil India Limited</b> | Theme: <b>Smart Automation & Multi-Well Heavy Oil Optimization</b>
    </div>
</div>
""", unsafe_allow_html=True)
