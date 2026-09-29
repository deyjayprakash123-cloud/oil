# 🛢️ Baghewala Digital Twin
### Well-to-Surface AI Digital Twin for Cyclic Steam Stimulation (CSS) and Sucker Rod Pump (SRP) Optimization in Heavy Oil Fields

**Organization:** Oil India Limited (OIL)  
**Initiative:** Smart India Hackathon 2026  
**Problem Statement ID:** 26120  
**Theme:** Smart Automation & Artificial Intelligence in Energy  
**Field Location:** Baghewala Heavy Oil / Bitumen Field, Bikaner-Nagaur Basin, Rajasthan, India  

---

## 📌 Executive Summary

The **Baghewala Heavy Oil Field** in Rajasthan presents one of India's most challenging upstream petroleum engineering frontiers. Featuring extra-heavy crude and bitumen (14°–17° API) with dead oil viscosities ranging from 5,000 to over 15,000 cP at native reservoir temperatures (~48°C), conventional cold production is technically infeasible. 

Oil India Limited utilizes **Cyclic Steam Stimulation (CSS)** (Huff-and-Puff) paired with **Sucker Rod Pumping (SRP)** systems. However, suboptimal steam allocation, prolonged soak times, high Steam-to-Oil Ratios (SOR > 4.5), and extreme downhole viscous drag often induce **sucker rod floating**, premature rod fatigue, and excessive energy consumption.

The **Baghewala Digital Twin** integrates:
1. **Rigorous Petroleum Engineering Physics** (Marx-Langenheim thermal dissipation, Arrhenius non-Newtonian viscosity thinning, and API RP 11L polished rod dynamics).
2. **Machine Learning & AI Optimization** (XGBoost production regression, Random Forest rod failure classification, and multi-objective SRP kinematics optimization).
3. **Interactive 3D Wellbore & Reservoir Visualization** (Plotly 3D thermal isosurfaces, dynamic dynacards, and real-time SCADA alerting).
4. **Resilient Edge-Ready Architecture** (SQLite local persistence, Streamlit UI, zero cloud vendor lock-in).

---

## 📁 Project Architecture

```text
baghewala_digital_twin/
├── app.py                      # Main Streamlit web application & 3-column cockpit
├── physics_engine.py           # First-principles petroleum engineering physics
├── data_generator.py           # Synthetic SCADA & CSS cycle data generator (±5% noise)
├── ai_optimizer.py             # XGBoost & Random Forest models, optimization & forecast
├── database.py                 # SQLite database management (baghewala.db)
├── css_optimizer.pkl           # Pre-trained XGBoost Regressor for production forecasting
├── rod_failure_model.pkl       # Pre-trained Random Forest Classifier for rod failure risk
├── requirements.txt            # Python dependencies
├── README.md                   # Technical documentation
└── .streamlit/
    └── config.toml             # Dark-theme industrial styling configuration
```

---

## 🔬 Mathematical & Physics Formulations

### 1. Reservoir Heating & Thermal Loss (Marx-Langenheim Model)
During steam injection into the Jodhpur sandstone pay zone, heat loss conducts into overburden and underburden rock:

$$\frac{dA}{dt} = \frac{H_o - Q_{loss}}{M_R h_t (T_s - T_R)}$$

$$t_D = \frac{4 k_{ob} M_{ob}}{M_R^2 h_t^2} t$$

$$F_1(t_D) = e^{t_D} \text{erfc}(\sqrt{t_D}) + 2 \sqrt{\frac{t_D}{\pi}} - 1$$

The effective heated radius $r_h(t) = \sqrt{A(t) / \pi}$ and subsequent post-soak thermal dissipation follow:

$$T(t) = T_R + (T_s - T_R) \cdot \exp(-\lambda t)$$

### 2. Heavy Oil Viscosity Thinning (Arrhenius Equation)
Baghewala crude displays steep exponential thermal thinning modeled via activation energy $E_a$:

$$\mu(T) = \mu_{ref} \cdot \exp\left( \frac{E_a}{R} \left( \frac{1}{T + 273.15} - \frac{1}{T_{ref} + 273.15} \right) \right)$$

- At native temperature ($48^\circ\text{C}$): $\mu \approx 8,500\text{ cP}$
- At steam-heated temperature ($200^\circ\text{C} - 280^\circ\text{C}$): $\mu < 25\text{ cP}$

### 3. Sucker Rod Pump Mechanics (API RP 11L)
Pumping heavy oil requires accounting for viscous drag and inertial acceleration:

- **Acceleration factor:** $\alpha = \frac{S \cdot N^2}{70,500}$
- **Buoyant Rod Weight:** $W_{rf} = W_r \left(1 - \frac{\rho_f}{\rho_s}\right)$
- **Peak Polished Rod Load (PPRL):**
  $$PPRL = W_{rf}(1 + \alpha) + F_o(1.05 + 0.1\alpha) + F_{visc\_drag}$$
- **Minimum Polished Rod Load (MPRL):**
  $$MPRL = W_{rf}(1 - \alpha) - F_{visc\_drag}$$
- **Rod Floating Condition:** If $MPRL \le 0\text{ lbs}$, downward gravitational acceleration is cancelled by upward viscous shear and deceleration, causing sucker rods to float, buckle, and fail.

### 4. Steam-to-Oil Ratio (SOR) & Pump Volumetric Efficiency
$$SOR = \frac{V_{steam\_injected} \text{ (bbl)}}{N_p \text{ (bbl oil)}}$$

Economic cutoff for Baghewala thermal recovery is defined at $SOR \le 4.5\text{ bbl/bbl}$.

$$\eta_v = \left(\frac{Q_{actual}}{0.1166 \cdot D_p^2 \cdot S \cdot N}\right) \times 100\%$$

---

## 🤖 AI Models & Predictive Analytics

1. **CSS Production Regressor (XGBoost)**:
   - **Trained on:** 10,000 synthetic physics-calibrated CSS cycles.
   - **Features:** Steam Volume, Injection Pressure, Soak Days, SPM, Stroke Length, Reservoir Temperature.
   - **Performance:** $R^2 > 0.96$, $RMSE < 2300\text{ bbl}$.
   - **Artifact:** `css_optimizer.pkl`.

2. **Rod Failure & Floating Classifier (Random Forest)**:
   - **Trained on:** Multi-variable operational states with cold slugging and viscous friction.
   - **Features:** SPM, Stroke Length, Viscosity, Rod Load, Pump Depth.
   - **Performance:** Accuracy $> 99.2\%$, AUC-ROC $> 0.99$.
   - **Artifact:** `rod_failure_model.pkl`.

3. **Multi-Objective Kinematics Optimizer**:
   - Searches for global optimal SPM and Stroke Length to simultaneously maximize gross oil production while enforcing $MPRL > 300\text{ lbs}$ (eliminating rod floating) and reducing motor electrical consumption by 12–20%.

4. **Predictive 7-Day Thermodynamic Forecast**:
   - Forecasts bottom-hole temperature decline, in-situ viscosity rebound, and expected BOPD for the upcoming 7 operational days.

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.10+ (Tested up to Python 3.13)
- Git

### 1. Clone & Setup
```bash
git clone <repository-url>
cd oil/baghewala_digital_twin
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Digital Twin Web Application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🖥️ User Interface Overview

The interface features an **Industrial Three-Column Dashboard** styled to Oil India Limited standards:

- **Left Panel ("What-If" Simulator):**
  - Interactive sliders for steam injection volume, injection pressure, soak time, and production duration.
  - Sucker rod pump sliders for SPM, stroke length, and pump setting depth.
  - Pre-configured asset selector for Baghewala wells (`BG-01`, `BG-04`, `BG-07`, `BG-12`, `BG-15`).
  - One-click **"Auto-AI"** optimization button.

- **Center Panel (Interactive Digital Twin):**
  - **Tab 1: 3D Reservoir Thermal Map:** 3D interactive Plotly visualization showing the vertical wellbore, downhole pump at ~1000m, perforated pay zone, and the ellipsoidal steam chamber colored by temperature (Red to Blue). Includes an interactive day scrubber.
  - **Tab 2: Production & Thermal Curves:** 4 synchronized charts depicting Reservoir Temperature vs Days, Viscosity Thinning vs Days (log scale), Daily Oil Production (BOPD), and Cumulative SOR with economic cutoff.
  - **Tab 3: SRP Performance:** Real-time API RP 11L load limits, Pump Efficiency Gauge, and Rod Floating Risk Indicator.

- **Right Panel (AI Insights & Real-Time Alerts):**
  - **AI Autonomous Optimizer Card:** Current SPM vs AI Recommended SPM (highlighted in emerald green), predicted production gain, and energy savings.
  - **Real-Time Diagnostic Alerts:** Dynamic Red alerts for rod floating, Yellow alerts for excessive SOR (>4.5), and Green badges for optimal status.
  - **7-Day Predictive Forecast:** Mini predictive chart of upcoming 7-day trajectory.
  - **Run Persistence & CSV Export:** Save scenarios to SQLite (`baghewala.db`) and export time-series to CSV.

---

## 📸 Screenshots

*(Dashboard screenshots placeholder)*
- **3D Thermal Plume & Wellbore Trajectory**
- **Hyperbolic Oil Production & Arrhenius Viscosity Curves**
- **API RP 11L Dynacard and Rod Floating Safety Gauge**
- **AI Recommendation Engine & SCADA Alert Stream**

---

## 👥 Smart India Hackathon 2026 Team

- **Team Name:** Baghewala AI Innovators
- **Problem Statement:** 26120 (Smart Automation for Oil India Limited)
- **Primary Domain:** Heavy Oil Thermal Recovery & Artificial Lift Automation

---

## 📄 License & Confidentiality
Developed for the **Smart India Hackathon 2026** in partnership with **Oil India Limited**. All engineering formulas and models comply with API RP 11L and standard open petroleum literature.
