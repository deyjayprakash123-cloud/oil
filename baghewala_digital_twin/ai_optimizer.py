"""
Baghewala Digital Twin - AI Optimizer Module (ai_optimizer.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

AI/ML Models:
1. XGBoost Regressor for CSS production optimization (css_optimizer.pkl)
2. Random Forest Classifier for SRP rod failure & floating risk (rod_failure_model.pkl)
3. Multi-objective SRP kinematics optimizer (optimize_srp_parameters)
4. Time-series predictive forecasting for reservoir temp & production (predict_future_state)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb

# Support direct and relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_generator import generate_css_cycle_data, generate_rod_failure_data
from physics_engine import calculate_rod_load, calculate_oil_viscosity

CSS_MODEL_FILE = os.path.join(current_dir, "css_optimizer.pkl")
ROD_MODEL_FILE = os.path.join(current_dir, "rod_failure_model.pkl")


# ============================================================================
# 1. TRAIN XGBOOST CSS OPTIMIZATION MODEL
# ============================================================================

def train_optimization_model(num_cycles: int = 10000, save_path: str = CSS_MODEL_FILE) -> Dict[str, Any]:
    """
    Generate 10,000 synthetic CSS cycles using data_generator.py.
    Train an XGBoost Regressor to predict Oil_Production given:
    Features: Steam_Volume, Injection_Pressure, Soak_Days, SPM, Stroke_Length, Reservoir_Temp
    Save the model as css_optimizer.pkl.

    Returns:
    --------
    Dict[str, Any]
        Model evaluation metrics (RMSE, R2 score).
    """
    print(f"Generating {num_cycles} synthetic CSS cycles for XGBoost training...")
    df = generate_css_cycle_data(num_cycles=num_cycles, random_seed=42)

    features = [
        "Steam_Volume",
        "Injection_Pressure",
        "Soak_Days",
        "SPM",
        "Stroke_Length",
        "Reservoir_Temp",
        "API_Gravity"
    ]
    target = "Oil_Production"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = float(r2_score(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Save model
    joblib.dump(model, save_path)
    print(f"CSS Optimizer Model saved to {save_path} (R² = {r2:.4f}, RMSE = {rmse:.2f})")

    return {
        "model": model,
        "r2_score": r2,
        "rmse": rmse,
        "features": features
    }


def get_css_optimizer_model():
    """Load or train CSS optimizer model."""
    if os.path.exists(CSS_MODEL_FILE):
        try:
            m = joblib.load(CSS_MODEL_FILE)
            if hasattr(m, "feature_names_in_") and "API_Gravity" in m.feature_names_in_:
                return m
        except Exception:
            pass
    metrics = train_optimization_model(num_cycles=3000)
    return metrics["model"]


# ============================================================================
# 2. TRAIN RANDOM FOREST ROD FAILURE CLASSIFIER
# ============================================================================

def train_rod_failure_classifier(num_samples: int = 5000, save_path: str = ROD_MODEL_FILE) -> Dict[str, Any]:
    """
    Generate synthetic rod failure data.
    Train a Random Forest Classifier to predict Rod_Failure_Risk (0 = safe, 1 = failure imminent).
    Features: SPM, Stroke_Length, Viscosity, API_Gravity, Rod_Load, Pump_Depth
    Save the model as rod_failure_model.pkl.
    """
    print(f"Generating {num_samples} samples for rod failure classifier...")
    df = generate_rod_failure_data(num_samples=num_samples, random_seed=101)

    features = ["SPM", "Stroke_Length", "Viscosity", "API_Gravity", "Rod_Load", "Pump_Depth"]
    target = "Rod_Failure_Risk"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    preds = rf_model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    joblib.dump(rf_model, save_path)
    print(f"Rod Failure Classifier saved to {save_path} (Accuracy = {acc:.4f})")

    return {
        "model": rf_model,
        "accuracy": acc,
        "features": features
    }


def get_rod_failure_model():
    """Load or train Rod failure classifier."""
    if os.path.exists(ROD_MODEL_FILE):
        try:
            m = joblib.load(ROD_MODEL_FILE)
            if hasattr(m, "feature_names_in_") and "API_Gravity" in m.feature_names_in_:
                return m
        except Exception:
            pass
    metrics = train_rod_failure_classifier(num_samples=3000)
    return metrics["model"]


# ============================================================================
# 3. OPTIMIZE SRP PARAMETERS
# ============================================================================

def optimize_srp_parameters(
    current_viscosity: float = 120.0,
    current_temp: float = 85.0,
    pump_depth: float = 1000.0,
    api_gravity: float = 15.5,
    steam_volume: float = 12000.0,
    injection_pressure: float = 1800.0,
    soak_days: int = 5
) -> Dict[str, Any]:
    """
    Optimize SRP parameters for a specific well.
    Wells with lower API gravity (e.g., BG-07 at 14.5°) have thicker oil and require
    lower SPM (~4-5 SPM) to prevent rod floating.
    Wells with higher API gravity (e.g., BG-15 at 18.0°) can safely handle ~8-10 SPM.
    """
    xgb_model = get_css_optimizer_model()
    rf_model = get_rod_failure_model()

    # Search space
    spm_candidates = np.linspace(4.0, 11.0, 29)         # step 0.25
    stroke_candidates = np.linspace(50.0, 120.0, 15)     # step 5 in

    best_score = -1e9
    best_params = {
        "optimal_spm": 5.0 if api_gravity < 15.5 else 8.5,
        "optimal_stroke_length": 80.0,
        "predicted_production": 0.0,
        "rod_load": 14000.0,
        "confidence": 0.92
    }

    # Evaluate candidates
    candidate_records = []
    for spm in spm_candidates:
        for stroke in stroke_candidates:
            candidate_records.append({
                "Steam_Volume": steam_volume,
                "Injection_Pressure": injection_pressure,
                "Soak_Days": soak_days,
                "SPM": spm,
                "Stroke_Length": stroke,
                "Reservoir_Temp": current_temp,
                "API_Gravity": api_gravity
            })

    candidate_df = pd.DataFrame(candidate_records)
    preds = xgb_model.predict(candidate_df)

    idx = 0
    for spm in spm_candidates:
        for stroke in stroke_candidates:
            pred_prod = float(preds[idx])
            idx += 1

            # Check rod mechanics with well-specific API gravity and viscosity
            rod_info = calculate_rod_load(
                spm=spm,
                stroke_length=stroke,
                fluid_viscosity=current_viscosity,
                pump_depth=pump_depth,
                api_gravity=api_gravity
            )

            pprl = rod_info["PPRL"]
            mprl = rod_info["MPRL"]
            is_floating = rod_info["Rod_Floating_Flag"]

            # Failure risk probability from classifier
            rf_features = pd.DataFrame([{
                "SPM": spm,
                "Stroke_Length": stroke,
                "Viscosity": current_viscosity,
                "API_Gravity": api_gravity,
                "Rod_Load": pprl,
                "Pump_Depth": pump_depth
            }])
            probs = rf_model.predict_proba(rf_features)[0]
            if len(rf_model.classes_) > 1 and 1 in rf_model.classes_:
                class_1_idx = list(rf_model.classes_).index(1)
                fail_prob = float(probs[class_1_idx])
            else:
                fail_prob = 1.0 if rf_model.classes_[0] == 1 else 0.0

            # Polished rod power proxy
            power_penalty = 0.00035 * (pprl - max(mprl, 0)) * (stroke / 12.0) * spm

            # Penalty function for constraint violations
            penalty = 0.0
            if is_floating or mprl < 300.0:
                penalty += 35000.0  # Massive penalty for rod floating/buckling
            if pprl > 22000.0:
                penalty += (pprl - 22000.0) * 8.0
            penalty += fail_prob * 8000.0

            # Well-specific fluid rheology guidance matching petroleum engineering guidelines
            if api_gravity <= 15.0 or current_viscosity >= 8000.0:
                # Thick oil (e.g. BG-07): strictly maintain SPM between 4.0 and 5.0 to prevent rod floating
                if spm > 5.0:
                    penalty += (spm - 5.0) * 12000.0
            elif api_gravity >= 17.5 or current_viscosity <= 6000.0:
                # Thinner oil (e.g. BG-15): optimal SPM range is 8.0 - 10.0 for high volume recovery
                if spm < 8.0:
                    penalty += (8.0 - spm) * 6000.0
                if spm > 10.0:
                    penalty += (spm - 10.0) * 8000.0
            else:
                # Intermediate crude (e.g. BG-01): optimal SPM range between 4.0 and 8.0
                if spm > 7.5:
                    penalty += (spm - 7.5) * 4000.0

            # Net score: production minus power penalty and constraint penalties
            score = pred_prod - power_penalty - penalty

            if score > best_score:
                best_score = score
                confidence = float(np.clip(1.0 - fail_prob - (0.15 if is_floating else 0.0), 0.65, 0.98))
                opt_spm_val = float(np.clip(np.round(spm, 1), 4.0, 12.0))
                best_params = {
                    "optimal_spm": opt_spm_val,
                    "optimal_stroke_length": float(np.round(stroke, 0)),
                    "predicted_production": float(np.round(pred_prod, 2)),
                    "rod_load": float(np.round(pprl, 2)),
                    "mprl": float(np.round(mprl, 2)),
                    "confidence": float(np.round(confidence, 3)),
                    "energy_saving_pct": float(np.round(max(5.0, 22.0 - (spm - 4.5) * 2.8), 1))
                }

    return best_params


# ============================================================================
# 4. PREDICT FUTURE STATE (7-Day Forecast)
# ============================================================================

def predict_future_state(simulation_df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
    """
    Forecast reservoir temperature, oil production, and rod load for the next
    `days_ahead` days based on the simulation trajectory.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Day, Reservoir_Temp, Oil_Production, Oil_Viscosity, Rod_Load
    """
    # Filter to production phase
    prod_data = simulation_df[simulation_df["Phase"] == "Production"].copy()
    if prod_data.empty:
        prod_data = simulation_df.tail(min(30, len(simulation_df))).copy()

    last_day = int(prod_data["Day"].iloc[-1])
    recent_days = np.arange(len(prod_data))
    future_days_rel = np.arange(len(prod_data), len(prod_data) + days_ahead)
    forecast_days = np.arange(last_day + 1, last_day + days_ahead + 1)

    # 1. Temperature forecast (exponential decline towards native 48°C)
    temps = prod_data["Reservoir_Temp"].values
    t_native = 48.0
    if len(temps) >= 5 and temps[0] > t_native:
        # Fit log decay: log(T - 48) ~ a - b*t
        log_diff = np.log(np.maximum(temps - t_native, 0.5))
        poly = np.polyfit(recent_days, log_diff, deg=1)
        pred_log = poly[0] * future_days_rel + poly[1]
        pred_temps = t_native + np.exp(pred_log)
    else:
        # Fallback gentle decline
        last_t = temps[-1] if len(temps) > 0 else 75.0
        pred_temps = [max(last_t - 0.8 * i, t_native) for i in range(1, days_ahead + 1)]

    pred_temps = np.round(np.clip(pred_temps, 48.0, 290.0), 2)

    # 2. Viscosity forecast from Arrhenius equation
    pred_visc = [calculate_oil_viscosity(t) for t in pred_temps]
    pred_visc = np.round(pred_visc, 2)

    # 3. Production forecast (hyperbolic decline)
    oil_rates = prod_data["Oil_Production"].values
    if len(oil_rates) >= 5 and oil_rates[0] > 0:
        # Fit power law / polynomial
        poly_oil = np.polyfit(recent_days, np.log(np.maximum(oil_rates, 1.0)), deg=1)
        pred_oil_log = poly_oil[0] * future_days_rel + poly_oil[1]
        pred_oil = np.exp(pred_oil_log)
    else:
        last_oil = oil_rates[-1] if len(oil_rates) > 0 else 85.0
        pred_oil = [max(last_oil * (0.98 ** i), 5.0) for i in range(1, days_ahead + 1)]

    pred_oil = np.round(np.clip(pred_oil, 5.0, 450.0), 2)

    # 4. Rod load forecast based on predicted viscosity
    last_rod = prod_data["Rod_Load"].iloc[-1] if not prod_data.empty else 14500.0
    pred_rod = []
    for v in pred_visc:
        # As oil cools and viscosity rises, viscous drag adds upward rod load
        drag_adder = 0.05 * (v ** 0.45) * 80.0
        pred_rod.append(np.round(last_rod + drag_adder, 2))

    forecast_df = pd.DataFrame({
        "Day": forecast_days,
        "Reservoir_Temp": pred_temps,
        "Oil_Production": pred_oil,
        "Oil_Viscosity": pred_visc,
        "Rod_Load": pred_rod
    })

    return forecast_df


if __name__ == "__main__":
    print("Training models...")
    train_optimization_model(num_cycles=2000)
    train_rod_failure_classifier(num_samples=2000)
    print("\nTesting SRP optimization...")
    res = optimize_srp_parameters(current_viscosity=140.0, current_temp=82.0, pump_depth=1000.0)
    print("Optimization Result:", res)
