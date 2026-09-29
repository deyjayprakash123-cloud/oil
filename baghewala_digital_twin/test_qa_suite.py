import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_data
import physics_engine as pe
import data_generator as dg
import ai_optimizer as ai
import database as db

results = []

# Step 1: Files
files = [
    'app.py', 'physics_engine.py', 'data_generator.py', 'ai_optimizer.py',
    'well_data.py', 'database.py', 'requirements.txt', 'README.md',
    os.path.join('.streamlit', 'config.toml')
]
base_dir = os.path.dirname(os.path.abspath(__file__))
missing = [f for f in files if not os.path.exists(os.path.join(base_dir, f))]
results.append(('Step 1: File Integrity', len(missing) == 0, f'Missing: {missing}'))

# Step 2: Physics Engine
t_curve = pe.calculate_reservoir_temperature(47.0, 250.0, 5, 60, pump_depth=1020)
t_min_ok = np.all(t_curve >= 46.99)
t_decay_ok = np.all(np.diff(t_curve[15:]) <= 0.05)
t_soak_zero = pe.calculate_reservoir_temperature(47.0, 250.0, 0, 60)
results.append(('Test 2.1: Reservoir Temperature Model', t_min_ok and t_decay_ok and (len(t_soak_zero) == 60), f't_min={np.min(t_curve):.1f}C, len_zero_soak={len(t_soak_zero)}'))

v_ref = pe.calculate_oil_viscosity(47.0, 7800.0, 47.0, api_gravity=15.8)
v_hot = pe.calculate_oil_viscosity(200.0, 7800.0, 47.0, api_gravity=15.8)
v_zero = pe.calculate_oil_viscosity(0.0, 7800.0, 47.0, api_gravity=15.8)
results.append(('Test 2.2: Oil Viscosity Model (Arrhenius)', abs(v_ref - 7800.0) < 0.1 and 10.0 < v_hot < 300.0 and v_zero <= 60000.0, f'v_ref={v_ref:.1f} cP, v_hot={v_hot:.1f} cP, v_zero={v_zero:.1f} cP'))

res_p = pe.calculate_rod_load(8.0, 80.0, 7800.0, 1020.0, 1.0)
p_pprl, p_mprl, p_float = res_p["PPRL"], res_p["MPRL"], res_p["Rod_Floating_Flag"]

res_b = pe.calculate_rod_load(8.0, 80.0, 9500.0, 980.0, 1.0)
b_pprl, b_mprl, b_float = res_b["PPRL"], res_b["MPRL"], res_b["Rod_Floating_Flag"]

res_z = pe.calculate_rod_load(0.0, 80.0, 7800.0, 1020.0, 1.0)
z_pprl, z_mprl, z_float = res_z["PPRL"], res_z["MPRL"], res_z["Rod_Floating_Flag"]
results.append(('Test 2.3: Rod Load & Floating Physics', (15000 <= p_pprl <= 30000) and p_mprl > 0 and not p_float and b_float and b_mprl < 0 and z_pprl > 0, f'p_pprl={p_pprl:.1f} lbs, p_mprl={p_mprl:.1f} lbs, BG-07 rod_float={b_float}'))

sor_norm = pe.calculate_steam_oil_ratio(500.0, 10000.0)
sor_zero = pe.calculate_steam_oil_ratio(0.0, 10000.0)
results.append(('Test 2.4: Steam-Oil Ratio Calculations', sor_norm == 20.0 and np.isinf(sor_zero), f'sor_norm={sor_norm}, sor_zero={sor_zero}'))

sim_df = pe.simulate_css_cycle(steam_volume=10000.0, spm=8.0, soak_days=5, production_days=60, reservoir_temp=47.0, oil_viscosity=7800.0, api_gravity=15.8, pump_depth=1020.0)
cols = ['Day', 'Phase', 'Reservoir_Temp', 'Oil_Viscosity', 'Oil_Production', 'Steam_Injected', 'SOR', 'Rod_Load', 'Pump_Efficiency', 'Rod_Floating_Flag']
cols_ok = all(c in sim_df.columns for c in cols)
phases_ok = set(sim_df['Phase'].unique()) == {'Injection', 'Soak', 'Production'}
inj_prod_zero = (sim_df[sim_df['Phase'].isin(['Injection', 'Soak'])]['Oil_Production'] == 0).all()
results.append(('Test 2.5: CSS Simulation Cycle Trajectory', cols_ok and phases_ok and inj_prod_zero and len(sim_df) >= 60, f'rows={len(sim_df)}, phases={list(sim_df["Phase"].unique())}'))

# Step 3: AI Optimizer
opt_bg01 = ai.optimize_srp_parameters(7800.0, 47.0, 1020.0, 15.8)
opt_bg15 = ai.optimize_srp_parameters(5500.0, 48.0, 1080.0, 18.0)
opt_bg07 = ai.optimize_srp_parameters(9500.0, 48.5, 980.0, 14.5)
spm_bg01_ok = 4.0 <= opt_bg01['optimal_spm'] <= 8.0
spm_bg15_ok = 8.0 <= opt_bg15['optimal_spm'] <= 10.0
spm_bg07_ok = 4.0 <= opt_bg07['optimal_spm'] <= 5.0
pred_future = ai.predict_future_state(sim_df)
pred_ok = len(pred_future) == 7 and (pred_future['Oil_Production'] >= 0).all()
results.append(('Step 3: AI Optimizer & Forecaster', spm_bg01_ok and spm_bg15_ok and spm_bg07_ok and pred_ok, f'BG-01 SPM={opt_bg01["optimal_spm"]}, BG-15 SPM={opt_bg15["optimal_spm"]}, BG-07 SPM={opt_bg07["optimal_spm"]}'))

# Step 4: Data Generator
css_data = dg.generate_css_cycle_data(1000)
css_ok = len(css_data) == 1000 and (css_data['Steam_Volume'] >= 5000).all() and (css_data['Steam_Volume'] <= 20000).all() and (css_data['SPM'] >= 4).all() and (css_data['SPM'] <= 12).all() and (css_data['Oil_Production'] > 0).all() and (css_data['SOR'] > 0).all() and not css_data.isnull().values.any()
rod_data = dg.generate_rod_failure_data(5000)
fail_rate = rod_data['Rod_Failure_Risk'].mean()
rod_ok = len(rod_data) == 5000 and set(rod_data['Rod_Failure_Risk'].unique()) == {0, 1} and (0.10 <= fail_rate <= 0.30)
results.append(('Step 4: Synthetic Data Generator', css_ok and rod_ok, f'css_rows={len(css_data)}, fail_rate={fail_rate*100:.2f}%'))

# Step 6: Database
db.init_database()
dummy_sim = {
    'steam_volume': 10000.0,
    'soak_days': 5,
    'spm': 8.0,
    'oil_produced': 450.0,
    'sor': 22.2,
    'rod_load': 18500.0,
    'pump_efficiency': 75.0,
    'rod_floating': 0
}
db.save_simulation(dummy_sim)
hist = db.get_simulation_history(limit=5)
db.save_alert('WARNING', 'Test QA Alert for Baghewala Field', 'WARNING')
alerts = db.get_alerts(limit=5)
results.append(('Step 6: SQLite Database Ops & Ring Buffer', len(hist) > 0 and len(alerts) > 0, f'hist_count={len(hist)}, alert_count={len(alerts)}'))

# Step 8: Error handling
sim_zero_steam = pe.simulate_css_cycle(steam_volume=0.0, spm=8.0, soak_days=5, production_days=60, reservoir_temp=47.0, oil_viscosity=7800.0)
sim_zero_spm = pe.simulate_css_cycle(steam_volume=10000.0, spm=0.0, soak_days=5, production_days=60, reservoir_temp=47.0, oil_viscosity=7800.0)
sim_zero_prod = pe.simulate_css_cycle(steam_volume=10000.0, spm=8.0, soak_days=5, production_days=0, reservoir_temp=47.0, oil_viscosity=7800.0)
results.append(('Step 8: Robust Boundary Error Handling', len(sim_zero_steam) > 0 and len(sim_zero_spm) > 0 and len(sim_zero_prod) > 0, 'Zero steam, zero SPM, zero prod days handled without crashing'))

print('\n==================== BAGHEWALA DIGITAL TWIN QA SUITE ====================')
all_pass = True
for name, passed, detail in results:
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_pass = False
    print(f'[{status}] {name}\n       -> {detail}')
print(f'========================================================================')
print(f'OVERALL STATUS: {"ALL TESTS PASSED" if all_pass else "FAILURES DETECTED"} ({sum(1 for _, p, _ in results)}/{len(results)})')
