"""
Baghewala Digital Twin - SQLite Database Module (database.py)
Oil India Limited | Smart India Hackathon 2026 (Problem Statement ID: 26120)

Manages SQLite database (baghewala.db) for storing simulation results,
AI optimization runs, and SCADA alarm records.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala.db")


def get_db_connection() -> sqlite3.Connection:
    """Returns a thread-safe connection to baghewala.db with row factory."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Create SQLite database baghewala.db with tables:
    - simulations (id, timestamp, steam_volume, soak_days, spm, oil_produced, sor, rod_load)
    - alerts (id, timestamp, alert_type, message, severity)
    Plus auxiliary tables for well assets and detailed scenario tracking.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Primary Required Table: simulations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        steam_volume REAL NOT NULL,
        soak_days INTEGER NOT NULL,
        spm REAL NOT NULL,
        oil_produced REAL NOT NULL,
        sor REAL NOT NULL,
        rod_load REAL NOT NULL,
        stroke_length REAL DEFAULT 80.0,
        pump_depth REAL DEFAULT 1000.0,
        pump_efficiency REAL DEFAULT 75.0,
        rod_floating INTEGER DEFAULT 0
    )
    """)

    # 2. Primary Required Table: alerts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        message TEXT NOT NULL,
        severity TEXT NOT NULL
    )
    """)

    # 3. Auxiliary table: wells for Baghewala Field overview
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wells (
        well_id TEXT PRIMARY KEY,
        well_name TEXT NOT NULL,
        formation TEXT DEFAULT 'Jodhpur Sandstone',
        latitude REAL,
        longitude REAL,
        depth_m REAL,
        api_gravity REAL,
        dead_oil_visc_cp REAL,
        current_status TEXT,
        spm REAL,
        stroke_in REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

    # Seed initial wells if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM wells")
    if cursor.fetchone()["cnt"] == 0:
        seed_wells(conn)

    # Seed a couple default alerts if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM alerts")
    if cursor.fetchone()["cnt"] == 0:
        seed_initial_alerts(conn)

    conn.close()


def seed_wells(conn):
    """Seed representative Baghewala wells."""
    cursor = conn.cursor()
    wells_data = [
        ("BG-01", "Baghewala-01 (Main Pad)", "Jodhpur Sandstone", 27.8421, 72.3120, 1020.0, 15.8, 7800.0, "Producing", 5.2, 100.0),
        ("BG-04", "Baghewala-04 (North Pad)", "Jodhpur Sandstone", 27.8590, 72.3280, 980.0, 14.9, 9200.0, "Producing (Alert)", 8.4, 120.0),
        ("BG-07", "Baghewala-07 (East Flank)", "Jodhpur Sandstone", 27.8340, 72.3410, 1060.0, 16.2, 6500.0, "Steam Injection", 0.0, 0.0),
        ("BG-12", "Baghewala-12 (Central Cluster)", "Bilara Formation", 27.8480, 72.3210, 1010.0, 15.2, 8400.0, "Soaking Phase", 0.0, 0.0),
        ("BG-15", "Baghewala-15 (South Extension)", "Jodhpur Sandstone", 27.8210, 72.2980, 1110.0, 14.2, 11500.0, "High Viscous Drag", 4.0, 90.0),
    ]
    cursor.executemany("""
    INSERT INTO wells (well_id, well_name, formation, latitude, longitude, depth_m, api_gravity, dead_oil_visc_cp, current_status, spm, stroke_in)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, wells_data)
    conn.commit()


def seed_initial_alerts(conn):
    """Seed initial sample alerts for telemetry stream."""
    cursor = conn.cursor()
    sample_alerts = [
        ("2026-09-29 10:15:00", "PUMP_OFF", "Well BG-04: High SPM (8.4) causing fluid pound & rod floating risk.", "CRITICAL"),
        ("2026-09-29 11:30:00", "THERMAL_EFFICIENCY", "Well BG-15: Bottomhole temp dropped to 62°C; SOR trending above 4.8.", "WARNING"),
        ("2026-09-29 14:00:00", "SYSTEM_HEALTH", "Well BG-01: SRP operating within optimal API RP 11L limits (5.2 SPM).", "INFO")
    ]
    cursor.executemany("""
    INSERT INTO alerts (timestamp, alert_type, message, severity)
    VALUES (?, ?, ?, ?)
    """, sample_alerts)
    conn.commit()


def save_simulation(results_dict: Dict[str, Any]) -> int:
    """
    Insert simulation results into the simulations table.

    Parameters:
    -----------
    results_dict : dict
        Keys: steam_volume, soak_days, spm, oil_produced, sor, rod_load,
              and optional stroke_length, pump_depth, pump_efficiency, rod_floating

    Returns:
    --------
    int
        Inserted record ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = results_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    steam_vol = float(results_dict.get("steam_volume", 0.0))
    soak_d = int(results_dict.get("soak_days", 0))
    spm_val = float(results_dict.get("spm", 0.0))
    oil_prod = float(results_dict.get("oil_produced", 0.0))
    sor_val = float(results_dict.get("sor", 0.0))
    rod_ld = float(results_dict.get("rod_load", 0.0))

    stroke_len = float(results_dict.get("stroke_length", 80.0))
    p_depth = float(results_dict.get("pump_depth", 1000.0))
    p_eff = float(results_dict.get("pump_efficiency", 75.0))
    rod_flt = int(results_dict.get("rod_floating", 0))

    cursor.execute("""
    INSERT INTO simulations (
        timestamp, steam_volume, soak_days, spm, oil_produced, sor, rod_load,
        stroke_length, pump_depth, pump_efficiency, rod_floating
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, steam_vol, soak_d, spm_val, oil_prod, sor_val, rod_ld,
        stroke_len, p_depth, p_eff, rod_flt
    ))

    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_simulation_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve past simulations for trend analysis.

    Parameters:
    -----------
    limit : int
        Maximum number of past runs to return

    Returns:
    --------
    List[Dict[str, Any]]
        List of simulation records ordered by timestamp descending
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM simulations ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def save_alert(alert_type: str, message: str, severity: str = "WARNING", cooldown_seconds: int = 180) -> int:
    """
    Log warnings and critical alerts with automatic deduplication cooldown
    to prevent flooding the database on rapid UI reruns or slider movements.

    Parameters:
    -----------
    alert_type : str
        e.g., 'ROD_FLOATING', 'HIGH_SOR', 'MOTOR_OVERLOAD', 'SYSTEM_OPTIMAL'
    message : str
        Descriptive message for engineers
    severity : str
        'INFO', 'WARNING', or 'CRITICAL'
    cooldown_seconds : int
        Seconds within which duplicate alerts are skipped (default 180s)

    Returns:
    --------
    int
        Inserted or existing alert ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if cooldown_seconds > 0:
        cursor.execute(
            "SELECT id, timestamp FROM alerts WHERE alert_type = ? AND message = ? ORDER BY id DESC LIMIT 1",
            (alert_type, message)
        )
        last_alert = cursor.fetchone()
        if last_alert:
            try:
                last_time = datetime.strptime(last_alert["timestamp"], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_time).total_seconds() < cooldown_seconds:
                    existing_id = int(last_alert["id"])
                    conn.close()
                    return existing_id
            except Exception:
                pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO alerts (timestamp, alert_type, message, severity)
    VALUES (?, ?, ?, ?)
    """, (timestamp, alert_type, message, severity.upper()))

    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def clear_alerts():
    """Clear all recorded alerts from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


def clear_simulation_history():
    """Clear simulation records from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM simulations")
    conn.commit()
    conn.close()


def cleanup_old_records(max_simulations: int = 1000, max_alerts: int = 1000):
    """Prune old records to ensure the SQLite database does not grow indefinitely."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    DELETE FROM simulations WHERE id NOT IN (
        SELECT id FROM simulations ORDER BY id DESC LIMIT ?
    )
    """, (max_simulations,))
    cursor.execute("""
    DELETE FROM alerts WHERE id NOT IN (
        SELECT id FROM alerts ORDER BY id DESC LIMIT ?
    )
    """, (max_alerts,))
    conn.commit()
    conn.close()


def get_all_wells() -> List[Dict[str, Any]]:
    """Retrieve all Baghewala field well records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM wells ORDER BY well_id ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

