import sqlite3
import random
from datetime import datetime
import time
import logging

DB_FILE = "transactions.db"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_db_connection():
    # Opens a new SQLite connection, enables WAL mode for concurrency
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode for better concurrent reads/writes
    return conn

def generate_transaction():
    # India bounding box (tighter bounds)
    # Lat ~ 8.0 to 37.1, Lon ~ 68.0 to 97.4
    latitude = random.uniform(8.0, 37.1)
    longitude = random.uniform(68.0, 97.4)

    beneficiary_id = random.randint(10000, 99999)

    # Majority normal amounts around 5,000 with some spread
    # ~85% normal, ~15% anomalies
    is_anom = random.random() < 0.15
    if not is_anom:
        # Normal amounts: around 5k with noise
        base = 5000
        noise = random.choice([0, 100, 200, -100, -200, 300, -300])
        amount = max(10, base + noise)
        fraud_type = None
    else:
        # Pick an anomaly type
        anomaly_kind = random.choice(["High Amount", "Low Amount", "Off-hours", "Geo Out-of-Bounds"])
        if anomaly_kind == "High Amount":
            amount = random.choice([20000, 50000, 70000, 100000])
            fraud_type = "High Amount"
        elif anomaly_kind == "Low Amount":
            amount = random.choice([20, 50, 70, 99])
            fraud_type = "Low Amount"
        elif anomaly_kind == "Off-hours":
            amount = random.choice([4800, 5000, 5200])
            fraud_type = "Off-hours"
        else:  # Geo Out-of-Bounds
            # Put a few outside India bounds intentionally
            # e.g., slightly outside the bbox
            latitude = random.choice([7.5, 37.8, 38.0])
            longitude = random.choice([67.5, 97.9, 98.5])
            amount = random.choice([4800, 5000, 5200])
            fraud_type = "Geo Out-of-Bounds"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scheme_type = random.choice([
        "PM-KISAN", "MGNREGA", "Jan Dhan", "Health Mission",
        "Atal Pension", "Housing", "Mid-Day Meal", "Scholarship"
    ])

    is_anomaly = 1 if fraud_type is not None else 0

    # status/anomaly_score left for monitor to fill
    return (beneficiary_id, amount, timestamp, latitude, longitude, scheme_type, is_anomaly, fraud_type, None, None)


def insert_transaction(conn, trx):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (
                beneficiary_id, amount, timestamp, latitude, longitude, scheme_type,
                is_anomaly, fraud_type, status, anomaly_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, trx)
        conn.commit()
        logging.info(f"Inserted transaction amount={trx[1]} anomaly={trx} timestamp={trx}")
    except sqlite3.Error as e:
        logging.error(f"Database insert error: {e}")

def simulate_live_feed():
    logging.info("Starting live transaction feed simulator... Press Ctrl+C to stop.")
    conn = get_db_connection()
    try:
        while True:
            trx = generate_transaction()
            insert_transaction(conn, trx)
            time.sleep(random.uniform(0.5, 3.0))
    except KeyboardInterrupt:
        logging.info("Simulator stopped by user.")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    simulate_live_feed()
