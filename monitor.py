import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import time
import logging

MODEL_PATH = "modeling/autoencoder_model.h5"
SCALER_PATH = "modeling/scaler.pkl"
CONFIG_PATH = "modeling/model_config.json"
DB_FILE = "transactions.db"
POLL_INTERVAL_SECONDS = 5
BATCH_SIZE = 50   # Number of transactions to process per batch

# Setup logging syslog-like with timestamps and info level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

INDIA_LAT_MIN, INDIA_LAT_MAX = 8.0, 37.1
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 97.4

def classify_anomaly(row, threshold_amount=5000):
    """
    Returns an interpretable fraud_type string for anomalies.
    Categories: High Amount, Low Amount, Off-hours, Geo Out-of-Bounds, Unusual Pattern.
    """
    amt = float(row.get("amount", 0.0))
    ts = pd.to_datetime(row.get("timestamp"))
    lat = row.get("latitude")
    lon = row.get("longitude")

    # Amount-based
    if amt >= 2.5 * threshold_amount:
        return "High Amount"
    if amt <= 0.05 * threshold_amount:
        return "Low Amount"

    # Time-based (off-hours: 00:00–05:59)
    try:
        hour = int(pd.to_datetime(ts).hour)
        if hour < 6:
            return "Off-hours"
    except Exception:
        pass

    # Geo-based
    try:
        if lat is not None and lon is not None:
            if not (INDIA_LAT_MIN <= float(lat) <= INDIA_LAT_MAX and INDIA_LON_MIN <= float(lon) <= INDIA_LON_MAX):
                return "Geo Out-of-Bounds"
    except Exception:
        pass

    # Fallback
    return "Unusual Pattern"


def load_artifacts(model_path, scaler_path, config_path):
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = config['threshold']
        logging.info(f"Artifacts loaded successfully. Threshold = {threshold}")
        return model, scaler, threshold
    except Exception as e:
        logging.error(f"Failed to load artifacts: {e}")
        return None, None, None

def engineer_features_for_prediction(df_row):
    df_row = df_row.copy()
    df_row['timestamp'] = pd.to_datetime(df_row['timestamp'])

    # Extract time features
    df_row['hour_of_day'] = df_row['timestamp'].dt.hour
    df_row['day_of_week'] = df_row['timestamp'].dt.dayofweek

    # Log amount
    df_row['log_amount'] = np.log1p(df_row['amount'])

    # Normalizing latitude and longitude using same bounds as training
    lat_min, lat_max = 8.0, 37.1
    lon_min, lon_max = 68.0, 97.4
    df_row['norm_latitude'] = (df_row['latitude'] - lat_min) / (lat_max - lat_min)
    df_row['norm_longitude'] = (df_row['longitude'] - lon_min) / (lon_max - lon_min)

    feature_columns = ['amount', 'log_amount', 'hour_of_day', 'day_of_week', 'norm_latitude', 'norm_longitude']
    return df_row[feature_columns]

def process_batch(trx_list, column_names, model, scaler, threshold, cursor, conn):
    df_batch = pd.DataFrame(trx_list, columns=column_names)
    results_status = []
    results_fraud = []

    try:
        features = engineer_features_for_prediction(df_batch)
        scaled_features = scaler.transform(features)
        reconstructions = model.predict(scaled_features, verbose=0)
        maes = np.mean(np.abs(scaled_features - reconstructions), axis=1)

        for i, row in df_batch.iterrows():
            trx_id = int(row["transaction_id"])
            score = float(maes[i])
            status = "ANOMALY" if score > threshold else "NORMAL"
            results_status.append((status, score, trx_id))

            # Derive fraud_type only if anomaly; otherwise None
            if status == "ANOMALY":
                fraud = classify_anomaly(row, threshold_amount=5000)
            else:
                fraud = None
            results_fraud.append((fraud, trx_id))

    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return

    # Batch update DB
    try:
        # Update status and score
        cursor.executemany("UPDATE transactions SET status = ?, anomaly_score = ? WHERE transaction_id = ?", results_status)
        # Update fraud_type
        cursor.executemany("UPDATE transactions SET fraud_type = ? WHERE transaction_id = ?", results_fraud)
        conn.commit()
        logging.info(f"Batch updated {len(results_status)} transactions.")
    except Exception as e:
        logging.error(f"Error updating batch to DB: {e}")

def monitor_transactions():
    model, scaler, threshold = load_artifacts(MODEL_PATH, SCALER_PATH, CONFIG_PATH)
    if model is None:
        logging.error("Model artifacts not loaded. Exiting.")
        return

    logging.info("Starting real-time monitoring...")

    while True:
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM transactions WHERE status IS NULL LIMIT ?", (BATCH_SIZE,))
            new_transactions = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

            if len(new_transactions) == 0:
                logging.info("No new transactions found. Sleeping...")
            else:
                logging.info(f"Processing {len(new_transactions)} new transactions.")
                process_batch(new_transactions, column_names, model, scaler, threshold, cursor, conn)

            conn.close()
            time.sleep(POLL_INTERVAL_SECONDS)

        except sqlite3.Error as db_err:
            logging.error(f"Database error: {db_err}. Retrying in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user.")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            break

if __name__ == "__main__":
    monitor_transactions()
