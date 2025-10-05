import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

DB_FILE = "transactions.db"
MODEL_PATH = "autoencoder_model.h5"
SCALER_PATH = "scaler.pkl"
CONFIG_PATH = "model_config.json"
RANDOM_SEED = 42

def load_normal_data(db_file):
    conn = sqlite3.connect(db_file)
    query = "SELECT * FROM transactions WHERE is_anomaly = 0"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df)} normal transactions.")
    return df

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['log_amount'] = np.log1p(df['amount'])

    # Normalize latitude and longitude between 0-1 using known India lat/lon bounds
    lat_min, lat_max = 8.0, 37.1
    lon_min, lon_max = 68.0, 97.4
    df['norm_latitude'] = (df['latitude'] - lat_min) / (lat_max - lat_min)
    df['norm_longitude'] = (df['longitude'] - lon_min) / (lon_max - lon_min)

    feat_cols = ['amount', 'log_amount', 'hour_of_day', 'day_of_week', 'norm_latitude', 'norm_longitude']
    print(f"Using features: {feat_cols}")
    return df[feat_cols]


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    e = Dense(16, activation='relu')(input_layer)
    e = Dense(8, activation='relu')(e)
    b = Dense(2, activation='relu')(e)
    d = Dense(8, activation='relu')(b)
    d = Dense(16, activation='relu')(d)
    output_layer = Dense(input_dim, activation='sigmoid')(d)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    autoencoder.summary()
    return autoencoder

def main():
    df = load_normal_data(DB_FILE)
    if df.empty:
        print("No data to train on. Exiting.")
        return
    features = engineer_features(df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    X_train, X_val = train_test_split(scaled_features, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Training data shape: {X_train.shape}; Validation data shape: {X_val.shape}")

    autoencoder = build_autoencoder(X_train.shape[1])
    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        shuffle=True,
        validation_data=(X_val, X_val),
        verbose=1
    )
    # Calculate anomaly threshold
    val_recon = autoencoder.predict(X_val)
    val_mae = np.mean(np.abs(val_recon - X_val), axis=1)
    threshold = float(np.mean(val_mae) + 3 * np.std(val_mae))
    print(f"Calculated anomaly threshold: {threshold}")

    # Save model, scaler, threshold
    autoencoder.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'threshold': threshold}, f)
    print(f"Config with threshold saved to {CONFIG_PATH}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    main()
