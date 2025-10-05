import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np

DBFILE = "transactions.db"
NUM_NORMAL = 10000
NUM_ANOMALOUS = 200

LATRANGE = (8.0, 37.1)
LONRANGE = (68.0, 97.4)

SCHEMENAMES = [
    "PM-KISAN", "MGNREGA", "Pradhan Mantri Jan Dhan Yojana",
    "National Health Mission", "Atal Pension Yojana",
    "Pradhan Mantri Awas Yojana", "Mid-Day Meal Scheme",
    "National Scholarship Portal"
]

def create_database():
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("""
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            beneficiary_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            scheme_type TEXT,
            is_anomaly INTEGER NOT NULL,
            fraud_type TEXT,
            status TEXT,
            anomaly_score REAL
        )
    """)
    conn.commit()
    conn.close()

def generate_normal_transactions(num):
    transactions = []
    datestart = datetime(2022, 1, 1)
    dateend = datetime(2023, 10, 1)
    for _ in range(num):
        bid = random.randint(10000, 50000)
        # Amount centered around 5000 with some noise
        base_amount = 5000
        noise = random.choice([0, 100, -100, 200, -200, 300, -300])
        amount = max(10, base_amount + noise)

        # Random day between datestart and dateend
        rand_days = random.randint(0, (dateend - datestart).days)
        rand_date = datestart + timedelta(days=rand_days)
        # Skip weekends
        while rand_date.weekday() >= 5:
            rand_date -= timedelta(days=1)

        # Random hour during business hours
        rand_hour = random.randint(9, 17)
        rand_minute = random.randint(0, 59)

        ts = rand_date.replace(hour=rand_hour, minute=rand_minute)

        lat = random.uniform(*LATRANGE)
        lon = random.uniform(*LONRANGE)
        scheme = random.choice(SCHEMENAMES)

        transactions.append((
            bid, amount, ts.strftime("%Y-%m-%d %H:%M:%S"),
            lat, lon, scheme, 0, None, None, None
        ))
    return transactions

def generate_anomalous_transactions(num):
    transactions = []
    now = datetime.now()

    # Unusual amounts anomalies
    for _ in range(num // 4):
        bid = random.randint(60000, 65000)
        amount = random.choice([50000, 75000, 150, 50])
        ts = now - timedelta(days=random.randint(1, 30))
        lat = random.uniform(*LATRANGE)
        lon = random.uniform(*LONRANGE)
        scheme = random.choice(SCHEMENAMES)

        transactions.append((
            bid, amount, ts.strftime("%Y-%m-%d %H:%M:%S"),
            lat, lon, scheme, 1, "Unusual Amount", None, None
        ))

    # Unusual timing anomalies
    for _ in range(num // 4):
        bid = random.randint(65000, 70000)
        amount = 2000
        base_date = now - timedelta(days=random.randint(1, 30))
        # Off-hours 0 to 4 AM
        if random.random() < 0.5:
            ts = base_date.replace(hour=random.randint(0, 4), minute=random.randint(0, 59))
        else:
            # Adjust to weekday if weekend
            while base_date.weekday() >= 5:
                base_date -= timedelta(days=1)
            ts = base_date.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))

        lat = random.uniform(*LATRANGE)
        lon = random.uniform(*LONRANGE)
        scheme = random.choice(SCHEMENAMES)

        transactions.append((
            bid, amount, ts.strftime("%Y-%m-%d %H:%M:%S"),
            lat, lon, scheme, 1, "Unusual Timing", None, None
        ))

    # High frequency burst anomalies
    for _ in range(10):
        bid = random.randint(70000, 75000)
        base_ts = now - timedelta(days=random.randint(1, 30), hours=random.randint(1, 12))
        lat = random.uniform(*LATRANGE)
        lon = random.uniform(*LONRANGE)
        scheme = random.choice(SCHEMENAMES)

        for i in range(10):
            amount = 2000
            ts = base_ts - timedelta(minutes=i*2)
            transactions.append((
                bid, amount, ts.strftime("%Y-%m-%d %H:%M:%S"),
                lat, lon, scheme, 1, "High Frequency", None, None
            ))

    # Collusion ring anomalies
    for _ in range(10):
        base_bid = random.randint(80000, 85000)
        lat = random.uniform(*LATRANGE)
        lon = random.uniform(*LONRANGE)
        scheme = random.choice(SCHEMENAMES)

        for i in range(10):
            bid = base_bid + i
            amount = 2000
            ts = now - timedelta(days=random.randint(1, 30), minutes=random.randint(1, 59))
            transactions.append((
                bid, amount, ts.strftime("%Y-%m-%d %H:%M:%S"),
                lat, lon, scheme, 1, "Collusion Ring", None, None
            ))

    return transactions

def insert_transactions(transactions):
    conn = sqlite3.connect(DBFILE)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO transactions (
            beneficiary_id, amount, timestamp, latitude, longitude,
            scheme_type, is_anomaly, fraud_type, status, anomaly_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, transactions)
    conn.commit()
    conn.close()

def main():
    print("Creating database and tables...")
    create_database()

    print("Generating normal transactions...")
    normal_txns = generate_normal_transactions(NUM_NORMAL)

    print("Generating anomalous transactions...")
    anomalous_txns = generate_anomalous_transactions(NUM_ANOMALOUS)

    all_txns = normal_txns + anomalous_txns
    random.shuffle(all_txns)

    print(f"Inserting {len(all_txns)} transactions...")
    insert_transactions(all_txns)

    print("Database generation complete.")

if __name__ == "__main__":
    main()
