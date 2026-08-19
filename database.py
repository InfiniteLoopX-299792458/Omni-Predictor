import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    """Initializes the SQLite database for tracking predictions."""
    conn = sqlite3.connect('predictions_ledger.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            predicted_direction TEXT,
            confidence REAL,
            actual_outcome TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(ticker, direction, confidence):
    """Logs a new prediction into the database."""
    conn = sqlite3.connect('predictions_ledger.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute('''
        INSERT INTO predictions (timestamp, ticker, predicted_direction, confidence, actual_outcome)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, ticker, direction, confidence, "Pending"))
    conn.commit()
    conn.close()

def get_ledger_data():
    """Fetches all logged predictions to display on the dashboard."""
    conn = sqlite3.connect('predictions_ledger.db')
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df