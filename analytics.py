import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta

class CareerAnalytics:
    def __init__(self, db_path: str = 'data/career_predictions.db'):
        self.db_path = db_path
        self.output_dir = 'analytics_output'
        os.makedirs(self.output_dir, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_predictions_data(self, days: int = 30) -> pd.DataFrame:
        """Get prediction data for the last n days."""
        query = """
        SELECT * FROM predictions 
        WHERE timestamp >= date('now', ?)
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[f'-{days} days'])

def main():
    analytics = CareerAnalytics()
    report = analytics.generate_report()
    print("Analytics Report:")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
