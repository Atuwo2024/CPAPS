
import mysql.connector
from mysql.connector import Error

from app import get_connection

conn = None
cursor = None

try:
    conn = get_connection()
    cursor = conn.cursor()

    print("\n========== CPAPS DATABASE TARGET INSPECTION ==========")

    print("\n1. COLUMNS IN v_crime_report")

    cursor.execute("DESCRIBE v_crime_report")

    columns = cursor.fetchall()

    for column in columns:
        print(column[0], "-", column[1])

    print("\n2. CHECKING POSSIBLE PREDICTION TARGETS")

    cursor.execute("""
        SELECT *
        FROM v_crime_report
        LIMIT 5
    """)

    names = [item[0] for item in cursor.description]

    rows = cursor.fetchall()

    targets = [
        "crime_type",
        "crime_category",
        "severity_level",
        "property_loss_ksh",
        "injuries",
        "injuries_reported",
        "fatalities"
    ]

    for target in targets:
        if target in names:
            print(f"\n{target}: AVAILABLE")

            index = names.index(target)

            values = [row[index] for row in rows]

            print("Sample values:", values)
        else:
            print(f"\n{target}: NOT AVAILABLE")

    print("\n========== INSPECTION COMPLETE ==========")

except Error as e:
    print("Database inspection error:", e)

finally:
    if cursor:
        cursor.close()

    if conn and conn.is_connected():
        conn.close()
