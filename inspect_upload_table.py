
from app import get_connection

conn = None
cursor = None

try:
    conn = get_connection()
    cursor = conn.cursor()

    print("\n========== CPAPS DATABASE TABLES ==========")

    cursor.execute("SHOW FULL TABLES")

    for row in cursor.fetchall():
        print(row)

    print("\n========== CRIME REPORT VIEW DEFINITION ==========")

    cursor.execute("SHOW CREATE VIEW v_crime_report")

    result = cursor.fetchone()
    print(result[1])

    print("\n========== INSPECTION COMPLETE ==========")

except Exception as e:
    print("Inspection error:", e)

finally:
    if cursor:
        cursor.close()

    if conn and conn.is_connected():
        conn.close()
