from app import get_connection

conn = None
cursor = None

tables = [
    "crime_incidents",
    "crime_types",
    "locations",
    "officers"
]

try:
    conn = get_connection()
    cursor = conn.cursor()

    for table in tables:
        print(f"\n========== {table.upper()} ==========")

        cursor.execute(f"SHOW CREATE TABLE `{table}`")
        result = cursor.fetchone()

        print(result[1])

        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        count = cursor.fetchone()[0]

        print(f"Existing records: {count}")

    print("\n========== SCHEMA INSPECTION COMPLETE ==========")

except Exception as e:
    print("Inspection error:", e)

finally:
    if cursor:
        cursor.close()

    if conn and conn.is_connected():
        conn.close()