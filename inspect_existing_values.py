from app import get_connection

conn = None
cursor = None

try:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    print("\n========== EXISTING CRIME TYPES ==========")

    cursor.execute("""
        SELECT crime_type_id, category, subcategory, severity_level
        FROM crime_types
        ORDER BY crime_type_id
    """)

    for row in cursor.fetchall():
        print(row)

    print("\n========== EXISTING LOCATIONS ==========")

    cursor.execute("""
        SELECT location_id, county, sub_county, ward,
               latitude, longitude, area_type
        FROM locations
        ORDER BY location_id
    """)

    for row in cursor.fetchall():
        print(row)

    print("\n========== EXISTING OFFICERS ==========")

    cursor.execute("""
        SELECT officer_id, badge_number, officer_rank,
               station, county
        FROM officers
        ORDER BY officer_id
    """)

    for row in cursor.fetchall():
        print(row)

    print("\n========== SAMPLE INCIDENT RECORDS ==========")

    cursor.execute("""
        SELECT incident_id, case_number, crime_type_id,
               location_id, officer_id, incident_date,
               incident_time, reported_date, status
        FROM crime_incidents
        ORDER BY incident_id DESC
        LIMIT 5
    """)

    for row in cursor.fetchall():
        print(row)

    print("\n========== INSPECTION COMPLETE ==========")

except Exception as e:
    print("Inspection error:", e)

finally:
    if cursor:
        cursor.close()

    if conn and conn.is_connected():
        conn.close()