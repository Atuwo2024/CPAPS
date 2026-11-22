import mysql.connector
import random
from datetime import datetime, timedelta

# ── DB CONNECTION 
def get_connection():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = "12345",   
        database = "cpaps_db"
    )

#  REFERENCE DATA
crime_type_ids  = list(range(1, 15)) 
location_ids    = list(range(1, 9))    
officer_ids     = list(range(1, 6))    

weather_options = ['Clear', 'Rainy', 'Cloudy', 'Night', 'Unknown']
status_options  = ['Open', 'Under Investigation', 'Closed', 'Cold Case']

# GENERATE RANDOM DATE 
def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

#  GENERATE RANDOM TIME 
def random_time():
    hour   = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:00"

#  GENERATE SUSPECTS 
first_names = [
    'Ali', 'Brian', 'Chege', 'David', 'Erick',
    'Faith', 'George', 'Hassan', 'Isaac', 'James',
    'Kevin', 'Lilian', 'Moses', 'Nancy', 'Owen',
    'Peter', 'Queen', 'Robert', 'Sarah', 'Thomas'
]
last_names = [
    'Mwangi', 'Ochieng', 'Kamau', 'Otieno', 'Njoroge',
    'Kipchoge', 'Wanjiku', 'Auma', 'Mutua', 'Nyambura',
    'Karanja', 'Adhiambo', 'Kiptoo', 'Wanjiru', 'Omondi'
]

def generate_suspects(cursor, count=100):
    print(f"Generating {count} suspects...")
    for _ in range(count):
        first  = random.choice(first_names)
        last   = random.choice(last_names)
        gender = random.choice(['Male', 'Female', 'Unknown'])
        age    = random.randint(16, 55)
        status = random.choice(['At Large', 'Arrested', 'Convicted', 'Released', 'Unknown'])
        cursor.execute("""
            INSERT INTO suspects (first_name, last_name, gender, age_estimate, status)
            VALUES (%s, %s, %s, %s, %s)
        """, (first, last, gender, age, status))
    print("✅ Suspects inserted.")

#  GENERATE CRIME INCIDENTS 
def generate_incidents(cursor, count=500):
    print(f"Generating {count} crime incidents...")
    for i in range(1, count + 1):
        case_number   = f"NPS/{random.randint(2020,2024)}/{i+100:04d}"
        crime_type_id = random.choice(crime_type_ids)
        location_id   = random.choice(location_ids)
        officer_id    = random.choice(officer_ids)
        incident_date = random_date()
        incident_time = random_time()
        reported_date = incident_date + timedelta(days=random.randint(0, 3))
        status        = random.choice(status_options)
        weather       = random.choice(weather_options)
        property_loss = round(random.uniform(0, 500000), 2) if random.random() > 0.4 else 0.00
        injuries      = random.choice([True, False])
        fatalities    = random.randint(0, 3) if random.random() < 0.1 else 0

        cursor.execute("""
            INSERT INTO crime_incidents (
                case_number, crime_type_id, location_id, officer_id,
                incident_date, incident_time, reported_date,
                status, weather_condition, property_loss_ksh,
                injuries_reported, fatalities
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            case_number, crime_type_id, location_id, officer_id,
            incident_date.strftime('%Y-%m-%d'), incident_time,
            reported_date.strftime('%Y-%m-%d'),
            status, weather, property_loss, injuries, fatalities
        ))
    print("✅ Crime incidents inserted.")

#  LINK SUSPECTS TO INCIDENTS 
def generate_incident_suspects(cursor, incident_count=500, suspect_count=100):
    print("Linking suspects to incidents...")
    roles = ['Primary', 'Accomplice', 'Person of Interest']
    # Get actual incident IDs from DB (skipping the 10 already inserted)
    cursor.execute("SELECT incident_id FROM crime_incidents WHERE incident_id > 10")
    incident_ids = [row[0] for row in cursor.fetchall()]

    for incident_id in random.sample(incident_ids, min(300, len(incident_ids))):
        num_suspects = random.randint(1, 3)
        for _ in range(num_suspects):
            suspect_id = random.randint(1, suspect_count)
            role       = random.choice(roles)
            arrested   = random.choice([True, False])
            cursor.execute("""
                INSERT IGNORE INTO incident_suspects (incident_id, suspect_id, role, arrested)
                VALUES (%s, %s, %s, %s)
            """, (incident_id, suspect_id, role, arrested))
    print("✅ Incident-suspect links inserted.")

#  MAIN RUNNER 
def main():
    print("\n🔌 Connecting to cpaps_db...")
    conn   = get_connection()
    cursor = conn.cursor()
    print("✅ Connected!\n")

    try:
        generate_suspects(cursor, count=100)
        conn.commit()

        generate_incidents(cursor, count=500)
        conn.commit()

        generate_incident_suspects(cursor, incident_count=500, suspect_count=100)
        conn.commit()

        # ── SUMMARY 
        print("\n📊 DATABASE SUMMARY:")
        tables = ['locations', 'crime_types', 'officers', 'suspects',
                  'crime_incidents', 'incident_suspects']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table:<25} → {count:>4} records")

    

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
