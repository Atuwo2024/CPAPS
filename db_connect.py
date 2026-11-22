import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",           
    password="12345",           
    database="cpaps_db"
)

if conn.is_connected():
    print("✅ Connected to cpaps_db successfully!")

cursor = conn.cursor()
cursor.execute("SELECT * FROM crime_incidents")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()