import io
import pandas as pd
import csv

headers = [
    "case_number",
    "incident_date",
    "crime_type",
    "crime_category",
    "severity_level",
    "county",
    "incident_time"
]

sample_data = [
    ["TEST-001", "2026-11-20", "Robbery", "Violent Crime",
     "High", "Nairobi", "14:30:00"]
]

# Create temporary CSV in memory
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(headers)
writer.writerows(sample_data)

csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")

# Create temporary Excel workbook in memory
excel_buffer = io.BytesIO()

with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    pd.DataFrame(sample_data, columns=headers).to_excel(
        writer, index=False
    )

excel_bytes = excel_buffer.getvalue()

# Test CSV reading
csv_result = csv.DictReader(
    io.StringIO(csv_bytes.decode("utf-8-sig"))
)

print("CSV TEST:")
print(list(csv_result))

# Test Excel reading
excel_result = pd.read_excel(
    io.BytesIO(excel_bytes),
    header=None,
    dtype=object,
    engine="openpyxl"
)

print("\nEXCEL TEST:")
print(excel_result)

print("\nBoth file formats were read successfully.")
from app import parse_upload_row

test_row = {
    "case_number": "TEST-001",
    "incident_date": "2026-11-20",
    "crime_type": "Robbery",
    "crime_category": "Violent Crime",
    "severity_level": "High",
    "county": "Nairobi",
    "incident_time": "14:30:00"
}

print("\nCPAPS VALIDATION TEST:")
validated = parse_upload_row(test_row, 2)
print(validated)
print("\nCPAPS validation passed.")

