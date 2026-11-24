
import os
import html
import folium
import mysql.connector
import pandas as pd
import numpy as np

from folium.plugins import HeatMap, MarkerCluster


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "templates")
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "crime_map.html")



def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="cpaps_db"
    )



def get_color(severity):

    severity = str(severity).strip().title()

    colors = {
        "Critical": "red",
        "High": "orange",
        "Medium": "blue",
        "Low": "green"
    }

    return colors.get(severity, "gray")


def get_icon(category):

    category = str(category).strip()

    icons = {
        "Violent Crime": "exclamation-sign",
        "Property Crime": "home",
        "Drug Offences": "tint",
        "Cyber Crime": "phone",
        "Public Order": "flag"
    }

    return icons.get(category, "info-sign")


def safe_text(value):

    if pd.isna(value):
        return "N/A"

    return html.escape(str(value))


def safe_number(value):

    try:
        if pd.isna(value):
            return 0

        return int(float(value))

    except (ValueError, TypeError, OverflowError):
        return 0



def load_crime_data():

    conn = None

    query = """
        SELECT
            cr.case_number,
            cr.incident_date,
            cr.crime_type,
            cr.crime_category,
            cr.severity_level,
            cr.county,
            cr.sub_county,
            cr.ward,
            cr.status,
            cr.property_loss_ksh,
            cr.fatalities,
            cr.injuries_reported,
            l.latitude,
            l.longitude

        FROM v_crime_report cr

        INNER JOIN crime_incidents ci
            ON cr.case_number = ci.case_number

        INNER JOIN locations l
            ON ci.location_id = l.location_id

        WHERE
            l.latitude IS NOT NULL
            AND l.longitude IS NOT NULL
    """

    try:

        conn = get_connection()

        df = pd.read_sql_query(query, conn)

        print(
            f"Successfully loaded {len(df)} crime records."
        )

        return df

    except mysql.connector.Error as e:

        print("Database error:", e)

        return pd.DataFrame()

    except Exception as e:

        print("Error loading crime data:", e)

        return pd.DataFrame()

    finally:

        if conn is not None and conn.is_connected():
            conn.close()


def build_crime_map():

    print("\nBuilding interactive CPAPS crime map...\n")

    # Load database records
    df = load_crime_data()

    if df.empty:

        print("No crime records found. Map cannot be generated.")

        return None


    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    if df.empty:

        print("No valid geographic coordinates available.")

        return None

    print(
        f"Valid georeferenced incidents: {len(df)}"
    )

    df["severity_level"] = (
        df["severity_level"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["crime_category"] = (
        df["crime_category"]
        .fillna("Other")
        .astype(str)
        .str.strip()
    )

    df["county"] = (
        df["county"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
        control_scale=True
    )

    # Public OpenStreetMap tiles.
    # No API key is required.

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=True
    ).add_to(m)


    # Small offsets prevent markers at identical
    # coordinates from completely overlapping.
    # Original coordinates are retained for analysis.

    np.random.seed(42)

    df["lat_offset"] = (
        df["latitude"]
        + np.random.uniform(-0.002, 0.002, len(df))
    )

    df["lon_offset"] = (
        df["longitude"]
        + np.random.uniform(-0.002, 0.002, len(df))
    )

    heat_layer = folium.FeatureGroup(
        name="Crime Heatmap",
        show=True
    )

    heat_data = df[
        ["latitude", "longitude"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=20,
        blur=15,
        min_opacity=0.3,
        max_zoom=10,
        gradient={
            0.2: "blue",
            0.5: "yellow",
            0.8: "orange",
            1.0: "red"
        }
    ).add_to(heat_layer)

    heat_layer.add_to(m)

    marker_layer = folium.FeatureGroup(
        name="Individual Crime Incidents",
        show=False
    )

    marker_cluster = MarkerCluster(
        name="Crime Clusters"
    ).add_to(marker_layer)

    for _, row in df.iterrows():

        color = get_color(row["severity_level"])

        icon = get_icon(row["crime_category"])

        case_number = safe_text(row["case_number"])
        crime_type = safe_text(row["crime_type"])
        category = safe_text(row["crime_category"])
        severity = safe_text(row["severity_level"])
        incident_date = safe_text(row["incident_date"])
        county = safe_text(row["county"])
        sub_county = safe_text(row["sub_county"])
        status = safe_text(row["status"])

        property_loss = safe_number(
            row["property_loss_ksh"]
        )

        fatalities = safe_number(
            row["fatalities"]
        )

        injuries = safe_number(
            row["injuries_reported"]
        )

        popup_html = f"""
        <div style="
            font-family: Arial;
            min-width: 230px;
            color: #333;
        ">

            <h3 style="
                color: #e63946;
                margin-bottom: 10px;
            ">
                {crime_type}
            </h3>

            <table style="
                font-size: 12px;
                width: 100%;
                border-collapse: collapse;
            ">

                <tr>
                    <td><b>Case Number</b></td>
                    <td>{case_number}</td>
                </tr>

                <tr>
                    <td><b>Date</b></td>
                    <td>{incident_date}</td>
                </tr>

                <tr>
                    <td><b>Category</b></td>
                    <td>{category}</td>
                </tr>

                <tr>
                    <td><b>Severity</b></td>
                    <td style="color:{color}">
                        <b>{severity}</b>
                    </td>
                </tr>

                <tr>
                    <td><b>County</b></td>
                    <td>{county}</td>
                </tr>

                <tr>
                    <td><b>Sub County</b></td>
                    <td>{sub_county}</td>
                </tr>

                <tr>
                    <td><b>Status</b></td>
                    <td>{status}</td>
                </tr>

                <tr>
                    <td><b>Property Loss</b></td>
                    <td>KES {property_loss:,}</td>
                </tr>

                <tr>
                    <td><b>Fatalities</b></td>
                    <td>{fatalities}</td>
                </tr>

                <tr>
                    <td><b>Injuries</b></td>
                    <td>{injuries}</td>
                </tr>

            </table>

        </div>
        """

        folium.Marker(

            location=[
                row["lat_offset"],
                row["lon_offset"]
            ],

            popup=folium.Popup(
                popup_html,
                max_width=350
            ),

            tooltip=f"{crime_type} - {county}",

            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="glyphicon"
            )

        ).add_to(marker_cluster)

    marker_layer.add_to(m)



    hotspot_layer = folium.FeatureGroup(
        name="County Hotspot Zones",
        show=False
    )

    county_totals = (

        df.groupby("county", dropna=False)

        .agg(
            incident_count=("case_number", "count"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean")
        )

        .reset_index()

    )

    max_count = county_totals["incident_count"].max()

    for _, row in county_totals.iterrows():

        count = int(row["incident_count"])

        radius = 1000 + (
            count / max_count
        ) * 9000

        county_name = safe_text(row["county"])

        popup_html = f"""
        <div style="font-family:Arial">

            <h3 style="color:#e63946">
                {county_name} County
            </h3>

            <p>
                <b>Total Crime Incidents:</b> {count}
            </p>

        </div>
        """

        folium.Circle(

            location=[
                row["latitude"],
                row["longitude"]
            ],

            radius=radius,

            color="#e63946",

            weight=2,

            fill=True,

            fill_color="#e63946",

            fill_opacity=0.15,

            tooltip=f"{county_name}: {count} incidents",

            popup=folium.Popup(
                popup_html,
                max_width=250
            )

        ).add_to(hotspot_layer)

    hotspot_layer.add_to(m)


    severity_options = [
        ("Critical", "red"),
        ("High", "orange"),
        ("Medium", "blue"),
        ("Low", "green")
    ]

    for severity, color in severity_options:

        sev_layer = folium.FeatureGroup(
            name=f"{severity} Severity",
            show=False
        )

        sev_df = df[
            df["severity_level"] == severity
        ]

        for _, row in sev_df.iterrows():

            folium.CircleMarker(

                location=[
                    row["lat_offset"],
                    row["lon_offset"]
                ],

                radius=6,

                color=color,

                weight=2,

                fill=True,

                fill_color=color,

                fill_opacity=0.8,

                tooltip=(
                    f"{safe_text(row['crime_type'])}"
                    f" - {safe_text(row['county'])}"
                )

            ).add_to(sev_layer)

        sev_layer.add_to(m)

   

    folium.LayerControl(
        collapsed=False,
        position="topright"
    ).add_to(m)

   

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: rgba(22,27,34,0.95);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        font-family: Arial;
        color: #e6edf3;
        min-width: 175px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.4);
    ">

        <h4 style="
            margin: 0 0 12px;
            color: #e63946;
        ">
            CPAPS LEGEND
        </h4>

        <div style="margin:6px 0">
            <span style="color:red">●</span>
            Critical Severity
        </div>

        <div style="margin:6px 0">
            <span style="color:orange">●</span>
            High Severity
        </div>

        <div style="margin:6px 0">
            <span style="color:#4a9eff">●</span>
            Medium Severity
        </div>

        <div style="margin:6px 0">
            <span style="color:green">●</span>
            Low Severity
        </div>

        <hr style="border-color:#30363d">

        <div style="
            font-size:11px;
            color:#8b949e;
            line-height:1.5;
        ">
            KCA University<br>
            CPAPS - Crime Pattern Analysis<br>
            and Prediction System
        </div>

    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )


    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    m.save(OUTPUT_PATH)


    print("\n===================================")

    print("CPAPS CRIME MAP GENERATED SUCCESSFULLY")

    print("===================================")

    print(f"Total incidents: {len(df)}")

    print(f"Counties covered: {len(county_totals)}")

    print("\nLayers generated:")

    print("1. OpenStreetMap Base Layer")
    print("2. Crime Heatmap")
    print("3. Individual Crime Incidents")
    print("4. Crime Clusters")
    print("5. County Hotspot Zones")
    print("6. Critical Severity")
    print("7. High Severity")
    print("8. Medium Severity")
    print("9. Low Severity")

    print(f"\nMap saved to: {OUTPUT_PATH}")

    print("===================================\n")

    return OUTPUT_PATH



if __name__ == "__main__":

    try:

        build_crime_map()

    except Exception as e:

        print("\nAn unexpected error occurred:")
        print(e)

