import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# SETTINGS
sns.set_theme(style="darkgrid")
OUTPUT_DIR = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  DB CONNECTION 
def get_connection():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = "12345",   
        database = "cpaps_db"
    )

# LOAD DATA 
def load_data():
    conn = get_connection()
    query = "SELECT * FROM v_crime_report"
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"✅ Loaded {len(df)} crime records\n")
    return df

#  CHART 1: Crime Types Distribution 
def plot_crime_types(df):
    plt.figure(figsize=(12, 6))
    order = df['crime_type'].value_counts().index
    sns.countplot(data=df, y='crime_type', order=order, palette='Reds_r')
    plt.title('Crime Types Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_crime_types.png")
    plt.show()
    print("✅ Chart 1 saved: Crime Types Distribution")

#  CHART 2: Crime by County (Hotspots) 
def plot_crime_by_county(df):
    plt.figure(figsize=(10, 5))
    order = df['county'].value_counts().index
    sns.countplot(data=df, x='county', order=order, palette='OrRd')
    plt.title('Crime Incidents by County', fontsize=16, fontweight='bold')
    plt.xlabel('County')
    plt.ylabel('Number of Incidents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_crime_by_county.png")
    plt.show()
    print("✅ Chart 2 saved: Crime by County")

#  CHART 3: Crime by Hour of Day 
def plot_crime_by_hour(df):
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='hour_of_day', palette='coolwarm')
    plt.title('Crime Incidents by Hour of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Hour (0 = Midnight, 12 = Noon)')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_crime_by_hour.png")
    plt.show()
    print("✅ Chart 3 saved: Crime by Hour of Day")

#  CHART 4: Monthly Crime Trend 
def plot_monthly_trend(df):
    df['year_month'] = pd.to_datetime(
        df['year_number'].astype(str) + '-' + df['month_number'].astype(str)
    )
    monthly = df.groupby('year_month').size().reset_index(name='count')
    plt.figure(figsize=(14, 5))
    plt.plot(monthly['year_month'], monthly['count'], marker='o', color='crimson', linewidth=2)
    plt.fill_between(monthly['year_month'], monthly['count'], alpha=0.2, color='crimson')
    plt.title('Monthly Crime Trend (2020-2024)', fontsize=16, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Number of Incidents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_monthly_trend.png")
    plt.show()
    print("✅ Chart 4 saved: Monthly Crime Trend")

#  CHART 5: Crime by Day of Week 
def plot_crime_by_day(df):
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='day_of_week', order=day_order, palette='magma')
    plt.title('Crime Incidents by Day of Week', fontsize=16, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_crime_by_day.png")
    plt.show()
    print("✅ Chart 5 saved: Crime by Day of Week")

#  CHART 6: Crime Severity Distribution 
def plot_severity(df):
    plt.figure(figsize=(8, 5))
    order = ['Critical', 'High', 'Medium', 'Low']
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c']
    sns.countplot(data=df, x='severity_level', order=order, palette=colors)
    plt.title('Crime Severity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_severity.png")
    plt.show()
    print("✅ Chart 6 saved: Severity Distribution")

#  CHART 7: Crime Category Breakdown 
def plot_crime_category(df):
    category_counts = df['crime_category'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette('Set2'),
        startangle=140
    )
    plt.title('Crime Category Breakdown', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_crime_category_pie.png")
    plt.show()
    print("✅ Chart 7 saved: Crime Category Breakdown")

#  CHART 8: Property Loss by Crime Type 
def plot_property_loss(df):
    loss_df = df.groupby('crime_type')['property_loss_ksh'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    loss_df.plot(kind='bar', color='steelblue')
    plt.title('Average Property Loss by Crime Type (KES)', fontsize=16, fontweight='bold')
    plt.xlabel('Crime Type')
    plt.ylabel('Average Loss (KES)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/8_property_loss.png")
    plt.show()
    print("✅ Chart 8 saved: Property Loss by Crime Type")

#  CHART 9: Case Status Distribution 
def plot_case_status(df):
    plt.figure(figsize=(8, 5))
    order = df['status'].value_counts().index
    sns.countplot(data=df, x='status', order=order, palette='Blues_r')
    plt.title('Case Status Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Status')
    plt.ylabel('Number of Cases')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/9_case_status.png")
    plt.show()
    print("✅ Chart 9 saved: Case Status Distribution")

# CHART 10: Heatmap — Crime Category vs Day of Week 
def plot_heatmap(df):
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = df.pivot_table(
        index='crime_category',
        columns='day_of_week',
        values='case_number',
        aggfunc='count',
        fill_value=0
    )
    pivot = pivot.reindex(columns=day_order, fill_value=0)
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
    plt.title('Crime Category vs Day of Week Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Crime Category')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_heatmap.png")
    plt.show()
    print("✅ Chart 10 saved: Heatmap")

#  PRINT SUMMARY STATS 
def print_summary(df):
    print("\n" + "="*50)
    print("       CPAPS - EDA SUMMARY STATISTICS")
    print("="*50)
    print(f"Total Incidents         : {len(df)}")
    print(f"Most Common Crime       : {df['crime_type'].mode()[0]}")
    print(f"Highest Crime County    : {df['county'].mode()[0]}")
    print(f"Peak Hour               : {int(df['hour_of_day'].mode()[0])}:00")
    print(f"Total Property Loss     : KES {df['property_loss_ksh'].sum():,.2f}")
    print(f"Total Fatalities        : {int(df['fatalities'].sum())}")
    print(f"Cases with Injuries     : {df['injuries_reported'].sum()}")
    print("="*50 + "\n")

#  MAIN 
if __name__ == "__main__":
    print("\n🚀 Starting Exploratory Data Analysis...\n")
    df = load_data()
    print_summary(df)
    plot_crime_types(df)
    plot_crime_by_county(df)
    plot_crime_by_hour(df)
    plot_monthly_trend(df)
    plot_crime_by_day(df)
    plot_severity(df)
    plot_crime_category(df)
    plot_property_loss(df)
    plot_case_status(df)
    plot_heatmap(df)
  
