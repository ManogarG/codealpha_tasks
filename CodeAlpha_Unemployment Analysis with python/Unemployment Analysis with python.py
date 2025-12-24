# Unemployment Analysis with Python
# CodeAlpha Data Science Internship - Task 2

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import kagglehub

# Set visualization style
plt.rcParams['figure.figsize'] = (12, 6)

# Step 1: Download the dataset from Kaggle
print("Downloading Unemployment in India dataset from Kaggle...")
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Path to dataset files:", path)

# Step 2: Load the dataset
df = pd.read_csv(path + '/Unemployment in India.csv')

print("\n=== Dataset Overview ===")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Step 3: Data Cleaning
print("\n=== Data Cleaning ===")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for common column name variations
print(f"Cleaned column names: {df.columns.tolist()}")

# Convert date column to datetime (adjust column name as needed)
date_column = None
for col in df.columns:
    if 'date' in col.lower():
        date_column = col
        break

if date_column:
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    print(f"\nDate column '{date_column}' converted to datetime")

# Handle missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
print(f"Rows remaining after cleaning: {len(df)}")

# Step 4: Exploratory Data Analysis
print("\n=== Exploratory Data Analysis ===")

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Find unemployment rate column
unemployment_col = None
for col in df.columns:
    if 'unemployment' in col.lower() or 'rate' in col.lower():
        if df[col].dtype in ['float64', 'int64']:
            unemployment_col = col
            break

if unemployment_col:
    print(f"\nUnemployment Rate Statistics:")
    print(f"Mean: {df[unemployment_col].mean():.2f}%")
    print(f"Median: {df[unemployment_col].median():.2f}%")
    print(f"Min: {df[unemployment_col].min():.2f}%")
    print(f"Max: {df[unemployment_col].max():.2f}%")
    print(f"Std Dev: {df[unemployment_col].std():.2f}%")

# Check for region/state column
region_col = None
for col in df.columns:
    if any(keyword in col.lower() for keyword in ['region', 'state', 'area']):
        region_col = col
        break

if region_col:
    print(f"\n{region_col} distribution:")
    print(df[region_col].value_counts())

# Step 5: Covid-19 Impact Analysis
print("\n=== Covid-19 Impact Analysis ===")

if date_column:
    # Define pre-Covid and Covid periods
    covid_start = pd.to_datetime('2020-03-01')
    
    df['Period'] = df[date_column].apply(
        lambda x: 'Pre-Covid' if x < covid_start else 'During Covid'
    )
    
    if unemployment_col:
        period_analysis = df.groupby('Period')[unemployment_col].agg(['mean', 'median', 'std'])
        print("\nUnemployment Rate by Period:")
        print(period_analysis)
        
        pre_covid_rate = df[df['Period'] == 'Pre-Covid'][unemployment_col].mean()
        covid_rate = df[df['Period'] == 'During Covid'][unemployment_col].mean()
        increase = covid_rate - pre_covid_rate
        percent_increase = (increase / pre_covid_rate) * 100
        
        print(f"\nPre-Covid Average Rate: {pre_covid_rate:.2f}%")
        print(f"During Covid Average Rate: {covid_rate:.2f}%")
        print(f"Increase: {increase:.2f}% ({percent_increase:.2f}% rise)")

# Step 6: Trend Analysis
print("\n=== Trend Analysis ===")

if date_column and unemployment_col:
    # Monthly trends
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    
    monthly_trend = df.groupby(['Year', 'Month'])[unemployment_col].mean().reset_index()
    print("\nMonthly unemployment trends calculated")
    
    # Yearly trends
    yearly_trend = df.groupby('Year')[unemployment_col].mean()
    print("\nYearly Average Unemployment Rate:")
    print(yearly_trend)

# Step 7: Regional Analysis
print("\n=== Regional Analysis ===")

if region_col and unemployment_col:
    regional_stats = df.groupby(region_col)[unemployment_col].agg(['mean', 'min', 'max'])
    regional_stats = regional_stats.sort_values('mean', ascending=False)
    
    print("\nTop 5 Regions with Highest Unemployment:")
    print(regional_stats.head())
    
    print("\nTop 5 Regions with Lowest Unemployment:")
    print(regional_stats.tail())

# Step 8: Visualizations
print("\n=== Creating Visualizations ===")

# Visualization 1: Unemployment Rate Over Time
if date_column and unemployment_col:
    plt.figure(figsize=(14, 6))
    plt.plot(df[date_column], df[unemployment_col], linewidth=2, color='blue', alpha=0.7)
    if 'Period' in df.columns:
        plt.axvline(x=covid_start, color='red', linestyle='--', linewidth=2, label='Covid-19 Start')
    plt.title('Unemployment Rate Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Unemployment Rate (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('unemployment_trend.png', dpi=300, bbox_inches='tight')
    print("Saved: unemployment_trend.png")
    plt.show()

# Visualization 2: Pre-Covid vs During Covid Comparison
if 'Period' in df.columns and unemployment_col:
    plt.figure(figsize=(10, 6))
    
    periods = df['Period'].unique()
    data_to_plot = [df[df['Period'] == period][unemployment_col].values for period in periods]
    
    bp = plt.boxplot(data_to_plot, labels=periods, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Unemployment Rate: Pre-Covid vs During Covid', fontsize=16, fontweight='bold')
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('Unemployment Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('covid_impact_boxplot.png', dpi=300, bbox_inches='tight')
    print("Saved: covid_impact_boxplot.png")
    plt.show()

# Visualization 3: Regional Comparison
if region_col and unemployment_col:
    plt.figure(figsize=(14, 8))
    top_regions = df.groupby(region_col)[unemployment_col].mean().sort_values(ascending=True).tail(10)
    
    plt.barh(range(len(top_regions)), top_regions.values, color='coral', edgecolor='darkred')
    plt.yticks(range(len(top_regions)), top_regions.index)
    plt.title('Top 10 Regions by Average Unemployment Rate', fontsize=16, fontweight='bold')
    plt.xlabel('Average Unemployment Rate (%)', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('regional_unemployment.png', dpi=300, bbox_inches='tight')
    print("Saved: regional_unemployment.png")
    plt.show()

# Visualization 4: Monthly Pattern Analysis
if 'Month' in df.columns and unemployment_col:
    plt.figure(figsize=(12, 6))
    monthly_pattern = df.groupby('Month')[unemployment_col].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.bar(range(1, 13), monthly_pattern.values, color='skyblue', edgecolor='navy', linewidth=1.5)
    plt.xticks(range(1, 13), month_names)
    plt.title('Average Unemployment Rate by Month (Seasonal Pattern)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Unemployment Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('seasonal_pattern.png', dpi=300, bbox_inches='tight')
    print("Saved: seasonal_pattern.png")
    plt.show()

# Step 9: Key Insights and Recommendations
print("\n" + "="*60)
print("=== KEY INSIGHTS AND POLICY RECOMMENDATIONS ===")
print("="*60)

insights = []

if 'Period' in df.columns and unemployment_col:
    if covid_rate > pre_covid_rate:
        insights.append(f"1. Covid-19 Impact: Unemployment increased by {percent_increase:.1f}% during the pandemic")
        insights.append(f"   - Pre-Covid average: {pre_covid_rate:.2f}%")
        insights.append(f"   - During Covid average: {covid_rate:.2f}%")

if region_col and unemployment_col:
    highest_region = regional_stats.index[0]
    highest_rate = regional_stats.iloc[0]['mean']
    lowest_region = regional_stats.index[-1]
    lowest_rate = regional_stats.iloc[-1]['mean']
    insights.append(f"\n2. Regional Disparity:")
    insights.append(f"   - Highest: {highest_region} ({highest_rate:.2f}%)")
    insights.append(f"   - Lowest: {lowest_region} ({lowest_rate:.2f}%)")
    insights.append(f"   - Gap: {highest_rate - lowest_rate:.2f} percentage points")

if 'Month' in df.columns and unemployment_col:
    peak_month = monthly_pattern.idxmax()
    low_month = monthly_pattern.idxmin()
    insights.append(f"\n3. Seasonal Trends:")
    insights.append(f"   - Peak unemployment: {month_names[peak_month-1]} ({monthly_pattern.max():.2f}%)")
    insights.append(f"   - Lowest unemployment: {month_names[low_month-1]} ({monthly_pattern.min():.2f}%)")

insights.append("\n4. Policy Recommendations:")
insights.append("   - Implement targeted job creation programs in high-unemployment regions")
insights.append("   - Develop sector-specific recovery plans for Covid-affected industries")
insights.append("   - Create seasonal employment schemes during high-unemployment months")
insights.append("   - Strengthen unemployment insurance and social safety nets")
insights.append("   - Invest in skill development and reskilling programs")

for insight in insights:
    print(insight)

print("\n" + "="*60)
print("=== Analysis Complete! ===")
print("="*60)
print(f"\nGenerated visualizations:")
print("  • unemployment_trend.png")
print("  • covid_impact_boxplot.png")
print("  • regional_unemployment.png")
print("  • seasonal_pattern.png")
print("\nAll files saved in the current directory.")