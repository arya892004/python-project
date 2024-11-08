import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Create and Save Sample Blood Pressure Data
data = {
    'date_time': [
        "2024-01-01 08:00:00", "2024-01-01 14:30:00", "2024-01-02 09:00:00",
        "2024-01-03 08:00:00", "2024-01-04 14:30:00", "2024-01-05 09:00:00",
        "2024-01-06 08:00:00", "2024-01-07 14:30:00", "2024-01-08 09:00:00"
    ],
    'systolic': [120, 125, 115, 130, 128, 119, 121, 126, 118],
    'diastolic': [80, 82, 78, 85, 83, 79, 81, 84, 80],
    'heart_rate': [72, 75, 70, 78, 77, 73, 74, 76, 71],
    'age': [35, 35, 35, 35, 35, 35, 35, 35, 35],
    'activity': ["resting", "exercising", "resting", "exercising", "resting", "resting", "exercising", "resting", "resting"]
}

# Create DataFrame and Save as CSV
df = pd.DataFrame(data)
df.to_csv("blood_pressure_data.csv", index=False)
print("Sample blood pressure data saved to 'blood_pressure_data.csv'.")

# Step 2: Load the CSV Data
data = pd.read_csv("blood_pressure_data.csv")
data['date_time'] = pd.to_datetime(data['date_time'])
data.set_index('date_time', inplace=True)

# Step 3: Visualize Blood Pressure Over Time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['systolic'], label='Systolic', color='blue', linewidth=1)
plt.plot(data.index, data['diastolic'], label='Diastolic', color='red', linewidth=1)
plt.xlabel('Date Time')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Blood Pressure Over Time')
plt.legend()
plt.show()

# Step 4: Distribution of Blood Pressure Readings
plt.figure(figsize=(10, 5))
sns.histplot(data['systolic'], kde=True, color='blue', label='Systolic')
sns.histplot(data['diastolic'], kde=True, color='red', label='Diastolic')
plt.xlabel('Blood Pressure (mmHg)')
plt.title('Distribution of Blood Pressure Readings')
plt.legend()
plt.show()

# Step 5: Correlation Analysis between Systolic and Diastolic Pressure
correlation, _ = pearsonr(data['systolic'], data['diastolic'])
print(f'Correlation between systolic and diastolic pressure: {correlation:.2f}')

# Step 6: Trend and Seasonality Analysis (using Systolic as an example)
decomposed = seasonal_decompose(data['systolic'], model='additive', period=1)
decomposed.plot()
plt.suptitle('Seasonal Decomposition of Systolic Blood Pressure')
plt.show()

# Step 7: Rolling Average (3-day window for demonstration)
data['systolic_rolling'] = data['systolic'].rolling(window=3).mean()
data['diastolic_rolling'] = data['diastolic'].rolling(window=3).mean()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['systolic_rolling'], label='Systolic (3-day Avg)', color='blue')
plt.plot(data.index, data['diastolic_rolling'], label='Diastolic (3-day Avg)', color='red')
plt.xlabel('Date Time')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('3-Day Rolling Average of Blood Pressure')
plt.legend()
plt.show()

# Step 8: Blood Pressure by Time of Day (if hourly data)
data['hour'] = data.index.hour
hourly_avg = data.groupby('hour')[['systolic', 'diastolic']].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg['systolic'], label='Systolic', color='blue')
plt.plot(hourly_avg.index, hourly_avg['diastolic'], label='Diastolic', color='red')
plt.xlabel('Hour of Day')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Average Blood Pressure by Hour of Day')
plt.legend()
plt.show()

# Step 9: Summary Report
summary = f"""
Blood Pressure Data Analysis Summary:
1. Correlation between Systolic and Diastolic: {correlation:.2f}
2. Observed Daily Patterns and Trends in Blood Pressure.
3. 3-Day Rolling Average of Systolic and Diastolic shows longer-term trends.
"""

print(summary)
with open("blood_pressure_analysis_summary.txt", "w") as file:
    file.write(summary)