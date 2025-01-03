# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Analysis
from scipy.stats import linregress
import statsmodels.api as sm

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load datasets
df = pd.read_csv("Customer_chrun.py/temperature.csv")
df1 = pd.read_csv("Customer_chrun.py/carbon_emmission.csv")

# Display the first 5 rows of each dataset
print(df.head())
print(df1.head())

# Dataset Information
print(df.info())
print(df1.info())
# Load datasets
df = pd.read_csv("Customer_chrun.py/temperature.csv")
df1 = pd.read_csv("Customer_chrun.py/carbon_emmission.csv")

# Display the first 5 rows of each dataset
print(df.head())
print(df1.head())

# Dataset Information
print(df.info())
print(df1.info())

# Calculate key statistics for temperature anomalies
temperature_stats = pd.DataFrame()
temperature_stats['mean'] = df.iloc[:, 4:].mean()
temperature_stats['median'] = df.iloc[:, 4:].median()
temperature_stats['variance'] = df.iloc[:, 4:].var()

# Calculate key statistics for CO2 concentrations
co2_stats = pd.Series({
    'mean': df1['Value'].mean(),
    'median': df1['Value'].median(),
    'variance': df1['Value'].var()
})

# Display summary statistics
temperature_summary = temperature_stats.describe()
co2_summary = co2_stats

# Prepare data for analysis
temperature_means = df.iloc[:, 4:].mean()
years = [int(col[1:]) for col in df.columns[4:]]
df1['Year'] = df1['Date'].str[:4].astype(int)
co2_means = df1.groupby('Year')['Value'].mean()

# Plot Temperature Anomalies Over Time
plt.figure(figsize=(12, 6))
plt.plot(years, temperature_means, marker='o', linestyle='-', label='Temperature Anomaly (°C)')
plt.title('Global Temperature Anomalies Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.legend()
plt.show()

# Plot CO₂ Concentrations Over Time
plt.figure(figsize=(12, 6))
plt.plot(co2_means.index, co2_means, marker='o', linestyle='-', label='CO2 Concentration (ppm)')
plt.title('Global CO₂ Concentrations Over Time')
plt.xlabel('Year')
plt.ylabel('CO₂ Concentration (ppm)')
plt.grid(True)
plt.legend()
plt.show()

# Merge data based on years
merged_data = pd.DataFrame({'Year': years, 'Temperature': temperature_means}).merge(
    co2_means.reset_index(), on='Year')

# Calculate and visualize correlation
correlation = merged_data['Temperature'].corr(merged_data['Value'])

plt.figure(figsize=(8, 6))
sns.heatmap(merged_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap: Temperature vs CO₂ Concentrations')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Value'], merged_data['Temperature'], alpha=0.7)
plt.title('Scatter Plot: Temperature vs CO₂ Concentrations')
plt.xlabel('CO₂ Concentration (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.show()

# Linear Regression Model for Trends
X_co2 = merged_data['Year'].values.reshape(-1, 1)
y_co2 = merged_data['Value'].values

co2_model = LinearRegression()
co2_model.fit(X_co2, y_co2)
co2_trend = co2_model.predict(X_co2)

y_temp = merged_data['Temperature'].values
temp_model = LinearRegression()
temp_model.fit(X_co2, y_temp)
temp_trend = temp_model.predict(X_co2)

# Plot Linear Trends
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(merged_data['Year'], merged_data['Value'], alpha=0.7, label='CO₂ Data')
plt.plot(merged_data['Year'], co2_trend, color='red', label='Trend')
plt.title('CO₂ Concentration Trend')
plt.xlabel('Year')
plt.ylabel('CO₂ Concentration (ppm)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(merged_data['Year'], merged_data['Temperature'], alpha=0.7, label='Temperature Data')
plt.plot(merged_data['Year'], temp_trend, color='red', label='Trend')
plt.title('Temperature Anomaly Trend')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()

plt.tight_layout()
plt.show()

# Monthly Average CO₂ Concentrations
df1['Month'] = pd.to_datetime(df1['Date'], format='%YM%m').dt.month
monthly_avg_co2 = df1.groupby('Month')['Value'].mean()

# Plot Seasonal Variations
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_co2.index, monthly_avg_co2.values, marker='o', linestyle='-', color='blue')
plt.title('Seasonal Variations in CO₂ Concentrations')
plt.xlabel('Month')
plt.ylabel('CO₂ Concentration (ppm)')
plt.grid(True)
plt.show()

# Create Lagged Variables
merged_data['CO2_Lag1'] = merged_data['Value'].shift(1)
merged_data['CO2_Lag2'] = merged_data['Value'].shift(2)
merged_data['CO2_Lag3'] = merged_data['Value'].shift(3)

lagged_data = merged_data.dropna()
X = lagged_data[['Value', 'CO2_Lag1', 'CO2_Lag2', 'CO2_Lag3']]
y = lagged_data['Temperature']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Clustering
clustering_data = merged_data[['Temperature', 'Value']]
scaled_data = StandardScaler().fit_transform(clustering_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Value'], merged_data['Temperature'], c=merged_data['Cluster'], cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering: Temperature vs CO₂ Concentrations')
plt.xlabel('CO₂ Concentration (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.show()

# Predict Temperature Anomalies for Different CO₂ Scenarios
current_co2 = merged_data['Value'].mean()
scenarios = {
    "Increase CO₂ by 10%": current_co2 * 1.10,
    "Decrease CO₂ by 10%": current_co2 * 0.90,
    "Increase CO₂ by 20%": current_co2 * 1.20,
    "Decrease CO₂ by 20%": current_co2 * 0.80,
}

predictions = {}
for scenario, co2_level in scenarios.items():
    predicted_temp = model.predict([[co2_level]])[0]
    predictions[scenario] = predicted_temp

print(predictions)
