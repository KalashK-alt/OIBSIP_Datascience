# Unemployment Analysis with Python (Custom for given CSV)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------
# Step 1: Load & Clean Dataset
# -----------------------------
file_path = "Unemployment in India.csv"
df = pd.read_csv(file_path)

# Rename columns
df = df.rename(columns={
    "Region": "State",
    " Date": "Date",
    " Frequency": "Frequency",
    " Estimated Unemployment Rate (%)": "Unemployment_Rate",
    " Estimated Employed": "Employed",
    " Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
    " Area": "Area"
})

# Remove extra spaces in Date column
df["Date"] = df["Date"].str.strip()

# Convert Date to datetime (dd-mm-yyyy format)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%d-%m-%Y")

# Drop missing rows (if any)
df = df.dropna()

print("✅ Dataset cleaned successfully")
print(df.info())
print(df.head())

# -----------------------------
# Step 2: Unemployment by State
# -----------------------------
plt.figure(figsize=(14,6))
sns.barplot(data=df, x="State", y="Unemployment_Rate", errorbar=None)
plt.xticks(rotation=90)
plt.title("Average Unemployment Rate by State", fontsize=14)
plt.xlabel("States")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Step 3: Urban vs Rural Unemployment
# -----------------------------
plt.figure(figsize=(6,5))
sns.boxplot(data=df, x="Area", y="Unemployment_Rate")
plt.title("Unemployment Rate in Urban vs Rural Areas", fontsize=14)
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Step 4: Trend Over Time
# -----------------------------
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="Date", y="Unemployment_Rate", hue="Area")
plt.title("Unemployment Trend Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Step 5: Interactive Visualization
# -----------------------------
fig = px.line(df, x="Date", y="Unemployment_Rate", color="State",
              title="Unemployment Rate Over Time by State")
fig.show()

# -----------------------------
# Step 6: Correlation Analysis
# -----------------------------
plt.figure(figsize=(8,5))
sns.heatmap(df[["Unemployment_Rate","Employed","Labour_Participation_Rate"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix", fontsize=14)
plt.show()

# -----------------------------
# Step 7: Covid Impact Analysis
# -----------------------------
# Add Covid period column
df["Covid_Period"] = df["Date"].apply(lambda x: "Pre-Covid" if x.year < 2020 else "Covid-2020")

# Average unemployment before vs during Covid
covid_comparison = df.groupby("Covid_Period")["Unemployment_Rate"].mean()

print("\n📊 Average Unemployment Rate Comparison:")
print(covid_comparison)

# Bar chart comparison
plt.figure(figsize=(6,5))
sns.barplot(x=covid_comparison.index, y=covid_comparison.values)
plt.title("Covid Impact on Unemployment (Pre-2020 vs 2020)", fontsize=14)
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.show()

# Line chart to see spike
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="Date", y="Unemployment_Rate", hue="Covid_Period", palette="Set1")
plt.title("Unemployment Trend Before vs During Covid", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Step 8: Insights
# -----------------------------
print("\n--- Key Insights ---")
print("1. Some states (like Haryana, Tripura, etc.) show higher unemployment rates.")
print("2. Urban areas generally have more volatility in unemployment compared to rural.")
print("3. During Covid period (2020), unemployment spiked sharply across most states.")
print("4. Unemployment Rate is negatively correlated with Employment levels.")
print("5. Covid-19 caused a significant increase in unemployment compared to pre-2020.")
