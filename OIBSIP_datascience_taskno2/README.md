# Objective

Unemployment is measured by the unemployment rate, which represents the number of unemployed people as a percentage of the total labour force.
During the Covid-19 pandemic, India witnessed a sharp increase in unemployment, making it a crucial area for analysis.

# Steps Involved
1. Data Collection

Used the dataset: Unemployment in India.csv

Contains unemployment data by State, Date, Area (Urban/Rural), and Rates.

2. Data Preprocessing

Cleaned column names and removed extra spaces.

Converted Date column to proper datetime format.

Dropped missing/invalid values.

3. Exploratory Data Analysis (EDA)

Analyzed unemployment rates across states.

Compared Urban vs Rural unemployment using boxplots.

Plotted trends over time.

Built an interactive Plotly graph for state-wise unemployment.

4. Correlation Analysis

Checked correlations between:

Unemployment Rate

Employment

Labour Participation Rate

Visualized using a heatmap.

5. Covid Impact Analysis

Split data into: Pre-Covid (before 2020) and Covid-2020.

Compared average unemployment before and during Covid.

Plotted bar and line charts to show the Covid spike.

6. Insights & Conclusion

Urban areas showed higher volatility in unemployment.

Some states (like Haryana, Tripura) had consistently high unemployment.

Covid-19 caused a sharp surge in unemployment across India.

Negative correlation observed between Employment and Unemployment Rate.

# Tools & Technologies Used

Python  – Core programming language

Pandas – Data cleaning and manipulation

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualization (barplots, lineplots, boxplots, heatmaps)

Plotly – Interactive visualizations

Jupyter Notebook / VS Code – Development environment

# Outcome
This project successfully analyzed unemployment trends in India using real-world data, with a particular focus on the impact of the COVID-19 pandemic. The analysis revealed significant differences across states and between rural and urban regions. Urban areas showed more fluctuation and higher average unemployment rates, while rural areas remained relatively more stable.
