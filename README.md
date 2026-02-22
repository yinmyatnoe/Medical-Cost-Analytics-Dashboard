# Medical Cost Analytics Dashboard

A data-driven Streamlit application that analyzes and predicts medical insurance costs based on lifestyle, health, and demographic factors.

## Project Overview

This project combines exploratory data analysis (EDA) with predictive modeling to:

- Identify cost drivers in medical insurance (smoking, age, chronic conditions)
- Visualize spending patterns across demographics and regions
- Predict individual medical costs using a regression model
- Support data-informed pricing and prevention strategies

Key Finding: Top 20% of customers drive approximately 80% of total medical spending (Pareto principle)

---

## Features

### Interactive Dashboard

- Real-time filtering by age, smoking status, region, BMI, and chronic conditions
- Business KPIs displayed prominently (average spend, cost variability, high-spend percentage)
- 5 analytical tabs with different perspectives on the data

### Data Analysis (Jupyter Notebook)

- Data cleaning and deduplication
- Feature engineering and validation
- Statistical visualizations and trend analysis
- Comprehensive exploratory data analysis with 6+ analytical charts

### Predictive Model

- Log-linear regression trained on 100K+ medical records
- Model inputs: age, BMI, risk score, smoking status, hospitalizations, chronic count, visits, days hospitalized
- R-squared = 0.94 (excellent predictive power)
- Key drivers: risk score, smoking, hospitalizations, chronic conditions

---


Markdown
## Project Structure

Medical-Cost-Analytics-Dashboard/ ├── README.md # Project documentation ├── SETUP.md # Detailed setup guide ├── requirements.txt # Python dependencies ├── data/ │ └── medical_insurance_cleaned.csv # 100K+ records, 65 features ├── notebooks/ │ └── analysis.ipynb # Full EDA and exploratory analysis ├── src/ │ └── app.py # Streamlit interactive dashboard ├── reports/ │ └── analysis_results.pdf # Statistical findings summary └── .gitignore # Git ignore rules
