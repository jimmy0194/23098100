# 23098100.py
# Fundamentals of Data Science - SOHAIL SHAIK - Student ID: 23098100

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Step 1: Load and prepare dataset ===
df_sohail = pd.read_csv('sales0.csv')
df_sohail['Date'] = pd.to_datetime(df_sohail['Date'], errors='coerce')
df_sohail.dropna(subset=['Date'], inplace=True)

# Extract time features
df_sohail['Year'] = df_sohail['Date'].dt.year
df_sohail['Month'] = df_sohail['Date'].dt.month
df_sohail['DayOfYear'] = df_sohail['Date'].dt.dayofyear

# === Step 2: Compute total items sold and revenue ===
df_sohail['ItemsSoldTotal'] = (
    df_sohail['NumberGroceryShop'] + df_sohail['NumberGroceryOnline'] +
    df_sohail['NumberNongroceryShop'] + df_sohail['NumberNongroceryOnline']
)
df_sohail['RevenueGrocery'] = (
    df_sohail['NumberGroceryShop'] * df_sohail['PriceGroceryShop'] +
    df_sohail['NumberGroceryOnline'] * df_sohail['PriceGroceryOnline']
)
df_sohail['RevenueNongrocery'] = (
    df_sohail['NumberNongroceryShop'] * df_sohail['PriceNongroceryShop'] +
    df_sohail['NumberNongroceryOnline'] * df_sohail['PriceNongroceryOnline']
)
df_sohail['RevenueTotal'] = df_sohail['RevenueGrocery'] + df_sohail['RevenueNongrocery']
df_sohail['UnitPriceAverage'] = df_sohail['RevenueTotal'] / df_sohail['ItemsSoldTotal']

# === Step 3: Monthly Average Items Sold + Fourier ===
monthly_avg_sales = df_sohail.groupby('Month')['ItemsSoldTotal'].mean()

df_year_2022 = df_sohail[df_sohail['Year'] == 2022]
sales_2022 = df_year_2022['ItemsSoldTotal'].values
days_2022 = np.arange(1, len(sales_2022) + 1)

N = len(sales_2022)
a0 = np.mean(sales_2022)
fourier_curve = np.full(N, a0 / 2)

for term in range(1, 9):
    an = 2 / N * np.sum(sales_2022 * np.cos(2 * np.pi * term * days_2022 / N))
    bn = 2 / N * np.sum(sales_2022 * np.sin(2 * np.pi * term * days_2022 / N))
    fourier_curve += an * np.cos(2 * np.pi * term * days_2022 / N) + bn * np.sin(2 * np.pi * term * days_2022 / N)

plt.figure(figsize=(14, 6))
plt.bar(monthly_avg_sales.index, monthly_avg_sales.values, color='teal', alpha=0.7, label='Monthly Avg Items Sold')
plt.plot(np.linspace(1, 12, N), fourier_curve, color='darkorange', linewidth=2.5, label='Fourier Approx (8 terms)')
plt.title('Figure 1 - Monthly Sales & Fourier Fit (ID: 23098100)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Average Daily Items Sold')
plt.xticks(ticks=np.arange(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(loc='upper left')
plt.text(11, max(monthly_avg_sales) + 100, 'Student ID: 23098100', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figure1_23098100.png')
plt.close()

# === Step 4: Scatter Plot + Regression ===
plt.figure(figsize=(10, 6))
plt.scatter(df_sohail['ItemsSoldTotal'], df_sohail['UnitPriceAverage'], c='slateblue', alpha=0.6, label='Daily Records')
plt.title('Figure 2 - Price vs Items Sold with Linear Fit (ID: 23098100)', fontsize=14)
plt.xlabel('Total Items Sold')
plt.ylabel('Average Unit Price')

x_vals = df_sohail['ItemsSoldTotal'].values.reshape(-1, 1)
y_vals = df_sohail['UnitPriceAverage'].values
model = LinearRegression().fit(x_vals, y_vals)
y_pred_line = model.predict(x_vals)
plt.plot(df_sohail['ItemsSoldTotal'], y_pred_line, color='crimson', linewidth=2, label='Linear Regression')

plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figure2_23098100.png')
plt.close()

# === Step 5: Calculate X & Y as Revenue Percentages ===
summer_months = [6, 7, 8]
autumn_months = [9, 10, 11]

total_revenue = df_sohail['RevenueTotal'].sum()
summer_revenue = df_sohail[df_sohail['Month'].isin(summer_months)]['RevenueTotal'].sum()
autumn_revenue = df_sohail[df_sohail['Month'].isin(autumn_months)]['RevenueTotal'].sum()

X_percent = round((summer_revenue / total_revenue) * 100, 2)
Y_percent = round((autumn_revenue / total_revenue) * 100, 2)

# === Step 6: Print Results ===
print("Student ID: 23098100")
print(f"X (Summer Revenue %): {X_percent}%")
print(f"Y (Autumn Revenue %): {Y_percent}%")