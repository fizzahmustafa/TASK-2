# Task 2: Predict Future Stock Prices (Short-Term)

## Objective
Use historical stock market data to train a machine learning model that predicts the **next day's closing price** based on features like Open, High, Low, and Volume.

---

## Dataset
**Yahoo Finance** — Retrieved using the `yfinance` Python library. This provides real historical OHLCV (Open, High, Low, Close, Volume) data for any publicly listed stock.

| Feature | Description |
|---|---|
| `Open` | Opening price of the stock for the day |
| `High` | Highest price reached during the day |
| `Low` | Lowest price reached during the day |
| `Volume` | Number of shares traded |
| `Close` | **Target** — Closing price (what we predict) |

---

## Requirements

Install the required libraries before running:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

---

## Project Structure

```
task2_stock_prediction/
│
├── stock_predictor.py      # Main script
├── README.md               # This file
└── plots/
    ├── actual_vs_predicted.png
    └── feature_importance.png  # (Random Forest only)
```

---

## Step-by-Step Instructions

### Step 1 — Load Historical Stock Data

```python
import yfinance as yf
import pandas as pd

# Select your stock ticker
TICKER = "AAPL"   # Try: TSLA, MSFT, GOOGL, AMZN

# Download 2 years of data
df = yf.download(TICKER, start="2022-01-01", end="2024-01-01")

print(df.shape)
print(df.head())
print(df.isnull().sum())
```

> **Note:** Replace `"AAPL"` with any valid stock ticker symbol.

### Step 2 — Feature Engineering

We predict the **next day's Close** using today's OHLV data.

```python
# Shift Close by -1 to create the target (next day's Close)
df['Target'] = df['Close'].shift(-1)

# Drop rows with NaN (last row has no "next day")
df.dropna(inplace=True)

# Define features and target
features = ['Open', 'High', 'Low', 'Volume']
X = df[features]
y = df['Target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Step 3 — Train/Test Split

```python
from sklearn.model_selection import train_test_split

# 80% training, 20% testing — no shuffling (time series data!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

> **Important:** Always use `shuffle=False` for time series data to avoid data leakage.

### Step 4A — Train a Linear Regression Model

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression Results ===")
print(f"MAE:  ${mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred_lr):.4f}")
```

### Step 4B — Train a Random Forest Model

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest Results ===")
print(f"MAE:  ${mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred_rf):.4f}")
```

### Step 5 — Plot Actual vs Predicted Prices

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

# Use the test set's date index for the x-axis
test_dates = df.index[-len(y_test):]

plt.plot(test_dates, y_test.values, label='Actual Close', color='steelblue', linewidth=1.5)
plt.plot(test_dates, y_pred_lr, label='Linear Regression', color='orange', linestyle='--', linewidth=1.5)
plt.plot(test_dates, y_pred_rf, label='Random Forest', color='green', linestyle='--', linewidth=1.5)

plt.title(f'{TICKER} — Actual vs Predicted Closing Prices', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150)
plt.show()
```

### Step 6 — Feature Importance (Random Forest Only)

```python
import seaborn as sns

importances = pd.Series(rf_model.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Random Forest — Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()
```

---

## Key Observations

- Stock price prediction using only OHLV features achieves **high R²** because today's Close is highly correlated with tomorrow's Open and High.
- **Random Forest** typically outperforms Linear Regression for capturing non-linear patterns.
- This is a **simplified model** — real-world trading uses sentiment analysis, technical indicators (RSI, MACD), and deep learning (LSTMs).
- Always evaluate with **time-ordered splits** — never shuffle stock data before splitting.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average dollar error per prediction |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **R²** | Coefficient of determination — 1.0 = perfect fit |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `yfinance` | Downloading historical stock data |
| `pandas` | Data manipulation |
| `scikit-learn` | Model training and evaluation |
| `matplotlib` | Plotting actual vs predicted prices |
| `seaborn` | Feature importance visualization |

---

## How to Run

```bash
python stock_predictor.py
```

Change the `TICKER` variable at the top of the script to switch stocks.
