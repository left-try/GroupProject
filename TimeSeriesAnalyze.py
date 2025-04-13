# ===============================
# Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from dateutil.relativedelta import relativedelta
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import ttest_rel, wilcoxon

df = pd.read_csv('FillnessData.csv')

# Inspect the first few rows and column names.
print("Original DataFrame columns:")


df.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)
df = df.drop(index=0).reset_index(drop=True)
df = df.drop(index=4).reset_index(drop=True)
df = df.drop(index=4).reset_index(drop=True)
df = df.map(lambda x: float(x.replace('%', '').strip()) / 100 if isinstance(x, str) and '%' in x else x)

print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
timestamps = pd.to_datetime(df.columns, errors='coerce')

ts_values = df.astype(float).mean(axis=0)
ts_df = pd.DataFrame({'Date': timestamps, 'Fill_Level': ts_values.values})
ts_df = ts_df.sort_values('Date').reset_index(drop=True)

plt.figure(figsize=(12, 5))
plt.plot(ts_df['Date'], ts_df['Fill_Level'], label='Average Fill Level')
plt.xlabel('Date')
plt.ylabel('Fill Level (fraction)')
plt.title('Trash Bin Fill Level Over Time')
plt.legend()
plt.show()

cutoff_date = pd.to_datetime("2024-06-01")
ts_pre = ts_df[ts_df['Date'] < cutoff_date].copy()
ts_post = ts_df[ts_df['Date'] >= cutoff_date].copy()

print("Number of pre-change observations:", len(ts_pre))
print("Number of post-change observations:", len(ts_post))

def create_lag_features(df, lag_days=3, col="Fill_Level"):
    df_copy = df.copy()
    for lag in range(1, lag_days + 1):
        df_copy[f'lag_{lag}'] = df_copy[col].shift(lag)
    return df_copy.dropna().reset_index(drop=True)

lag_days = 1
ts_pre_lag = create_lag_features(ts_pre, lag_days=lag_days, col="Fill_Level")
ts_post_lag = create_lag_features(ts_post, lag_days=lag_days, col="Fill_Level")

feature_cols = [f'lag_{i}' for i in range(1, lag_days+1)]
X_pre = ts_pre_lag[feature_cols].values
y_pre = ts_pre_lag["Fill_Level"].values

split_idx = int(0.8 * len(ts_pre_lag))
X_train, X_val = X_pre[:split_idx], X_pre[split_idx:]
y_train, y_val = y_pre[:split_idx], y_pre[split_idx:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.05, 0.1],
    'gamma': ['scale', 'auto']
}
svr_model = SVR(kernel='rbf')
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train_scaled)
best_svr = grid_search.best_estimator_
print("Best SVR parameters (pre-change):", grid_search.best_params_)

y_val_pred_scaled = best_svr.predict(X_val_scaled)
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
print("Pre-change Validation MAE:", mean_absolute_error(y_val, y_val_pred))
print("Pre-change Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))

X_post = ts_post_lag[feature_cols].values
y_post = ts_post_lag["Fill_Level"].values
X_post_scaled = scaler_X.transform(X_post)

y_post_pred_scaled = best_svr.predict(X_post_scaled)
y_post_pred = scaler_y.inverse_transform(y_post_pred_scaled.reshape(-1, 1)).ravel()

mae_post = mean_absolute_error(y_post, y_post_pred)
rmse_post = np.sqrt(mean_squared_error(y_post, y_post_pred))
print("Post-change MAE:", mae_post)
print("Post-change RMSE:", rmse_post)

diff = y_post - y_post_pred
mean_diff = np.mean(diff)

print("\nMean difference (Actual - Predicted) in post-change period: {:.4f}".format(mean_diff))
if mean_diff > 0:
    print("Interpretation: The actual fill levels are higher than expected. This suggests that, with fewer bins, waste is concentrating more in each bin.")
elif mean_diff < 0:
    print("Interpretation: The actual fill levels are lower than expected. This suggests that the bin reduction might have been beneficial.")
else:
    print("Interpretation: The actual fill levels closely match the expected values.")

t_stat, p_val = ttest_rel(y_post, y_post_pred)
w_stat, p_val_w = wilcoxon(y_post, y_post_pred)
print("\nPaired t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_val))
print("Wilcoxon test: statistic = {:.4f}, p-value = {:.4f}".format(w_stat, p_val_w))

plt.figure(figsize=(12, 6))
plt.plot(ts_post_lag['Date'], y_post, label='Actual Fill Level (Post-change)', marker='o')
plt.plot(ts_post_lag['Date'], y_post_pred, label='Predicted (Pre-change SVR)', marker='x')
plt.xlabel('Date')
plt.ylabel('Fill Level (fraction)')
plt.title('Actual vs. Predicted Fill Levels After Bin Reduction')
plt.legend()
plt.show()
