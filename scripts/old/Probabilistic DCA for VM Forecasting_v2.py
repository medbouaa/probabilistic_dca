import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from my_pandas_extensions.dca_functions_v2 import *




# Set seed for reproducibility
np.random.seed(42)

# Load production data
prod_data_path = "00_data_wrangled/AF-6h_daily_prod.csv"
prod_data = pd.read_csv(prod_data_path)

# Load well data
wells_data_path = "00_data_wrangled/wells_final_Q12020.csv"
wells_df = pd.read_csv(wells_data_path)

# Study well integration
study_well = "AF-6(h)"
well_info = wells_df[wells_df["well_name"] == study_well] if study_well in wells_df["well_name"].values else None

# Filter production data for the selected well
data_tbl = prod_data.copy()
data_tbl = data_tbl[['cum_eff_prod_day', 'oil_month_bpd', 'cum_oil_bbl']]
data_tbl.columns = ['x', 'y', 'cum_oil']

# Identify the starting point for decline analysis (maximum production rate)
max_prod_idx = data_tbl['y'].idxmax()
max_prod_day = data_tbl['x'].iloc[max_prod_idx]

# Filter data from max production day onwards
data_tbl = data_tbl[data_tbl['x'] >= max_prod_day]

# Apply Loess smoothing and calculate residuals
best_span, models_df = optimum_half_window(data_tbl)

# Compute rolling standard deviation (sigma_k)
models_df = sigma_k_calc(models_df, half_window_width=int(best_span * len(models_df['x']) / 2))

# Extend last non-NaN values to fill boundaries
models_df['roll_sd'].fillna(method='bfill', inplace=True)
models_df['roll_sd'].fillna(method='ffill', inplace=True)
models_df['roll_sigma2'] = models_df['roll_sd'] ** 2

# Generate Monte Carlo samples with row-wise sorting
sample_sorted_tbl = sample_sorted_datasets(models_df)

# Compute Monte Carlo statistics separately
sample_std = sample_sorted_tbl.iloc[:, 3:].std(axis=1)
sample_p10 = sample_sorted_tbl.iloc[:, 3:].apply(lambda x: np.percentile(x, 10), axis=1)
sample_mean = sample_sorted_tbl.iloc[:, 3:].mean(axis=1)
sample_p50 = sample_sorted_tbl.iloc[:, 3:].apply(lambda x: np.percentile(x, 50), axis=1)
sample_p90 = sample_sorted_tbl.iloc[:, 3:].apply(lambda x: np.percentile(x, 90), axis=1)

# Create DataFrame with computed statistics
sample_sorted_stats_tbl = pd.DataFrame({
    'x': sample_sorted_tbl['x'],
    'y': sample_sorted_tbl['y'],
    'sample_std': sample_std,
    'sample_p10': sample_p10,
    'sample_mean': sample_mean,
    'sample_p50': sample_p50,
    'sample_p90': sample_p90
})

# Plot sorted sample data
plt.figure(figsize=(10, 6))
plt.scatter(sample_sorted_stats_tbl['x'], sample_sorted_stats_tbl['y'], color='red', alpha=0.5, label='Original Data')
plt.plot(sample_sorted_stats_tbl['x'], sample_sorted_stats_tbl['sample_mean'], color='black', label='Mean Trend')
plt.plot(sample_sorted_stats_tbl['x'], sample_sorted_stats_tbl['sample_p10'], color='red', linestyle='dashed', label='P10')
plt.plot(sample_sorted_stats_tbl['x'], sample_sorted_stats_tbl['sample_p90'], color='red', linestyle='dashed', label='P90')

# Plot individual sampled points
sample_indices = [1, 10, 200, 300, 400, 500, 600, 700, 800, 990, 1000]
colors = ['blue'] * 10 + ['black']
for idx, color in zip(sample_indices, colors):
    if f'sample_{idx}' in sample_sorted_tbl.columns:
        plt.scatter(sample_sorted_tbl['x'], sample_sorted_tbl[f'sample_{idx}'], color=color, alpha=0.5, s=10)

plt.title("Sorted Sampled Data")
plt.xlabel("Days")
plt.ylabel("Oil Rate, bbl/day")
plt.legend()
plt.grid(True)
plt.show()









































