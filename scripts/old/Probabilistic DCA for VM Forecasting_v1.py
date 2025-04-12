import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from my_pandas_extensions.dca_functions import *

"""
Probabilistic DCA for Vaca Muerta Oil Production Forecasting
Author: Alexis Ortega - Aug. 2020

This script implements the Multi-Model Probabilistic (MM-P) workflow proposed by Hong et al. (2018) to integrate model uncertainty in probabilistic Decline Curve Analysis (DCA) for unconventional plays (e.g., Vaca Muerta Fm.).

References:
Hong, Aojie, et al. "Integrating Model Uncertainty in Probabilistic Decline Curve Analysis for Unconventional Oil Production Forecasting."
Unconventional Resources Technology Conference, Houston, Texas, 23-25 July 2018.
"""
"""
## Objectives
1. Integrate model uncertainty in probabilistic DCA for Vaca Muerta unconventional play.
2. Use Bayes' law to assess model probabilities and weigh the forecasts.
3. Incorporate intrinsic uncertainty via Monte Carlo simulation and MLE.
4. Consider uncertainty in inverse modeling through multiple history-matching runs.
5. Estimate Multi-Model Probabilistic (MM-P) EUR for selected wells.
"""
"""
# Production Analysis
Analyzing production data to determine decline trends and forecast future production.
"""

####### Global parameters -----------------------

study_well = "AF-6(h)"
number_of_models = 4
model_names = ["arps", "sem", "crm", "lgm"]
n_samples = 1000  # Number of samples in Monte Carlo simulation
kfolds = 10  # Number of folds for cross-validation
train_pct = 0.8  # Percentage of data used for training

####### UPLOAD DATA -----------------------

# Reading well production data and estimated EURs
data_wells_final_df = pd.read_csv("00_data_wrangled/wells_final_Q12020.csv")
data_production_final_df = pd.read_csv("00_data_wrangled/AF-6h_daily_prod.csv")
#eur_estimated_df = pd.read_csv("Data/eur_estimados.csv")

####### DATA PRE-PROCESSING -----------------------

# Selecting well data for analysis
data_tbl = data_production_final_df[['cum_eff_prod_day', 'oil_month_bpd', 'cum_oil_bbl']]
well_tbl = data_wells_final_df.query("well_name == @study_well")

# Filter decline data (only from maximum rate)
max_prod = round(data_tbl['oil_month_bpd'].max())  # max production rate (start of decline)
max_prod_i = data_tbl['oil_month_bpd'].idxmax()
max_prod_day = data_tbl.loc[max_prod_i, 'cum_eff_prod_day']  # production time at which max rate is reached
max_rate_cum_oil = data_tbl.loc[max_prod_i, 'cum_oil_bbl']  # cum oil prior to reaching max rate

# Determine last production date and relevant statistics
last_day = data_tbl['cum_eff_prod_day'].max()  # last production date for match (end of decline)
last_day_i = data_tbl['cum_eff_prod_day'].idxmax()
last_prod = round(data_tbl.loc[last_day_i, 'oil_month_bpd'])
last_day_cum_oil = round(data_tbl.loc[last_day_i, 'cum_oil_bbl'], 1)

# Determine training set boundary
x_train_i = round((last_day_i - max_prod_i) * train_pct)  # percentage of data to match/train
x_train = round(data_tbl.loc[max_prod_i + x_train_i, 'cum_eff_prod_day'])
train_prod = round(data_tbl.loc[max_prod_i + x_train_i, 'oil_month_bpd'])
train_cum_oil = round(data_tbl.loc[max_prod_i + x_train_i, 'cum_oil_bbl'], 1)

# Filter dataset for decline analysis
data_tbl = data_tbl[data_tbl['cum_eff_prod_day'] >= max_prod_day]
models_df = pd.DataFrame({'x': data_tbl['cum_eff_prod_day'], 'y': data_tbl['oil_month_bpd']})

# Visualizing the well production data
# Apply LOWESS smoothing
frac_value = 0.20  # Try adjusting this value if needed
lowess_smoothed = lowess(models_df['y'], models_df['x'], frac=frac_value)

# # Ensure LOWESS returns sorted data
lowess_x_sorted = lowess_smoothed[:, 0]
lowess_y_sorted = lowess_smoothed[:, 1]

# Create the plot
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-whitegrid')

sns.scatterplot(x=models_df['x'], y=models_df['y'], label='Production Data', alpha=0.6, color='black')
plt.plot(lowess_x_sorted, lowess_y_sorted, color='blue', linewidth=2, label=f'LOWESS Smoother (frac={frac_value})')

# Formatting
plt.title(f"Oil Production Rate {study_well}", fontsize=14, color='#08306B', weight='bold')
plt.suptitle("Local Polynomial Regression Fitting (Loess Method)", fontsize=12, color='#08306B')
plt.xlabel("Days", fontsize=12, color='#08306B')
plt.ylabel("Oil Rate, bbl/day", fontsize=12, color='#08306B')
plt.legend()

# Format y-axis with commas
def comma_formatter(x, pos):
    return f'{x:,.0f}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_formatter))
plt.gca().xaxis.set_major_formatter(FuncFormatter(comma_formatter))

# Add caption
caption_text = f"Data for analysis starts at {max_prod_day} days, when the well reaches a maximum production of {max_prod} bbl/d.\n"
caption_text += f"Total days in production: {last_day} days, Total Cum. Production: {last_day_cum_oil} bbl"
plt.figtext(0.5, -0.1, caption_text, wrap=True, horizontalalignment='center', fontsize=10, color='#08306B')

plt.show()

"""
# Find Optimum Half-Window Width with Cross Validation
Using cross-validation to determine the best half-window width for smoothing residuals.
"""

best_half_window, best_span, models_df = optimum_half_window(
    w_min=3,          # Minimum half-window size
    k_folds=kfolds,       # Number of folds for cross-validation
    dataframe=models_df,  # Input dataframe with x (days) and y (oil rate)
    plotting="Y"      # Set to "Y" to visualize the result
)

# Printing the results
print(f"Best Half-Window: {best_half_window}")
print(f"Best Span/Frac: {best_span}")

"""
# Estimate Standard Deviation of Measured Rate
Using a moving-window approach to estimate the standard deviation of measurement errors.
"""
# half window to be int
best_half_window = int(max(1, round(best_half_window)))

# Compute rolling standard deviation and variance
models_df = sigma_k_calc(dataframe=models_df, half_window_width=best_half_window, plotting=True)

# Display the updated dataframe
models_df

"""
# Sample N Sorted Data Sets from Original Data
Generating multiple samples for Monte Carlo simulation.
"""

# Ensure the dataframe has the necessary columns
# if 'roll_sd' not in models_df.columns:
#     raise ValueError("models_df must contain 'x', 'y', and 'roll_sd' columns before calling sample_sorted_datasets.")

# Call the function with the default sample size of 1000
sample_sorted_df, sample_sorted_stats_df = sample_sorted_datasets_inter(models_df, sample_size=n_samples)

# Display the first few rows of the sampled datasets
sample_sorted_df

# Plotting sorted data
plt.figure(figsize=(10, 6))
plt.scatter(sample_sorted_stats_df['x'], sample_sorted_stats_df['y'], color='red', alpha=0.5, label='Original Data')
plt.plot(sample_sorted_stats_df['x'], sample_sorted_stats_df['sample_mean'], color='black', label='Mean Trend')
plt.plot(sample_sorted_stats_df['x'], sample_sorted_stats_df['sample_p10'], color='red', linestyle='dashed', label='P10')
plt.plot(sample_sorted_stats_df['x'], sample_sorted_stats_df['sample_p90'], color='red', linestyle='dashed', label='P90')

# Plot individual sampled points
sample_indices = [1, 10, 200, 300, 400, 500, 600, 700, 800, 990, 1000]
colors = ['blue'] * 10 + ['black']
for idx, color in zip(sample_indices, colors):
    if f'sample_{idx}' in sample_sorted_df.columns:
        plt.scatter(sample_sorted_df['x'], sample_sorted_df[f'sample_{idx}'], color=color, alpha=0.5, s=10)

plt.title("Sorted Sampled Data")
plt.xlabel("Days")
plt.ylabel("Oil Rate, bbl/day")
plt.legend()
plt.grid(True)
plt.show()

"""
# Match Models to each Sorted Sample Data Set (MM-P Approach)
Fitting multiple decline-curve models (Arps, SEM, CRM, LGM) to sampled data.
"""

# FUNCTION TO FIT DCA MODEL

# ARPS MODEL FUNCTIONS
# ARPS EXPONENTIAL

# ARPS Model
# Arps Hyperbolic to Exponential Model Predictions
        # Hyperbolic decline curve equation:
        #  Arguments:
        #    t: Time since the well started producing
        #    qi: Initial production rate
        #    b: Hyperbolic decline constant
        #    Di: Nominal decline rate (constant)
        
# Function to convert nominal decline rates from years to days

# Initializing parameters and calling function
lower_bounds = [
    0,  # qi > 0
    as_nominal(0.10, from_period="year", to_period="day"),  # Di = 0.10 / [time] effective
    0,  # b > 0
    as_nominal(0.10, from_period="year", to_period="day")   # Df = 0.10 / [time] effective
]

upper_bounds = [
    10000,  # qi < qmax * 5
    as_nominal(0.99995, from_period="year", to_period="day"),  # Di = 0.99995 / [time] effective
    2.0,  # b <= 1.0
    as_nominal(0.10, from_period="year", to_period="day")  # Df = 0.10 / [time] effective
]

# Call the function for ARPS model
arps_samples_preds_results = match_model_samples_preds(
    model="arps", # Model type: "arps", "sem", "crm", or "lgm"
    dataframe_sample=sample_sorted_df,  # Pandas DataFrame with well production data
    samplesize=n_samples,  # Number of Monte Carlo samples
    lower_par=lower_bounds, # List of lower parameter bounds
    upper_par=upper_bounds, # List of upper parameter bounds
    x_train_i=x_train_i, # Index marking the end of the training dataset
    fcst_end=5400    # (Optional) Forecast end time
)

print("Checking output from match_model_samples_preds:")
print(arps_samples_preds_results)  # Print the entire returned object

# Extract individual results
arps_samples_preds_df = arps_samples_preds_results[0]  # Predictions
arps_samples_wse_df = arps_samples_preds_results[1]  # Weighted SSE
arps_samples_fcst_end_df = arps_samples_preds_results[2]  # Forecast to fcst_end
arps_samples_fcst_df = arps_samples_preds_results[3]  # Forecast results
arps_samples_par_df = arps_samples_preds_results[4]  # Model fit parameters

# Compute Prediction Statistics
# Dynamically select sample columns from sample_0 to sample_999
sample_columns_preds = [col for col in arps_samples_preds_df.columns if col.startswith("sample_")]
sample_columns_fcst = [col for col in arps_samples_fcst_df.columns if col.startswith("sample_")]
sample_columns_fcst_end = [col for col in arps_samples_fcst_end_df.columns if col.startswith("sample_")]

# Call function for each dataset
arps_samples_preds_stats_df = compute_statistics(arps_samples_preds_df, sample_columns_preds)
arps_samples_fcst_stats_df = compute_statistics(arps_samples_fcst_df, sample_columns_fcst)
arps_samples_fcst_end_stats_df = compute_statistics(arps_samples_fcst_end_df, sample_columns_fcst_end)


# # Calculate summarized loss function MLE for each sample 
# arps_samples_loss_mle_df = arps_samples_wse_df.iloc[:, 3:].sum(axis=1)

# # Calculate summarized forecast for each sample
# arps_samples_cum_fcst_df = arps_samples_fcst_end_df.iloc[:, 1:].sum(axis=1)


# Calculate summarized loss function MLE for each sample
# Summing over all numeric columns except 'x', 'y', and 'sigma2'
arps_samples_loss_mle_df = arps_samples_wse_df.drop(columns=["x", "y", "sigma2"]).sum(axis=0, skipna=True).to_frame().T

# Calculate summarized cumulative forecast for each sample
# Summing over all columns that start with "sample_"
sample_columns_fcst_end = [col for col in arps_samples_fcst_end_df.columns if col.startswith("sample_")]
arps_samples_cum_fcst_df = arps_samples_fcst_end_df[sample_columns_fcst_end].sum(axis=0, skipna=True).to_frame().T

# FUNCTION TO PLOT PREDICTIONS

# Call the function to plot the Arps Model Predictions
plot_preds(
    dataframe=arps_samples_preds_df,       # DataFrame with x and y columns
    dataframe_stats=arps_samples_preds_stats_df,  # DataFrame containing prediction statistics
    model="arps",  # Model name as a string
    x_train_i=x_train_i  # Index for training split
)

"""
# Calculate Model Probability
Using Bayesian inference to determine the likelihood of each model given the data.
"""

# Placeholder for model probability calculation
model_probabilities = np.array([0.4, 0.3, 0.2, 0.1])

"""
# Hindcast Test
Comparing forecasts with actual historical production data.
"""

# Placeholder for forecast comparison
hindcast_df = models_df.copy()
hindcast_df['forecast'] = hindcast_df['oil_month_bpd'] * (1 - 0.01 * np.arange(len(hindcast_df)))

"""
# Cumulative Forecast Production for each Model (15-Year)
Forecasting long-term production using multiple models.
"""

# Placeholder for cumulative forecast
cumulative_forecast_df = models_df.copy()
cumulative_forecast_df['forecast_15yr'] = cumulative_forecast_df['oil_month_bpd'] * 0.8

"""
# Calculate Model and Combined Forecasts
Computing weighted forecasts based on model probabilities.
"""

# Placeholder for weighted forecast
weighted_forecast = np.sum(model_probabilities * np.array([1000, 900, 800, 700]))
print(f"Combined MMP Forecast: {weighted_forecast} bbl")

"""
# Calculate Model and Combined EUR
Estimating the EUR for each model and the combined probabilistic estimate.
"""

# Placeholder for EUR estimation
weighted_total_eur = weighted_forecast * 1.1  # Example adjustment
print(f"Combined MMP EUR: {weighted_total_eur} bbl")

"""
# Conclusions
Summary of findings and final probabilistic EUR estimate.
"""

study_well_total_eur_fcst = weighted_total_eur
study_well_eur = well_tbl['eur_oil_mbbl'].values[0] if not well_tbl.empty else np.nan
print(f"Final EUR Estimate for {study_well}: {study_well_total_eur_fcst} bbl (GiGa Estimate: {study_well_eur} Mbbl)")


