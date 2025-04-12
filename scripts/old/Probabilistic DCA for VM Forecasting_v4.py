import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from my_pandas_extensions.dca_functions_v4 import *

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
best_half_window, best_span, models_df = crossval_loess(
    dataframe=models_df,
    k_folds=kfolds,
    #w_min=3,          
    #dataframe=models_df,
    plotting="True"      
)

# Printing the results
print(f"Best Half-Window: {best_half_window}")
print(f"Best Span/Frac: {best_span}")
models_df.head()

"""
# Estimate Standard Deviation of Measured Rate
Using a moving-window approach to estimate the standard deviation of measurement errors.
"""
# Compute rolling standard deviation and variance
models_df = rolling_std(dataframe=models_df, half_window=best_half_window, plotting=True)

# Printing the results
models_df.head()

"""
# Sample N Sorted Data Sets from Original Data
Generating multiple samples for Monte Carlo simulation.
"""
# Call the function with the default sample size of 1000
sample_sorted_df, sample_sorted_stats_df = sample_sorted_datasets(models_df, n_samples=n_samples)

# Printing the results
sample_sorted_df.head()
sample_sorted_stats_df.head()

"""
# Match Models to each Sorted Sample Data Set (MM-P Approach)
Fitting multiple decline-curve models (Arps, SEM, CRM, LGM) to sampled data.
"""
# FUNCTION TO FIT DCA MODEL

# ARPS EXPONENTIAL
# Arps Hyperbolic to Exponential Model Predictions
        # Hyperbolic decline curve equation:
        #  Arguments:
        #    t: Time since the well started producing
        #    qi: Initial production rate
        #    b: Hyperbolic decline constant
        #    Di: Nominal decline rate (constant)


# 1) Figure out the row index for your training cutoff
#    Let’s say 'train_cutoff' is the row index in the filtered dataset 
#    that corresponds to x_train days:
train_cutoff =  x_train_i

# 2) Separate training vs test
train_df = sample_sorted_df.iloc[:train_cutoff+1].copy()  # from row 0 to train_cutoff
test_df  = sample_sorted_df.iloc[train_cutoff+1:].copy()  # the remainder

# 3) Fit the Arps model on the training portion
arps_fit_results = fit_arps_for_samples(train_df)

print(arps_fit_results['params'].head()) 
# => first few sets of [qi, Di, b, Df] for sample_1, sample_2, etc.

print(arps_fit_results['sse'][:10]) 
# => SSE array for the first 10 columns

pred_for_sample1 = arps_fit_results['predictions']['sample_1']
# => array of fitted rates vs. time for that sample’s data

# 2) Gather predictions into a 2D array (time steps x samples)
pred_matrix = gather_arps_predictions(arps_fit_results, train_df)

# 3) Compute stats: p10, mean, p90, etc.
pred_stats = compute_pred_stats(pred_matrix)

# 4) Plot
plot_arps_predictions(train_df, pred_stats)

############################################################
## Forecast the Testing Period Using Those Parameters
# Extract the best-fit parameters for each sample from arps_fit_results['params'].
# Evaluate the model at each time in test_df['x'].
# Store these predictions in a new 2D array or dictionary.
    
test_forecasts = forecast_arps_for_test(arps_fit_results, test_df)
test_pred_stats = compute_pred_stats_test(test_forecasts)

plot_arps_test(test_df, test_pred_stats)

###############################################################
## Define a Forecast Function for “Future Days”
# We can reuse much of the logic from the “test forecast” step. The main difference: instead of test_df['x'], we’ll generate a brand-new array of times from last_day + 1 to last_day + 5400.

# 1) Suppose from your entire dataset (train + test or just the max of the dataset), 
#    you find last_day = the final day with production data
last_day_in_dataset = sample_sorted_df['x'].max()

# 2) Use your training-fit Arps parameters
arps_fit_results  = fit_arps_for_samples(sample_sorted_df)
# or whichever method you used

df_params = arps_fit_results['params']
p50_params = df_params.median()
print(p50_params)

# 3) Forecast next 15 years = 5400 days
t_fore, forecast_matrix = forecast_arps_for_future(arps_fit_results,
                                                   last_time=last_day_in_dataset,
                                                   days_ahead=5400)

# 4) Summarize
future_pred_stats = compute_pred_stats_future(forecast_matrix)

# 5) Plot if desired

plot_future_forecast(t_fore, future_pred_stats)

###################################################################
### Integrate Rates for “Cumulative” Forecast

# If you also need cumulative production over that future period, do something like:

# Then you can do:
future_cums = compute_cumulative(forecast_matrix, dt=1.0)  # shape (#samples,)

# And compute p10/p50/p90 across that distribution:
cum_p10 = np.nanpercentile(future_cums, 10)
cum_p50 = np.nanpercentile(future_cums, 50)
cum_mean = np.nanmean(future_cums)
cum_p90 = np.nanpercentile(future_cums, 90)

p10_eur = last_day_cum_oil + cum_p10
p50_eur = last_day_cum_oil + cum_p50
mean_eur = last_day_cum_oil + cum_mean
p90_eur = last_day_cum_oil + cum_p90


# dt = 1.0  # 1 day
# cum_forecast = np.cumsum(forecast_matrix, axis=0) * dt  # shape (N_forecast, M)
# cum_stats = {
#     'p10':  np.nanpercentile(cum_forecast[-1, :], 10),
#     'p50':  np.nanpercentile(cum_forecast[-1, :], 50),
#     'p90':  np.nanpercentile(cum_forecast[-1, :], 90),
#     'mean': np.nanmean(cum_forecast[-1, :]),
# }

# total_cum = last_day_cum_oil + cum_forecast
# p10_eur  = np.nanpercentile(total_cum[-1, :], 10)
# p50_eur  = np.nanpercentile(total_cum[-1, :], 50)
# p90_eur  = np.nanpercentile(total_cum[-1, :], 90)
# mean_eur = np.nanmean(total_cum[-1, :])