import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path # Using pathlib for better path handling

# Imports needed for the integrated functions
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm # Added for fit_model_for_samples


# ---- Assume other local imports are still available ----
# These are imports for functions NOT provided by the user yet
try:
    # from src.probabilistic_dca.my_dca_models.data_processing import (...) # Removed
    # from src.probabilistic_dca.my_dca_models.fitting import (...) # Removed
    from src.probabilistic_dca.my_dca_models.utilities import (
        load_all_model_train_results, calc_model_probabilities, compute_marginal_model_probs, rank_models_by_probability, combine_forecasts_across_models
    )
    from src.probabilistic_dca.my_dca_models.plotting import (
        plot_model_predictions, plot_hindcast, plot_future_forecast, plot_post_prob_models, boxplot_eur
    )
    from src.probabilistic_dca.my_dca_models.arps_model import ArpsModel      # Arps: piecewise hyperbolic->exponential
    from src.probabilistic_dca.my_dca_models.sem_model import SEMModel        # SEM: Stretched Exponential Model
    from src.probabilistic_dca.my_dca_models.crm_model import CRMModel        # CRM: Capacitance-Resistance Model
    from src.probabilistic_dca.my_dca_models.lgm_model import LGMModel        # LGM: Logistic Growth Model
except ImportError as e:
    print(f"Error importing remaining local modules: {e}")
    print("Please ensure the 'src' directory is in the Python path or installed.")
    exit()

# -------------------------------------------------------------------
# INTEGRATED DATA PROCESSING FUNCTIONS (from previous step)
# -------------------------------------------------------------------

######################################################################
# data pre-processing (Integrated)
######################################################################

def data_processing(prod_df=None, train_pct=0.8, plotting=False, frac_value=0.2):
    """
    Processes production data to find decline start, smooth data, and prepare for analysis.
    NOTE: This function now expects prod_df to have columns 'x', 'y', 'z'.
    """
    if prod_df is None:
        raise ValueError("A valid dataframe with 'x', 'y', 'z' columns is required.")

    if not all(col in prod_df.columns for col in ['x', 'y', 'z']):
         raise ValueError("Input dataframe must contain 'x', 'y', and 'z' columns.")

    # Filter decline data (only from maximum rate)
    max_prod = round(prod_df['y'].max())  # max production rate (start of decline)
    max_prod_i = prod_df['y'].idxmax()
    max_prod_day = prod_df.loc[max_prod_i, 'x']  # production time at which max rate is reached
    # max_rate_cum_oil = prod_df.loc[max_prod_i, 'z']  # cum oil prior to reaching max rate (Not used later)

    # Determine last production date and relevant statistics
    last_day = prod_df['x'].max()  # last production date for match (end of decline)
    last_day_i = prod_df['x'].idxmax()
    #last_prod = round(prod_df.loc[last_day_i, 'y']) # Not used later
    last_day_cum_oil = round(prod_df.loc[last_day_i, 'z'], 1)

    # Determine training set boundary (relative to start of decline)
    # Ensure indices are valid before calculating difference
    if max_prod_i is None or last_day_i is None:
         raise ValueError("Could not determine max production index or last day index.")

    # Calculate number of points *after* max production
    decline_points_count = prod_df[prod_df['x'] >= max_prod_day].shape[0]
    if decline_points_count <= 1:
         print("Warning: Very few points after max production. Training split might be small.")
         # Handle edge case: if only 1 point or fewer after max, train index is 0
         x_train_i = 0
    else:
         # Calculate index relative to the start of the decline data subset
         # The index should be based on the count of points in the decline phase
         x_train_i = round((decline_points_count - 1) * train_pct) # -1 because index is 0-based

    # Filter dataset for decline analysis (keep only data from max production onwards)
    decline_df = prod_df[prod_df['x'] >= max_prod_day].copy().reset_index(drop=True)

    if decline_df.empty:
         raise ValueError("No data remaining after filtering for decline phase (x >= max_prod_day).")

    # Adjust x_train_i to be within the bounds of the decline_df
    x_train_i = min(x_train_i, decline_df.shape[0] - 1)
    x_train_i = max(x_train_i, 0) # Ensure it's not negative

    # Create the DataFrame needed for subsequent steps (only x and y)
    models_df = pd.DataFrame({'x': decline_df['x'], 'y': decline_df['y']})

    if plotting:
        # Visualizing the well production data (using decline_df)

        # Apply LOWESS smoothing to the decline data
        if models_df.shape[0] > 1: # Need at least 2 points for lowess
            try:
                # Ensure frac_value is reasonable
                frac_value = max(min(frac_value, 1.0), 2.0/models_df.shape[0]) # Basic bounds check
                lowess_smoothed = lowess(models_df['y'], models_df['x'], frac=frac_value)
                lowess_x_sorted = lowess_smoothed[:, 0]
                lowess_y_sorted = lowess_smoothed[:, 1]
            except Exception as e:
                 print(f"Warning: LOWESS smoothing failed - {e}. Plotting raw data only.")
                 lowess_x_sorted, lowess_y_sorted = None, None
        else:
             print("Warning: Not enough data points for LOWESS smoothing.")
             lowess_x_sorted, lowess_y_sorted = None, None


        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        sns.scatterplot(x=models_df['x'], y=models_df['y'], label='Decline Production Data', alpha=0.6, color='black')
        if lowess_x_sorted is not None:
             plt.plot(lowess_x_sorted, lowess_y_sorted, color='blue', linewidth=2, label=f'LOWESS Smoother (frac={frac_value})')

        # Formatting
        plt.title(f"Oil Production Rate (Decline Phase)", fontsize=14, color='#08306B', weight='bold')
        plt.suptitle("Local Polynomial Regression Fitting (Loess Method)", fontsize=12, color='#08306B')
        plt.xlabel("Days", fontsize=12, color='#08306B')
        plt.ylabel("Oil Rate, bbl/day", fontsize=12, color='#08306B')
        plt.legend()

        # Format y-axis with commas
        def comma_formatter(x_val, pos):
            return f'{x_val:,.0f}'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_formatter))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(comma_formatter))

        # Add caption
        caption_text = f"Data for analysis starts at {max_prod_day} days (Max Rate: {max_prod} bbl/d).\n"
        caption_text += f"Total days in production: {last_day} days, Total Cum. Production at end: {last_day_cum_oil} bbl"
        plt.figtext(0.5, -0.1, caption_text, wrap=True, horizontalalignment='center', fontsize=10, color='#08306B')

        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to prevent caption overlap
        plt.show()

    # Return values based on the full original dataset, but the index relative to the decline start
    # models_df contains only the decline data ('x', 'y')
    return last_day, last_day_cum_oil, x_train_i, models_df


######################################################################
# “optimum_half_window” - Cross Validation for local smoothing (Integrated)
######################################################################

def crossval_loess(dataframe=None, k_folds=10, window_range=None, plotting=False):
    """
    Performs K-Fold cross-validation to find the optimal LOWESS span (frac).
    Adds 'model_residuals' column to the input dataframe based on the best fit.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
        k_folds (int): Number of folds for cross-validation.
        window_range (range, optional): Range of half-window sizes to test. Defaults to range(3, 9).
        plotting (bool): Whether to plot CV results.

    Returns:
        tuple: (best_halfwindow, best_span, dataframe_with_residuals)
    """
    if dataframe is None:
        raise ValueError("A valid dataframe with 'x' and 'y' columns is required.")
    if not all(col in dataframe.columns for col in ['x', 'y']):
         raise ValueError("Input dataframe must contain 'x' and 'y' columns.")
    if dataframe.shape[0] < k_folds:
         print(f"Warning: Number of data points ({dataframe.shape[0]}) is less than k_folds ({k_folds}). Reducing k_folds.")
         k_folds = max(2, dataframe.shape[0]) # Ensure at least 2 folds if possible

    x = dataframe['x'].values
    y = dataframe['y'].values

    if window_range is None:
        # Ensure window range doesn't lead to frac > 1 or too small
        max_w = int((len(x) - 1) / 2) -1 # Max possible half-window
        default_max_w = min(max_w, 8) # Default max is 8 or max possible
        default_min_w = 3
        if default_min_w > default_max_w:
             print(f"Warning: Not enough data points ({len(x)}) for default window range. Using minimal range.")
             window_range = range(max(1, default_max_w), default_max_w + 1) # Try at least one value if possible
        else:
             window_range = range(default_min_w, default_max_w + 1)

    best_span = None
    best_score = float('inf')
    best_w = None

    X = np.array(x)
    Y = np.array(y)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=123)

    # We store (w, span, mean_mse) in a list
    candidate_results = []

    N = len(X)
    if N <= 1:
         print("Warning: Only 1 data point. Cannot perform LOWESS cross-validation.")
         dataframe["model_residuals"] = 0.0 # Assign zero residual
         return window_range[0] if window_range else 3, 1.0, dataframe # Return defaults


    for w in window_range:
        # Calculate span, ensuring it's within valid range (0, 1]
        span = min(1.0, max( (2*w + 1)/N, 1e-6 )) # Avoid span=0

        fold_mses = []
        for train_idx, test_idx in kf.split(X):
            if len(train_idx) < 2: continue # Need at least 2 points to fit lowess

            x_train, y_train = X[train_idx], Y[train_idx]
            x_test, y_test   = X[test_idx], Y[test_idx]

            # Sort training data for lowess
            order = np.argsort(x_train)
            x_sorted = x_train[order]
            y_sorted = y_train[order]

            try:
                # Fit lowess on train data
                fitted = lowess(endog=y_sorted, exog=x_sorted, frac=span, return_sorted=True)

                # Interpolate to predict test points
                # Ensure fitted[:,0] covers the range of x_test
                y_pred_test = np.interp(x_test, fitted[:,0], fitted[:,1], left=fitted[0,1], right=fitted[-1,1])

                fold_mses.append(mean_squared_error(y_test, y_pred_test))
            except Exception as e:
                print(f"Warning: LOWESS fit/interp failed for fold (w={w}, span={span:.3f}): {e}")
                fold_mses.append(float('inf')) # Penalize failures

        if not fold_mses: # If all folds failed
             mean_mse = float('inf')
        else:
             mean_mse = np.mean([m for m in fold_mses if np.isfinite(m)]) # Average finite MSEs
             if not np.isfinite(mean_mse): mean_mse = float('inf') # Handle case where all were inf


        candidate_results.append((w, span, mean_mse))
        if mean_mse < best_score:
            best_score = mean_mse
            best_span = span
            best_w = w

    # Check if a best window was found
    if best_w is None:
         print("Warning: Could not find a best window via cross-validation. Using defaults.")
         best_w = window_range[0] if window_range else 3
         best_span = min(1.0, max( (2*best_w + 1)/N, 1e-6 ))

    # Perform final lowess fit on all data using the best span
    try:
        order = np.argsort(X)
        X_sorted = X[order]
        Y_sorted = Y[order]
        final_fit = lowess(endog=Y_sorted, exog=X_sorted, frac=best_span, return_sorted=True)

        # Interpolate back to original X order to compute residuals
        Y_smooth = np.interp(X, final_fit[:,0], final_fit[:,1], left=final_fit[0,1], right=final_fit[-1,1])
        dataframe["model_residuals"] = Y - Y_smooth
    except Exception as e:
        print(f"Error during final LOWESS fit or residual calculation: {e}")
        # Assign zero residuals as a fallback?
        dataframe["model_residuals"] = 0.0


    if plotting and candidate_results:
        # Plot the candidate results
        wvals = [cr[0] for cr in candidate_results]
        mses  = [cr[2] for cr in candidate_results]
        plt.figure(figsize=(8, 5))
        plt.plot(wvals, mses, marker='o', linestyle='-')
        if best_w is not None:
             plt.scatter([best_w], [best_score], color='red', s=100, label=f'Best w={best_w} (MSE={best_score:.2e})', zorder=5)
             plt.legend()
        plt.title("Cross-validation MSE vs. LOWESS Half-Window")
        plt.xlabel("Half-window size (w)")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.grid(True)
        plt.show()

    return best_w, best_span, dataframe


#####################################################################
# “sigma_k_calc” - Rolling/moving window stdev of residuals (Integrated)
#####################################################################

def rolling_std(dataframe, half_window, plotting=True):
    """
    Calculates the rolling standard deviation of 'model_residuals'.
    Adds 'roll_sd' and 'roll_sigma2' columns to the dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'x' and 'model_residuals'.
        half_window (int): Half-size of the rolling window.
        plotting (bool): Whether to plot residuals and rolling SD.

    Returns:
        pd.DataFrame: DataFrame with added 'roll_sd' and 'roll_sigma2'.
    """
    if 'model_residuals' not in dataframe.columns:
         raise ValueError("DataFrame must contain 'model_residuals' column.")
    if half_window is None or half_window < 0:
         print("Warning: Invalid half_window provided. Using half_window=0.")
         half_window = 0

    arr = dataframe['model_residuals'].values
    N = len(arr)
    window_size = 2*half_window + 1

    # Use pandas rolling for efficiency if possible
    if N > 0:
        # center=True mimics the manual loop behavior
        # min_periods=2 ensures std dev is calculated only if >1 non-NA point in window
        roll_sd_series = pd.Series(arr).rolling(window=window_size, center=True, min_periods=2).std(ddof=1)
        # Fill NaNs resulting from min_periods or edges. A common strategy is backfill/forwardfill.
        roll_sd_series = roll_sd_series.fillna(method='bfill').fillna(method='ffill')
        # Ensure no zero SD - replace with small value if necessary
        roll_sd_series = roll_sd_series.replace(0, 1e-9)
        # Handle case where all values might still be NaN (e.g., very short series)
        roll_sd = roll_sd_series.fillna(1e-9).values
    else:
        roll_sd = np.array([])


    # Compute rolling variance
    roll_var = roll_sd ** 2  # Variance

    # Assign values back to dataframe
    dataframe['roll_sd'] = roll_sd
    dataframe['roll_sigma2'] = roll_var

    # Plot results if required
    if plotting and N > 0:
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(dataframe['x'], dataframe['model_residuals'], label='Residuals', color='gray', linestyle='-', alpha=0.7)
            plt.scatter(dataframe['x'], dataframe['model_residuals'], color='blue', s=10, label='Residual Points')
            plt.title('Model Residuals from LOWESS Fit')
            plt.xlabel('Days')
            plt.ylabel('Residuals (Rate - Smoothed Rate)')
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.legend()
            plt.grid(True, axis='y', linestyle=':')
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(dataframe['x'], dataframe['roll_sd'], label=f'Rolling SD (Window={window_size})', color='blue')
            # plt.scatter(dataframe['x'], dataframe['roll_sd'], color='red', s=10, label='Rolling SD Points') # Scatter might be too dense
            plt.title('Rolling Standard Deviation of Residuals')
            plt.xlabel('Days')
            plt.ylabel('Standard Deviation')
            plt.legend()
            plt.grid(True, axis='y', linestyle=':')
            plt.ylim(bottom=0) # SD cannot be negative
            plt.show()
        except Exception as e:
            print(f"Warning: Could not plot rolling_std results - {e}")


    return dataframe


######################################################################
# “sample_sorted_datasets” (Integrated)
######################################################################

def sample_sorted_datasets(dataframe, n_samples=1000, seed=123, plotting=True):
    """
    Generates Monte Carlo samples based on mean ('y') and rolling SD ('roll_sd').
    Sorts samples for each time step as described in SPE-194503-PA.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'x', 'y', 'roll_sd'.
        n_samples (int): Number of samples to generate per time step.
        seed (int): Random seed for reproducibility.
        plotting (bool): Whether to plot the sampled data statistics.

    Returns:
        tuple: (df_samples, df_samples_stats)
               df_samples: DataFrame with 'x', 'y', 'sigma2', and 'sample_1'...'sample_n' columns.
               df_samples_stats: DataFrame with 'x', 'y', 'sigma2', and P10/Mean/P50/P90 of samples.
    """
    if not all(col in dataframe.columns for col in ['x', 'y', 'roll_sd']):
         raise ValueError("DataFrame must contain 'x', 'y', and 'roll_sd' columns.")
    if dataframe.empty:
         print("Warning: Input dataframe is empty. Cannot generate samples.")
         # Return empty dataframes with expected columns
         cols_samples = ['x', 'y', 'sigma2'] + [f"sample_{j+1}" for j in range(n_samples)]
         cols_stats = ['x', 'y', 'sigma2', 'sample_p10', 'sample_mean', 'sample_p50', 'sample_p90']
         return pd.DataFrame(columns=cols_samples), pd.DataFrame(columns=cols_stats)


    rng = np.random.default_rng(seed)

    # Pull arrays for convenience
    x = dataframe['x'].values
    y = dataframe['y'].values
    sd = dataframe['roll_sd'].values  # Use the calculated rolling SD
    sigma2 = dataframe.get('roll_sigma2', sd**2).values # Get variance if available, else calculate
    N = len(x)

    # Create an array of shape (N, n_samples)
    # Each row i => all samples for that time step
    sampled_matrix = np.zeros((N, n_samples))

    for i in range(N):
        # Draw from Normal distribution N(mean=y[i], std=sd[i])
        # Ensure scale (sd) is positive
        current_sd = max(sd[i], 1e-9) # Use small positive value if sd is zero or negative
        draws = rng.normal(loc=y[i], scale=current_sd, size=n_samples)
        # Ensure draws are non-negative if rate cannot be negative
        draws[draws < 0] = 0
        # Sort descending, as per paper's method description
        draws_sorted = np.sort(draws)[::-1]
        # Place them in row i
        sampled_matrix[i, :] = draws_sorted

    # Build the primary output DataFrame
    col_names = [f"sample_{j}" for j in range(n_samples)] # Use 0-based index for consistency
    df_samples = pd.DataFrame(sampled_matrix, columns=col_names)

    # Insert original data and variance at the beginning
    df_samples.insert(0, 'x', x)
    df_samples.insert(1, 'y', y)
    df_samples.insert(2, 'sigma2', sigma2)

    # Compute Monte Carlo statistics (P10, Mean, P50, P90) from the sampled matrix
    # Use nan-safe functions
    sample_p10 = np.nanpercentile(sampled_matrix, 10, axis=1)
    sample_mean = np.nanmean(sampled_matrix, axis=1)
    sample_p50 = np.nanpercentile(sampled_matrix, 50, axis=1)
    sample_p90 = np.nanpercentile(sampled_matrix, 90, axis=1)

    df_samples_stats = pd.DataFrame({
        'x': x,
        'y': y,
        'sigma2': sigma2,
        'sample_p10': sample_p10,
        'sample_mean': sample_mean,
        'sample_p50': sample_p50,
        'sample_p90': sample_p90
    })

    # Plot results if required
    if plotting and N > 0:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df_samples_stats['x'], df_samples_stats['y'], 'o', color='gray', alpha=0.6, markersize=4, label='Original Data (Mean)')
            plt.plot(df_samples_stats['x'], df_samples_stats['sample_mean'], color='black', label='Mean of Samples', linewidth=1.5)
            plt.plot(df_samples_stats['x'], df_samples_stats['sample_p10'], color='red', linestyle='dashed', label='P10 of Samples', linewidth=1)
            plt.plot(df_samples_stats['x'], df_samples_stats['sample_p90'], color='red', linestyle='dashed', label='P90 of Samples', linewidth=1)
            plt.fill_between(df_samples_stats['x'], df_samples_stats['sample_p10'], df_samples_stats['sample_p90'], color='red', alpha=0.1, label='P10-P90 Range')


            # Plot a few individual sample realizations (optional, can be slow/cluttered)
            # sample_indices_to_plot = [0, n_samples // 2, n_samples - 1] # e.g., highest, median, lowest rank
            # colors = ['blue', 'green', 'purple']
            # for i, sample_idx in enumerate(sample_indices_to_plot):
            #     col_name = f'sample_{sample_idx}'
            #     if col_name in df_samples.columns:
            #         plt.plot(df_samples['x'], df_samples[col_name], color=colors[i], alpha=0.3, linewidth=0.5, label=f'Sample Rank {sample_idx}')


            plt.title(f"Sorted Monte Carlo Sample Realizations ({n_samples} Samples)")
            plt.xlabel("Days")
            plt.ylabel("Oil Rate, bbl/day")
            plt.legend()
            plt.grid(True, axis='y', linestyle=':')
            plt.ylim(bottom=0)
            plt.show()
        except Exception as e:
            print(f"Warning: Could not plot sample_sorted_datasets results - {e}")


    # df_samples has shape (N, 3 + n_samples)
    return df_samples, df_samples_stats

# -------------------------------------------------------------------
# INTEGRATED FITTING AND RESULTS FUNCTIONS
# -------------------------------------------------------------------

######################################################################
# “fit_model_for_samples” (Integrated)
######################################################################

def fit_model_for_samples(model_class, sample_df, seed=None):
    """
    Fits a decline model to each Monte Carlo sample using the same initial parameters.

    Args:
        model_class: A class that inherits from a base decline model structure
                     (expected to have .initialize_parameters(), .fit(), .predict(), _bounds attribute).
        sample_df: DataFrame must have columns ['x','sigma2','sample_0', 'sample_1', ...].
        seed: Optional random seed for reproducible parameter initialization.

    Returns:
        dict: A dictionary with:
              'params': DataFrame of fitted parameters (samples x parameters).
              'sse': NumPy array of Sum of Squared Errors for each sample fit.
              'predictions': Dictionary {sample_col_name: prediction_array}.
    """
    # Find sample columns automatically
    sample_cols = sample_df.filter(regex=r'^sample_\d+$').columns
    if not sample_cols.any():
        raise ValueError("No columns starting with 'sample_' found in sample_df.")

    t_data = sample_df['x'].values
    # Ensure sigma2 is present and positive
    if 'sigma2' not in sample_df.columns:
         raise ValueError("'sigma2' column missing from sample_df.")
    var_data = sample_df['sigma2'].values.copy()
    var_data[var_data <= 0] = 1e-12 # Avoid division by zero in SSE

    # Initialize lists/dicts to store results
    param_list = []
    sse_list = []
    predictions = {} # Store predictions as {col_name: array}

    # --------------------------------------------------------
    # Compute the P50 curve across all sample columns
    # --------------------------------------------------------
    sample_values = sample_df[sample_cols].values  # shape: (n_timesteps, n_samples)
    p50_curve = np.nanpercentile(sample_values, 50, axis=1)

    # --------------------------------------------------------
    # Use the P50 curve to generate the shared initial guess
    # --------------------------------------------------------
    try:
        model_tmp = model_class()
        # Check if initialize_parameters exists and accepts expected args
        # Assuming it takes t_data, q_data, and optionally var_data, seed
        shared_initial_guess = model_tmp.initialize_parameters(
            t_data=t_data, q_data=p50_curve, var_data=var_data, seed=seed
        )
        print(f"[{model_class.__name__} Init Guess from P50 Curve] {shared_initial_guess}")
    except AttributeError:
         print(f"Warning: {model_class.__name__} has no 'initialize_parameters' method. Fitting may fail without good guess.")
         shared_initial_guess = None # Or try to get default params if possible
    except Exception as e:
         print(f"Warning: Error initializing parameters for {model_class.__name__} using P50 curve: {e}")
         shared_initial_guess = None

    # Determine the expected number of parameters (needed for fallback on failure)
    try:
         num_params = len(model_tmp._bounds) # Assuming _bounds attribute exists
    except AttributeError:
         print(f"Warning: Cannot determine number of parameters for {model_class.__name__} from _bounds. Fallback on fit failure might be incorrect.")
         # Try getting length from initial guess if available, otherwise make a guess
         num_params = len(shared_initial_guess) if shared_initial_guess is not None else 3 # Default guess

    # --------------------------------------------------------
    # Fit each sample using the same initial guess
    # --------------------------------------------------------
    # Use tqdm for progress bar
    for col in tqdm(sample_cols, desc=f"Fitting {model_class.__name__} samples"):
        q_data = sample_df[col].values

        # Check for NaNs in this sample's data
        if np.isnan(q_data).any():
            print(f"Warning: NaN values found in data for {col}. Skipping fit.")
            sse_list.append(float('inf')) # Assign high SSE for NaN data
            param_list.append([np.nan] * num_params)
            predictions[col] = np.full_like(q_data, np.nan)
            continue

        # Create a new model instance for each sample fit
        model = model_class()
        # Assign the shared initial guess if available
        if shared_initial_guess is not None:
             model._initial_guess = shared_initial_guess

        try:
            # Fit the model to this sample's data
            # Assuming .fit() returns the best parameters
            best_params = model.fit(t_data, q_data, var_data, sample_id=col) # Pass sample_id if needed by fit

            if best_params is None:
                 raise RuntimeError("Fit method returned None")

            # Predict using the fitted parameters
            # Assuming .predict() uses the parameters stored within the model instance after fitting
            q_pred = model.predict(t_data)

            # Compute Sum of Squared Errors (weighted by inverse variance)
            resid = (q_data - q_pred)
            sse = np.sum((resid**2) / var_data) # var_data already checked for non-positive

            # Store results
            sse_list.append(sse)
            param_list.append(best_params)
            predictions[col] = q_pred # Store prediction array

        except Exception as e:
            print(f"Fit failed for {col}: {e}")
            # Store fallback values indicating failure
            sse_list.append(float('inf')) # Use infinity for failed fits
            param_list.append([np.nan] * num_params)
            predictions[col] = np.full_like(q_data, np.nan) # Store NaN array for prediction

    # Post-process results
    # Ensure parameter list has consistent structure even with failures
    param_list = [p if isinstance(p, (list, np.ndarray)) and len(p) == num_params else [np.nan] * num_params for p in param_list]

    # Create DataFrame for parameters
    # Assuming model parameters have names defined in the class, e.g., model_tmp.param_names
    try:
         param_names = model_tmp.param_names
    except AttributeError:
         param_names = [f'param_{i}' for i in range(num_params)] # Default names
    df_params = pd.DataFrame(param_list, index=sample_cols, columns=param_names)

    # Return results dictionary
    return {
        'params': df_params,
        'sse': np.array(sse_list),
        'predictions': predictions # Return dict of {col_name: array}
    }

######################################################################
# “gather_sse_matrix” (Integrated)
######################################################################

def gather_sse_matrix(model_results_dict, model_names):
    """
    Gathers SSE arrays from a results dictionary into a matrix.

    Args:
        model_results_dict (dict): Dictionary where keys are model names and values
                                   are dicts containing at least {'sse': np.array}.
                                   e.g., {'arps': {'sse': [...]}, 'sem': {'sse': [...]}}
        model_names (list): List of model names defining the order of rows in the output matrix.

    Returns:
        np.ndarray: SSE matrix of shape (M, N), where M is the number of models
                    and N is the number of samples. Returns None if input is invalid.
    """
    if not model_results_dict or not model_names:
        print("Error: Invalid input dictionary or model names list.")
        return None

    all_sse = []
    first_sse_len = -1

    for m in model_names:
        if m not in model_results_dict or 'sse' not in model_results_dict[m]:
            print(f"Error: SSE data missing for model '{m}'.")
            return None # Or handle by skipping/filling with NaN? Returning None is safer.

        sse_array = model_results_dict[m]['sse']

        if not isinstance(sse_array, np.ndarray):
             try:
                 sse_array = np.array(sse_array)
             except Exception as e:
                 print(f"Error: Could not convert SSE for model '{m}' to NumPy array: {e}")
                 return None

        if sse_array.ndim != 1:
             print(f"Error: SSE array for model '{m}' is not 1-dimensional (shape: {sse_array.shape}).")
             return None

        if first_sse_len == -1:
            first_sse_len = len(sse_array)
        elif len(sse_array) != first_sse_len:
            print(f"Error: SSE arrays have inconsistent lengths (model '{m}' has {len(sse_array)}, expected {first_sse_len}).")
            return None

        all_sse.append(sse_array)

    if not all_sse: # If model_names was empty or no valid SSE found
        print("Error: No valid SSE arrays found to stack.")
        return None

    # Stack the SSE arrays vertically
    sse_matrix = np.vstack(all_sse)  # shape (M, N)
    return sse_matrix

######################################################################
# “gather_predictions” (Integrated)
######################################################################

def gather_predictions(fit_results, sample_df):
    """
    Gathers predicted rates from fit results into a matrix.

    Args:
        fit_results (dict): Dictionary containing at least {'predictions': {sample_col: array}}.
        sample_df (pd.DataFrame): DataFrame used for fitting, containing sample columns
                                  (e.g., 'sample_0', 'sample_1', ...) to determine order.

    Returns:
        np.ndarray: Prediction matrix of shape (N_time, M_samples).
                    Returns None if input is invalid.
    """
    if not isinstance(fit_results, dict) or 'predictions' not in fit_results:
        print("Error: Invalid fit_results dictionary or 'predictions' key missing.")
        return None
    if not isinstance(sample_df, pd.DataFrame):
        print("Error: Invalid sample_df provided.")
        return None

    predictions_dict = fit_results['predictions']
    if not isinstance(predictions_dict, dict):
        print("Error: 'predictions' in fit_results is not a dictionary.")
        return None

    # Identify sample columns from sample_df to ensure correct order and count
    sample_cols = sample_df.filter(regex=r'^sample_\d+$').columns
    if not sample_cols.any():
        print("Error: No 'sample_' columns found in sample_df.")
        return None

    # Get dimensions
    N_time = len(sample_df) # Number of time steps from the input df
    M_samples = len(sample_cols)

    # Initialize prediction matrix
    pred_matrix = np.full((N_time, M_samples), np.nan) # Initialize with NaN

    # Fill matrix column by column using the order from sample_cols
    for j, col_name in enumerate(sample_cols):
        if col_name in predictions_dict:
            pred_array = predictions_dict[col_name]
            # Basic validation of the prediction array
            if isinstance(pred_array, np.ndarray) and pred_array.shape == (N_time,):
                pred_matrix[:, j] = pred_array
            else:
                print(f"Warning: Prediction for '{col_name}' is invalid or has wrong shape. Expected ({N_time},), got {type(pred_array)} shape {getattr(pred_array, 'shape', 'N/A')}. Filling with NaN.")
        else:
            print(f"Warning: Prediction for sample column '{col_name}' not found in predictions_dict. Filling with NaN.")

    return pred_matrix

######################################################################
# “forecast_from_params” (Integrated)
######################################################################

def forecast_from_params(model_class, param_df, t_array):
    """
    Generates forecasts for a given time array using fitted parameters for each sample.

    Args:
        model_class: The class of the decline model (e.g., ArpsModel).
                     Expected to be initializable with params: model_class(params=...).
                     Expected to have a .predict(t_array) method.
        param_df (pd.DataFrame): DataFrame where index = sample names (e.g., 'sample_0')
                                 and columns = parameter names.
        t_array (np.ndarray): Array of time points at which to generate forecasts.

    Returns:
        np.ndarray: Forecast matrix of shape (N_time, M_samples).
                    Returns None if input is invalid.
    """
    if not isinstance(param_df, pd.DataFrame) or param_df.empty:
        print("Error: Invalid or empty parameter DataFrame provided.")
        return None
    if not isinstance(t_array, np.ndarray) or t_array.ndim != 1:
        print("Error: Invalid t_array provided (must be 1D NumPy array).")
        return None

    sample_names = param_df.index
    M_samples = len(sample_names)
    N_time = len(t_array)

    if M_samples == 0 or N_time == 0:
        print("Warning: No samples or no time points specified for forecast.")
        return np.zeros((N_time, M_samples)) # Return empty matrix of correct shape

    # Initialize forecast matrix
    forecast_matrix = np.full((N_time, M_samples), np.nan) # Initialize with NaN

    # Iterate through each sample (row in param_df)
    for j, sample_name in enumerate(sample_names):
        row_params = param_df.loc[sample_name].values

        # Check if parameters for this sample are valid (not all NaN)
        if np.isnan(row_params).all():
            print(f"Warning: Parameters for sample '{sample_name}' are all NaN. Skipping forecast.")
            continue # Leave column as NaN

        try:
            # Instantiate the model with the parameters for this sample
            # Assumes model can be instantiated like this
            model = model_class(params=row_params)

            # Generate prediction for this sample
            q_pred = model.predict(t_array)

            # Validate prediction shape
            if isinstance(q_pred, np.ndarray) and q_pred.shape == (N_time,):
                forecast_matrix[:, j] = q_pred
            else:
                 print(f"Warning: Prediction for sample '{sample_name}' has wrong shape. Expected ({N_time},), got {getattr(q_pred, 'shape', 'N/A')}. Filling with NaN.")

        except Exception as e:
            print(f"Error forecasting for sample '{sample_name}' with params {row_params}: {e}")
            # Leave column as NaN on error

    return forecast_matrix

######################################################################
# “compute_forecast_stats” (Integrated)
######################################################################

def compute_forecast_stats(pred_matrix):
    """
    Computes statistics (P10, P50, Mean, P90) across samples for each time step.

    Args:
        pred_matrix (np.ndarray): Prediction matrix of shape (N_time, M_samples).

    Returns:
        dict: Dictionary containing arrays for 'p10', 'p50', 'mean', 'p90',
              each of shape (N_time,). Returns None if input is invalid.
    """
    if not isinstance(pred_matrix, np.ndarray) or pred_matrix.ndim != 2:
        print("Error: Invalid prediction matrix provided (must be 2D NumPy array).")
        return None

    N_time, M_samples = pred_matrix.shape

    if M_samples == 0:
        print("Warning: Prediction matrix has no samples (columns). Returning NaN stats.")
        nan_array = np.full(N_time, np.nan)
        return {'p10': nan_array, 'p50': nan_array, 'mean': nan_array, 'p90': nan_array}

    try:
        # Use nan-aware functions to handle potential NaNs from failed fits/forecasts
        p10 = np.nanpercentile(pred_matrix, 10, axis=1)
        p50 = np.nanpercentile(pred_matrix, 50, axis=1) # Median
        mean = np.nanmean(pred_matrix, axis=1)
        p90 = np.nanpercentile(pred_matrix, 90, axis=1)

        return {'p10': p10, 'p50': p50, 'mean': mean, 'p90': p90}

    except Exception as e:
        print(f"Error computing forecast stats: {e}")
        return None


# -------------------------------------------------------------------
# 1) GLOBAL PARAMETERS & CONFIGURATION (Remains the same)
# -------------------------------------------------------------------
STUDY_WELL = "AF-6(h)"
N_SAMPLES = 1000      # Monte Carlo samples (Consistent with sample_sorted_datasets default)
KFOLDS = 10           # cross-validation folds
TRAIN_PCT = 0.8       # fraction of data used for training
DAYS_FUTURE = 5400    # Forecast horizon (e.g., 15 years * 360)
BASE_SEED = 123       # Base seed for reproducibility

# Data paths (Consider moving to a config file/class)
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
DATA_DIR = Path("src/probabilistic_dca/data") # Adjust if needed

RESULTS_DIR = DATA_DIR / "model_fit_results"
WELL_INFO_FILE = DATA_DIR / "wells_final_Q12020.csv"
PRODUCTION_FILE = DATA_DIR / f"{STUDY_WELL}_daily_prod.csv" # Assuming filename pattern
SAMPLE_SORTED_FILE = RESULTS_DIR / "sample_sorted_df.csv"

# Column names (Consider moving to a config file/class)
TIME_COLUMN = 'cum_eff_prod_day'
RATE_COLUMN = 'oil_month_bpd'
CUM_PROD_COLUMN = 'cum_oil_bbl'

# --- Model Configuration ---
AVAILABLE_MODELS = {
    "arps": ArpsModel, "sem": SEMModel, "crm": CRMModel, "lgm": LGMModel
}
MODELS_TO_RUN = ["all"] # Or e.g., ["arps", "sem"]

# -------------------------------------------------------------------
# 2) DATA LOADING FUNCTION (Remains the same)
# -------------------------------------------------------------------
def load_data(well_info_path, production_path, time_col, rate_col, cum_col, study_well_name):
    # (Code omitted for brevity - same as previous version)
    try:
        if not well_info_path.exists(): print(f"Error: Well info file not found at {well_info_path}"); return None, None
        if not production_path.exists(): print(f"Error: Production file not found at {production_path}"); return None, None
        data_wells_final_df = pd.read_csv(well_info_path)
        data_production_final_df = pd.read_csv(production_path)
    except Exception as e: print(f"Error reading CSV files: {e}"); return None, None
    if not all(c in data_production_final_df.columns for c in [time_col, rate_col, cum_col]): print(f"Error: Required columns missing in {production_path}"); return None, None
    well_df = data_production_final_df[[time_col, rate_col, cum_col]].copy()
    well_df.dropna(subset=[time_col, rate_col], inplace=True)
    if well_df.empty: print(f"Warning: No data remaining for well {study_well_name}."); return None, data_wells_final_df
    print(f"Loaded production data for well {study_well_name}. Shape: {well_df.shape}")
    return well_df, data_wells_final_df

# -------------------------------------------------------------------
# 3) DATA PREPROCESSING FUNCTION (Wrapper - Remains the same)
# -------------------------------------------------------------------
def preprocess_data_wrapper(well_df, time_col, rate_col, cum_col, train_pct, kfolds, n_samples, study_well_name, results_dir, sample_sorted_file):
    # (Code omitted for brevity - same as previous version, calls integrated functions)
    print("\n--- Starting Data Preprocessing ---")
    # 3.1) LOF
    print("Step 3.1: Performing LOF Outlier Detection...")
    X_lof = well_df[[rate_col]].values; min_neighbors=2; n_neighbors_param=min(16, X_lof.shape[0]-1)
    if X_lof.shape[0] < min_neighbors or n_neighbors_param < min_neighbors:
        print(f"Warning: Not enough data points ({X_lof.shape[0]}) for LOF. Skipping."); data_tbl=pd.DataFrame({'x':well_df[time_col],'y':well_df[rate_col],'z':well_df[cum_col]}); data_tbl_full=data_tbl.copy(); data_tbl_full['lof_flag']=1
    else:
        try: lof=LocalOutlierFactor(n_neighbors=n_neighbors_param, contamination=0.05); lof_labels=lof.fit_predict(X_lof); data_tbl_full=pd.DataFrame({'x':well_df[time_col],'y':well_df[rate_col],'z':well_df[cum_col],'lof_flag':lof_labels}); data_tbl=data_tbl_full[data_tbl_full['lof_flag']==1].drop(columns='lof_flag').reset_index(drop=True); print(f"LOF removed {data_tbl_full.shape[0]-data_tbl.shape[0]} outliers.")
        except Exception as e: print(f"Error during LOF: {e}. Skipping."); data_tbl=pd.DataFrame({'x':well_df[time_col],'y':well_df[rate_col],'z':well_df[cum_col]}); data_tbl_full=data_tbl.copy(); data_tbl_full['lof_flag']=1
    if data_tbl.empty: print("Error: No data remaining after LOF."); return None,None,None,None
    # 3.2) Plot Outliers
    print("Step 3.2: Plotting Production Curve with Outliers...")
    try: plt.figure(figsize=(12,6)); sns.scatterplot(data=data_tbl_full,x='x',y='y',hue=data_tbl_full['lof_flag'].map({1:'Inlier',-1:'Outlier'}),palette={'Inlier':'tab:blue','Outlier':'tab:red'},alpha=0.7); plt.xlabel('Days'); plt.ylabel('Rate'); plt.title(f"{study_well_name} Outliers"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    except Exception as e: print(f"Warning: Could not plot outliers - {e}")
    # 3.3) Call data_processing
    print("Step 3.3: Processing data...")
    try: last_day, last_day_cum_oil, x_train_i, models_df = data_processing(prod_df=data_tbl, train_pct=train_pct, frac_value=0.4, plotting=True)
    except Exception as e: print(f"Error during data_processing call: {e}"); return None,None,None,None
    # 3.4) Call crossval_loess
    print("Step 3.4: Cross-validating LOESS...")
    try: best_half_window, best_span, models_df = crossval_loess(dataframe=models_df, k_folds=kfolds, plotting=True); print(f"Best LOESS: w={best_half_window}, span={best_span:.3f}")
    except Exception as e: print(f"Error during crossval_loess call: {e}"); return None,None,None,None
    # 3.5) Call rolling_std
    print("Step 3.5: Calculating Rolling SD...")
    try: models_df = rolling_std(dataframe=models_df, half_window=best_half_window, plotting=True)
    except Exception as e: print(f"Error during rolling_std call: {e}"); return None,None,None,None
    # 3.6) Call sample_sorted_datasets
    print("Step 3.6: Generating Samples...")
    try: sample_sorted_df, sample_stats_df = sample_sorted_datasets(models_df, n_samples=n_samples, seed=BASE_SEED, plotting=True)
    except Exception as e: print(f"Error during sample_sorted_datasets call: {e}"); return None,None,None,None
    # 3.7) Save samples
    print(f"Step 3.7: Saving sampled data...")
    try: results_dir.mkdir(parents=True,exist_ok=True); sample_sorted_df.to_csv(sample_sorted_file,index=False); print("Saved.")
    except Exception as e: print(f"Error saving samples: {e}")
    return sample_sorted_df, last_day, last_day_cum_oil, x_train_i

# -------------------------------------------------------------------
# 4) DATA SPLITTING FUNCTION (Remains the same)
# -------------------------------------------------------------------
def split_data(sample_sorted_df, train_cutoff_idx):
    # (Code omitted for brevity - same as previous version)
    print("\n--- Splitting Data into Train/Test ---")
    if sample_sorted_df is None or sample_sorted_df.empty: print("Error: Cannot split, input DF is None/empty."); return None,None
    if train_cutoff_idx is None or not isinstance(train_cutoff_idx,(int,np.integer)): print(f"Error: Invalid train_cutoff_idx: {train_cutoff_idx}."); return None,None
    n_decline_points = len(sample_sorted_df)
    if train_cutoff_idx < 0 or train_cutoff_idx >= n_decline_points: print(f"Error: train_cutoff_idx {train_cutoff_idx} out of bounds for length {n_decline_points}"); return None,None
    train_df = sample_sorted_df.iloc[:train_cutoff_idx+1].copy()
    test_df = sample_sorted_df.iloc[train_cutoff_idx+1:].copy()
    print(f"Training data shape (decline): {train_df.shape}"); print(f"Testing data shape (decline): {test_df.shape}")
    return train_df, test_df

# -------------------------------------------------------------------
# 5) MODEL TRAINING FUNCTION (Calls integrated fit_model_for_samples)
# -------------------------------------------------------------------
def train_models(train_df, models_to_fit, base_seed, results_dir):
    # (Code largely the same, but calls integrated fit_model_for_samples)
    print("\n--- Training Models on Training Data ---")
    model_train_results = {}
    if train_df is None or train_df.empty: print("Error: Training data missing."); return model_train_results
    if not models_to_fit: print("Error: No models selected."); return model_train_results

    t_train = train_df["x"].values; q_train_orig = train_df["y"].values
    sample_cols = train_df.filter(regex=r'^sample_\d+$').columns; num_samples_in_df = len(sample_cols)
    if num_samples_in_df == 0: print("Error: No 'sample_' columns found."); return model_train_results

    for m_index, (m_name, m_class) in enumerate(models_to_fit.items()):
        print(f"\n--- Fitting {m_name.upper()} Model ---")
        model_results_dir = results_dir / m_name; model_results_dir.mkdir(parents=True, exist_ok=True)
        param_path = model_results_dir/f"{m_name}_parameters.csv"; sse_path = model_results_dir/f"{m_name}_sse.csv"; pred_path = model_results_dir/f"{m_name}_predictions.csv"

        # Fit model across samples using integrated function
        print(f"Fitting {m_name.upper()} across {num_samples_in_df} samples...")
        try:
            # Pass base_seed for initial guess generation inside fit_model_for_samples
            fit_results = fit_model_for_samples(
                model_class=m_class,
                sample_df=train_df, # Contains x, y, sigma2, sample_0, ...
                seed=base_seed # Seed for initial guess consistency
            )
        except Exception as e:
            print(f"Error calling fit_model_for_samples for {m_name}: {e}")
            continue # Skip this model

        model_train_results[m_name] = fit_results

        # Extract and save (using results structure from fit_model_for_samples)
        param_df = fit_results.get("params")
        sse_arr = fit_results.get("sse")
        preds_dict = fit_results.get("predictions") # This is now a dict {col: array}

        if param_df is None or sse_arr is None or preds_dict is None:
            print(f"Warning: Missing results after fitting {m_name}. Skipping save.")
            if m_name in model_train_results: del model_train_results[m_name]
            continue

        try:
            param_df.to_csv(param_path)
            pd.DataFrame({"sample": param_df.index, "sse": sse_arr}).to_csv(sse_path, index=False)

            # Convert predictions dict to DataFrame for saving
            # Ensure columns match the order in sample_cols used for fitting
            pred_matrix_for_save = np.full((len(t_train), num_samples_in_df), np.nan)
            for i, col_name in enumerate(sample_cols):
                 if col_name in preds_dict:
                     pred_matrix_for_save[:, i] = preds_dict[col_name]

            pred_df = pd.DataFrame(pred_matrix_for_save, columns=sample_cols)
            pred_df.insert(0, "x", t_train)
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved results for {m_name.upper()} to {model_results_dir}")
        except Exception as e:
             print(f"Error saving results for {m_name}: {e}")
             print(f"Warning: Could not save results for {m_name}.")

    return model_train_results


# -------------------------------------------------------------------
# 6) ANALYZE TRAINING FITS FUNCTION (Calls integrated gather_predictions, compute_forecast_stats)
# -------------------------------------------------------------------
def analyze_training_fits(model_train_results, train_df, models_to_analyze):
    # (Code largely the same, calls integrated functions)
    print("\n--- Analyzing Training Fits ---")
    if train_df is None or train_df.empty: print("Skipping: Training data missing."); return

    for m_name in models_to_analyze:
        if m_name not in model_train_results: print(f"Warning: Results for '{m_name}' not found."); continue
        print(f"\n--- Analyzing {m_name.upper()} fit on TRAIN set ---")
        fit_results = model_train_results[m_name]
        try:
            # Use integrated gather_predictions
            pred_matrix_train = gather_predictions(fit_results, train_df)
            if pred_matrix_train is None or pred_matrix_train.size==0: print(f"Warning: gather_predictions failed for {m_name}."); continue
            # Use integrated compute_forecast_stats
            train_stats = compute_forecast_stats(pred_matrix_train)
            if train_stats is None: print(f"Warning: compute_forecast_stats failed for {m_name}."); continue

            plot_model_predictions(train_df, train_stats, title_suffix=f"{m_name.upper()} Train Fit")
            df_params = fit_results.get("params");
            if df_params is not None: print(f"{m_name.upper()} - P50 Params:\n{df_params.median()}\n")
            else: print(f"Warning: Params DF not found for {m_name}.")
        except Exception as e: print(f"Error analyzing training fit for {m_name}: {e}")

# -------------------------------------------------------------------
# 7) CALCULATE MODEL PROBABILITIES FUNCTION (Calls integrated gather_sse_matrix)
# -------------------------------------------------------------------
def calculate_model_probabilities(model_train_results, model_names):
    # (Code largely the same, calls integrated gather_sse_matrix)
    print("\n--- Calculating Model Probabilities ---")
    model_dict_for_probs = {}; valid_model_names = []
    for m_name in model_names:
        if m_name in model_train_results and isinstance(model_train_results[m_name],dict) and "sse" in model_train_results[m_name]:
            sse_data = model_train_results[m_name]["sse"]
            if sse_data is not None: model_dict_for_probs[m_name]={"sse":sse_data}; valid_model_names.append(m_name)
            else: print(f"Warning: SSE is None for '{m_name}'. Excluding.")
        else: print(f"Warning: SSE not found for '{m_name}'. Excluding.")
    if len(valid_model_names)<2: print("Error: Need >= 2 models with SSE."); return None,None,[]

    try:
        # Use integrated gather_sse_matrix
        sse_matrix = gather_sse_matrix(model_dict_for_probs, valid_model_names)
        if sse_matrix is None: raise ValueError("gather_sse_matrix failed.")
        prob_matrix = calc_model_probabilities(sse_matrix)
        marginal_probs = compute_marginal_model_probs(prob_matrix, valid_model_names)
        ranked_models = rank_models_by_probability(marginal_probs)
    except Exception as e: print(f"Error calculating probabilities: {e}"); return None,None,[]

    print("\n--- Marginal Posterior Probability ---"); [print(f"{n}: {p:.3f}") for n,p in ranked_models]
    try: plot_post_prob_models(ranked_models)
    except Exception as e: print(f"Warning: Could not plot probabilities - {e}")
    return prob_matrix, ranked_models, valid_model_names

# -------------------------------------------------------------------
# 8) HINDCAST FUNCTION (Calls integrated forecast_from_params, compute_forecast_stats)
# -------------------------------------------------------------------
def perform_hindcast(test_df, model_train_results, models_to_analyze, model_classes_map):
    # (Code largely the same, calls integrated functions)
    print("\n--- Performing Hindcast ---")
    if test_df is None or test_df.empty: print("Skipping: Test data missing."); return
    t_test = test_df["x"].values

    for m_name in models_to_analyze:
        if m_name not in model_train_results or not isinstance(model_train_results[m_name],dict) or "params" not in model_train_results[m_name] or model_train_results[m_name]["params"] is None: print(f"Warning: Params not found for '{m_name}'. Skipping."); continue
        if m_name not in model_classes_map: print(f"Warning: Class not found for '{m_name}'. Skipping."); continue

        print(f"\n--- Hindcasting {m_name.upper()} ---")
        fit_res=model_train_results[m_name]; param_df=fit_res["params"]; m_class=model_classes_map[m_name]
        try:
            # Use integrated forecast_from_params
            test_fc_matrix = forecast_from_params(m_class, param_df, t_test)
            if test_fc_matrix is None or test_fc_matrix.size==0: print(f"Warning: forecast_from_params failed for {m_name}."); continue
            # Use integrated compute_forecast_stats
            test_stats = compute_forecast_stats(test_fc_matrix)
            if test_stats is None: print(f"Warning: compute_forecast_stats failed for {m_name}."); continue
            plot_hindcast(test_df, test_stats, title_suffix=f"{m_name.upper()} Hindcast")
        except Exception as e: print(f"Error during hindcast for {m_name}: {e}")

# -------------------------------------------------------------------
# 9) FUTURE FORECAST FUNCTION (Calls integrated forecast_from_params, compute_forecast_stats)
# -------------------------------------------------------------------
def forecast_future(last_day, days_future, model_train_results, models_to_analyze, model_classes_map, last_day_cum_oil):
    # (Code largely the same, calls integrated functions)
    print("\n--- Generating Future Forecasts ---")
    future_forecasts_dict={}; model_eur_stats_dict={}
    if last_day is None or last_day_cum_oil is None: print("Error: Missing last_day info."); return future_forecasts_dict,model_eur_stats_dict
    if not models_to_analyze: print("Warning: No models for forecast."); return future_forecasts_dict,model_eur_stats_dict
    t_future = np.arange(last_day+1, last_day+1+days_future)

    for m_name in models_to_analyze:
        if m_name not in model_train_results or not isinstance(model_train_results[m_name],dict) or "params" not in model_train_results[m_name] or model_train_results[m_name]["params"] is None: print(f"Warning: Params not found for '{m_name}'. Skipping."); continue
        if m_name not in model_classes_map: print(f"Warning: Class not found for '{m_name}'. Skipping."); continue

        print(f"\n--- Long-Term Forecast: {m_name.upper()} ---")
        fit_res=model_train_results[m_name]; param_df=fit_res["params"]; m_class=model_classes_map[m_name]
        try:
            # Use integrated forecast_from_params
            fc_matrix = forecast_from_params(m_class, param_df, t_future)
            if fc_matrix is None or fc_matrix.size==0: print(f"Warning: forecast_from_params failed for {m_name}."); continue
            # Use integrated compute_forecast_stats
            fc_stats = compute_forecast_stats(fc_matrix)
            if fc_stats is None: print(f"Warning: compute_forecast_stats failed for {m_name}."); continue
            plot_future_forecast(t_future, fc_stats, title_suffix=f"{m_name.upper()} Future Forecast")

            dt=1.0; cum_matrix=np.cumsum(fc_matrix,axis=0)*dt; cum_dist=cum_matrix[-1,:]; valid_cum_dist=cum_dist[np.isfinite(cum_dist)]
            if valid_cum_dist.size==0: print(f"Warning: Non-finite cum values for {m_name}."); eur_stats={k:np.nan for k in ["p10","p25","p50","mean","p75","p90"]}
            else: p10_cum,p50_cum=np.nanpercentile(valid_cum_dist,[10,50]); mean_cum=np.nanmean(valid_cum_dist); p90_cum=np.nanpercentile(valid_cum_dist,90); eur_stats={"p10":last_day_cum_oil+p10_cum,"p25":last_day_cum_oil+np.nanpercentile(valid_cum_dist,25),"p50":last_day_cum_oil+p50_cum,"mean":last_day_cum_oil+mean_cum,"p75":last_day_cum_oil+np.nanpercentile(valid_cum_dist,75),"p90":last_day_cum_oil+p90_cum}
            model_eur_stats_dict[m_name]=eur_stats

            yr_str=f"{days_future/365:.1f}" if days_future>0 else "0.0"
            p10_c_str,p50_c_str,mean_c_str,p90_c_str=(f"{v:.1f}" if not np.isnan(v) else "NaN" for v in [p10_cum,p50_cum,mean_cum,p90_cum])
            p10_e_str,p50_e_str,mean_e_str,p90_e_str=(f"{eur_stats[k]:.1f}" if not np.isnan(eur_stats[k]) else "NaN" for k in ['p10','p50','mean','p90'])
            print(f"{m_name.upper()} {yr_str}-Yr Future CUM P10={p10_c_str}, P50={p50_c_str}, Mean={mean_c_str}, P90={p90_c_str}")
            print(f"{m_name.upper()} {yr_str}-Yr EUR P10={p10_e_str}, P50={p50_e_str}, Mean={mean_e_str}, P90={p90_e_str}")
            future_forecasts_dict[m_name]=fc_matrix.T
        except Exception as e: print(f"Error during future forecast for {m_name}: {e}")
    return future_forecasts_dict,model_eur_stats_dict

# -------------------------------------------------------------------
# 10) COMBINED RESULTS FUNCTION (Remains the same)
# -------------------------------------------------------------------
def generate_combined_results(future_forecasts_dict, prob_matrix, model_names_in_prob, model_eur_stats_dict, last_day_cum_oil, study_well_name):
    # (Code omitted for brevity - same as previous version)
    print("\n--- Generating Multi-Model Combined Forecast ---")
    if not future_forecasts_dict or prob_matrix is None or not model_names_in_prob: print("Error: Missing inputs."); return None
    if last_day_cum_oil is None: print("Error: last_day_cum_oil is None."); return None
    model_future_arrays=[]; valid_model_indices_in_prob=[]; final_model_names_combined=[]
    for i,m_name in enumerate(model_names_in_prob):
        if m_name in future_forecasts_dict and future_forecasts_dict[m_name] is not None:
            if future_forecasts_dict[m_name].shape[0]==prob_matrix.shape[1]: model_future_arrays.append(future_forecasts_dict[m_name]); valid_model_indices_in_prob.append(i); final_model_names_combined.append(m_name)
            else: print(f"Warning: Sample count mismatch for '{m_name}'. Excluding.")
        else: print(f"Debug: Model '{m_name}' not found in forecasts.")
    if not model_future_arrays or not valid_model_indices_in_prob: print("Error: No valid forecasts align."); return None
    aligned_prob_matrix = prob_matrix[valid_model_indices_in_prob,:]
    try:
        future_forecast_tensor=np.stack(model_future_arrays,axis=0); combined_future_forecast=combine_forecasts_across_models(future_forecast_tensor,aligned_prob_matrix)
        if combined_future_forecast is None or combined_future_forecast.size==0: print("Error: combine_forecasts failed."); return None
        dt=1.0; cum_combined=np.cumsum(combined_future_forecast,axis=1)*dt; final_cum_samples=cum_combined[:,-1]; valid_final_cum=final_cum_samples[np.isfinite(final_cum_samples)]
        if valid_final_cum.size==0: print("Error: Non-finite combined cum values."); combined_stats={k:np.nan for k in ["p10","p25","p50","mean","p75","p90"]}
        else: combined_stats={"p10":last_day_cum_oil+np.nanpercentile(valid_final_cum,10),"p25":last_day_cum_oil+np.nanpercentile(valid_final_cum,25),"p50":last_day_cum_oil+np.nanpercentile(valid_final_cum,50),"mean":last_day_cum_oil+np.nanmean(valid_final_cum),"p75":last_day_cum_oil+np.nanpercentile(valid_final_cum,75),"p90":last_day_cum_oil+np.nanpercentile(valid_final_cum,90)}
        p10_str,p50_str,mean_str,p90_str=(f"{combined_stats[k]:.1f}" if not np.isnan(combined_stats[k]) else "NaN" for k in ['p10','p50','mean','p90'])
        print("\n--- Combined Forecast EUR Statistics ---"); print(f"P10: {p10_str}"); print(f"P50: {p50_str}"); print(f"Mean: {mean_str}"); print(f"P90: {p90_str}")
        eur_data=[]
        for m_name in final_model_names_combined:
             if m_name in model_eur_stats_dict: stats=model_eur_stats_dict[m_name]; eur_data.append({"model_name":m_name.upper(),"y10":stats.get('p10',np.nan),"y25":stats.get('p25',np.nan),"y50":stats.get('p50',np.nan),"ymean":stats.get('mean',np.nan),"y75":stats.get('p75',np.nan),"y90":stats.get('p90',np.nan)})
        eur_data.append({"model_name":"Combined","y10":combined_stats.get('p10',np.nan),"y25":combined_stats.get('p25',np.nan),"y50":combined_stats.get('p50',np.nan),"ymean":combined_stats.get('mean',np.nan),"y75":combined_stats.get('p75',np.nan),"y90":combined_stats.get('p90',np.nan)})
        df_eur=pd.DataFrame(eur_data); boxplot_eur(df_eur,title=f"EUR Distribution for {study_well_name}"); print(f"\nCombined P50 EUR: {p50_str} bbl")
        return df_eur
    except Exception as e: print(f"Error generating combined results: {e}"); return None


# -------------------------------------------------------------------
# MAIN EXECUTION SCRIPT (Remains the same)
# -------------------------------------------------------------------
def main():
    """Main function to run the probabilistic DCA workflow."""
    # (Code omitted for brevity - same as previous version)
    if MODELS_TO_RUN == ["all"]: models_to_fit = AVAILABLE_MODELS
    else: models_to_fit = {name: AVAILABLE_MODELS[name] for name in MODELS_TO_RUN if name in AVAILABLE_MODELS};
    if not models_to_fit: print(f"Error: No specified models available."); return
    print(f"Running analysis for Well: {STUDY_WELL}"); print(f"Models selected: {list(models_to_fit.keys())}")
    well_data_raw, _ = load_data(WELL_INFO_FILE, PRODUCTION_FILE, TIME_COLUMN, RATE_COLUMN, CUM_PROD_COLUMN, STUDY_WELL)
    if well_data_raw is None: print("Exiting: Failed to load data."); return
    sample_sorted_df, last_day, last_day_cum_oil, x_train_i = None, None, None, None
    if SAMPLE_SORTED_FILE.exists():
        print(f"Loading existing samples: {SAMPLE_SORTED_FILE}");
        try: sample_sorted_df = pd.read_csv(SAMPLE_SORTED_FILE); print("Loaded samples."); print("Recalculating metadata..."); _, temp_last_day, temp_last_day_cum_oil, temp_x_train_i = preprocess_data_wrapper(well_data_raw, TIME_COLUMN, RATE_COLUMN, CUM_PROD_COLUMN, TRAIN_PCT, KFOLDS, N_SAMPLES, STUDY_WELL, RESULTS_DIR, SAMPLE_SORTED_FILE);
        if temp_x_train_i is not None: last_day, last_day_cum_oil, x_train_i = temp_last_day, temp_last_day_cum_oil, temp_x_train_i
        else: print("Error: Failed recalculating metadata."); sample_sorted_df = None
        except Exception as e: print(f"Error loading/processing existing samples: {e}"); sample_sorted_df = None
    if sample_sorted_df is None: print("Running full preprocessing..."); sample_sorted_df, last_day, last_day_cum_oil, x_train_i = preprocess_data_wrapper(well_data_raw, TIME_COLUMN, RATE_COLUMN, CUM_PROD_COLUMN, TRAIN_PCT, KFOLDS, N_SAMPLES, STUDY_WELL, RESULTS_DIR, SAMPLE_SORTED_FILE)
    if sample_sorted_df is None or x_train_i is None: print("Exiting: Preprocessing failed."); return
    train_df, test_df = split_data(sample_sorted_df, x_train_i)
    if train_df is None or test_df is None: print("Exiting: Data split failed."); return
    print("\n--- Loading/Training Models ---"); model_train_results = {}
    try: loaded_results = load_all_model_train_results(list(models_to_fit.keys()), base_dir=RESULTS_DIR); model_train_results.update(loaded_results); print(f"Loaded: {list(loaded_results.keys())}"); models_needing_training = {n: k for n, k in models_to_fit.items() if n not in loaded_results};
    if models_needing_training: print(f"Training: {list(models_needing_training.keys())}"); trained_results = train_models(train_df, models_needing_training, BASE_SEED, RESULTS_DIR); model_train_results.update(trained_results)
    else: print("All results loaded.")
    except Exception as e: print(f"Load failed ({e}). Training all..."); model_train_results = train_models(train_df, models_to_fit, BASE_SEED, RESULTS_DIR)
    if not model_train_results: print("Training/loading failed. Exiting."); return
    models_with_results = list(model_train_results.keys()); print(f"Analyzing models: {models_with_results}")
    analyze_training_fits(model_train_results, train_df, models_with_results)
    prob_matrix, ranked_models, model_names_in_prob = calculate_model_probabilities(model_train_results, models_with_results)
    if prob_matrix is None: print("Probability calculation failed."); model_names_in_prob = []
    perform_hindcast(test_df, model_train_results, model_names_in_prob, models_to_fit)
    future_forecasts, model_eur_stats = forecast_future(last_day, DAYS_FUTURE, model_train_results, model_names_in_prob, models_to_fit, last_day_cum_oil)
    if future_forecasts and model_eur_stats and model_names_in_prob and prob_matrix is not None:
        df_eur_results = generate_combined_results(future_forecasts, prob_matrix, model_names_in_prob, model_eur_stats, last_day_cum_oil, STUDY_WELL)
        if df_eur_results is not None: print("\n--- Final EUR Summary ---"); print(df_eur_results);
        try: eur_results_path=RESULTS_DIR/f"{STUDY_WELL}_eur_summary.csv"; df_eur_results.to_csv(eur_results_path,index=False); print(f"Saved EUR summary: {eur_results_path}")
        except Exception as e: print(f"Error saving EUR summary: {e}")
    else: print("Skipping combined results.")
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
