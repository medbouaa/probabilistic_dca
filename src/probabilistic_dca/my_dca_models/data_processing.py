import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt

from probabilistic_dca.my_dca_models.plotting import plot_data_processing
from probabilistic_dca.my_dca_models.plotting import plot_crossval_loess
from probabilistic_dca.my_dca_models.plotting import plot_rolling_std
from probabilistic_dca.my_dca_models.plotting import plot_sample_sorted_datasets

from probabilistic_dca.config import TRAIN_PCT

######################################################################
# data pre-processing
######################################################################

def data_processing(prod_df=None, train_pct=TRAIN_PCT,  frac_value=0.4, plotting=False):
      
    if prod_df is None:
        raise ValueError("A valid dataframe is required.")
    
    # Filter decline data (only from maximum rate)
    max_prod = round(prod_df['y'].max())  # max production rate (start of decline)
    max_prod_i = prod_df['y'].idxmax()
    max_prod_day = prod_df.loc[max_prod_i, 'x']  # production time at which max rate is reached
    max_rate_cum_oil = prod_df.loc[max_prod_i, 'z']  # cum oil prior to reaching max rate
    
    # Determine last production date and relevant statistics
    last_day = prod_df['x'].max()  # last production date for match (end of decline)
    last_day_i = prod_df['x'].idxmax()
    #last_prod = round(prod_df.loc[last_day_i, 'y'])
    last_day_cum_oil = round(prod_df.loc[last_day_i, 'z'], 1)

    # Determine training set boundary
    x_train_i = round((last_day_i - max_prod_i) * train_pct)  # percentage of data to match/train
    #x_train = round(prod_df.loc[max_prod_i + x_train_i, 'x'])
    #train_prod = round(prod_df.loc[max_prod_i + x_train_i, 'y'])
    #train_cum_oil = round(prod_df.loc[max_prod_i + x_train_i, 'z'], 1)

    # Filter dataset for decline analysis
    prod_df = prod_df[prod_df['x'] >= max_prod_day]
    
    models_df = pd.DataFrame({'x': prod_df['x'], 'y': prod_df['y']})
    
    # If user wants a plot, call the separate function
    if plotting:
        fig = plot_data_processing(
            models_df=models_df,
            max_prod_day=max_prod_day,
            max_prod=max_prod,
            last_day=last_day,
            last_day_cum_oil=last_day_cum_oil,
            frac_value=frac_value
        )
        # either show it or return it
        plt.show()
    
    return last_day, last_day_cum_oil, x_train_i, models_df
    

######################################################################
# “optimum_half_window” - Cross Validation for local smoothing
######################################################################

def crossval_loess(dataframe=None, k_folds=10, window_range=None, plotting=False):
    """
    This is a rough Python approximation of your optimum_half_window() logic.
    Instead of caret + repeated CV, we do a simple KFold cross-validation
    over a set of “spans” or “frac” for LOWESS.
    
    - x, y: data arrays
    - k_folds: e.g. 10
    - window_range: e.g. [3..8]
    - returns best_halfwindow, best_span, residuals
    """
    if dataframe is None:
        raise ValueError("A valid dataframe is required.")

    x = dataframe['x'].values
    y = dataframe['y'].values    
    
    if window_range is None:
        window_range = range(3, 9)  # or up to 8 as in your R code
    
    # We will map each window to a “span” = (2*w+1)/len(x)
    # Then run KFold CV, measuring MSE.
    
    best_span = None
    best_score = 1e15
    best_w = None
    
    X = np.array(x)
    Y = np.array(y)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=123)
    
    # We store (w, span, mean_mse) in a list
    candidate_results = []
    
    N = len(X)
    for w in window_range:
        span = (2*w + 1)/N
        # We do a simple K-fold: For each fold, we fit a “lowess” on train, measure MSE on test
        fold_mses = []
        for train_idx, test_idx in kf.split(X):
            x_train, y_train = X[train_idx], Y[train_idx]
            x_test, y_test   = X[test_idx], Y[test_idx]
            
            # Fit a lowess curve on train
            # statsmodels.lowess can’t do “training vs test” directly, so we do:
            # We can just do the entire dataset, ignoring that it’s train. 
            # A more faithful approach is a custom local regression on (x_train, y_train) only.
            # For demonstration, I'll do the entire set minus test. 
            
            # Sort data for lowess
            order = np.argsort(x_train)
            x_sorted = x_train[order]
            y_sorted = y_train[order]
            
            # Actually fit lowess on train
            fitted = lowess(endog=y_sorted, exog=x_sorted, frac=span, return_sorted=True)
            
            # Interpolate to predict test
            # fitted[:,0] = sorted x, fitted[:,1] = predicted y
            y_pred_test = np.interp(x_test, fitted[:,0], fitted[:,1])
            
            fold_mses.append(mean_squared_error(y_test, y_pred_test))
        
        mean_mse = np.mean(fold_mses)
        candidate_results.append((w, span, mean_mse))
        if mean_mse < best_score:
            best_score = mean_mse
            best_span = span
            best_w = w
    
    # Having found best_span, best_w, we do final lowess on all data:
    order = np.argsort(X)
    X_sorted = X[order]
    Y_sorted = Y[order]
    final_fit = lowess(endog=Y_sorted, exog=X_sorted, frac=best_span, return_sorted=True)
    # Compute residuals
    # Interpolate for each x
    Y_smooth = np.interp(X, final_fit[:,0], final_fit[:,1])
    dataframe["model_residuals"] = Y - Y_smooth

    # If user wants a plot, call the separate function
    if plotting:
        fig = plot_crossval_loess(
            candidate_results=candidate_results,
        )
        # either show it or return it
        plt.show()
    
    return best_w, best_span, dataframe


#####################################################################
# “sigma_k_calc” - Rolling/moving window stdev of residuals
#####################################################################

def rolling_std(dataframe, half_window, plotting=False):
    """
    Approximate your `sigma_k_calc` function:
    - half_window: integer
    - returns an array of rolling stdev with window=2*half_window+1
    """
    roll_sd = []
    #arr = np.array(residuals)
    arr = dataframe['model_residuals'].values
    N = len(arr)
    window_size = 2*half_window + 1
    
    for i in range(N):
        left = max(0, i-half_window)
        right = min(N, i+half_window+1)  # +1 because slice excludes end
        chunk = arr[left:right]
        sd = np.std(chunk, ddof=1) if len(chunk)>1 else 1e-9
        roll_sd.append(sd)
    
    roll_sd = np.array(roll_sd)
    
    # Compute rolling variance
    roll_var = roll_sd ** 2  # Variance
    
    # Assign values back to dataframe
    dataframe['roll_sd'] = roll_sd
    dataframe['roll_sigma2'] = roll_var
        
    # If user wants a plot, call the separate function
    if plotting:
        fig = plot_rolling_std(
            dataframe=dataframe,
        )
        # either show it or return it
        plt.show()

    return dataframe


######################################################################
# 4) “sample_sorted_datasets”
######################################################################

def sample_sorted_datasets(dataframe, n_samples=1000, seed=123, plotting=False):
    """
    For each data point i, draw 'n_samples' from N(mean = y[i], std = sd[i]),
    sort them in descending order, and store them in row i.
    This yields a final DataFrame with shape (N, n_samples):
      rows = time steps,
      columns = sample realizations.

    R-style approach:
      - data.table: row = time i, col = sample_j
      - each row i is the set of sorted draws for day i
    """
    rng = np.random.default_rng(seed)

    # Pull arrays for convenience
    x = dataframe['x'].values
    y = dataframe['y'].values
    sd = dataframe['roll_sd'].values  # or whatever column holds stdev
    N = len(x)

    # Create an array of shape (N, n_samples)
    # Each row i => all samples for that time step
    sampled_matrix = np.zeros((N, n_samples))

    for i in range(N):
        # Draw from Normal
        draws = rng.normal(loc=y[i], scale=sd[i], size=n_samples)
        # Sort descending, just like R: sort(day_sample, decreasing=TRUE)
        draws_sorted = np.sort(draws)[::-1]
        # Place them in row i
        sampled_matrix[i, :] = draws_sorted

    # Now build a DataFrame with columns = [sample_1, sample_2, ..., sample_n]
    col_names = [f"sample_{j+1}" for j in range(n_samples)]
    df_samples = pd.DataFrame(sampled_matrix, columns=col_names)

    # Optionally, you might want to keep x, y, or sigma in the same DataFrame
    # at the left, so that each row still knows which day it corresponds to:
    df_samples.insert(0, 'x', x)
    df_samples.insert(1, 'y', y)
    df_samples.insert(2, 'sigma2', sd**2)  # or roll_sigma2, etc.

    # Compute Monte Carlo statistics separately
    df_samples_stats = df_samples.copy()
    df_samples_stats['sample_std'] = df_samples.iloc[:, 3:].std(axis=1)
    df_samples_stats['sample_p10'] = df_samples.iloc[:, 3:].apply(lambda x: np.percentile(x, 10), axis=1)
    df_samples_stats['sample_mean'] = df_samples.iloc[:, 3:].mean(axis=1)
    df_samples_stats['sample_p50'] = df_samples.iloc[:, 3:].apply(lambda x: np.percentile(x, 50), axis=1)
    df_samples_stats['sample_p90'] = df_samples.iloc[:, 3:].apply(lambda x: np.percentile(x, 90), axis=1)

    df_samples_stats = df_samples_stats[['x', 'y', 'sigma2', 'sample_p10', 'sample_mean', 'sample_p50', 'sample_p90']]
    
    # If user wants a plot, call the separate function
    fig = None
    if plotting:
        fig = plot_sample_sorted_datasets(
            df_samples=df_samples,
            df_samples_stats=df_samples_stats,
        )
        # either show it or return it
        plt.show()

    # Now df_samples has shape (N, n_samples+3).  Each row is a time step.
    return df_samples, df_samples_stats, fig
