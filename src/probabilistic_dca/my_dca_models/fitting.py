
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import time

from multiprocessing import Value
from threading import Thread


######################################################################
# “timing_decorator”
######################################################################

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"⏱️ {func.__name__} completed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

######################################################################
# “fit_single_sample”
######################################################################
def fit_single_sample(
    sample_idx,
    col,
    model_class,
    sample_df,
    seed,
    n_inits,               # max number of attempts
    num_trials,
    use_shared_p50_init,
    shared_initial_guess,
    sse_threshold,         # good “enough” SSE
    min_improvement_frac,  # stop if improvement < 1%
):
    """
    Fit a single sample with multiple initializations, with an adaptive early stop
    based on SSE threshold and improvement fraction.
    """
    t_data = sample_df['x'].values
    var_data = sample_df['sigma2'].values
    q_data = sample_df[col].values

    rng = np.random.default_rng(seed + sample_idx if seed is not None else None)

    best_sse = np.inf
    best_params = None
    best_pred = None
    solver_used = None
    early_stop_reason = "max_attempts"  # default

    for attempt in range(n_inits):
        model = model_class()

        # Generate init guess
        if use_shared_p50_init and shared_initial_guess is not None:
            init_guess = [
                val + (0.1 * val * rng.uniform(-0.5, 0.5)) for val in shared_initial_guess
            ]
        else:
            sample_seed = (seed + sample_idx*100 + attempt) if seed is not None else None
            init_guess = model.initialize_parameters(
                t_data=t_data,
                q_data=q_data,
                var_data=var_data,
                seed=sample_seed,
                num_trials=num_trials
            )

        model._initial_guess = init_guess

        try:
            fit_params = model.fit(t_data, q_data, var_data, sample_id=col)
            if fit_params is None:
                continue  # skip if fit returned None

            q_pred = model.predict(t_data)
            resid = q_data - q_pred
            sse = np.sum((resid**2) / (var_data + 1e-12))

            if sse < best_sse:
                improvement = (
                    (best_sse - sse) / best_sse
                    if best_sse < np.inf else 1.0
                )
                best_sse = sse
                best_params = fit_params
                best_pred = q_pred
                solver_used = model.last_solver
            else:
                improvement = 0.0

            # ------ Adaptive Early Break ------
            if best_sse < sse_threshold:
                early_stop_reason = "sse_threshold"
                break
            elif best_sse < (sse_threshold * 1.5) and improvement < min_improvement_frac:
                early_stop_reason = "low_improvement"
                break

        except Exception as e:
            print(f"Sample {col} (attempt={attempt}): {e}")
            continue

    if best_params is None:
        # all attempts failed
        best_params = [np.nan]*len(model._bounds)
        best_sse = 1e15
        best_pred = np.full_like(q_data, np.nan)
        solver_used = "fail"
        early_stop_reason = "fit_failed"

    return col, best_params, best_sse, best_pred, solver_used, early_stop_reason

######################################################################
# “fit_model_for_samples_mstart_para”
######################################################################

@timing_decorator
def fit_model_for_samples_mstart_para(
    model_class,
    sample_df,
    seed=None,
    n_inits=10,
    num_trials=5,
    use_shared_p50_init=False,
    n_jobs=-1,
    sse_threshold=250.0,
    min_improvement_frac=0.01,
):
    """
    Parallel version: Fits a decline model to each Monte Carlo sample using multiple initializations.
    """
    from joblib import Parallel, delayed
    import numpy as np
    import pandas as pd

    sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]
    rng = np.random.default_rng(seed)

    # Optional shared P50 initial guess
    shared_initial_guess = None
    if use_shared_p50_init:
        sample_values = sample_df[sample_cols].values
        p50_curve = np.nanpercentile(sample_values, 50, axis=1)
        model_tmp = model_class()
        shared_initial_guess = model_tmp.initialize_parameters(
            t_data=sample_df['x'].values,
            q_data=p50_curve,
            var_data=sample_df['sigma2'].values,
            seed=seed,
            num_trials=num_trials
        )
        print(f"[Shared P50 Init Guess] {shared_initial_guess}")

    # Parallel execution
    parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)

    results = parallel(
        delayed(fit_single_sample)(
            i, col,
            model_class,
            sample_df,
            seed,
            n_inits,
            num_trials,
            use_shared_p50_init,
            shared_initial_guess,
            sse_threshold,
            min_improvement_frac
        )
        for i, col in enumerate(sample_cols)
    )

    # Collect results
    param_list, sse_list, predictions, solver_list, early_stop_list = [], [], {}, [], []
    for col, params, sse, pred, solver, stop_reason in results:
        param_list.append(params)
        sse_list.append(sse)
        predictions[col] = pred
        solver_list.append(solver)
        early_stop_list.append(stop_reason)

    df_params = pd.DataFrame(param_list, index=sample_cols)

    # ✅ Solver success summary
    solver_counts = pd.Series(solver_list).value_counts(normalize=True) * 100
    print("\n✅ Solver Success Summary:")
    for solver, pct in solver_counts.items():
        print(f"   {solver}: {pct:.1f}%")

    return {
        'params': df_params,
        'sse': np.array(sse_list),
        'predictions': predictions,
        'solver_used': solver_list,
        'early_stop': early_stop_list
    }

######################################################################
# “gather_sse_matrix”
######################################################################
    
def gather_sse_matrix(model_results_dict, model_names):
    """
    model_results_dict: e.g. {
      "arps": {"sse": <array shape (N,)>, "forecast": <array shape (T, N) or (N,)>},
      "sem":  {"sse": <array shape (N,)>, "forecast": ...},
      ...
    }
    model_names: list of strings, e.g. ["arps","sem","crm","lgm"]

    Returns:
      - sse_matrix shape (M, N) 
         (M=#models, N=#samples)
    """
    # Let's find N by checking the first SSE
    M = len(model_names)
    # Gather SSE into a 2D array
    all_sse = []
    for m in model_names:
        sse_array = model_results_dict[m]["sse"]  # shape (N,)
        all_sse.append(sse_array)
    sse_matrix = np.vstack(all_sse)  # shape (M, N)
    return sse_matrix

######################################################################
# “gather_predictions”
######################################################################

# Gather Predictions into a 2D Array
# We want a shape (N, n_samples) array. Each row i is time-step i, each column # j is the model’s predicted rate for sample j.

def gather_predictions(arps_fit_results, sample_df):
    """
    Gathers the predicted rates from each sample’s best-fit model into 
    a single 2D array: pred_matrix[time_i, sample_j].
    
    sample_df: the DataFrame used for training (or forecasting),
               which has the 'x' times (and 'y' if we want to plot actual data).
    """
    predictions_dict = arps_fit_results['predictions']
    
    # Identify which columns are sample_j
    sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]

    # The number of time steps
    N = len(sample_df)
    # The number of samples
    M = len(sample_cols)
    
    # pred_matrix: shape (N, M)
    pred_matrix = np.zeros((N, M))
    
    # Fill in columns from the dictionary
    for j, col in enumerate(sample_cols):
        pred_matrix[:, j] = predictions_dict[col]
    
    return pred_matrix

######################################################################
# “forecast_from_params”
######################################################################

def forecast_from_params(model_class, param_df, t_array):
    """
    Use a class-based model to predict rates for each sample at times 't_array',
    given 'param_df' with best-fit parameters for each sample row.
    
    Returns: forecast_matrix of shape (#times, #samples),
             each column j is the predicted rates for sample_j.
    """
    sample_cols = param_df.index  # e.g. Index(['sample_1','sample_2',...])
    M = len(sample_cols)
    N = len(t_array)

    forecast_matrix = np.zeros((N, M))

    # For each sample, build a model with those parameters
    for j, col in enumerate(sample_cols):
        row_params = param_df.loc[col].values  # e.g. [qi, Di, b, Df]
        model = model_class(params=row_params)  # instantiate with those best-fit params
        # model.predict(t_array)
        q_pred = model.predict(t_array)
        forecast_matrix[:, j] = q_pred

    return forecast_matrix

######################################################################
# “compute_forecast_stats”
######################################################################

# Compute P10, Mean, P90 at Each Time Step
# Once we have the 2D array pred_matrix, we can compute the desired statistics # across the sample dimension (axis=1 for each time step).

def compute_forecast_stats(pred_matrix):
    """
    Given an (N, M) matrix of predictions:
       N = # time steps,
       M = # samples,
    compute p10, mean, p50, p90, etc. along axis=1.
    Returns a dict of { 'p10': ..., 'mean': ..., 'p90': ... } 
    each shape (N,).
    """
    p10 = np.nanpercentile(pred_matrix, 10, axis=1)
    p90 = np.nanpercentile(pred_matrix, 90, axis=1)
    mean = np.nanmean(pred_matrix, axis=1)
    # optionally p50, etc.
    p50  = np.nanmedian(pred_matrix, axis=1)
    
    return {
        'p10':  p10,
        'p50':  p50,
        'mean': mean,
        'p90':  p90
    }
