
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import time

from multiprocessing import Value
from threading import Thread


######################################################################
# “parallel_worker”
######################################################################
def parallel_worker(
    i,
    col,
    model_class,
    sample_df,
    seed,
    n_inits,
    num_trials,
    use_shared_p50_init,
    shared_initial_guess,
    sse_threshold,
    min_improvement_frac,
    counter
):
    from probabilistic_dca.my_dca_models.fitting import fit_single_sample  # safe local import

    result = fit_single_sample(
        i, col, model_class, sample_df, seed,
        n_inits, num_trials,
        use_shared_p50_init,
        shared_initial_guess,
        sse_threshold,
        min_improvement_frac
    )
    with counter.get_lock():
        counter.value += 1

    return result

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
    sse_threshold,   # good “enough” SSE
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
            # 1) SSE is already "good enough"
            if best_sse < sse_threshold:
                break

            # 2) If improvement is tiny, no need to keep trying
            # if improvement < min_improvement_frac:
            #     break
            elif best_sse < (sse_threshold * 1.5) and improvement < min_improvement_frac:
                # We’re close enough that if improvement is tiny, we probably won't do much better
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

    return col, best_params, best_sse, best_pred, solver_used


# def fit_single_sample(
#     sample_index,
#     sample_col,
#     model_class,
#     sample_df,
#     seed,
#     n_inits,
#     num_trials,
#     use_shared_p50_init,
#     shared_initial_guess,
#     sse_threshold,
#     min_improvement_frac,
# ):
#     """
#     Fit a decline curve model to a single Monte Carlo sample.

#     Parameters:
#     ----------
#     sample_index : int
#         Index of the sample.
#     sample_col : str
#         Column name of the sample in sample_df.
#     model_class : class
#         Decline model class (ArpsModel, SEMModel, etc.).
#     sample_df : pd.DataFrame
#         DataFrame containing the sample data.
#     seed : int
#         Random seed for reproducibility.
#     n_inits : int
#         Number of initializations.
#     num_trials : int
#         Number of trials for parameter initialization.
#     use_shared_p50_init : bool
#         Whether to use a shared P50 curve initialization.
#     shared_initial_guess : dict or None
#         Shared initial guess parameters if using shared init.
#     sse_threshold : float
#         Error threshold to accept the fit.
#     min_improvement_frac : float
#         Minimum relative improvement for early stopping.

#     Returns:
#     -------
#     tuple
#         (sample_col, best_params, best_sse, best_prediction, solver_used)
#     """

#     # Extract data for this sample
#     t_data = sample_df['x'].values
#     q_data = sample_df[sample_col].values
#     var_data = sample_df['sigma2'].values

#     best_params = None
#     best_sse = np.inf
#     best_prediction = None
#     solver_used = "None"

#     # Initialize model
#     model = model_class()

#     # Multi-start fitting loop
#     for init_num in range(n_inits):
#         # Seed for reproducibility
#         trial_seed = None if seed is None else seed + init_num

#         # Generate initial guess
#         try:
#             if use_shared_p50_init and shared_initial_guess is not None:
#                 initial_guess = shared_initial_guess
#             else:
#                 initial_guess = model.initialize_parameters(
#                     t_data=t_data,
#                     q_data=q_data,
#                     var_data=var_data,
#                     seed=trial_seed,
#                     num_trials=num_trials,
#                 )
#         except Exception as e:
#             # Failed to generate initial guess, skip this trial
#             continue

#         # Perform optimization
#         try:
#             fitted_params, sse, solver = model.fit(
#                 t_data=t_data,
#                 q_data=q_data,
#                 var_data=var_data,
#                 initial_guess=initial_guess,
#                 sse_threshold=sse_threshold,
#                 min_improvement_frac=min_improvement_frac,
#                 seed=trial_seed,
#             )
#         except Exception as e:
#             # If optimization fails, skip this trial
#             continue

#         # Check if this is the best so far
#         if sse < best_sse:
#             best_sse = sse
#             best_params = fitted_params
#             best_prediction = model.predict(t_data, fitted_params)
#             solver_used = solver

#             # Early stopping if sse is very low
#             if best_sse <= sse_threshold:
#                 break

#     # If no successful fit, return NaNs
#     if best_params is None:
#         param_dim = model.param_dim
#         best_params = np.full(param_dim, np.nan)
#         best_sse = np.nan
#         best_prediction = np.full_like(q_data, np.nan)
#         solver_used = "None"

#     return (
#         sample_col,
#         best_params,
#         best_sse,
#         best_prediction,
#         solver_used,
#     )
    
######################################################################
# “fit_model_for_samples_mstart_para”
######################################################################

# @timing_decorator
# def fit_model_for_samples_mstart_para(
#     model_class,
#     sample_df,
#     seed=None,
#     n_inits=10,
#     num_trials=5,
#     use_shared_p50_init=False,
#     n_jobs=-1,
#     sse_threshold=250.0,
#     min_improvement_frac=0.01,
# ):
#     """
#     Parallel version: Fits a decline model to each Monte Carlo sample using multiple initializations.
#     Includes solver success logging and runtime timing.
#     """
#     sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]
#     rng = np.random.default_rng(seed)

#     # Optional shared P50 initial guess
#     shared_initial_guess = None
#     if use_shared_p50_init:
#         sample_values = sample_df[sample_cols].values
#         p50_curve = np.nanpercentile(sample_values, 50, axis=1)
#         model_tmp = model_class()
#         shared_initial_guess = model_tmp.initialize_parameters(
#             t_data=sample_df['x'].values,
#             q_data=p50_curve,
#             var_data=sample_df['sigma2'].values,
#             seed=seed,
#             num_trials=num_trials
#         )
#         print(f"[Shared P50 Init Guess] {shared_initial_guess}")

#     parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)

#     results = parallel(
#         delayed(fit_single_sample)(
#             i, col, model_class, sample_df, seed,
#             n_inits, num_trials, 
#             use_shared_p50_init, 
#             shared_initial_guess,
#             sse_threshold,
#             min_improvement_frac            
#         )
#         for i, col in enumerate(tqdm(sample_cols, desc="Fitting samples (parallel)"))
#     )

#     # Collect results
#     param_list, sse_list, predictions, solver_list = [], [], {}, []
#     for col, params, sse, pred, solver in results:
#         param_list.append(params)
#         sse_list.append(sse)
#         predictions[col] = pred
#         solver_list.append(solver)

#     df_params = pd.DataFrame(param_list, index=sample_cols)

#     # ✅ Solver success summary
#     solver_counts = pd.Series(solver_list).value_counts(normalize=True) * 100
#     print("\n✅ Solver Success Summary:")
#     for solver, pct in solver_counts.items():
#         print(f"   {solver}: {pct:.1f}%")

#     return {
#         'params': df_params,
#         'sse': np.array(sse_list),
#         'predictions': predictions,
#         'solver_used': solver_list
#     }


# @timing_decorator
# def fit_model_for_samples_mstart_para(
#     model_class,
#     sample_df,
#     seed=None,
#     n_inits=10,
#     num_trials=5,
#     use_shared_p50_init=False,
#     n_jobs=-1,
#     sse_threshold=250.0,
#     min_improvement_frac=0.01,
#     chunk_size=100,  # ✅ Add chunking parameter
#     status_placeholder=None,  # ✅ For UI updates (optional)
# ):
#     """
#     Parallel version: Fits a decline model to each Monte Carlo sample using multiple initializations.
#     Includes solver success logging, runtime timing, and optional chunked parallelization.
#     """
#     sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]
#     rng = np.random.default_rng(seed)

#     # Optional shared P50 initial guess
#     shared_initial_guess = None
#     if use_shared_p50_init:
#         sample_values = sample_df[sample_cols].values
#         p50_curve = np.nanpercentile(sample_values, 50, axis=1)
#         model_tmp = model_class()
#         shared_initial_guess = model_tmp.initialize_parameters(
#             t_data=sample_df['x'].values,
#             q_data=p50_curve,
#             var_data=sample_df['sigma2'].values,
#             seed=seed,
#             num_trials=num_trials
#         )
#         print(f"[Shared P50 Init Guess] {shared_initial_guess}")

#     total_samples = len(sample_cols)
#     completed_samples = 0

#     # Prepare tasks in chunks
#     def chunks(lst, n):
#         """Yield successive n-sized chunks from lst."""
#         for i in range(0, len(lst), n):
#             yield lst[i:i + n]

#     results = []
#     parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)

#     for chunk in tqdm(list(chunks(list(enumerate(sample_cols)), chunk_size)), desc="Chunked Fitting"):
#         chunk_results = parallel(
#             delayed(fit_single_sample)(
#                 i, col, model_class, sample_df, seed,
#                 n_inits, num_trials,
#                 use_shared_p50_init,
#                 shared_initial_guess,
#                 sse_threshold,
#                 min_improvement_frac
#             )
#             for i, col in chunk
#         )
#         results.extend(chunk_results)

#         # ✅ Update progress bar
#         completed_samples += len(chunk)
#         if status_placeholder:
#             status_placeholder.info(f"🔄 Fitting progress: {completed_samples}/{total_samples} samples")

#     # Collect results
#     param_list, sse_list, predictions, solver_list = [], [], {}, []
#     for col, params, sse, pred, solver in results:
#         param_list.append(params)
#         sse_list.append(sse)
#         predictions[col] = pred
#         solver_list.append(solver)

#     df_params = pd.DataFrame(param_list, index=sample_cols)

#     # ✅ Solver success summary
#     solver_counts = pd.Series(solver_list).value_counts(normalize=True) * 100
#     print("\n✅ Solver Success Summary:")
#     for solver, pct in solver_counts.items():
#         print(f"   {solver}: {pct:.1f}%")

#     return {
#         'params': df_params,
#         'sse': np.array(sse_list),
#         'predictions': predictions,
#         'solver_used': solver_list
#     }


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
    param_list, sse_list, predictions, solver_list = [], [], {}, []
    for col, params, sse, pred, solver in results:
        param_list.append(params)
        sse_list.append(sse)
        predictions[col] = pred
        solver_list.append(solver)

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
        'solver_used': solver_list
    }
    
######################################################################
# “fit_model_for_samples”
######################################################################

def fit_model_for_samples(model_class, sample_df, seed=None):
    """
    Fits a decline model to each Monte Carlo sample using the same initial parameters.
    
    model_class: a class that inherits from BaseDeclineModel (not an instance)   
    sample_df: must have columns ['x','sigma2','sample_1'..'sample_n'] 
    seed: optional, for reproducible parameter initialization

    Returns:
    A dictionary with fitted parameters, SSE values, and predictions.    
    """
    
    sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]
    t_data = sample_df['x'].values
    var_data = sample_df['sigma2'].values

    predictions = {}
    param_list = []
    sse_list = []

    # --------------------------------------------------------
    # Compute the P50 curve across all sample_i columns
    # --------------------------------------------------------
    sample_values = sample_df[sample_cols].values  # shape: (n_timesteps, n_samples)
    p50_curve = np.nanpercentile(sample_values, 50, axis=1)

    # --------------------------------------------------------
    # Use the P50 curve to generate the shared initial guess
    # --------------------------------------------------------
    model_tmp = model_class()
    shared_initial_guess = model_tmp.initialize_parameters(
        t_data=t_data, q_data=p50_curve, var_data=var_data, seed=seed
    )
    print(f"[Init Guess from P50 Curve] {shared_initial_guess}")
    
    # --------------------------------------------------------
    # Fit each sample using the same initial guess
    # --------------------------------------------------------

    with tqdm(sample_cols, desc="Fitting samples") as progress:
        for i, col in enumerate(progress):
            q_data = sample_df[col].values
            
            # New instance for each sample
            model = model_class()
            model._initial_guess = shared_initial_guess  # reuse same initial guess
            
            # # initialize random starting point (use per-sample seed for reproducibility)
            # sample_seed = seed + i if seed is not None else None          
            # model._initial_guess = model.initialize_parameters(
            #     t_data=t_data, q_data=q_data, var_data=var_data, seed=sample_seed)            
               
            try:
                # fit (use the same instance)
                best_params = model.fit(t_data, q_data, var_data, sample_id=col)
                # predict
                q_pred = model.predict(t_data)
                # compute SSE
                resid = (q_data - q_pred)
                sse = np.sum((resid**2)/(var_data + 1e-12))
                sse_list.append(sse)
                param_list.append(best_params)
                predictions[col] = q_pred
            except Exception as e:
                print(f"Fit failed for {col} : {e}")
                sse_list.append(1e15)
                #param_list.append([np.nan]*len(model._initial_guess))
                param_list.append([np.nan]*len(model._bounds))
                predictions[col] = np.full_like(q_data, np.nan)
    
    # Ensure missing fits are handled
    param_list = [p if p is not None else [np.nan] * len(model._bounds) for p in param_list]
    df_params = pd.DataFrame(param_list, index=sample_cols)
    
    return {
        'params': df_params,
        'sse': np.array(sse_list),
        'predictions': predictions
    }

######################################################################
# “fit_model_for_samples_mstart”
######################################################################

def fit_model_for_samples_mstart(
    model_class,
    sample_df,
    seed=None,
    n_inits=3,
    use_shared_p50_init=False
):
    """
    Fits a decline model to each Monte Carlo sample using multiple random initializations.
    """

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    sample_cols = [c for c in sample_df.columns if c.startswith('sample_')]
    t_data = sample_df['x'].values
    var_data = sample_df['sigma2'].values

    predictions = {}
    param_list = []
    sse_list = []

    rng = np.random.default_rng(seed)

    # Optional: shared P50 initialization
    if use_shared_p50_init:
        sample_values = sample_df[sample_cols].values
        p50_curve = np.nanpercentile(sample_values, 50, axis=1)
        model_tmp = model_class()
        shared_initial_guess = model_tmp.initialize_parameters(
            t_data=t_data,
            q_data=p50_curve,
            var_data=var_data,
            seed=seed
        )
        print(f"[Shared P50 Init Guess] {shared_initial_guess}")
    else:
        shared_initial_guess = None

    with tqdm(sample_cols, desc="Fitting samples") as progress:
        for i, col in enumerate(progress):
            q_data = sample_df[col].values

            best_sse = np.inf
            best_params = None
            best_pred = None

            for attempt in range(n_inits):
                model = model_class()

                if use_shared_p50_init and shared_initial_guess is not None:
                    init_guess = [
                        val + (0.1 * val * rng.uniform(-0.5, 0.5)) for val in shared_initial_guess
                    ]
                else:
                    sample_seed = (seed + i * 100 + attempt) if seed is not None else None
                    init_guess = model.initialize_parameters(
                        t_data=t_data,
                        q_data=q_data,
                        var_data=var_data,
                        seed=sample_seed,
                        num_trials=3  # Keep internal trials reasonable
                    )

                model._initial_guess = init_guess

                try:
                    fit_params = model.fit(t_data, q_data, var_data, sample_id=col)
                    if fit_params is None:
                        continue

                    q_pred = model.predict(t_data)
                    resid = q_data - q_pred
                    sse = np.sum((resid**2) / (var_data + 1e-12))

                    if sse < best_sse:
                        best_sse = sse
                        best_params = fit_params
                        best_pred = q_pred

                except Exception as e:
                    # Continue to next attempt if current fails
                    print(f"Attempt failed for sample {col} (try {attempt}): {e}")
                    continue

            # Handle final best result
            if best_params is None:
                best_params = [np.nan] * len(model._bounds)
                best_sse = 1e15
                best_pred = np.full_like(q_data, np.nan)

            param_list.append(best_params)
            sse_list.append(best_sse)
            predictions[col] = best_pred

    df_params = pd.DataFrame(param_list, index=sample_cols)

    return {
        'params': df_params,
        'sse': np.array(sse_list),
        'predictions': predictions
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
