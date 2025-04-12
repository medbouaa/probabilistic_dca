import numpy as np
import scipy.optimize as opt
from abc import ABC, abstractmethod
    
class BaseDeclineModel:
    """
    Base class for decline-curve models.
    Subclasses typically override:
      - self._initial_guess
      - self._bounds
      - self._rate_function(t, params)
      - optional self._objective(...) if needed
    Then implement fit(t_data, q_data, var_data) & predict(t).
    """

    def __init__(self, params=None):
        self.params = params  # best-fit parameters or None
        self._initial_guess = None
        self._bounds = None

    def initialize_parameters(self, num_trials=5, t_data=None, q_data=None, var_data=None, seed=None):
        """
        Hybrid approach:
        1. Generates multiple random starts.
        2. Evaluates the `_objective` function for each.
        3. Selects best initialization.
        """
        if self._bounds is None:
            raise ValueError("Parameter bounds are not defined in the model.")

        if t_data is None or q_data is None:
            raise ValueError("Must provide t_data and q_data for parameter initialization.")

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        best_params = None
        best_loss = np.inf

        # for _ in range(num_trials):
        #     # Generate random initial parameters within bounds
        #     initial_guess = [rng.uniform(low, high) for low, high in self._bounds]
        
        for _ in range(num_trials):
            initial_guess = []
            for low, high in self._bounds:
                if low > 0 and high / low > 100:
                    initial_guess.append(np.exp(rng.uniform(np.log(low), np.log(high))))
                else:
                    initial_guess.append(rng.uniform(low, high))
            
            # Evaluate loss using the `_objective` function
            loss = self._objective(initial_guess, t_data, q_data, var_data)

            if loss < best_loss:
                best_loss = loss
                best_params = initial_guess

        return best_params

    @abstractmethod
    def _rate_function(self, t, p):
        """ Subclasses must define the decline curve function """
        # raise NotImplementedError("Subclasses must implement _rate_function")
        pass

    def _objective(self, p, t_data, q_data, var_data):
        """
        Computes weighted SSE (Sum of Squared Errors).
        Default: (q_data - q_model)^2 / var_data.
        Subclasses can override if needed.
        """
        q_model = self._rate_function(t_data, p)
        if var_data is None:
            var_data = np.ones_like(q_data)
        resid = q_data - q_model
        return np.sum((resid**2)/(var_data + 1e-12)) # Prevent division by zero

    def fit(self, t_data, q_data, var_data=None, sample_id=None):
        """
        Minimizes self._objective(...) using L-BFGS-B or similar.
        Fit the model to observed production data.
        Uses Trust-Region Constrained optimization by default.
        
        Parameters:
        - t_data: Time data
        - q_data: Production data
        - var_data: Optional variance data
        - sample_id: Optional identifier for status reporting        
        """
        # if self._initial_guess is None or self._bounds is None:
        #     raise ValueError("Subclass must define _initial_guess and _bounds.")

        # if sample_id is not None:
        #     print(f"Fitting sample {sample_id}...")
        
        self.last_solver = None  # ✅ Track solver

        if self._initial_guess is None:
            self._initial_guess = self.initialize_parameters(
                t_data=t_data, q_data=q_data, var_data=var_data
            )

        def objective_wrapper(p):
            return self._objective(p, t_data, q_data, var_data)

        try:
            res = opt.minimize(
                objective_wrapper,
                self._initial_guess,
                method='trust-constr',
                options={"maxiter": 2000, "verbose": 0},
                bounds=self._bounds
            )

            if res.success:
                self.params = res.x
                self.last_solver = 'trust-constr'
                return self.params

            # Fallback solver
            res = opt.minimize(
                objective_wrapper,
                self._initial_guess,
                method='L-BFGS-B',
                options={"maxiter": 2000, "maxfun": 10000, "disp": False},
                bounds=self._bounds
            )

            if res.success:
                self.params = res.x
                self.last_solver = 'L-BFGS-B'
                return self.params

            print(f"Warning: Fit failed for sample {sample_id}: {res.message}")
            return None

        except Exception as e:
            print(f"Exception during fitting sample {sample_id}: {e}")
            return None

    def predict(self, t):
        """
        Evaluate the model rate at times t using self.params
        Predict production rate q(t) using fitted parameters.
        """
        if self.params is None:
            raise ValueError("Must call fit(...) before predict(...)")
        return self._rate_function(t, self.params)
    
