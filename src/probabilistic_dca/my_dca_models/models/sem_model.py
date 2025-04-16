import numpy as np
from probabilistic_dca.my_dca_models.base_models import BaseDeclineModel

# You can adjust the bounds if you want to limit n to ≤ 1 or some other range.
# The default _objective from BaseDeclineModel uses Weighted SSE. If you prefer a different approach, override _objective.

# SEM Model Predictions
  #    Arguments:
  #        t: Time since the well started producing
  #        qi: Initial production rate
  #        tau: Characteristic time parameter
  #        n: Exponent parameter


class SEMModel(BaseDeclineModel):
    """
    SEM: q(t) = qi * exp( - ( t / Tau )^n )
    params = [qi, Tau, n]
    """
    def __init__(self, params=None):
        super().__init__(params=params)
        self.name = "sem"
        # Typical guesses, tune as needed:
        # self._initial_guess = [1000.0, 100.0, 1.0]  
        # Bounds: qi>0, Tau>0, n>0 
        self._bounds = [
            (1e-6, 10000),  # qi
            (1e-6, 100),  # Tau
            (1e-6, 1.0)   # n
        ]
        self._initial_guess = None  # Do not initialize here

    def _rate_function(self, t, p):
        qi, Tau, n = p
        # SEM formula
        return qi * np.exp(-((t / Tau)**n))
