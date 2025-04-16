import numpy as np
from probabilistic_dca.my_dca_models.base_models import BaseDeclineModel

## LGM Model Predictions
#Arguments:
    #        t: Time since the well started producing
    #        a: A constant
    #        K: Carrying capacity
    #        n: Hyperbolic exponent parameter


class LGMModel(BaseDeclineModel):
    """
    LGM: q(t) = (a*K*n * t^(n-1)) / (a + t^n)^2
    params = [a, K, n]
    """
    def __init__(self, params=None):
        super().__init__(params=params)
        self.name = "lgm"
        # Some typical guess:
        #self._initial_guess = [300.0, 2500000, 0.6]
        # Bounds: a>0, K>0, 0<n<some range
        self._bounds = [
            (5, 350),   # a
            (40000, 5000000),   # K
            (0.5, 1.0)    # n
        ]
        self._initial_guess = None  # Do not initialize here
    
    def _rate_function(self, t, p):
        a, K, n = p
        # clamp t>1e-9 if needed
        eps = 1e-9
        t_clamped = np.maximum(t, eps)
        t_pow = t_clamped**n

        denom = (a + t_pow)**2
        denom = np.maximum(denom, eps)
        # logistic growth formula
        q_val = (a * K * n * t_clamped**(n - 1)) / denom
        return q_val