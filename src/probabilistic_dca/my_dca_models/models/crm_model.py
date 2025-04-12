import numpy as np
from probabilistic_dca.my_dca_models.base_models import BaseDeclineModel

# If you want to handle extremely small times differently or skip them, you can adapt.
# The default _objective = Weighted SSE is typically enough, but you can override if needed.

# CRM Model Predictions
# Arguments:
    #        t: Time since the well started producing
    #        dP: Difference between the initial reservoir pressure and the assumed constant flowing bottomhole pressure
    #        beta: Linear transient flow parameter
    #        J_inf: Constant productivity index that the well will reach at boundary-dominated flow
    #        ct_Vp: Total compressibility * Drainage pore volume


class CRMModel(BaseDeclineModel):
    """
    CRM: q(t) = dP * [ beta/sqrt(t) + J_inf ] * exp( - (2*beta*sqrt(t) + J_inf*t)/(ctVp) )
    params = [dP, beta, J_inf, ctVp]
    """
    def __init__(self, params=None):
        super().__init__(params=params)
        # Typical guess
        #self._initial_guess = [2000.0, 0.5, 0.1, 200.0]  # [dP, beta, J_inf, ctVp]
        # Example bounds; you can adapt:
        self._bounds = [
            (0.001, 5000),    # dP
            (0.01, 1.0),   # beta
            (0.001, 0.3),    # J_inf
            (1.0,   1500)    # ctVp
        ]
        self._initial_guess = None  # Do not initialize here
    
    def _rate_function(self, t, p):
        dP, beta, J_inf, ctVp = p
        # clamp t>=1e-9 or so to avoid sqrt(0)
        t_clamped = np.where(t < 1e-9, 1e-9, t)

        val = dP * ( beta/np.sqrt(t_clamped) + J_inf ) * \
              np.exp( - (2.0 * beta * np.sqrt(t_clamped) + J_inf*t_clamped)/ctVp )
        # clamp negative to small positive
        val = np.maximum(val, 1e-9)
        return val