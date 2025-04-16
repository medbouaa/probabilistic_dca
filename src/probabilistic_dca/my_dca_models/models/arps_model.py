import numpy as np
from probabilistic_dca.my_dca_models.base_models import BaseDeclineModel

# Arps Hyperbolic to Exponential Model Predictions
        # Hyperbolic decline curve equation:
        #  Arguments:
        #    t: Time since the well started producing
        #    qi: Initial production rate
        #    b: Hyperbolic decline constant
        #    Di: Nominal decline rate (constant)

class ArpsModel(BaseDeclineModel):
    """
    Arps piecewise hyperbolic->exponential decline model.
    params = [qi, Di, b, Df]
    """
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "arps"
        # typical guesses, tune as needed
        #self._initial_guess = [1000.0, 0.02, 0.5, 0.000288]
        self._bounds = [
            (1e-6, 5000),    # qi
            (0.000288, 0.02711),   # Di
            (0.0,   2.0),   # b
            #(0.0001404, 0.0002283)    # Df, 0.0001404, 0.0002283
        ]
        self._initial_guess = None  # Do not initialize here

    def _rate_function(self, t, p):
        # piecewise hyperbolic->exponential
        #def _rate_function(self, t, p)
        qi, Di, b = p
        
        Df = 0.000288 # 10% annual effective decline rate, constant
        
        def hyperbolic_q(t_):
            base = 1.0 + b * Di * t_
            base = np.where(base <= 0, 1e-8, base)  # Prevent invalid values
            return qi * base ** (-1.0 / b)

        def exponential_q(qt, Df_, dt):
            return qt * np.exp(-Df_*dt)

        # transition
        eps = 1e-12
        if b>eps and Di>eps and Df>eps:
            t_trans = (Di/Df - 1.0)/(b*Di)
        else:
            t_trans = np.inf
        q_trans = hyperbolic_q(t_trans) if t_trans < 1e12 else qi
        
        out = []
        for val in t:
            if val <= t_trans:
                out.append(hyperbolic_q(val))
            else:
                out.append(exponential_q(q_trans, Df, val - t_trans))
        return np.array(out)