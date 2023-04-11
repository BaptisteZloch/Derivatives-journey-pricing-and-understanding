from typing import Literal
import numpy as np
import scipy.stats as stats


def vega(
    S,
    K,
    tau,
    r,
    sigma,
    option_type: Literal["call", "put"] = "call",
):
    """Vega of an option.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price.
    tau : float
        Time to maturity.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    call : bool, optional
        True if the option is a call option, False if the option is a put option.

    Returns
    -------
    float
        Vega of the option.
    """
    return (
        S
        * stats.norm.pdf(
            (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * tau**0.5)
        )
        * tau**0.5
    )
