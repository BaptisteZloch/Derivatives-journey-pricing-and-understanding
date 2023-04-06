import numpy as np
import scipy.stats as stats
import numpy.typing as npt
from typing import Literal


def monte_carlo_spot_price(
    s0: float | int,
    r: float | int,
    tau: float | int,
    sigma: float | int,
    nb_simulation: int = 10000,
) -> npt.NDArray:
    """Monte Carlo simulation of the spot price.

    Args:
        s0 (float | int): The initial spot price.
        r (float | int): The risk-free interest rate.
        tau (float | int): The time to maturity.
        sigma (float | int): The volatility.
        nb_simulation (int, optional): The number of simulation to run, could be long. Defaults to 10000.

    Returns:
        npt.NDArray: The simulated spot prices.
    """
    WT = np.random.normal(0.0, np.sqrt(tau), nb_simulation)
    return s0 * np.exp((r - 0.5 * sigma**2) * tau + sigma * WT)


def monte_carlo_option_price(
    s0: float | int,
    k: float | int,
    r: float | int,
    tau: float | int,
    sigma: float | int,
    nb_simulation: int = 10000,
    option_type: Literal["call", "put"] = "call",
) -> tuple[np.floating, np.floating]:
    """Monte Carlo simulation of the option price.

    Args:
        s0 (float | int): The initial spot price.
        k (float | int): The strike price.
        r (float | int): The risk-free interest rate.
        tau (float | int): The time to maturity.
        sigma (float | int): The volatility.
        nb_simulation (int, optional): The number of simulation to run, could be long. Defaults to 10000.
        option_type: Literal["call", "put"]: The type of option to price. Defaults to "call".

    Returns:
        tuple[np.floating, np.floating]: The simulated option price and its standard error.
    """
    st = monte_carlo_spot_price(s0, r, tau, sigma, nb_simulation)
    option_prices: npt.NDArray = (
        np.exp(-r * tau) * np.maximum(st - k, 0)
        if option_type == "call"
        else np.exp(-r * tau) * np.maximum(st - k, 0)
    )
    return np.mean(option_prices), np.std(option_prices) / np.sqrt(nb_simulation)


def black_scholes_option_price(
    s: float | int,
    k: float | int,
    r: float | int,
    tau: float | int,
    sigma: float | int,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes european option pricing model.

    Args:
        s (float | int): The actual spot price.
        k (float | int): The strike price.
        r (float | int): The risk-free interest rate.
        tau (float | int): The time to maturity.
        sigma (float | int): The volatility.
        option_type: Literal["call", "put"]: The type of option to price. Defaults to "call".

    Returns:
        float: The simulated option price.
    """
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * tau**0.5)
    d2 = d1 - sigma * np.sqrt(tau)
    pv = k * np.exp(-r * tau)

    match option_type:
        case "put":
            return pv * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)
        case "call":
            return s * stats.norm.cdf(d1) - pv * stats.norm.cdf(d2)
        case _:
            raise ValueError(f"Unknown option type: {option_type}")


def from_call_to_put(
    call_price: float,
    s: float | int,
    k: float | int,
    r: float | int,
    tau: float | int,
) -> float:
    """Using the put call parity it computes the put price from the call price.

    Args:
        call_price (float): The current call price.
        s (float | int): The actual spot price.
        k (float | int): The strike price.
        r (float | int): The risk-free interest rate.
        tau (float | int): The time to maturity.

    Returns:
        float: The put price.
    """
    return call_price - s + k * np.exp(-r * tau)


def from_put_to_call(
    put_price: float,
    s: float | int,
    k: float | int,
    r: float | int,
    tau: float | int,
) -> float:
    """Using the put call parity it computes the call price from the put price.

    Args:
        put_price (float): The put price.
        s (float | int): The actual spot price.
        k (float | int): The strike price.
        r (float | int): The risk-free interest rate.
        tau (float | int): The time to maturity.

    Returns:
        float: The call price.
    """
    return s - k * np.exp(-r * tau) + put_price
