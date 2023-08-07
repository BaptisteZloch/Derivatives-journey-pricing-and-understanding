import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Literal, Union, Optional
from datetime import datetime


def generate_ABM_asset_paths(
    n_paths: int,
    T: Union[float, int],
    mu: Union[float, int],
    sigma: Union[float, int],
    s0: Union[float, int],
    average_path: bool = True,
) -> pd.DataFrame:
    """Generate some Arithmetic Brownian Motion paths.

    Args:
    ----
        n_paths (int): Number of paths to generate.
        T (Union[float, int]): The period in years.
        mu (Union[float, int]): The drift term (expected return annualized)
        sigma (Union[float, int]): The volatility term (standard deviation annualized).
        s0 (Union[float, int]): The initial condition (starting point)
        average_path (bool, optional): Whether or not to average all the paths. Defaults to True.

    Returns:
    ----
        pd.DataFrame: The dataframe containing the paths.
    """
    assert sigma > 0, "Error, sigma must be positive."
    n_steps = int(365 * T)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = (mu - 0.5 * (sigma**2)) * dt + sigma * np.random.normal(
        0, (dt**0.5), (n_paths, n_steps)
    )
    paths = np.cumsum(paths, axis=1)
    return __from_path_to_dataframe(paths, average_path)


def generate_Heston_ABM_asset_paths(
    n_paths: int,
    T: Union[float, int],
    mu: Union[float, int],
    s0: Union[float, int],
    kappa: Union[float, int] = 2,
    theta: Union[float, int] = 0.015,
    eta: Union[float, int] = 0.15,
    average_path: bool = True,
) -> pd.DataFrame:
    """Generate Arithmetic Brownian Motion paths using a stochastic volatility (modeled as a mean reverting process)

    Args:
    ----
        n_paths (int): Number of paths to generate.
        T (Union[float, int]): The period in years.
        mu (Union[float, int]): The drift term (expected return annualized)
        sigma (Union[float, int]): The volatility term (standard deviation annualized).
        s0 (Union[float, int]): The initial condition (starting point)
        kappa (Union[float, int], optional): The mean reverting speed. Defaults to 2.
        theta (Union[float, int], optional): The long term volatility mean. Defaults to 0.015.
        eta (Union[float, int], optional): The volatility of volatility. Defaults to 0.15.
        average_path (bool, optional): Whether or not to average all the paths. Defaults to True.

    Returns:
    ----
        pd.DataFrame: The dataframe containing the paths.
    """
    n_steps = int(365 * T)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    vol = __generate_heston_volatility_paths(
        n_paths=n_paths, T=T, kappa=kappa, theta=theta, eta=eta, s0=0.25**2
    )
    paths[:, 1:] = (mu - 0.5 * (vol[:, 1:] ** 2)) * dt + vol[:, 1:] * np.random.normal(
        0, (dt**0.5), (n_paths, n_steps)
    )
    paths = np.cumsum(paths, axis=1)
    return __from_path_to_dataframe(paths, average_path)


def generate_Heston_GBM_asset_paths(
    n_paths: int,
    T: Union[float, int],
    mu: Union[float, int],
    s0: Union[float, int],
    kappa: Union[float, int] = 2,
    theta: Union[float, int] = 0.015,
    eta: Union[float, int] = 0.15,
    average_path: bool = True,
) -> pd.DataFrame:
    """Generate Geometric Brownian Motion paths using a stochastic volatility (modeled as a mean reverting process)

    Args:
    ----
        n_paths (int): Number of paths to generate.
        T (Union[float, int]): The period in years.
        mu (Union[float, int]): The drift term (expected return annualized)
        sigma (Union[float, int]): The volatility term (standard deviation annualized).
        s0 (Union[float, int]): The initial condition (starting point)
        kappa (Union[float, int], optional): The mean reverting speed. Defaults to 2.
        theta (Union[float, int], optional): The long term volatility mean. Defaults to 0.015.
        eta (Union[float, int], optional): The volatility of volatility. Defaults to 0.15.
        average_path (bool, optional): Whether or not to average all the paths. Defaults to True.

    Returns:
    ----
        pd.DataFrame: The dataframe containing the paths.
    """
    n_steps = int(365 * T)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.log(s0)
    vol = __generate_heston_volatility_paths(
        n_paths=n_paths, T=T, kappa=kappa, theta=theta, eta=eta, s0=0.25**2
    )
    paths[:, 1:] = (mu - 0.5 * (vol[:, 1:] ** 2)) * dt + vol[:, 1:] * np.random.normal(
        0, (dt**0.5), (n_paths, n_steps)
    )
    paths = np.cumprod(np.exp(paths), axis=1)

    return __from_path_to_dataframe(paths, average_path)


def generate_GBM_asset_paths(
    n_paths: int,
    T: Union[float, int],
    mu: Union[float, int],
    sigma: Union[float, int],
    s0: Union[float, int],
    average_path: bool = True,
) -> pd.DataFrame:
    """Generate some Arithmetic Brownian Motion paths.

    Args:
    ----
        n_paths (int): Number of paths to generate.
        T (Union[float, int]): The period in years.
        mu (Union[float, int]): The drift term (expected return annualized)
        sigma (Union[float, int]): The volatility term (standard deviation annualized).
        s0 (Union[float, int]): The initial condition (starting point)
        average_path (bool, optional): Whether or not to average all the paths. Defaults to True.

    Returns:
    ----
        pd.DataFrame: The dataframe containing the paths.
    """
    assert sigma > 0, "Error, sigma must be positive."
    n_steps = int(365 * T)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.log(s0)
    paths[:, 1:] = (mu - 0.5 * (sigma**2)) * dt + sigma * np.random.normal(
        0, (dt**0.5), (n_paths, n_steps)
    )
    paths = np.cumprod(np.exp(paths), axis=1)

    return __from_path_to_dataframe(paths, average_path)


def generate_OU_asset_paths(
    n_paths: int,
    T: Union[float, int],
    mu: Union[float, int],
    sigma: Union[float, int],
    theta: Union[float, int],
    s0: Union[float, int],
    average_path: bool = True,
) -> pd.DataFrame:
    """Ornstein Uhlenbeck process.

    Args:
    ----
        n_paths (int): The number of path to generate.
        T (Union[float, int]): The period in years.
        mu (Union[float, int]): The mean which the process will oscillate around
        sigma (Union[float, int]): The oscillation volatility
        theta (Union[float, int]): The oscillation speed around the mean.
        s0 (Union[float, int]): The starting point of the series, it could be the mean.
        average_path (bool, optional): _description_. Defaults to True.

    Returns:
    ----
        pd.DataFrame: The dataFrame containing the paths.
    """
    assert theta > 0 and sigma > 0, "Error, sigma and theta cannot be negative."

    n_steps = int(365 * T)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0

    for j in range(n_paths):
        for i in range(1, n_steps + 1):
            paths[j, i] = (
                paths[j, i - 1]
                + theta * (mu - paths[j, i - 1]) * dt
                + sigma * np.random.normal(0, np.sqrt(dt))
            )

    return __from_path_to_dataframe(paths, average_path)


def __generate_heston_volatility_paths(
    n_paths: int,
    T: Union[float, int] = 2,
    kappa: Union[float, int] = 4,
    theta: Union[float, int] = 0.12,
    eta: Union[float, int] = 0.9,
    s0: Union[float, int] = 0.1,
) -> npt.NDArray[np.float32]:
    """Generate volatility paths using an Ornstein Uhlenbeck process mean reverting process.

    Args:
        n_paths (int): The number of path to generate.
        T (Union[float, int], optional): The total period in years. Defaults to 2.
        kappa (Union[float, int], optional): Rate of mean reversion. Defaults to 4.
        theta (Union[float, int], optional): Long run average volatility. Defaults to 0.02.
        eta (Union[float, int], optional): Volatility of volatility. Defaults to 0.9.
        s0 (Union[float, int], optional): The initial condition of volatility. Defaults to 0.1.

    Returns:
        npt.NDArray[np.float32]: The numpy array containing the paths.
    """
    return (
        generate_OU_asset_paths(
            n_paths=n_paths,
            T=T,
            mu=theta,
            sigma=eta,
            theta=kappa,
            s0=s0,
            average_path=False,
        )
        .to_numpy()
        .T
    )


def __generate_poisson_jump(
    n_paths: int,
    T: Union[float, int],
    mean: Union[float, int],
    sigma: Union[float, int],
    lamb: Union[float, int],
) -> npt.NDArray[np.float64]:
    n_steps = int(365 * T)
    dt = T / n_steps
    Nt = np.cumsum(np.random.poisson(lamb * dt, size=(n_paths, n_steps + 1)), 1)
    jumps = np.zeros((n_paths, n_steps + 1))
    mask = Nt[:, 1:] > Nt[:, :-1]
    jumps[:, 1:][mask] = np.random.normal(mean, sigma, np.sum(mask))
    return jumps


def __from_path_to_dataframe(
    paths: npt.NDArray,
    average_path: bool = True,
) -> pd.DataFrame:
    if average_path is True:
        return pd.DataFrame(
            np.mean(paths, axis=0).T,
            columns=["Price"],
            index=pd.bdate_range(end=datetime.now(), periods=paths.shape[-1]),
        )

    return pd.DataFrame(
        paths.T,
        columns=[f"Price_{i+1}" for i in range(paths.shape[0])],
        index=pd.bdate_range(end=datetime.now(), periods=paths.shape[-1]),
    )
