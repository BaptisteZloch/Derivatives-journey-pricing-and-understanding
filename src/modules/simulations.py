import numpy as np
import numpy.typing as npt
from typing import Literal
import matplotlib.pyplot as plt


def generate_brownian_paths(
    n_paths: int,
    n_steps: int,
    T: int,
    mu: float | int,
    sigma: float | int,
    s0: float | int,
    brownian_type: Literal["ABM", "GBM"] = "ABM",
    get_time: bool = True,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.log(s0)

    for i in range(0, n_paths):
        for j in range(n_steps):
            paths[i, j + 1] = (
                paths[i, j]
                + (mu * 0.5 * sigma**2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt))
            )

    if brownian_type == "GBM":
        paths = np.exp(paths)

    return np.linspace(0, T, n_steps + 1), paths if get_time is True else paths


def account_evolution(risk_free_rate: float, time: float) -> float:
    return np.exp(risk_free_rate * time)


def plot_paths_with_distribution(
    time: npt.NDArray[np.float64],
    paths: npt.NDArray[np.float64],
    title: str = "Simulated Brownian Motion",
) -> None:
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for y in paths:
        ax1.plot(time, y)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{paths.shape[0]} simulations")
    ax1.grid(True)
    ax2.hist(
        paths[:, -1],
        density=True,
        bins=75,
        facecolor="blue",
        alpha=0.3,
        label="Frequency of X(T)",
    )
    ax2.hist(
        paths[:, paths.shape[-1] // 4],
        density=True,
        bins=75,
        facecolor="red",
        alpha=0.3,
        label="Frequency of X(T/4)",
    )
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution at X(T) and X(T/4)")
    ax2.legend()
    ax2.grid(True)


# n_sim = 100
# n_step = 100
# mu = 3
# sigma = 0.5
# s0 = 100

# time, sim =  generate_brownian_paths(n_sim, n_step, 1, mu, sigma, s0, "ABM")

# plot_paths(time, sim, f"Arithmetic Brownian Motion {n_sim} simulations, {n_step} steps, mu={mu}, sigma={sigma}, s0={s0}")
