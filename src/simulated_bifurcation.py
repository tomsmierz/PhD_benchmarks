import numpy as np

from math import sqrt
from typing import Optional
from tqdm import tqdm

from src.utils import calculate_energy_matrix


def wall(x: np.ndarray, y: np.ndarray):
    mask = np.abs(x) > 1
    x[mask] = np.sign(x[mask])
    y[mask] = 0
    return x, y


def discrete_simulated_bifurcation_cpu(J, h, num_steps, time_step, num_trajectories: int,
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1

    N, _ = J.shape
    mean_J = np.sqrt(np.sum(np.square(J)) / (N * (N - 1)))
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling

    a = np.linspace(0, a_0, num=num_steps)

    x = np.zeros((N, num_trajectories))
    y = np.random.uniform(-0.1, 0.1, (N, num_trajectories))

    for t in tqdm(range(num_steps), desc="Symulowana Bifurkacja"):
        y += (-1 * (a_0 - a[t]) * x + c_0 * (J @ np.sign(x) + h.reshape((N, 1)))) * time_step  # y(t+1); x(t), x(t)
        x += a_0 * y * time_step  # x(t + 1); y(t+1)

        x, y = wall(x, y)

    x = np.sign(x)
    return x, calculate_energy_matrix(J, h, x)
