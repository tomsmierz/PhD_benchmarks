import os
import time

import cupy as cp
import numpy as np
import pandas as pd

from typing import Optional
from tqdm import tqdm

from src.utils import read_instance, PEGASUS_ROOT, SQUARE1_ROOT, calculate_energy_matrix, calculate_energy_gpu


# def parrarel_annealing_gpu(J, h, step_size: float, lambda_t_max: float, num_steps: int, num_trajectories: int,
#                            schedule: Optional[list] = None, dtype=cp.float32):
#     n = len(h)
#     x = cp.zeros((n, num_trajectories), dtype=dtype)  # stan podstawowy dla H_innit = sum(x**2)
#     momentum = cp.zeros((n, num_trajectories), dtype=dtype)
#     state = cp.random.choice([dtype(-1.), dtype(1.)], size=(n, num_trajectories))  # losowy stan początkowy
#     step_size = dtype(step_size)
#
#     if schedule is None:
#         schedule = [dtype(lambda_t_max * (1 - i / (num_steps - 1))) for i in
#                     range(num_steps)]  # dlaczego nie linspace? chodzi o typy
#
#     kernel = cp.RawModule(path="cuda_kernels/pa_kernel.ptx")
#     parrarel_annealing_step = kernel.get_function("parrarel_annealing_step")
#
#     threadsperblock = 256  # Ilość wątków w bloku,
#     blockspergrid_x = num_trajectories  # każdy blok zajmuje się trajektorią
#     blockspergrid_y = (n + threadsperblock - 1) // threadsperblock  # wystarczająca ilość bloków by pomieścić całą kolumnę
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
#
#     x_new = cp.empty_like(x)
#     momentum_new = cp.empty_like(momentum)
#     state_new = cp.empty_like(state)
#
#     for k in tqdm(range(num_steps), desc="wyżarzanie równoległe GPU"):
#         lambda_t = schedule[k]
#         A = cp.empty((n, num_trajectories), dtype=dtype)
#         cp.matmul(J, state, out=A)
#         parrarel_annealing_step(blockspergrid, (threadsperblock,), (A, h, x, momentum,
#                                                                     lambda_t, step_size, n,
#                                                                     momentum_new, x_new, state_new))
#         momentum = momentum_new
#         x = x_new
#         state = state_new
#
#     return state, calculate_energy_gpu(J, h, state)



def parallel_annealing_gpu(
                            J, h,
                            step_size: float,
                            lambda_t_max: float,
                            num_steps: int,
                            num_trajectories: int,
                            schedule: Optional[list] = None,
):
    # IMPORTANT: this kernel is float32-only as written
    dtype = cp.float32

    J = cp.asarray(J, dtype=dtype)
    h = cp.asarray(h, dtype=dtype)

    n = h.size
    M = num_trajectories

    x = cp.zeros((n, M), dtype=dtype)
    m = cp.zeros_like(x)
    state = cp.random.choice([dtype(-1.), dtype(1.)], size=(n, M))

    if schedule is None:
        # host-side scalars are fine
        schedule = [dtype(lambda_t_max * (1.0 - i / (num_steps - 1))) for i in range(num_steps)]

    # Preallocate A to avoid per-iter allocation
    A = cp.empty((n, M), dtype=dtype)


    module = cp.RawModule(path="cuda_kernels/pa_kernel.ptx")
    step_kernel = module.get_function("parallel_annealing_step_rowmajor")

    # threads = 256
    # grid = ((M + threads - 1) // threads, n)

    threadsperblock = 256  # Ilość wątków w bloku,
    blockspergrid_x = (M + threadsperblock -1) // threadsperblock
    blockspergrid_y = n
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    step_size_f = dtype(step_size)
    n_i32 = cp.int32(n)
    M_i32 = cp.int32(M)

    for k in tqdm(range(num_steps), desc="parallel annealing"):
        lambda_t = dtype(schedule[k])

        # Reuse A
        cp.matmul(J, state, out=A)

        # In-place update
        step_kernel(
            blockspergrid, (threadsperblock,),
            (A, h, x, m, lambda_t, step_size_f, n_i32, M_i32, state)
        )

    return state, calculate_energy_gpu(J, h, state)


def calculate_gradient_matrix(J: np.ndarray, h: np.ndarray, x: np.ndarray, state: np.ndarray, lambda_t: float) -> np.ndarray:
    n = len(h)
    # używamy brodcastingu który jest wykonywany automatycznie w bibliotece numpy
    return -1 * J @ state - h.reshape((n, 1)) + lambda_t * x


def parallel_annealing_cpu(J, h, step_size: float, lambda_t_max: float, num_steps: int, num_trajectories: int,
                           schedule: Optional[np.ndarray] = None, schedule_endpoint: Optional[float] = 0):
    n = len(h)
    x = np.zeros((n, num_trajectories))  # stan podstawowy dla H_innit = sum(x**2)
    momentum = np.zeros((n, num_trajectories))
    state = np.random.choice([-1, 1], size=(n, num_trajectories))  # losowy stan początkowy

    if schedule is None:
        schedule = np.linspace(lambda_t_max, schedule_endpoint, num=num_steps)

    for k in tqdm(range(num_steps), desc="wyżarzanie równoległe"):
        lambda_t = schedule[k]
        gradient = calculate_gradient_matrix(J, h, x, state, lambda_t)
        momentum = (1 - step_size) * momentum - step_size * gradient
        momentum = np.clip(momentum, -1, 1)
        x += momentum
        x = np.clip(x, -1, 1)
        state = np.sign(x)

    return state, calculate_energy_matrix(J, h, state)


if __name__ == '__main__':

    # test = os.path.join(PEGASUS_ROOT, "P4", "CBFM-P", "instances", "001_sg.txt")
    folder = os.path.join(SQUARE1_ROOT, "instances")
    df_best = pd.DataFrame(columns=["Instance", "Energy", "Time"], index=None)

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        J, h = read_instance(filepath)

        J = cp.asarray(J, dtype=cp.float32)
        h = cp.asarray(h, dtype=cp.float32)

        # Kernel compilation
        _, _ = parallel_annealing_gpu(J, h, step_size=0.01, lambda_t_max=10, num_steps=10, num_trajectories=50)

        # Computation
        start = time.time()
        state, energy = parallel_annealing_gpu(J, h, step_size=0.1, lambda_t_max=10, num_steps=1000, num_trajectories=2**11)
        end = time.time()

        elapsed = end - start

        name = filename.split(".")[0]
        df_best.loc[-1] = [name, min(energy), elapsed]
        df_best.index = df_best.index + 1
        df_best.sort_values("Instance")

    print(df_best)






