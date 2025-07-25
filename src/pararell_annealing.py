import os
import cupy as cp
from typing import Optional
from tqdm import tqdm

from src.utils import read_instance, PEGASUS_ROOT


def calculate_energy_gpu(J: cp.ndarray, h: cp.ndarray, state: cp.ndarray):
    # Zakładamy, że J jest hermitowska z czynnikiem 1/2
    n, _ = J.shape
    A = cp.multiply(-1/2, J)
    B = cp.matmul(A, state) - h.reshape(n, 1)
    C = cp.multiply(state, B)
    return cp.sum(C, axis=0)


def parrarel_annealing_gpu(J, h, step_size: float, lambda_t_max: float, num_steps: int, num_trajectories: int,
                           schedule: Optional[list] = None, dtype=cp.float32):
    n = len(h)
    x = cp.zeros((n, num_trajectories), dtype=dtype)  # stan podstawowy dla H_innit = sum(x**2)
    momentum = cp.zeros((n, num_trajectories), dtype=dtype)
    state = cp.random.choice([dtype(-1.), dtype(1.)], size=(n, num_trajectories))  # losowy stan początkowy
    step_size = dtype(step_size)

    if schedule is None:
        schedule = [dtype(lambda_t_max * (1 - i / (num_steps - 1))) for i in
                    range(num_steps)]  # dlaczego nie linspace? chodzi o typy

    # cu_path = os.path.join("cuda_kernels", "pa_kernel.cu")
    # ptx_path = os.path.join("cuda_kernels", "pa_kernel.ptx")
    # command = f"nvcc --ptx --define-macro N={n} --define-macro M={num_trajectories} {cu_path} -o {ptx_path}"

    # os.system(command)
    kernel = cp.RawModule(path="cuda_kernels/pa_kernel.ptx")
    parrarel_annealing_step = kernel.get_function("parrarel_annealing_step")

    threadsperblock = 256  # Ilość wątków w bloku,
    blockspergrid_x = num_trajectories  # każdy blok zajmuje się trajektorią
    blockspergrid_y = (
                                  n + threadsperblock - 1) // threadsperblock  # wystarczająca ilość bloków by pomieścić całą kolumnę
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    x_new = cp.empty_like(x)
    momentum_new = cp.empty_like(momentum)
    state_new = cp.empty_like(state)

    for k in tqdm(range(num_steps), desc="wyżarzanie równoległe GPU"):
        lambda_t = schedule[k]
        A = cp.matmul(J, state)
        parrarel_annealing_step(blockspergrid, (threadsperblock,), (A, h, x, momentum,
                                                                    lambda_t, step_size, n,
                                                                    momentum_new, x_new, state_new))
        momentum = momentum_new
        x = x_new
        state = state_new

    return state, calculate_energy_gpu(J, h, state)


if __name__ == '__main__':



    test = os.path.join(PEGASUS_ROOT, "P4", "CBFM-P", "instances", "001_sg.txt")
    J, h = read_instance(test)

    J = cp.asarray(J, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)

    state, energy = parrarel_annealing_gpu(J, h, step_size=0.01, lambda_t_max=10, num_steps=1000, num_trajectories=5000)
    print(min(energy))