import cupy as cp
from tqdm import tqdm


def select_lowest(energies, states, num_states):
    indices = cp.argpartition(energies, num_states)[:num_states]

    low_energies = energies[indices]
    low_states = states[indices]
    return low_energies, low_states


def sort_by_key(energies, states, num_states):
    order = cp.argsort(energies)[:num_states]

    low_energies = energies[order]
    low_states = states[order]
    return low_energies, low_states


def brute_force_gpu(Q, num_states: int, sweep_size_exponent: int = 10, threadsperblock: int = 256):
    N, _ = Q.shape
    if N > 64:
        raise ValueError("Za wysoka wartość N. Ta implementacja wspiera co najwyżej 64 spiny (64-bitowy integer)")
    sweep_size = 2 ** sweep_size_exponent
    num_chunks = 2 ** (N - sweep_size_exponent)

    brute_force_kernel = cp.RawModule(path="cuda_kernels/brute_force_kernel.ptx")
    compute_energies = brute_force_kernel.get_function("compute_energies")

    blockspergrid = sweep_size // threadsperblock

    final_energies = cp.array([])
    final_states = cp.array([])

    for i in tqdm(range(num_chunks), desc="wyczerpujące przeszukiwanie"):

        energies = cp.empty(sweep_size, dtype=cp.float32)
        states = cp.empty(sweep_size, dtype=cp.int64)
        compute_energies((blockspergrid,), (threadsperblock,),
                         (Q, cp.int32(N), cp.int32(sweep_size_exponent), energies, states, cp.int64(i)))

        low_energies, low_states = select_lowest(energies, states, num_states)

        final_energies = cp.concatenate((final_energies, low_energies))
        final_states = cp.concatenate((final_states, low_states))
        if i != 0:
            final_energies, final_states = select_lowest(final_energies, final_states, num_states)

    return sort_by_key(final_energies, final_states, num_states)