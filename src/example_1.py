import os
import dimod
import time

import networkx as nx
import numpy as np
import cupy as cp
import pandas as pd


from src.pararell_annealing import parallel_annealing_cpu
from src.simulated_annealing import simulated_annealing_cpu
from src.bruteforce import brute_force_gpu, brute_force_cpu
from src.simulated_bifurcation import discrete_simulated_bifurcation_cpu
from src.utils import ROOT, read_instance, calculate_energy, calculate_energy_matrix, read_instance_dict, dict_to_matrix

rng = np.random.default_rng()


def generate_instances(num: int):
    graph = nx.hexagonal_lattice_graph(1, 2)
    for idx in range(num):
        for u, v, data in graph.edges(data=True):
            data["J"] = rng.choice([-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0])

        lin_index = {node: i for i, node in enumerate(graph.nodes)}
        J = nx.to_numpy_array(graph, weight="J")
        h = [rng.choice([-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]) for _ in graph.nodes()]

        with open(os.path.join(ROOT, "data", "example", f"test_hex_{idx+1}.txt"), "w") as f:
            for i, node in enumerate(graph.nodes):
                f.write(f"{lin_index[node]+1} {lin_index[node]+1} {h[i]}\n")
            for idx, (i, j, data) in enumerate(graph.edges(data=True)):
                f.write(f"{lin_index[i]+1} {lin_index[j]+1} {data["J"]}\n")


def calculate_tts(p_s, p_t, elapsed):
    if p_s < 1:
     return elapsed * (np.log(1 - p_t)) / (np.log(1 - p_s))
    else:
        return elapsed

def main():
    df = pd.DataFrame({"instance": [], "gs": [], "tts_bf_gpu": [],
                       "tts_sa":[], "tts_pa":[], "tts_sbm":[]})
    solutions = {}
    ground_states = {1: -31.25, 2: -27.25, 3: -29.25, 4: -26.75, 5: -26.75,
                     6: -29.75, 7: -31.0, 8: -28.0, 9: -29.75, 10: -32.25}
    tts_bf = {1: 31.05, 2: 30.69 ,3: 35.21, 4: 35.27, 5: 35.06, 6: 35.09, 7: 35.46 , 8: 30.61, 9: 35.14 , 10: 35.22}

    p_t = 0.99

    for idx in range(1, 11):
        solutions["instance"] = idx
        J, h = read_instance(os.path.join(ROOT, "data", "example", f"hex_{idx}.txt"), convention="dwave")
        J_m, h_m = read_instance(os.path.join(ROOT, "data", "example", f"hex_{idx}.txt"))

        # bqm = dimod.BinaryQuadraticModel(vartype="SPIN")
        # bqm = bqm.from_ising(h, J)
        # Q, offset = bqm.to_qubo()
        # # print(offset)
        # Q = dict_to_matrix(Q)
        # Q_gpu = cp.asarray(Q, dtype=cp.float32)

        # _, energy = brute_force_cpu(J, h)
        # print(energy)
        # start = time.time()
        # energies_qpu, _ = brute_force_gpu(Q_gpu, 10, sweep_size_exponent=15)
        # end = time.time()
        # elapsed = end - start
        # print(energies_qpu + offset, elapsed)

        start = time.time()
        _, energies_pa = parallel_annealing_cpu(J_m, h_m, 0.2, 10, 400, 2**10)
        end = time.time()
        elapsed_pa = end - start


        p_s = np.sum(energies_pa == ground_states[idx]) / len(energies_pa)
        tts_pa = calculate_tts(p_s, p_t, elapsed_pa)

        solutions["tts_pa"] = tts_pa

        start = time.time()
        _, energies_sbm = discrete_simulated_bifurcation_cpu(J_m, h_m, 400, 0.25, 2**10)
        end = time.time()
        elapsed_sbm = end - start

        p_s = np.sum(energies_sbm == ground_states[idx]) / len(energies_sbm)
        tts_sbm = calculate_tts(p_s, p_t, elapsed_sbm)

        solutions["tts_sbm"] = tts_sbm

        start = time.time()
        _, energies_sa = simulated_annealing_cpu(J_m, h_m, 2**10, 400)
        end = time.time()
        elapsed_sa = end - start

        p_s = np.sum(energies_sa == ground_states[idx]) / len(energies_sa)
        tts_sa = calculate_tts(p_s, p_t, elapsed_sa)

        solutions["tts_sa"] = tts_sa
        solutions["gs"] = ground_states[idx]
        solutions["tts_bf_gpu"] = tts_bf[idx]


        df.loc[len(df)] = solutions
        df.to_csv(os.path.join(ROOT, "results", "example", "hex_30.csv"), index=False)

    print(df)


if __name__ == '__main__':
    main()
