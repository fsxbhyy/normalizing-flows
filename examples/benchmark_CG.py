import torch
import pandas as pd
from funcs_sigma import *
from parquetAD import FeynmanDiagram, load_leaf_info
from nsf_annealing import generate_model

import os
import torch.utils.benchmark as benchmark
import numpy as np
from memory_profiler import memory_usage
import time
import gc

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

max_order = 5
beta = 32.0
batch_size = 10000
# Neval = 20
Neval = 1


def benchmark_NF(order, beta, batch_size):
    model_state_dict_path = "nfm_o{0}_beta{1}_l2c32b8_state_Re0630.pt".format(
        order, beta
    )

    root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
    num_loops = [2, 6, 15, 39, 111, 448]
    partition = [(order, 0, 0)]
    name = "sigma"
    df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
    with torch.no_grad():
        # loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns[:num_loops[order-1]]]).T
        loopBasis = torch.Tensor(
            df.iloc[: order + 1, : num_loops[order - 1]].to_numpy()
        )
    leafstates = []
    leafvalues = []

    for key in partition:
        key_str = "".join(map(str, key))
        state, values = load_leaf_info(root_dir, name, key_str)
        leafstates.append(state)
        leafvalues.append(values)

    print("Loading normalizing-flow model")
    start_time = time.time()
    state_dict = torch.load(
        model_state_dict_path, map_location=device
    )  # ["model_state_dict"]
    nfm_batchsize = state_dict["p.samples"].shape[0]

    diagram_nfm = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], nfm_batchsize
    )
    diagram = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], batch_size
    )

    partial_state_dict = {
        k: v for k, v in state_dict.items() if k in state_dict and "p." not in k
    }
    nfm = generate_model(
        diagram_nfm,
        num_blocks=2,
        num_hidden_channels=32,
        num_bins=8,
    )
    nfm_state_dict = nfm.state_dict()
    nfm_state_dict.update(partial_state_dict)
    nfm.load_state_dict(nfm_state_dict)
    nfm.p = diagram
    nfm = nfm.to(device)
    nfm.eval()
    print("Loading model takes {:.3f}s \n".format(time.time() - start_time))
    gc.collect()

    def flow_func():
        nfm.p.samples[:], nfm.p.log_q[:] = nfm.q0(batch_size)
        for flow in nfm.flows:
            nfm.p.samples[:], nfm.p.log_det[:] = flow(nfm.p.samples)
            nfm.p.log_q -= nfm.p.log_det

    t1 = benchmark.Timer(
        stmt="flow_func()",
        globals={"flow_func": flow_func},
        label="Normalizing-flow (order {0} beta {1})".format(order, beta),
        sub_label="z->x flow (batchsize {0})".format(batch_size),
    )
    print(t1.timeit(Neval))


def benchmark_diagram(order, beta, batch_size):
    root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
    dim = 4 * order - 1
    num_loops = [2, 6, 15, 39, 111, 448]
    partition = [(order, 0, 0)]
    name = "sigma"
    df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
    with torch.no_grad():
        # loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns[:num_loops[order-1]]]).T
        loopBasis = torch.Tensor(
            df.iloc[: order + 1, : num_loops[order - 1]].to_numpy()
        )
    leafstates = []
    leafvalues = []

    for key in partition:
        key_str = "".join(map(str, key))
        state, values = load_leaf_info(root_dir, name, key_str)
        leafstates.append(state)
        leafvalues.append(values)

    diagram = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], batch_size
    )
    diagram = diagram.to(device)

    var = torch.rand(batch_size, dim, device=device)

    t1 = benchmark.Timer(
        stmt="diagram.prob(var)",
        globals={"diagram": diagram, "var": var},
        label="Self-energy diagram (order {0} beta {1})".format(order, beta),
        sub_label="Evaluating integrand (batchsize {0})".format(batch_size),
    )

    # t2 = benchmark.Timer(
    #     stmt="diagram.prob(var, 1)",
    #     globals={"diagram": diagram, "var": var},
    #     label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    #     sub_label="Evaluating integrand with jit.script (batchsize {0})".format(
    #         batch_size
    #     ),
    # )

    print(t1.timeit(Neval))
    # print(t2.timeit(Neval))


def benchmark_graph(batch_size, Neval):
    root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
    Sigma_module = torch.jit.load(
        os.path.join(root_dir, f"traced_Sigma_{batch_size:.0e}.pt")
    )
    # sigma_diagram = sigma_diagram.to(device)
    Sigma_module = Sigma_module.to(device)
    num_leafs = [2, 8, 27, 84, 274, 1272]
    for o in range(1, 7):
        inputs = torch.rand(batch_size, num_leafs[o - 1], device=device)

        if o == 1:
            eval_graph = eval_graph100
            eval_graph_jit = Sigma_module.func100
        elif o == 2:
            eval_graph = eval_graph200
            eval_graph_jit = Sigma_module.func200
        elif o == 3:
            eval_graph = eval_graph300
            eval_graph_jit = Sigma_module.func300
        elif o == 4:
            eval_graph = eval_graph400
            eval_graph_jit = Sigma_module.func400
        elif o == 5:
            eval_graph = eval_graph500
            eval_graph_jit = Sigma_module.func500
        elif o == 6:
            eval_graph = eval_graph600
            eval_graph_jit = Sigma_module.func600
        else:
            raise ValueError("Order not supported")

        # t1 = benchmark.Timer(
        #     stmt="eval_graph(inputs)",
        #     globals={"eval_graph": eval_graph, "inputs": inputs},
        #     label="Self-energy computational graph (order {0})".format(o),
        #     sub_label="Evaluating graph (batchsize {0})".format(batch_size),
        # )
        t2 = benchmark.Timer(
            stmt="eval_graph_jit(inputs)",
            globals={"eval_graph_jit": eval_graph_jit, "inputs": inputs},
            label="Self-energy computational graph (order {0})".format(o),
            sub_label="Evaluating graph with jit.trace (batchsize {0})".format(
                batch_size
            ),
        )

        # print(t1.timeit(Neval))
        # mem_usage_func = memory_usage((func, (inputs,)), interval=0.1, timeout=1)
        # print(f"Memory usage for func (order {o}): {max(mem_usage_func)} MiB")

        print(t2.timeit(Neval))
        mem_usage_func_jit = memory_usage(
            (eval_graph_jit, (inputs,)), interval=0.1, timeout=1
        )
        print(
            f"Memory usage for eval_graph_jit (order {o}): {max(mem_usage_func_jit)} MiB"
        )

        # if o == 5 or o == 6:
        #     t3 = benchmark.Timer(
        #         stmt="func_jit0(inputs)",
        #         globals={"func_jit0": func_jit0, "inputs": inputs},
        #         label="Self-energy computational graph (order {0})".format(o),
        #         sub_label="Evaluating graph with jit.script (batchsize {0})".format(
        #             batch_size
        #         ),
        #     )
        #     print(t3.timeit(Neval))
        #     mem_usage_func_jit0 = memory_usage(
        #         (func_jit0, (inputs,)), interval=0.1, timeout=1
        #     )
        #     print(
        #         f"Memory usage for func_jit0 (order {o}): {max(mem_usage_func_jit0)} MiB"
        #     )
        print("\n")


if __name__ == "__main__":
    # for o in range(1, max_order + 1):
    #     benchmark_diagram(o, beta, batch_size)
    # benchmark_diagram(6, beta, batch_size)
    # benchmark_graph(batch_size, Neval)
    benchmark_NF(5, beta, batch_size)
