import torch
import pandas as pd
from funcs_sigma import *
from parquetAD import FeynmanDiagram, load_leaf_info
import os
import torch.utils.benchmark as benchmark
import numpy as np
from memory_profiler import memory_usage

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

order = 5
dim = 4 * order - 1
beta = 10.0
batch_size = 10000
Neval = 40


def benchmark_diagram(order, beta, batch_size):
    root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
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
        stmt="diagram.prob(var, 0)",
        globals={"diagram": diagram, "var": var},
        label="Self-energy diagram (order {0} beta {1})".format(order, beta),
        sub_label="Evaluating integrand (batchsize {0})".format(batch_size),
    )

    t2 = benchmark.Timer(
        stmt="diagram.prob(var, 1)",
        globals={"diagram": diagram, "var": var},
        label="Self-energy diagram (order {0} beta {1})".format(order, beta),
        sub_label="Evaluating integrand with jit.script (batchsize {0})".format(
            batch_size
        ),
    )

    # t3 = benchmark.Timer(
    #     stmt="diagram.prob_jit(var)",
    #     globals={"diagram": diagram, "var": var},
    #     label="Sigma function (order {0} beta {1})".format(order, beta),
    #     sub_label="Evaluating integrand with jit.trace (batchsize {0})".format(
    #         batch_size
    #     ),
    # )

    print(t1.timeit(Neval))
    print(t2.timeit(Neval))
    # print(t3.timeit(Neval))


def benchmark_graph(batch_size, Neval):
    sigma_diagram = torch.jit.load("funcs_sigma/traced_Sigma_1e4.pt")
    sigma_diagram = sigma_diagram.to(device)
    num_leafs = [2, 8, 27, 84, 274, 1272]
    for o in range(1, 7):
        inputs = torch.rand(batch_size, num_leafs[o - 1], device=device)
        if o == 1:
            func = func_sigma_o100.graphfunc
            func_jit = sigma_diagram.func100
        elif o == 2:
            func = func_sigma_o200.graphfunc
            func_jit = sigma_diagram.func200
        elif o == 3:
            func = func_sigma_o300.graphfunc
            func_jit = sigma_diagram.func300
        elif o == 4:
            func = func_sigma_o400.graphfunc
            func_jit = sigma_diagram.func400
        elif o == 5:
            func = func_sigma_o500.graphfunc
            func_jit0 = func_sigma_o500_jit.graphfunc
            func_jit = sigma_diagram.func500
        elif o == 6:
            func = func_sigma_o600.graphfunc
            func_jit0 = func_sigma_o600_jit.graphfunc
            func_jit = sigma_diagram.func600
        else:
            raise ValueError("Order not supported")

        t1 = benchmark.Timer(
            stmt="func(inputs)",
            globals={"func": func, "inputs": inputs},
            label="Self-energy computational graph (order {0})".format(o),
            sub_label="Evaluating graph (batchsize {0})".format(batch_size),
        )
        t2 = benchmark.Timer(
            stmt="func_jit(inputs)",
            globals={"func_jit": func_jit, "inputs": inputs},
            label="Self-energy computational graph (order {0})".format(o),
            sub_label="Evaluating graph with jit.trace (batchsize {0})".format(
                batch_size
            ),
        )

        print(t1.timeit(Neval))
        mem_usage_func = memory_usage((func, (inputs,)), interval=0.1, timeout=1)
        print(f"Memory usage for func (order {o}): {max(mem_usage_func)} MiB")

        print(t2.timeit(Neval))
        mem_usage_func_jit = memory_usage(
            (func_jit, (inputs,)), interval=0.1, timeout=1
        )
        print(f"Memory usage for func_jit (order {o}): {max(mem_usage_func_jit)} MiB")

        if o == 5 or o == 6:
            t3 = benchmark.Timer(
                stmt="func_jit0(inputs)",
                globals={"func_jit0": func_jit0, "inputs": inputs},
                label="Self-energy computational graph (order {0})".format(o),
                sub_label="Evaluating graph with jit.script (batchsize {0})".format(
                    batch_size
                ),
            )
            print(t3.timeit(Neval))
            mem_usage_func_jit0 = memory_usage(
                (func_jit0, (inputs,)), interval=0.1, timeout=1
            )
            print(
                f"Memory usage for func_jit0 (order {o}): {max(mem_usage_func_jit0)} MiB"
            )
        print("\n")


if __name__ == "__main__":
    # benchmark_diagram(order, beta, batch_size)
    benchmark_graph(batch_size, Neval)
