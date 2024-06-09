import torch
from sigma_module import Sigma
import time

batch_size = 10000

num_leafs = [2, 8, 27, 84, 274, 1272]
example_inputs = [torch.rand(batch_size, num_leafs[i]) for i in range(6)]

module = Sigma()
start_time = time.time()
traced_module = torch.jit.trace_module(
    module,
    {
        "func100": example_inputs[0],
        "func200": example_inputs[1],
        "func300": example_inputs[2],
        "func400": example_inputs[3],
        "func500": example_inputs[4],
        "func600": example_inputs[5],
    },
)
print("Tracing time:", time.time() - start_time)

traced_module.save("traced_Sigma.pt")
