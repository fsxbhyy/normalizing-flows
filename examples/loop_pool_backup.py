import torch
import torch.nn.functional as F
class LoopPool:
    def __init__(self, name, dim, loop_num, basis=None, dtype=torch.float64):
        self.name = name
        self.dim = dim
        self.loop_num = loop_num
        if basis is None:
            # Initializing an empty basis and loops if no basis is provided.
            self.basis = torch.empty((loop_num, 0), dtype=dtype)
            self.loops = torch.empty((dim, 0), dtype=dtype)
        else:
            # Initializing with a given basis
            assert len(basis) > 0, "basis cannot be empty"
            assert all(len(x) == loop_num for x in basis), "All basis vectors must be of the same length"
            self.basis = torch.stack([torch.tensor(x, dtype=dtype) for x in basis], dim=1)
            print("test:", (dim, self.basis.size(1)))
            self.loops = torch.empty((dim, self.basis.size(1)), dtype=dtype)

    def __getitem__(self, idx):
        return self.basis[:, idx]

    def __setitem__(self, idx, value):
        self.basis[:, idx] = torch.tensor(value, dtype=self.basis.dtype)

    def size(self):
        return self.basis.size(1)

    def update(self, variable=None):
        if variable is None:
            variable = torch.rand((self.dim, self.loop_num), dtype=self.loops.dtype)
        else:
            variable = variable.to(self.basis.dtype)
        assert variable.size(0) == self.dim, "Variable dimension must match LoopPool dimension"
        print(variable, self.basis)
        self.loops = torch.mm(variable[:, :self.loop_num], self.basis)

    def loop(self, idx):
        return self.loops[:, idx]

    def has_loop(self):
        return self.dim > 0 and self.loop_num > 0

    def append_basis(self, basis_vector):
        assert len(basis_vector) <= self.loop_num, "Basis vector length cannot exceed the number of independent loops"
        
        # Ensure the basis_vector is a tensor
        basis_vector = torch.tensor(basis_vector, dtype=self.basis.dtype)
        
        # Pad basis_vector if it is shorter than loop_num
        if len(basis_vector) < self.loop_num:
            padding_size = self.loop_num - len(basis_vector)
            # Pad at the end of the tensor
            basis_vector = F.pad(basis_vector, (0, padding_size), "constant", 0)
        
        # Check for existing basis vector
        for i in range(self.size()):
            if torch.allclose(self.basis[:, i], basis_vector, atol=1e-16): 
                return i

        # Append new basis vector
        self.basis = torch.cat((self.basis, basis_vector.unsqueeze(1)), dim=1)
        new_loops = torch.rand((self.dim, 1), dtype=self.loops.dtype)
        self.loops = torch.cat((self.loops, new_loops), dim=1)
        return self.size() - 1