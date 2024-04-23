import torch
import torch.nn.functional as F

import numpy as np

class LoopPool:
    def __init__(self, name, dim, loop_num, basis=None, dtype=np.float64):
        self.name = name
        self.dim = dim
        self.loop_num = loop_num
        if basis is None:
            # Initializing an empty basis and loops if no basis is provided.
            self.basis = np.empty((loop_num, 0), dtype=dtype)
            self.loops = np.empty((dim, 0), dtype=dtype)
        else:
            # Initializing with a given basis
            assert len(basis) > 0, "basis cannot be empty"
            assert all(len(x) == loop_num for x in basis), "All basis vectors must be of the same length"
            self.basis = np.stack([np.array(x, dtype=dtype) for x in basis], axis=1)
            print("test:", (dim, self.basis.shape[1]))
            self.loops = np.empty((dim, self.basis.shape[1]), dtype=dtype)

    def __getitem__(self, idx):
        return self.basis[:, idx]

    def __setitem__(self, idx, value):
        self.basis[:, idx] = np.array(value, dtype=self.basis.dtype)

    def size(self):
        return self.basis.shape[1]

    def update(self, variable=None):
        if variable is None:
            variable = np.random.rand(self.dim, self.loop_num).astype(self.loops.dtype)
        else:
            variable = variable.astype(self.basis.dtype)
        assert variable.shape[0] == self.dim, "Variable dimension must match LoopPool dimension"
        self.loops = np.dot(variable[:, :self.loop_num], self.basis)

    def loop(self, idx):
        return self.loops[:, idx]

    def has_loop(self):
        return self.dim > 0 and self.loop_num > 0

    def append_basis(self, basis_vector):
        assert len(basis_vector) <= self.loop_num, "Basis vector length cannot exceed the number of independent loops"
        
        # Ensure the basis_vector is a numpy array
        basis_vector = np.array(basis_vector, dtype=self.basis.dtype)
        
        # Pad basis_vector if it is shorter than loop_num
        if len(basis_vector) < self.loop_num:
            padding_size = self.loop_num - len(basis_vector)
            # Pad at the end of the array
            basis_vector = np.pad(basis_vector, (0, padding_size), 'constant')
        
        # Check for existing basis vector
        for i in range(self.size()):
            if np.allclose(self.basis[:, i], basis_vector, atol=1e-16): 
                return i

        # Append new basis vector
        self.basis = np.concatenate((self.basis, basis_vector[:, np.newaxis]), axis=1)
        new_loops = np.random.rand(self.dim, 1).astype(self.loops.dtype)
        self.loops = np.concatenate((self.loops, new_loops), axis=1)
        return self.size() - 1