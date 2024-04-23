import unittest
import torch
from loop_pool import LoopPool  # Assuming the LoopPool class is in a module named your_module

class TestLoopPool(unittest.TestCase):
    def test_loop_pool_initialization_and_indexing(self):
        loopbasis = [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0, 0.0]
        ]
        loop_pool = LoopPool('K', 3, 4, loopbasis)
        self.assertEqual(loop_pool.size(), len(loopbasis))
        self.assertTrue(torch.allclose(loop_pool[0], torch.tensor([1.0, 1.0, 0.0, 0.0]).to(loop_pool.basis.dtype)))
        self.assertTrue(torch.allclose(loop_pool[-1], torch.tensor([1.0, 0.0, -1.0, 0.0]).to(loop_pool.basis.dtype)))

        loop_pool[1] = torch.tensor([1.0, 0.0, -1.0, 0.0]).to(loop_pool.basis.dtype)
        self.assertTrue(torch.allclose(loop_pool[1], torch.tensor([1.0, 0.0, -1.0, 0.0]).to(loop_pool.basis.dtype)))

    def test_loop_pool_append_and_update(self):
        dim, N = 3, 4
        loop_pool = LoopPool('K', dim, N)

        basis1 = [1.0, 0.0, 0.0, 1.0]
        basis2 = [1.0, 1.0, 0.0, 0.0]
        basis3 = [1.0, 0.0, -1.0, 1.0]

        idx1 = loop_pool.append_basis(basis1)
        idx2 = loop_pool.append_basis(basis2)
        idx3 = loop_pool.append_basis(basis2)
        idx4 = loop_pool.append_basis(basis1)
        idx5 = loop_pool.append_basis(basis3)

        self.assertEqual(loop_pool.size(), 3)
        self.assertEqual(idx1, idx4)
        self.assertEqual(idx2, idx3)

        varK = torch.rand((dim, N))
        loop_pool.update(varK)

        self.assertTrue(torch.allclose(loop_pool.loop(0), torch.matmul(varK, torch.tensor(basis1)).to(loop_pool.basis.dtype)))
        self.assertTrue(torch.allclose(loop_pool.loop(1), torch.matmul(varK, torch.tensor(basis2)).to(loop_pool.basis.dtype)))
        self.assertTrue(torch.allclose(loop_pool.loop(2), torch.matmul(varK, torch.tensor(basis3)).to(loop_pool.basis.dtype)))

if __name__ == '__main__':
    unittest.main()