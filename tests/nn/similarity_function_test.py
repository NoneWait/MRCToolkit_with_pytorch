import unittest
from mrc.nn.similarity_function import TriLinear
import torch


class TestSimilarityFunction(unittest.TestCase):
    def setUp(self):
        # [batch, c_len, d]
        self.t0 = torch.randn(16, 10, 100)
        # [batch, q_len, d]
        self.t1 = torch.randn(16, 20, 100)
        # self.t1.required_grad_()

        self.trilinear = TriLinear(100, bias=True)

    def test_TriLinear(self):
        out = self.trilinear(self.t0, self.t1)

        # 只测试大小是否一致，无法测试具体是否准确
        self.assertEqual(list(out.size()), [16, 10, 20])


if __name__ == '__main__':
    unittest.main()
