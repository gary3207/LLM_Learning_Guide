import numpy as np
import torch

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
result = (x * 2).cpu().numpy()
np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
print("GPU 연산·NumPy 변환 성공:", result)
