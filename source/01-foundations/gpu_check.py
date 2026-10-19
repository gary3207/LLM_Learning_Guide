import torch

print("PyTorch:", torch.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch에서 GPU를 사용할 수 없습니다."

print("GPU:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print("GPU 계산 결과:", (x * 2).cpu().tolist())
