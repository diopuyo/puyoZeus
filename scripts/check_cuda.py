"""CUDA 利用状況確認用 (Z-3C 高速化検証)。"""
import torch

print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
