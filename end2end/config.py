import torch

if not torch.cuda.is_available():
    CUDA_DEVICE = "cpu"
elif torch.cuda.device_count() == 1:
    CUDA_DEVICE = "cuda:0"  # for Krish
else:
    CUDA_DEVICE = "cuda:6"  # for Yuhao