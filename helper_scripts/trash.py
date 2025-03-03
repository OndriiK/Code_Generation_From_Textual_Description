import torch

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Supports Mixed Precision (FP16): {torch.cuda.get_device_capability(0)[0] >= 7}")
else:
    print("CUDA is not available on this system.")