import torch
print(torch.cuda.is_available())  # True bedeutet CUDA-Unterstützung vorhanden
print(torch.__version__)  # Die vollständige Version mit möglichem CUDA-Suffix