import torch
print(torch.__config__.show())
print(torch.backends.mkldnn.is_available())