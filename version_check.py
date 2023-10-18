import sys

import torch
import torchvision
from mmcv.ops import get_compiler_version, get_compiling_cuda_version

import mmpose

print("Python version:", sys.version)
print("torch version:", torch.__version__, torch.cuda.is_available())
print("torchvision version:", torchvision.__version__)
print("mmpose version:", mmpose.__version__)
print("cuda version:", get_compiling_cuda_version())
print("compiler information:", get_compiler_version())
