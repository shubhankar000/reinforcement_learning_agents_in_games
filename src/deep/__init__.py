"""
Supress torch nnpack warnings on VM. Applied to init so every parallel worker inherits it
"""

import torch

torch.backends.nnpack.set_flags(False)
