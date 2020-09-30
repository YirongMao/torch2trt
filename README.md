# torch2trt-Custom
This project is forked from https://github.com/NVIDIA-AI-IOT/torch2trt.

This forked version shows how to add a new tensorrt plugin. 

You can find hwo to add a custom plugin:  [flattenconcat] (https://github.com/YirongMao/TensorRT-Custom-Plugin). I will detail how to transfer this plugin from pytorch to tensorrt.

(1) create a class from torch.nn.Module


    ```python
    import torch
    class FlatCat(torch.nn.Module):
        def __init__(self):
            super(FlatCat, self).__init__()

        def forward(self, x, y):
            x = x.view(x.shape[0], -1, 1, 1)
            y = y.view(y.shape[0], -1, 1, 1)
            return torch.cat([x, y], 1)
    ```
    
The corresponding code is in [custom_plugins.py]
    
  
    
    
    
