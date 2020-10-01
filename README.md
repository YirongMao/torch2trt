# torch2trt-custom
This project is forked from https://github.com/NVIDIA-AI-IOT/torch2trt.

This forked version shows how to add a new tensorrt plugin. 

You can find hwo to add a custom plugin:  [flattenconcat](https://github.com/YirongMao/TensorRT-Custom-Plugin). I will detail how to transfer this plugin from pytorch to tensorrt.

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
    
The corresponding code is in [custom_plugins.py](https://github.com/YirongMao/torch2trt/blob/master/torch2trt/custom_plugins.py)

(2) import custom_plugin.py https://github.com/YirongMao/torch2trt/blob/master/torch2trt/torch2trt.py#L6
    
(3) create a new converter:
```python
@tensorrt_converter('FlatCat.forward')
def convert_flatcat(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[2]
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    plg_registry = trt.get_plugin_registry()
    plg_creator = plg_registry.get_plugin_creator("FlattenConcatCustom", "1", "")
    axis_pf = trt.PluginField("axis", np.array([1], np.int32), trt.PluginFieldType.INT32)
    batch_pf = trt.PluginField("ignoreBatch", np.array([0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([axis_pf, batch_pf])
    fn = plg_creator.create_plugin("FlattenConcatCustom1", pfc)
    layer = ctx.network.add_plugin_v2([input_a_trt, input_b_trt], fn)
    output = ctx.method_return
    output._trt = layer.get_output(0)
```

The corresponding code is in [flattenconcat.py](https://github.com/YirongMao/torch2trt/blob/master/torch2trt/converters/flattenconcat.py)

(4) After that, it's ready to transfer the torch model with flattenconcat module into tensorrt:
```python
# create example data
x = torch.ones((1, 4, 2, 2)).cuda()
y = torch.ones((1, 3, 2, 2)).cuda()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = FlatCat()

    def forward(self, x, y):
        return self.layer(x, y)

model = Model().cuda()
z = model.forward(x, y)
print(z.shape)
print(model)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x, y], max_batch_size=10)
with open('flatcat.engine', 'wb') as f:
    f.write(model_trt.engine.serialize())
```
The corresponding code is in [convert_flattencat.py](https://github.com/YirongMao/torch2trt/blob/master/torch2trt/convert_flattencat.py)
    
    
