from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import tensorrt as trt
import ctypes

nvinfer = ctypes.CDLL("path-to-tensorrt/TensorRT-6.0.1.5/lib/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
pg = ctypes.CDLL("path-to-/libflatten_concat.so", mode=ctypes.RTLD_GLOBAL)
print('register FlatCat converter')
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
