import torch
from torch2trt import torch2trt
from torch2trt.custom_plugins import *

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
print('Done')
