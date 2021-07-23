from model import *
import torch.onnx
import torch
import netron

model = DRQN_CustomNet(32, 3, 512, 2)
model.eval()
x = torch.rand(1,10,32)
# o = model(x)

traced_mod = torch.jit.trace(model, x)
traced_mod.save("img/dqrn.pt")