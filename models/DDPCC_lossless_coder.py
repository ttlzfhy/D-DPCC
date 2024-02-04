import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *

class get_model(nn.Module):
    def __init__(self, channels=8):
        super(get_model, self).__init__()
        self.ref1 = DownsampleLayer(1, 1, 1, 3, resnet=None)
        self.ref2 = DownsampleLayer(1, 1, 1, 3, resnet=None)
        self.compressor = LosslessCompressor()

    def forward(self, f1, f2, device, epoch=99999):
        num_points = f2.size()[0]
        ref1 = self.ref1(f2)
        ref2 = self.ref2(ref1)
        ref2 = ME.SparseTensor(torch.ones_like(ref2.C[:, :1], dtype=torch.float32), coordinates=ref2.C, tensor_stride=4, device=f2.device)
        bits, quant, cls, target = self.compressor(ref2, num_points)
        bpp = bits / num_points
        return bpp, quant, cls, target