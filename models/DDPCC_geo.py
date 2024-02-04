import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *

class get_model(nn.Module):
    def __init__(self, channels=8):
        super(get_model, self).__init__()
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3)
        self.inter_prediction = inter_prediction(64, 64, 48)
        self.enc3 = DownsampleLayer(64, 64, 32, 3)
        self.enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)

        self.dec1 = UpsampleLayer(channels, 64, 64, 3)
        self.dec2 = UpsampleLayer(64, 32, 32, 3)
        self.dec3 = UpsampleLayer(32, 16, 16, 3)

        self.BitEstimator = BitEstimator(channels, 3)
        self.MotionBitEstimator = BitEstimator(48, 3)
        self.crit = torch.nn.BCEWithLogitsLoss()

    def forward(self, f1, f2, device, epoch=99999):
        num_points = f2.C.size(0)

        ys1, ys2 = [f1, 0, 0, 0, 0], [f2, 0, 0, 0, 0]
        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

        # feature extraction
        ys1[1] = self.enc1(ys1[0])
        ys1[2] = self.enc2(ys1[1])
        ys2[1] = self.enc1(ys2[0])
        ys2[2] = self.enc2(ys2[1])

        # inter prediction
        residual, predicted_point2, quant_motion = self.inter_prediction(ys1[2], ys2[2], stride=4)

        # residual compression
        quant_motion_F = quant_motion.F.unsqueeze(0)
        ys2[3] = self.enc3(residual)
        ys2[4] = self.enc4(ys2[3])
        quant_y = quant(ys2[4].F.unsqueeze(0), training=self.training)

        # bit rate calculation
        p = self.BitEstimator(quant_y+0.5) - self.BitEstimator(quant_y-0.5)
        bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
        motion_p = self.MotionBitEstimator(quant_motion_F+0.5) - self.MotionBitEstimator(quant_motion_F-0.5)
        motion_bits = torch.sum(torch.clamp(-1.0 * torch.log(motion_p + 1e-10) / math.log(2.0), 0, 50))
        factor = 0.95
        if self.training:
            motion_bits = factor * motion_bits
        bpp = (bits + motion_bits) / num_points

        # point cloud reconstruction
        y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=ys2[4].coordinate_map_key,
                                   coordinate_manager=ys2[4].coordinate_manager, device=ys2[4].device)

        out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, ys2[2], True, residual=predicted_point2)
        out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], ys2[1], True, 1 if self.training else 1)
        out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], ys2[0], True, 1 if self.training else 1)
        return ys2, out2, out_cls2, target2, keep2, bpp


if __name__ == '__main__':
    from dataset_lossy import *
    import os

    torch.manual_seed(0)

    d_model = 32
    seq_len = 2000
    batch_size = 1
    num_heads = 4
    k_dim = 8

    tmp_dir = os.getcwd()
    '''    '''
    feat1 = torch.randint(low=0, high=2, size=(seq_len, 1), dtype=torch.float32)
    # coord1 = torch.randint(low=0, high=2000, size=(seq_len, 3), dtype=torch.float32)
    coord1 = [[2 * y for i in range(3)] for y in range(seq_len)]
    coord1 = torch.Tensor(coord1)
    coords1, feats1 = ME.utils.sparse_collate(coords=[coord1], feats=[feat1])
    # input1 = ME.SparseTensor(coordinates=coords1, features=feats1, tensor_stride=1)
    input1 = ME.SparseTensor(coordinates=coords1, features=feats1)

    feat2 = torch.randint(low=0, high=2, size=(seq_len, 1), dtype=torch.float32)
    # coord2 = torch.randint(low=0, high=2000, size=(seq_len, 3), dtype=torch.float32)
    coord2 = [[2 * y + 1 for i in range(3)] for y in range(seq_len)]
    coord2 = torch.Tensor(coord2)
    coords2, feats2 = ME.utils.sparse_collate(coords=[coord2], feats=[feat2])
    # input2 = ME.SparseTensor(coordinates=coords2, features=feats2, tensor_stride=1)
    input2 = ME.SparseTensor(coordinates=coords2, features=feats2)

    model_test = get_model(channels=8)
    _, out2, _, _, _, _ = model_test(input1, input2, device='cpu')  # device='cpu' may error in unpooling
    output = out2[-1]
    print(output.C.shape)  # output.C is final output points. 16-channel .F makes no sense