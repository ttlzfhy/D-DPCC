import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pytorch3d.ops import knn_points

from models.resnet import ResNet, InceptionResNet
import math


def sort_by_coor_sum(f, stride=None):
    if stride is None:
        stride = f.tensor_stride[0]
    xyz, feature = f.C, f.F
    maximum = xyz.max() + 1
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx], feature[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)
    return f_


def coordinate_sort_by_coor_sum(xyz):
    maximum = xyz.max() + 1
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_ = xyz[idx]
    return xyz_


def index_points(points, idx):
    """
    Input:
        points: input points data, [B*C, N, 1]
        idx: sample index data, [B*C, N, K]
    Return:
        new_points:, indexed points data, [C, N, K]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # B*C,1,1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 1,N,K
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def quant(x, training=False, qs=1):
    if training:
        compressed_x = x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5) * qs
    else:
        compressed_x = torch.round(x / qs) * qs
    return compressed_x


def merge_two_frames(f1, f2):
    stride = f1.tensor_stride[0]
    f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros_like(f1.F)], dim=-1), coordinates=f1.C,
                          tensor_stride=stride, device=f1.device)
    f2_ = ME.SparseTensor(torch.cat([torch.zeros_like(f2.F), f2.F], dim=-1), coordinates=f2.C,
                          tensor_stride=stride, coordinate_manager=f1_.coordinate_manager, device=f1.device)
    merged_f = f1_ + f2_
    merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C, tensor_stride=stride, device=merged_f.device)
    return merged_f


def index_by_channel(point1, idx, K=3):
    B, N1, C = point1.size()
    _, N2, C, __ = idx.size()  # (B, N2, C, K)
    point1_ = point1.transpose(1, 2).reshape(-1, N1, 1)  # (B*C, N1, 1)
    idx_ = idx.transpose(1, 2).reshape(-1, N2, K)  # (B*C, N2, K)
    knn_point1 = index_points(point1_, idx_).reshape(B, C, N2, K).transpose(1, 2)
    return knn_point1


def get_target_by_sp_tensor(out, target_sp_tensor):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 3] \
                         + coords[:, 2] * step \
                         + coords[:, 1] * step * step \
                         + coords[:, 0] * step * step * step
            return coords_sum

        step = max(out.C.max(), target_sp_tensor.C.max()) + 1

        out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
        in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C, step)

        # test whether each element of a 1-D array is also present in a second array.
        target = torch.isin(out_sp_tensor_coords_1d, in_sp_tensor_coords_1d)

    return target


def get_coords_nums_by_key(out, target_key):
    with torch.no_grad():
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])

        ins = cm.get_kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=1,
            region_type=1)

        row_indices_per_batch = out._batchwise_row_indices
        print(ins)
        print(row_indices_per_batch)

        coords_nums = [len(np.in1d(row_indices, ins[0]).nonzero()[0]) for _, row_indices in
                       enumerate(row_indices_per_batch)]

    return coords_nums


def keep_adaptive(out, coords_nums, rho=1.0):
    with torch.no_grad():
        keep = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        #  get row indices per batch.
        # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coordinate_map_key)
        row_indices_per_batch = out._batchwise_row_indices

        for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
            coords_num = min(len(row_indices), ori_coords_num * rho)  # select top k points.
            values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
            keep[row_indices[indices]] = True
    return keep


class inter_prediction(nn.Module):
    def __init__(self, input, hidden=64, output=8, kernel_size=2):
        super(inter_prediction, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels=input + input, out_channels=hidden, kernel_size=3, stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1,
                                             bias=True,
                                             dimension=3)
        self.down1 = DownsampleWithPruning(hidden, hidden, 3, 2, ResNet)
        self.up2 = DeconvWithPruning(hidden, hidden)
        self.down2 = DownsampleWithPruning(hidden, hidden, 3, 2, None)
        self.motion_compressor = ME.MinkowskiConvolution(in_channels=hidden, out_channels=output, kernel_size=2,
                                                         stride=2,
                                                         bias=True,
                                                         dimension=3)
        self.motion_decompressor1 = DeconvWithPruning(output, hidden)
        self.motion_decompressor2 = DeconvWithPruning(hidden, hidden)
        self.high_resolution_motion_generator = ME.MinkowskiConvolution(in_channels=hidden, out_channels=input * 3,
                                                                        kernel_size=3, stride=1,
                                                                        bias=True,
                                                                        dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)
        self.low_resolution_motion_generator = ME.MinkowskiConvolution(in_channels=hidden, out_channels=input * 3,
                                                                       kernel_size=3, stride=1,
                                                                       bias=True,
                                                                       dimension=3)
        self.conv_ref = ME.MinkowskiConvolution(in_channels=input, out_channels=1, kernel_size=2, stride=2,
                                                bias=True,
                                                dimension=3)
        self.conv_ref2 = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=2, stride=2,
                                                 bias=True,
                                                 dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def prune(self, f1, f2):
        mask = get_target_by_sp_tensor(f1, f2)
        out = self.pruning(f1, mask.to(f1.device))
        return out

    def get_downsampled_coordinate(self, x, stride, return_sorted=False):
        pc = ME.SparseTensor(torch.ones([x.size(0), 1], dtype=torch.float32, device=x.device), coordinates=x, tensor_stride=stride)
        downsampled = self.conv_ref2(pc)
        if return_sorted:
            downsampled = sort_by_coor_sum(downsampled, stride)
        return downsampled.C

    def decoder_side(self, quant_motion, f1, f2_coor, ys2_4_coor):
        ys2_4_coor_ = ME.SparseTensor(torch.ones([ys2_4_coor.size(0), 1], dtype=torch.float32, device=ys2_4_coor.device), ys2_4_coor, tensor_stride=8)
        f2_coor_ = ME.SparseTensor(torch.ones([f2_coor.size(0), 1], dtype=torch.float32, device=f2_coor.device), f2_coor, tensor_stride=4)
        reconstructed_motion1 = self.motion_decompressor1(quant_motion, ys2_4_coor_)

        # motion compensation
        # get motion flow m
        reconstructed_motion2 = self.motion_decompressor2(reconstructed_motion1, f2_coor_)
        m_f = self.high_resolution_motion_generator(reconstructed_motion2)

        m_c = self.low_resolution_motion_generator(reconstructed_motion1)
        m_c = self.unpooling(m_c)
        m_c = self.prune(m_c, f2_coor_)
        m = m_c + m_f

        # 3DAWI
        f2_coor_, m = sort_by_coor_sum(f2_coor_, 4), sort_by_coor_sum(m, 4)
        motion = m.F
        xyz1, xyz2, point1 = f1.C / 4, m.C / 4, f1.F
        xyz1, xyz2, point1 = xyz1[:, 1:].unsqueeze(0), xyz2[:, 1:].unsqueeze(0), point1.unsqueeze(0)
        B, N, C = 1, f2_coor_.size()[0], f1.size()[1]
        motion = motion.reshape(B, N, C, 3)
        xyz2_ = (xyz2.unsqueeze(2) + motion).reshape(B, -1, 3)
        dist, knn_index1_, __ = knn_points(xyz2_, xyz1, K=3)
        dist += 1e-8
        knn_index1_ = knn_index1_.reshape(B, N, C, 3)
        knn_point1_ = index_by_channel(point1, knn_index1_, 3)
        dist = dist.reshape(B, N, C, 3)
        weights = 1 / dist
        weights = weights / torch.clamp(weights.sum(dim=3, keepdim=True), min=3)
        predicted_point2 = (weights * knn_point1_).sum(dim=3).squeeze(0)
        predicted_f2 = ME.SparseTensor(predicted_point2, coordinates=f2_coor_.C, tensor_stride=4)
        return predicted_f2

    def forward(self, f1, f2, stride=8, training=True):
        # motion estimation
        assert f1.tensor_stride[0] == stride
        merged_f = merge_two_frames(f1, f2)
        out1 = self.relu(self.conv1(merged_f))
        e_o = self.relu(self.conv2(out1))
        ref = self.conv_ref(f2)
        e_c = self.down1(e_o)
        u1 = self.up2(e_c, e_o)
        delta_e = e_o - u1
        e_f = self.down2(delta_e, ref)
        e_c = self.prune(e_c, ref)
        e = e_c + e_f

        # motion compression
        compressed_motion2 = self.motion_compressor(e)
        quant_compressed_motion = ME.SparseTensor(quant(compressed_motion2.F, training=self.training),
                                                  coordinate_map_key=compressed_motion2.coordinate_map_key,
                                                  coordinate_manager=compressed_motion2.coordinate_manager)
        reconstructed_motion1 = self.motion_decompressor1(quant_compressed_motion, ref)

        # motion compensation
        # get motion flow m
        reconstructed_motion2 = self.motion_decompressor2(reconstructed_motion1, f2)
        m_f = self.high_resolution_motion_generator(reconstructed_motion2)

        m_c = self.low_resolution_motion_generator(reconstructed_motion1)
        m_c = self.unpooling(m_c)
        m_c = self.prune(m_c, f2)
        m = m_c + m_f

        # 3DAWI
        f2, m = sort_by_coor_sum(f2, stride), sort_by_coor_sum(m, stride)
        motion = m.F
        xyz1, xyz2, point1, point2 = f1.C / stride, m.C / stride, f1.F, f2.F
        xyz1, xyz2, point1, point2 = xyz1[:, 1:].unsqueeze(0), xyz2[:, 1:].unsqueeze(0), point1.unsqueeze(
            0), point2.unsqueeze(0)
        B, N, C = point2.size()
        motion = motion.reshape(B, N, C, 3)
        xyz2_ = (xyz2.unsqueeze(2) + motion).reshape(B, -1, 3)
        dist, knn_index1_, __ = knn_points(xyz2_, xyz1, K=3)
        dist += 1e-8
        knn_index1_ = knn_index1_.reshape(B, N, C, 3)
        knn_point1_ = index_by_channel(point1, knn_index1_, 3)
        dist = dist.reshape(B, N, C, 3)
        weights = 1 / dist
        weights = weights / torch.clamp(weights.sum(dim=3, keepdim=True), min=3)
        predicted_point2 = (weights * knn_point1_).sum(dim=3).squeeze(0)
        predicted_f2 = ME.SparseTensor(predicted_point2, coordinates=f2.C, coordinate_manager=f2.coordinate_manager,
                                       tensor_stride=stride, device=f2.device)

        # get residual
        residual_f2 = f2 - predicted_f2
        residual_f2 = ME.SparseTensor(residual_f2.F, coordinates=residual_f2.C, tensor_stride=stride, device=f2.device)
        if training:
            return residual_f2, predicted_f2, quant_compressed_motion
        else:
            return residual_f2, predicted_f2, quant_compressed_motion, m


class DownsampleLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2, resnet=InceptionResNet):
        super(DownsampleLayer, self).__init__()
        self.resnet = resnet
        self.conv = ME.MinkowskiConvolution(
            in_channels=input,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=output,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        if resnet is not None:
            self.block = self.make_layer(resnet, block_layers, output)
        self.relu = ME.MinkowskiReLU()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.down(self.relu(self.conv(x)))
        if self.resnet is not None:
            out = self.block(self.relu(out))
        return out


class DownsampleWithPruning(nn.Module):
    def __init__(self, input, output, block_layers, kernel=2, resnet=InceptionResNet):
        super(DownsampleWithPruning, self).__init__()
        self.resnet = resnet
        self.down = ME.MinkowskiConvolution(
            in_channels=input,
            out_channels=output,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        if resnet is not None:
            self.block = self.make_layer(resnet, block_layers, output)
        self.relu = ME.MinkowskiReLU()
        self.pruning = ME.MinkowskiPruning()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x, ref=None):
        out = self.down(x)
        if self.resnet is not None:
            out = self.block(self.relu(out))
        if ref is not None:
            mask = get_target_by_sp_tensor(out, ref)
            out = self.pruning(out, mask.to(out.device))
        return out


class UpsampleLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2):
        super(UpsampleLayer, self).__init__()
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=input,
            out_channels=hidden,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        self.conv = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=output,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, block_layers, output)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels=output,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
        return nn.Sequential(*layers)

    def get_cls(self, x):
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        out_cls = self.conv_cls(out)
        return out_cls

    def evaluate(self, x, adaptive, num_points, rho=1, residual=None, lossless=False):
        training = self.training
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        if residual is not None:
            residual = ME.SparseTensor(residual.F, coordinates=residual.C,
                                       coordinate_manager=out.coordinate_manager)
            out = out + residual
            out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=4)
        out_cls = self.conv_cls(out)

        if adaptive:
            coords_nums = num_points
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, keep

    def forward(self, x, target_label, adaptive, rho=1, residual=None, lossless=False):
        training = self.training
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        if residual is not None:
            residual = ME.SparseTensor(residual.F, coordinates=residual.C,
                                       coordinate_manager=out.coordinate_manager)
            out = out + residual
            out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=4)
        out_cls = self.conv_cls(out)
        target = get_target_by_sp_tensor(out, target_label)

        if adaptive:
            coords_nums = [len(coords) for coords in target_label.decomposed_coordinates]
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable
        if training or residual is not None:
            keep += target
        elif lossless:
            keep = target

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, target, keep


class DeconvWithPruning(nn.Module):
    def __init__(self, input, output):
        super(DeconvWithPruning, self).__init__()
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=input,
            out_channels=output,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def forward(self, x, ref=None):
        out = self.up(x)
        if ref is not None:
            mask = get_target_by_sp_tensor(out, ref)
            out = self.pruning(out, mask.to(x.device))
        return out


class Bitparm(nn.Module):
    # save params
    def __init__(self, channel, dimension=4, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        para = [1 for i in range(dimension)]
        para[dimension - 1] = -1
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    def __init__(self, channel, dimension=3):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel, dimension=dimension)
        self.f2 = Bitparm(channel, dimension=dimension)
        self.f3 = Bitparm(channel, dimension=dimension)
        self.f4 = Bitparm(channel, dimension=dimension, final=True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class LosslessCompressor(nn.Module):
    # for coding C(ys2) losslessly
    def __init__(self):
        super(LosslessCompressor, self).__init__()
        self.compressor1 = DownsampleLayer(1, 16, 32, 3)
        self.compressor2 = ME.MinkowskiConvolution(in_channels=32, out_channels=4, kernel_size=3,
                                                   stride=1,
                                                   bias=True,
                                                   dimension=3)
        self.decompressor1 = UpsampleLayer(4, 16, 32, 3, kernel=2)
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.bitEstimator = BitEstimator(4, 3)

    def get_cls(self, pc):
        return self.decompressor1.get_cls(pc)

    def forward(self, pc, num_points, sort_coordinates=False):
        out1 = self.compressor1(pc)
        out2 = self.compressor2(out1)
        quant_out2 = ME.SparseTensor(quant(out2.F, training=self.training),
                                     coordinate_map_key=out2.coordinate_map_key,
                                     coordinate_manager=out2.coordinate_manager,
                                     device=out2.device)
        if sort_coordinates:
            quant_out2 = sort_by_coor_sum(quant_out2, 8)
        out3, cls, target, keep = self.decompressor1(quant_out2, pc, True)
        bits1 = self.bce(cls.F.squeeze(),
                         target.type(cls.F.dtype).to(pc.device)) / math.log(2)
        p = self.bitEstimator(quant_out2.F + 0.5) - self.bitEstimator(quant_out2.F - 0.5)
        bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
        bits = bits1 + bits
        return bits, quant_out2, cls, target
