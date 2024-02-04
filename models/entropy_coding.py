import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchac


def factorized_entropy_coding(bitEstimator, input):
    input = input.squeeze(0)
    min_v = input.min().detach().float()
    max_v = input.max().detach().float()
    symbols = torch.arange(min_v, max_v + 1, device=input.device)
    symbols = symbols.reshape(1, -1, 1).repeat(1, 1, input.shape[-1])
    input_norm = input - min_v
    min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
    input_norm = input_norm.to(torch.int16)
    ''' get pmf '''
    pmf = bitEstimator(symbols+0.5)-bitEstimator(symbols-0.5)
    # print(symbols, pmf)
    pmf = pmf.squeeze(0).permute(1, 0)
    ''' pmf to cdf '''
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    ''' arithmetic coding '''
    out_cdf = cdf_with_0.unsqueeze(0).repeat(input_norm.shape[0], 1, 1).detach().cpu()
    strings = torchac.encode_float_cdf(out_cdf.cpu(), input_norm.cpu(), check_input_bounds=True)
    return strings, min_v, max_v


def binary_entropy_coding(p, x):
    """
    for losslessly compress 2x down coord's occupancy
    """
    ''' get pmf '''
    pmf = torch.cat([1 - p, p], dim=-1)
    ''' pmf to cdf '''
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    ''' arithmetic coding '''
    out_cdf = cdf_with_0.detach().cpu()
    x = x.detach().cpu().to(torch.int16)
    strings1 = torchac.encode_float_cdf(out_cdf, x, check_input_bounds=True)
    return strings1


def factorized_entropy_decoding(bitEstimator, shape, bitstream, min_v, max_v, device):
    symbols = torch.arange(min_v, max_v + 1).to(device)
    symbols = symbols.reshape(1, -1, 1).repeat(1, 1, shape[-1])
    ''' get pmf '''
    pmf = bitEstimator(symbols+0.5)-bitEstimator(symbols-0.5)
    pmf = pmf.squeeze(0).permute(1, 0)
    ''' pmf to cdf '''
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    ''' arithmetic decoding '''
    out_cdf = cdf_with_0.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
    values = torchac.decode_float_cdf(out_cdf, bitstream)
    values = values + min_v
    return values


def binary_entropy_decoding(p, bitstream):
    pmf = torch.cat([1 - p, p], dim=-1)
    ''' pmf to cdf '''
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    ''' arithmetic decoding '''
    out_cdf = cdf_with_0.detach().cpu()
    values = torchac.decode_float_cdf(out_cdf, bitstream)
    return values



