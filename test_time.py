import argparse
import importlib
import logging
import sys
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'PCGCv2'))

from models.model_utils import *
from tqdm import tqdm
from dataset_owlii import *
from models.entropy_coding import *
from GPCC.gpcc_wrapper import *
from PCGCv2.eval import test_one_frame
import pandas as pd


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Test Script')
    parser.add_argument('--model', type=str, default='DDPCC_geo')
    parser.add_argument('--lossless_model', type=str, default='DDPCC_lossless_coder')
    parser.add_argument('--log_name', type=str, default='aaa')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device [default: 0]')
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--ckpt_dir', type=str,
                        default='./ddpcc_ckpts')
    parser.add_argument('--pcgcv2_ckpt_dir', type=str,
                        default='./pcgcv2_ckpts')
    parser.add_argument('--frame_count', type=int, default=100, help='number of frames to be coded')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to store results (in csv format)')
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the bitstream of previous frame')
    parser.add_argument('--dataset_dir', type=str, default='/home/zhaoxudong/Owlii_10bit')
    return parser.parse_args()


def log_string(string):
    logger.info(string)
    print(string)


def encode(f1, f2, bitstream_filename, gpcc_bitstream_filename):
    ys1, ys2 = [f1, 0, 0, 0, 0], [f2, 0, 0, 0, 0]
    # feature extraction
    ys1[1] = model.enc1(ys1[0])
    ys1[2] = model.enc2(ys1[1])
    ys2[1] = model.enc1(ys2[0])
    ys2[2] = model.enc2(ys2[1])

    # inter prediction
    residual, predicted_f2, quant_motion = model.inter_prediction(ys1[2], ys2[2], stride=4)
    quant_motion = sort_by_coor_sum(quant_motion, 16)
    quant_motion_F = quant_motion.F.unsqueeze(0)

    # residual compression
    ys2[3] = model.enc3(residual)
    ys2[4] = model.enc4(ys2[3])
    ys2[4] = sort_by_coor_sum(ys2[4], 8)
    quant_y = quant(ys2[4].F.unsqueeze(0), training=model.training)

    # encode C_{x_t}^2
    ys2_2 = ME.SparseTensor(torch.ones_like(ys2[2].C[:, :1], dtype=torch.float32), coordinates=ys2[2].C,
                            tensor_stride=4)
    ys2_2 = sort_by_coor_sum(ys2_2, 4)
    _, ys2_2_feature, ys2_2_cls, ys2_2_target = lossless_model.compressor(ys2_2, ys2_2.size()[0],
                                                                          sort_coordinates=True)
    p = torch.sigmoid(ys2_2_cls.F)

    # entropy coding
    motion_bitstream, min_v_motion, max_v_motion = factorized_entropy_coding(model.MotionBitEstimator,
                                                                             quant_motion_F)
    residual_bitstream, min_v_res, max_v_res = factorized_entropy_coding(model.BitEstimator, quant_y)
    ys2_2_feature_bitstream, min_v_res2, max_v_res2 = factorized_entropy_coding(
        lossless_model.compressor.bitEstimator, ys2_2_feature.F)
    ys2_2_bitstream = binary_entropy_coding(p, ys2_2_target)
    ys2_4_C = ys2[4].decomposed_coordinates[0].detach().cpu().numpy()
    write_ply_data(os.path.join(tmp_dir, 'ys2_4.ply'), ys2_4_C / 8)
    gpcc_encode(os.path.join(tmp_dir, 'ys2_4.ply'), gpcc_bitstream_filename)
    file = open(bitstream_filename, 'wb')
    file.write(np.array(min_v_motion, dtype=np.int8).tobytes())
    file.write(np.array(max_v_motion, dtype=np.int8).tobytes())
    file.write(np.array(min_v_res, dtype=np.int8).tobytes())
    file.write(np.array(max_v_res, dtype=np.int8).tobytes())
    file.write(np.array(min_v_res2, dtype=np.int8).tobytes())
    file.write(np.array(max_v_res2, dtype=np.int8).tobytes())
    file.write(np.array(quant_y.shape[1], dtype=np.int16).tobytes())
    file.write(np.array(quant_motion.shape[0], dtype=np.int16).tobytes())
    file.write(np.array(ys2[0].shape[0], dtype=np.int32).tobytes())
    file.write(np.array(ys2[1].shape[0], dtype=np.int32).tobytes())
    file.write(np.array(len(motion_bitstream), dtype=np.int16).tobytes())
    file.write(np.array(len(ys2_2_feature_bitstream), dtype=np.int16).tobytes())
    file.write(np.array(len(ys2_2_bitstream), dtype=np.int16).tobytes())
    file.write(motion_bitstream)
    file.write(ys2_2_feature_bitstream)
    file.write(ys2_2_bitstream)
    file.write(residual_bitstream)
    file.close()


def decode(f1, bitstream_filename, gpcc_bitstream_filename):
    ys1 = [f1, 0, 0]
    file = open(bitstream_filename, 'rb')
    min_v_motion_, max_v_motion_, min_v_res_, max_v_res_, min_v_res2_, max_v_res2_ = np.frombuffer(
        file.read(6), dtype=np.int8)
    quant_y_length, quant_motion_length = np.frombuffer(
        file.read(4), dtype=np.int16)
    num_points_0, num_points_1 = np.frombuffer(
        file.read(8), dtype=np.int32)
    motion_bitstream_length, ys2_2_feature_bitstream_length, ys2_2_bitstream_length = np.frombuffer(
        file.read(6), dtype=np.int16)
    motion_bitstream_ = file.read(motion_bitstream_length)
    ys2_2_feature_bitstream_ = file.read(ys2_2_feature_bitstream_length)
    ys2_2_bitstream_ = file.read(ys2_2_bitstream_length)
    residual_bitstream_ = file.read()
    ys1[1] = model.enc1(ys1[0])
    ys1[2] = model.enc2(ys1[1])
    quant_y_F = factorized_entropy_decoding(model.BitEstimator, [quant_y_length, 8],
                                            residual_bitstream_,
                                            min_v_res_, max_v_res_, device).to(device).to(torch.float32)
    quant_motion_F_ = factorized_entropy_decoding(model.MotionBitEstimator, [quant_motion_length, 48],
                                                  motion_bitstream_, min_v_motion_, max_v_motion_,
                                                  device).to(device).to(torch.float32)
    ys2_2_feature_F = factorized_entropy_decoding(lossless_model.compressor.bitEstimator,
                                                  [quant_y_length, 4], ys2_2_feature_bitstream_,
                                                  min_v_res2_, max_v_res2_, device).to(device).to(
        torch.float32)

    gpcc_decode(gpcc_bitstream_filename, os.path.join(tmp_dir, 'recon_ys2_4.ply'))
    recon_ys2_4_C = 8 * torch.tensor(read_point_cloud(os.path.join(tmp_dir, 'recon_ys2_4.ply')),
                                     dtype=torch.int32, device=device)
    recon_ys2_4_C = torch.cat([torch.zeros_like(recon_ys2_4_C[:, :1]), recon_ys2_4_C], dim=-1)
    recon_ys2_4_C = coordinate_sort_by_coor_sum(recon_ys2_4_C)
    recon_ys2_2_feature = ME.SparseTensor(ys2_2_feature_F, coordinates=recon_ys2_4_C, tensor_stride=8)
    recon_ys2_2_cls = lossless_model.compressor.get_cls(recon_ys2_2_feature)
    recon_p = torch.sigmoid(recon_ys2_2_cls.F)
    ys2_2_mask = binary_entropy_decoding(recon_p, ys2_2_bitstream_).to(torch.bool).to(device)
    recon_ys2_2_C = ME.MinkowskiPruning()(recon_ys2_2_cls, ys2_2_mask).C
    y2_recon_ = ME.SparseTensor(quant_y_F, coordinates=recon_ys2_4_C, tensor_stride=8)

    motion_coor = model.inter_prediction.get_downsampled_coordinate(recon_ys2_4_C, 8,
                                                                    return_sorted=True)
    recon_quant_motion = ME.SparseTensor(quant_motion_F_, coordinates=motion_coor, tensor_stride=16)
    recon_predicted_f2 = model.inter_prediction.decoder_side(recon_quant_motion, ys1[2], recon_ys2_2_C,
                                                             recon_ys2_4_C)

    # point cloud reconstruction
    out2[0], out_cls2[0], target2[0], keep2[0] = model.dec1(y2_recon_, recon_predicted_f2, True,
                                                            residual=recon_predicted_f2)
    out2[1], out_cls2[1], keep2[1] = model.dec2.evaluate(out2[0], True, [num_points_1], 1)
    out2[2], out_cls2[2], keep2[2] = model.dec3.evaluate(out2[1], True, [num_points_0], 1)

    recon_f2 = ME.SparseTensor(torch.ones_like(out2[2].F[:, :1]), coordinates=out2[2].C)
    recon_f2_C = recon_f2.decomposed_coordinates[0].detach().cpu().numpy()
    f2_C = f2.decomposed_coordinates[0].detach().cpu().numpy()
    return recon_f2_C, f2_C, recon_f2

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(int(args.gpu))
    device = torch.device('cuda')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./%s.txt' % args.log_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    tmp_dir = args.tmp_dir
    # tmp_dir = './tmp_'+''.join(random.sample('0123456789', 10))
    tmp_dir_ = Path(tmp_dir)
    tmp_dir_.mkdir(exist_ok=True)
    results_dir = args.results_dir
    results_dir_ = Path(results_dir)
    results_dir_.mkdir(exist_ok=True)

    # load model
    log_string('PARAMETER ...')
    log_string(args)
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(channels=args.channels)

    LOSSLESS_MODEL = importlib.import_module(args.lossless_model)
    lossless_model = LOSSLESS_MODEL.get_model()
    lossless_checkpoint = torch.load(os.path.join(args.ckpt_dir, 'lossless_coder.pth'))
    lossless_model.load_state_dict(lossless_checkpoint['model_state_dict'])
    lossless_model = lossless_model.to(device).eval()

    results = {
        'basketball': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'num_of_bits': [], 'num_of_points': [], 'bpip': []},
        'dancer': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'num_of_bits': [], 'num_of_points': [], 'bpip': []},
        'exercise': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'num_of_bits': [], 'num_of_points': [], 'bpip': []},
        'model': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'num_of_bits': [], 'num_of_points': [], 'bpip': []}
    }
    '''
    start testing
    5: basketballplayer
    6: dancer
    8: exercise
    13: model
    '''
    # ckpts = {
    #     'r1_0.025bpp.pth': 'r1.pth',
    #     'r2_0.05bpp.pth': 'r2.pth',
    #     'r3_0.10bpp.pth': 'r3.pth',
    #     'r4_0.15bpp.pth': 'r4.pth',
    #     'r5_0.25bpp.pth': 'r5.pth',
    #     'r6_0.3bpp.pth': 'r6.pth',
    #     'r7_0.4bpp.pth': 'r7.pth',
    # }
    ckpts = {
        'r3_0.10bpp.pth': 'r1.pth',
        'r4_0.15bpp.pth': 'r2.pth',
        'r5_0.25bpp.pth': 'r3.pth',
        'r6_0.3bpp.pth': 'r4.pth',
        'r7_0.4bpp.pth': 'r5.pth'
    }
    with torch.no_grad():
        for pcgcv2_ckpt in ckpts:
            ddpcc_ckpt = os.path.join(args.ckpt_dir, ckpts[pcgcv2_ckpt])
            pcgcv2_ckpt = os.path.join(args.pcgcv2_ckpt_dir, pcgcv2_ckpt)
            checkpoint = torch.load(ddpcc_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            for sequence in (0, 1, 2, 3):
                dataset = Dataset(root_dir=args.dataset_dir, split=[sequence], type='test', format='ply')
                sequence_name = dataset.sequence_list[sequence]
                log_string('start testing sequence ' + sequence_name + ', rate point ' + ddpcc_ckpt)
                log_string('f bpp     d1PSNR  d2PSNR')
                d1_psnr_sum = 0
                d2_psnr_sum = 0
                bpp_sum = 0
                bits_sum = 0
                num_points_sum = 0

                # encode the first frame
                xyz, point, xyz1, point1 = collate_pointcloud_fn([dataset[0]])
                f1 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                bpp, d1psnr, d2psnr, f1 = test_one_frame(f1, pcgcv2_ckpt, os.path.join(tmp_dir, 'pcgcv2'))
                f1 = ME.SparseTensor(torch.ones_like(f1.F[:, :1]), coordinates=f1.C)
                log_string(str(0) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                bpp_sum += bpp
                d1_psnr_sum += d1psnr
                d2_psnr_sum += d2psnr
                num_points_sum += (f1.size()[0] * 1.0)
                bits_sum += (f1.size()[0] * bpp)

                for i in range(1, args.frame_count):
                    if args.overwrite:
                        bitstream_filename = os.path.join(tmp_dir, 'bitstream.bin')
                        gpcc_bitstream_filename = os.path.join(tmp_dir, 'ys2_4.bin')
                    else:
                        bitstream_filename = os.path.join(tmp_dir, 'bitstream_' + str(i) + '.bin')
                        gpcc_bitstream_filename = os.path.join(tmp_dir, 'ys2_4_' + str(i) + '.bin')
                    out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                    xyz, point, xyz1, point1 = collate_pointcloud_fn([dataset[i-1]])
                    f2 = ME.SparseTensor(features=point1, coordinates=xyz1, device=device)
                    num_points = f2.size()[0]

                    encoding_st = time.time()
                    encode(f1, f2, bitstream_filename, gpcc_bitstream_filename)
                    encoding_et = time.time()

                    log_string('encoding time: ' + str(encoding_et - encoding_st))
                    ddpcc_bpp = os.path.getsize(bitstream_filename) * 8 / num_points
                    gpcc_bpp = os.path.getsize(gpcc_bitstream_filename) * 8 / num_points
                    bpp = ddpcc_bpp + gpcc_bpp

                    decoding_st = time.time()
                    recon_f2_C, f2_C, recon_f2 = decode(f1, bitstream_filename, gpcc_bitstream_filename)
                    decoding_et = time.time()
                    log_string('decoding time: ' + str(decoding_et - decoding_st))

                    # D1 D2
                    # write_ply_data(os.path.join(tmp_dir, 'f2.ply'), f2_C)
                    write_ply_open3d_normal(os.path.join(tmp_dir, 'f2.ply'), f2_C)
                    write_ply_data(os.path.join(tmp_dir, 'f2_recon.ply'), recon_f2_C)
                    PSNRs = pc_error(os.path.join(tmp_dir, 'f2.ply'), os.path.join(tmp_dir, 'f2_recon.ply'), 1024,
                                     normal=True)
                    d1psnr = PSNRs['mseF,PSNR (p2point)'][0]
                    d2psnr = PSNRs['mseF,PSNR (p2plane)'][0]
                    log_string(str(i) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                    f1 = recon_f2
                    bpp_sum += bpp
                    d1_psnr_sum += d1psnr
                    d2_psnr_sum += d2psnr
                    num_points_sum += (num_points * 1.0)
                    bits_sum += (num_points * bpp)
                bpp_avg = bpp_sum / args.frame_count
                d1_psnr_avg = d1_psnr_sum / args.frame_count
                d2_psnr_avg = d2_psnr_sum / args.frame_count
                bpip = bits_sum / num_points_sum
                results[sequence_name]['bpp'].append(bpp_avg)
                results[sequence_name]['d1-psnr'].append(d1_psnr_avg)
                results[sequence_name]['d2-psnr'].append(d2_psnr_avg)
                results[sequence_name]['num_of_points'].append(num_points_sum)
                results[sequence_name]['num_of_bits'].append(bits_sum)
                results[sequence_name]['bpip'].append(bpip)
                log_string(dataset.sequence_list[sequence] + ' average bpp: ' + str(bpp_avg))
                log_string(dataset.sequence_list[sequence] + ' average d1-psnr: ' + str(d1_psnr_avg))
                log_string(dataset.sequence_list[sequence] + ' average d2-psnr: ' + str(d2_psnr_avg))

    for sequence_name in results:
        df = pd.DataFrame(results[sequence_name])
        df.to_csv(os.path.join(results_dir, sequence_name + '.csv'), index=False)
