import argparse
import time


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('DDPCC')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='DDPCC_geo', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.0008, type=float,
                        help='learning rate in training [default: 0.0008]')
    parser.add_argument('--gpu', type=str, default='7', help='specify gpu device [default: 0]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--lamb', default=10.0, type=float, help='loss=lambda*D+R')
    parser.add_argument('--pretrained', default='', type=str, help='path of pretrained model')
    parser.add_argument('--exp_name', default='I10', type=str, help='directory to store the result')
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--channels', default=8, type=int, help='bottleneck size')
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--dataset_dir', default='/home/zhaoxudong/dataset_npy', type=str)
    return parser.parse_args()
args = parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import importlib
import logging
import random
import shutil
from pathlib import Path
import torch
import sys
from tqdm import tqdm
from dataset_lossy import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
# torch.autograd.set_detect_anomaly(True)


def log_string(str):
    logger.info(str)
    print(str)


if __name__ == '__main__':

    '''mkdir'''
    if args.exp_name is None:
        args.exp_name = ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 10))
    log_root_dir = Path('./log')
    log_root_dir.mkdir(exist_ok=True)
    model_dir = log_root_dir.joinpath(args.model)
    model_dir.mkdir(exist_ok=True)
    experiment_dir = model_dir.joinpath(args.exp_name)
    experiment_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('log')
    checkpoint_dir = experiment_dir.joinpath('checkpoint')
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    '''logger'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''tensorboardX'''
    writer = SummaryWriter(comment='_add_regularization')

    '''dataset'''
    train_dataset = Dataset(root_dir=args.dataset_dir, split=[0, 1, 2, 3], type='train', scaling_factor=1)
    val_dataset = Dataset(root_dir=args.dataset_dir, split=[0, 1, 2, 3], type='val', scaling_factor=1)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                    collate_fn=collate_pointcloud_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                  collate_fn=collate_pointcloud_fn)

    '''model'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/model_utils.py', str(experiment_dir))
    model = MODEL.get_model(channels=args.channels)

    '''pretrained'''
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoint/best_model.pth', map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Detected existed model')
        exist = True
    except:
        log_string('No existing model')
        start_epoch = 0
        exist = False

    if len(args.pretrained) != 0:
        checkpoint = torch.load(args.pretrained)
        start_epoch = 0
        import collections

        old_paras = model.state_dict()
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k1 = k.replace('module.', '')
            if k1 in old_paras:
                new_state_dict[k1] = v
        old_paras.update(new_state_dict)
        model.load_state_dict(old_paras)
        log_string('Finetuning')

    if not args.cpu:
        model = model.cuda()

    '''optimizer'''
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    if exist:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('use exist optimizer')
        best_loss_test = 99999999
    else:
        best_loss_test = 99999999
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    for i in range(start_epoch % 15):  # % scheduler step_size
        scheduler.step()

    '''training'''
    log_string('start training')
    step = 0
    for epoch in range(start_epoch, args.epoch):
        log_string('\nEpoch: ' + str(epoch+1))
        total_bpp = 0
        total_cls = 0
        total_loss = 0
        counter = 0
        model.train()
        device = torch.device('cuda' if not args.cpu else 'cpu')
        if epoch >= 10 or len(args.pretrained) != 0:
            lamb = args.lamb
        else:
            lamb = 20.0
        log_string('current lambda: ' + str(lamb))
        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            try:
                xyz, point, xyz1, point1 = data
                xyz, point, xyz1, point1 = xyz.to(torch.float32), point.to(torch.float32), xyz1.to(
                    torch.float32), point1.to(torch.float32)
                # print(xyz.size(), xyz1.size())
                f1 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                f2 = ME.SparseTensor(features=point1, coordinates=xyz1, device=device)
                ys2, out2, out_cls2, targets2, keeps2, bpp = model(f1, f2, device, epoch)
                distortion = 0
                distortion_ = 0
                num_losses = len(out_cls2)
                for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
                    curr_loss = model.crit(out_cl.F.squeeze(),
                                           target.type(out_cl.F.dtype).to(device))
                    distortion += curr_loss / float(num_losses)
                loss = distortion * lamb + bpp
                # print(bpp, distortion)
                step = (step + 1) % args.batch_size
                loss.backward()
                if step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_bpp += bpp.item()
                total_cls += distortion.item()
                total_loss += loss.item()
                counter += 1
            except (MemoryError, RuntimeError, ValueError) as error:
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                time.sleep(2)
                log_string(str(error))
        avg_bpp = total_bpp / counter
        avg_cls = total_cls / counter
        avg_loss = total_loss / counter

        log_string('\naverage bpp: ' + str(avg_bpp))
        log_string('\naverage cls: ' + str(avg_cls))
        log_string('\naverage loss: ' + str(avg_loss))
        writer.add_scalar('Train Loss', avg_loss, epoch)
        writer.add_scalar('Train bpp', avg_bpp, epoch)
        writer.add_scalar('Train cls', avg_cls, epoch)
        scheduler.step()
        if epoch % 5 == 0:
            savepath = str(checkpoint_dir) + '/' + str(epoch) + '.pth'
            state = {
                'epoch': epoch,
                'loss': avg_bpp,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        log_string('evaluating')
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        test_total_bpp = 0
        test_total_cls = 0
        test_total_loss = 0
        test_counter = 0
        with torch.no_grad():
            for batch_id, data in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader), smoothing=0.9):
                xyz, point, xyz1, point1 = data
                xyz, point, xyz1, point1 = xyz.to(torch.float32), point.to(torch.float32), xyz1.to(
                    torch.float32), point1.to(torch.float32)
                f1 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                f2 = ME.SparseTensor(features=point1, coordinates=xyz1, device=device)
                ys2, out2, out_cls2, targets2, keeps2, bpp = model(f1, f2, device)
                distortion = 0
                distortions = []
                num_losses = len(out_cls2)
                for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
                    curr_loss = model.crit(out_cl.F.squeeze(),
                                           target.type(out_cl.F.dtype).to(device))
                    distortion += curr_loss / float(num_losses)
                loss = distortion * args.lamb + bpp
                test_total_bpp += bpp.item()
                test_total_loss += loss.item()
                test_total_cls += distortion.item()
                test_counter += 1
        test_avg_bpp = test_total_bpp / test_counter
        test_avg_cls = test_total_cls / test_counter
        test_avg_loss = test_total_loss / test_counter
        if (test_avg_loss < best_loss_test):
            logger.info('Save model...')
            best_loss_test = test_avg_loss
            savepath = str(checkpoint_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'loss': test_avg_bpp,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        log_string('\ntest average bpp: ' + str(test_avg_bpp))
        log_string('\ntest average cls: ' + str(test_avg_cls))
        log_string('\ntest average loss: ' + str(test_avg_loss))
        writer.add_scalar('Test Loss', test_avg_loss, epoch)
        writer.add_scalar('Test bpp', test_avg_bpp, epoch)
        writer.add_scalar('Test cls', test_avg_cls, epoch)

