import time

import numpy as np
import open3d

import os
import numpy as np
import subprocess
import pandas as pd


def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv12.
    You can download and install TMC13 from
    https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    subp = subprocess.Popen('./GPCC/tmc3' +
                            ' --mode=0' +
                            ' --positionQuantizationScale=1' +
                            ' --trisoupNodeSizeLog2=0' +
                            ' --neighbourAvailBoundaryLog2=8' +
                            ' --intra_pred_max_node_size_log2=6' +
                            ' --inferredDirectCodingMode=0' +
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath=' + filedir +
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return


def gpcc_decode(bin_dir, rec_dir, show=False):
    subp = subprocess.Popen('./GPCC/tmc3' +
                            ' --mode=1' +
                            ' --compressedStreamPath=' + bin_dir +
                            ' --reconstructedDataPath=' + rec_dir +
                            ' --outputBinaryPly=0'
                            ,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def pc_error(infile1, infile2, res, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)",
                "h.       1(p2point)", "h.,PSNR  1(p2point)"]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)",
                "h.       2(p2point)", "h.,PSNR  2(p2point)"]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)",
                "h.        (p2point)", "h.,PSNR   (p2point)"]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF
    # print('./utils/pc_error' +
    #                       ' -a '+infile1+
    #                       ' -b '+infile2+
    #                       # ' -n '+infile1+
    #                       ' --hausdorff=1 '+
    #                       ' --resolution='+str(res-1))

    command = str('./GPCC/pc_error' +
                  ' -a ' + infile1 +
                  ' -b ' + infile2 +
                  # ' -n '+infile1+
                  ' --hausdorff=1 ' +
                  ' --resolution=' + str(res - 1))

    if normal:
        headers += haders_p2plane
        command = str(command + ' -n ' + infile1)


    results = {}

    start = time.time()
    subp = subprocess.Popen(command,
                            shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')  # python3.
        # if show:
        #     print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c = subp.stdout.readline()
        # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])


def write_ply_data(filename, coords):
    if os.path.exists(filename):
        os.system('rm ' + filename)
    f = open(filename, 'a+')
    # print('data.shape:',data.shape)
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    for _, point in enumerate(coords):
        f.writelines([str(point[0]), ' ', str(point[1]), ' ', str(point[2]), '\n'])
    f.close()
    return


def write_ply_open3d_normal(filename, coords, dtype='int32', knn=20):
    pcd = open3d.geometry.PointCloud() 
    pcd.points = open3d.utility.Vector3dVector(coords.astype(dtype)) 
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=knn)) 
    open3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    f = open(filename)
    lines = f.readlines() 
    lines[4] = 'property float x\n' 
    lines[5] = 'property float y\n' 
    lines[6] = 'property float z\n' 
    lines[7] = 'property float nx\n' 
    lines[8] = 'property float ny\n' 
    lines[9] = 'property float nz\n' 
    fo = open(filename, "w")
    fo.writelines(lines) 
    return


def read_point_cloud(filename):
    pcd = open3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points)
