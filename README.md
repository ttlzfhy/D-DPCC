# D-DPCC
This is the code of D-DPCC: Deep Dynamic Point Cloud Compression via 3D Motion Prediction.

Link of the paper: https://www.ijcai.org/proceedings/2022/0126.pdf



# Reference
If you want to cite our work, please use the following reference:

@inproceedings{ijcai2022p126,
  title     = {D-DPCC: Deep Dynamic Point Cloud Compression via 3D Motion Prediction},
  author    = {Fan, Tingyu and Gao, Linyao and Xu, Yiling and Li, Zhu and Wang, Dong},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {898--904},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/126},
  url       = {https://doi.org/10.24963/ijcai.2022/126},
}



# Requirements
(Also shown in ```requirements.txt```)

cuda~=11.5.50

numpy~=1.21.2

open3d~=0.14.1

pandas~=1.2.3

torch~=1.10.0

MinkowskiEngine~=0.5.4

pytorch3d~=0.6.1

tqdm~=4.62.3

tensorboardX~=2.5

matplotlib~=3.5.1

h5py~=3.6.0

torchac~=0.9.3

setuptools~=58.0.4

scipy~=1.7.3

scikit-learn~=1.0.2



# Train and Test
## Train D-DPCC models:
```shell
python trainer.py --batch_size=4 --gpu=7 --lamb=10 --exp_name=I10 --dataset_dir='/home/zhaoxudong/dataset_npy'
```
## Train lossless model for the compression of 2x downsampled coordinates:
```shell
python trainer_lossless.py --dataset_dir='/home/zhaoxudong/dataset_npy'
```
In fact, the pretrained model is ```lossless_coder.pth```. You probably needn't to retrain this model.

## Test:
Estimate the bitrate with factorized entropy model, without practical and separate encoding and decoding process:
```shell
python test_owlii.py --log_name='aaa' --gpu=1 --frame_count=32 --results_dir='results' --tmp_dir='tmp' --dataset_dir='/home/zhaoxudong/Owlii_10bit'
```
With separate encoding and decoding process, which generates real bitstream, 
and calculate encoding and decoding time.
```shell
python test_time.py --log_name='aaa' --gpu=1 --frame_count=32 --results_dir='results' --tmp_dir='tmp' --dataset_dir='/home/zhaoxudong/Owlii_10bit'
```

## Probable problems in testing:
- If ```./GPCC/tmc3: Permission denied```:
```shell
chmod 777 ./GPCC/tmc3
```

- If ```./GPCC/pc_error: Permission denied```:
```shell
chmod 777 ./GPCC/pc_error
```

- The folder ```PCGCv2``` need to be copied and in both the parent and current directory.



# Results
Shown in the folder ```results_csv```. 

See details in the MPEG proposal: M60267 “[AI-3DGC] D-DPCC Test Results on 10 bit Owlii”, 2022/7. 
