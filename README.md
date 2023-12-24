# LACNN


Tensorflow and Pytorch implementations for the paper "Attention to Lesion: Lesion-Aware Convolutional Neural Network for Retinal Optical Coherence Tomography Image Classification". Email: chongwangsmu@gmail.com

## Overview:

The algorithm uses lesion-related domain knowledge for retinal disease diagnosis based on optical coherence tomography (OCT) images. The lesion attention map is adopted to weight the convolutional feature representations, making the classifier focus more on lesion regions for classification.


![image](https://github.com/runningcw/LACNN/blob/master/LACNN-torch/LACNN.png)


## Dataset:

The UCSD OCT dataset can be found [here](https://data.mendeley.com/datasets/rscbjbr9sj).

For the NEH OCT dataset, please refer to this [paper](https://ieeexplore.ieee.org/document/8166817).


## [Tensorflow version](https://github.com/runningcw/LACNN/tree/master/LACNN-tensorflow): (tensorflow 1.3+, python 2.7)

Usage:

(1) Generate_attenmap.py to create attention maps for UCSD dataset. (The tensorflow version of the trained LDN model is deprecated, please use the torch version)

(2) LACNN_train.py to train LACNN, VGG16.npy file can be found [here](https://github.com/machrisaa/tensorflow-vgg).                                       

(3) LACNN_test.py to test LACNN.



## [Pytorch version](https://github.com/runningcw/LACNN/tree/master/LACNN-torch): (1.9.0+, python 3.7)

A pretrained lesion detection network (LDN) is [available](https://drive.google.com/drive/folders/1I2Ov3nbuRTWdOqTVli1IuZ8fWffnESUc?usp=drive_link). 

Usage:

(1) LDN training and test.

(2) LACNN traning and test.


## Citation:
   
__If you use the code, please consider citing the following paper:__

```
@article{fang2019attention,
  title={Attention to lesion: Lesion-aware convolutional neural network for retinal optical coherence tomography image classification},
  author={Fang, Leyuan and Wang, Chong and Li, Shutao and Rabbani, Hossein and Chen, Xiangdong and Liu, Zhimin},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={8},
  pages={1959--1970},
  year={2019},
  publisher={IEEE}
}
```
