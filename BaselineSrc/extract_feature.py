# @Author :ZhenyuZhang
# @CrateTime   :2021/4/4 17:50
# @File   :extract_feature.py
# @Email  :201718018670196

'''
提取特征，包含不同的特征方法'
'''

import argparse
import sys
import os
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch.nn.functional as F    #ByZZY
import time                         #ByZZY
from text_preprocess_zzy import text_read
import torch
from torch import nn
from tensorboardX import SummaryWriter
from eval_metrics import compute_eer
import scipy.io as scio
import soundfile as sf
from joblib import Parallel, delayed
from CQCC import *                 #ByZZY
from torch.optim.lr_scheduler import StepLR
import pysptk
import util_frontend as nii_front_end






def compute_SdDCQCC_feats(x):
    sr=16000
    audio=x
    y = audio
    B = 360  #89
    fmax = int(sr / 2)
    fmin = fmax / 2 ** 9
    d = 16   #改完117不变。不会更改cqcc维度
    cf = 24   #默认19，加上0系数刚好20；现在为了凑齐矩阵64*3  126矩阵
    ZsdD = 'ZsdD'  #原始的包括阿static和delat和deltaa
    # ZsdD = 's'
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec = cqcc(y, sr, B, fmax, fmin, d, cf, ZsdD) #返回的有相应的 delat和delta+delta
    CQcc2=CQcc[:72,]
    return CQcc2



def get_wav(x):
    return np.reshape(x, (1, -1))