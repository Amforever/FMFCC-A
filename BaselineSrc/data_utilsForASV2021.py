"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    根据ptrol文件路径读取文件名称
"""
import torch
import collections
import os
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import h5py
import random  #因为数据太多，只提取一部分数据，随机置乱zzy



ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None,
        is_train=True, sample_size=None,              #sample_size=None，1000
        is_logical=True, feature_name=None,is_eval=False,protocols_dir=None,file_dir=None):


        self.sysid_dict = {
            'N01': 1,
            'N01_ND1': 2,
            'N01_ND02': 3,
            'N01_MP3': 4,
            'N01_AAC': 5,
            'A01': 6,
            'A01_ND1': 7,
            'A01_ND02': 8,
            'A01_MP3': 9,
            'A01_AAC': 10,
            'A02': 11,
            'A02_ND1': 12,
            'A02_ND02': 13,
            'A02_MP3': 14,
            'A02_AAC': 15,
            'A03': 16,
            'A03_ND1': 17,
            'A03_ND02': 18,
            'A03_MP3': 19,
            'A03_AAC': 20,
            'A04': 21,
            'A04_ND1': 22,
            'A04_ND02': 23,
            'A04_MP3': 24,
            'A04_AAC': 25,
            'A05': 26,
            'A05_ND1': 27,
            'A05_ND02': 28,
            'A05_MP3': 29,
            'A05_AAC': 30,
            'A06': 31,
            'A06_ND1': 32,
            'A06_ND02': 33,
            'A06_MP3': 34,
            'A06_AAC': 35,
            'A07': 36,
            'A07_ND1': 37,
            'A07_ND02': 38,
            'A07_MP3': 39,
            'A07_AAC': 40,
            'A08': 41,
            'A08_ND1': 42,
            'A08_ND02': 43,
            'A08_MP3': 44,
            'A08_AAC': 45,
            'A09': 46,
            'A09_ND1': 47,
            'A09_ND02': 48,
            'A09_MP3': 49,
            'A09_AAC': 50,
            'A10': 51,
            'A10_ND1': 52,
            'A10_ND02': 53,
            'A10_MP3': 54,
            'A10_AAC': 55,
            'A11': 56,
            'A11_ND1': 57,
            'A11_ND02': 58,
            'A11_MP3': 59,
            'A11_AAC': 60,
            'A12': 61,
            'A12_ND1': 62,
            'A12_ND02': 63,
            'A12_MP3': 64,
            'A12_AAC': 65,
            'A13': 66,
            'A13_ND1': 67,
            'A13_ND02': 68,
            'A13_MP3': 69,
            'A13_AAC': 70,
            'N71':71
        }
        self.is_eval = is_eval
        self.protocols_fname=protocols_dir
        self.files_dir=file_dir
        self.transform = transform                          ###############################################数据封装 封装 封装
        self.files_meta = self.parse_protocols_file(self.protocols_fname)   #把protocol文件中 所有的行的文件都进行了分解，还算快
        self.length=self.files_meta.__len__()
        # print(self.files_meta[:][2])
        self.x_data = [i[2] for i in self.files_meta]  #b = [i[0] for i in a]  从a中的每一行取第一个元素。
        # print(b)
        self.y_data=[i[4] for i in self.files_meta]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # x = self.data_x[idx]
        # y = self.data_y[idx]
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)  #根据 文件元路径 读取音频
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id        #data_y也就是key 是0和1分别表示bonafide和spoof，sys_id是合成的类别，分为0~19

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        if self.is_eval:
            return ASVFile(speaker_id=tokens[1],             #''
                file_name=tokens[0],                  #0                                #这个是比赛时读取测试集protocol文件的内容，现在测试的文件和比赛时不一样，
                path=os.path.join(self.files_dir, tokens[0] ), #'0'              #现在19年发布的数据时测试集的protocol是和训练集、测试集的protocol时一样的。
                sys_id=self.sysid_dict[tokens[3]],                                               #0  3
                key=int(tokens[2] == '1'))       #文件中1 表示真；所以key中1表示真，0表示假；这个地方是主要 确定标签的地方，int() 表示如果protrol中key值是1（）就标定为1；key值为0 标定位0；
        return ASVFile(speaker_id=tokens[1],
            file_name=tokens[0],
            path=os.path.join(self.files_dir, tokens[0]),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[2] == '1'))

    def parse_protocols_file(self, protocols_fname):      ##解析protocol文件中每一行内容的细节。即每一行记录的元组 meta
        lines = open(protocols_fname).readlines()
        #lines=random.shuffle(lines)                 #lineShuffled  ZZY
        #lines=lines[:1000]             #lineShuffledTop1000    ZZY

        # lineShuffled=random.sample(lines, len(lines))       #ZZY
        # lineShuffledTop1000=lineShuffled[:20]             #ZZY
        lineShuffledTop1000 = lines[:]    #20
        files_meta = map(self._parse_line, lineShuffledTop1000)  #lines   lineShuffledTop1000
        return list(files_meta)

    def get_weights_FromTrainDataset(self):
        label_info = self.y_data
        # num_zero_class = sum( [x for x in label_info if x==1]) #human为1 value为待检测多媒体文件的【真实概率值】
        # num_one_class = sum( [xx for xx in label_info if xx==0])   #spoof为0
        num_one_class = len([x for x in label_info if x==1])#human为1 value为待检测多媒体文件的【真实概率值】
        num_zero_class=  len([xx for xx in label_info if xx==0])    #spoof为0
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights









