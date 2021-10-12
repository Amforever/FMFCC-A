"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
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

# LOGICAL_DATA_ROOT = 'D:/AudioForensics/ASVspoof2019LA/LA'   #data_logical
# PHISYCAL_DATA_ROOT = 'data_physical'

LOGICAL_DATA_ROOT = '/home1/zzy/Forensics/ASV2019LA'   #data_logical
PHISYCAL_DATA_ROOT = 'data_physical'

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None,
        is_train=True, sample_size=None,              #sample_size=None，1000
        is_logical=True, feature_name=None, is_eval=False,
        eval_part=0):
        if is_logical:
            data_root = LOGICAL_DATA_ROOT
            track = 'LA'
        else:
            data_root = PHISYCAL_DATA_ROOT
            track = 'PA'
        if is_eval:
            data_root = os.path.join('eval_data', data_root)
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        v1_suffix = ''
        if is_eval and track == 'PA':
            v1_suffix='_v1'
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'SS_1': 1, # Wavenet vocoder
            'SS_2': 2, # Conventional vocoder WORLD
            'SS_4': 3, # Conventional vocoder MERLIN
            'US_1': 4, # Unit selection system MaryTTS
            'VC_1': 5, # Voice conversion using neural networks
            'VC_4': 6, # transform function-based voice conversion
            ## For eval unknown spoof methods
            'A07': 16,
            'A08': 17,
            'A09': 18,
            'A10': 19,
            'A11': 20,
            'A12': 21,
            'A13': 22,
            'A14': 23,
            'A15': 24,
            'A16': 25,
            'A17': 26,
            'A18': 27,
            'A19': 28,
            # For PA:
            'AA':7,
            'AB':8,
            'AC':9,
            'BA':10,
            'BB':11,
            'BC':12,
            'CA':13,
            'CB':14,
            'CC': 15
        }
        self.is_eval = is_eval
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval_{}.trl'.format(eval_part) if is_eval else 'train.trn' if is_train else 'dev.trl'  #train.trn  或者  train.trn_agu
        self.protocols_dir = os.path.join(self.data_root,
            '{}_protocols/'.format(self.prefix))                                        #protocol协议文件的路径。prefix前缀ASVspoof2019_{}'.format(LA OR PA)
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name )+v1_suffix, 'flac')
        self.protocols_fname = os.path.join(self.protocols_dir,                        #确定最后的ASVspoof2019  cm    .txt protocols文件的名字
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        self.cache_fname = 'cache_{}{}_{}_{}.npy'.format(self.dset_name,
        '_part{}'.format(eval_part) if is_eval else '',track, feature_name)
        self.cache_matlab_fname = 'cache_{}{}_{}_{}.mat'.format(
            self.dset_name, '_part{}'.format(eval_part) if is_eval else '',
             track, feature_name)
        self.transform = transform                          ###############################################数据封装 封装 封装
        if os.path.exists(self.cache_fname):                             ##一旦训练过一次，将特征数据自动保存在，当前目录，从而再次运行时会自动调用。
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        elif feature_name == 'cqcc':
            if os.path.exists(self.cache_matlab_fname):
                self.data_x, self.data_y, self.data_sysid = self.read_matlab_cache(self.cache_matlab_fname)
                self.files_meta = self.parse_protocols_file(self.protocols_fname)
                print('Dataset loaded from matlab cache ', self.cache_matlab_fname)
                torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta),
                           self.cache_fname, pickle_protocol=4)
                print('Dataset saved to cache ', self.cache_fname)
            else:
                print("Matlab cache for cqcc feature do not exist.")
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)   #把protocol文件中 所有的行的文件都进行了分解，还算快
            # print(self.files_meta.__len__() )
            # data = list(map(self.read_file, self.files_meta))                     #按照read_file函数，根据file_meta数据，读取wav音频的数据
            # self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))   #压缩完之后，把所有的数据 标签 都合成一行。
            # if self.transform:
            #     # self.data_x = list(map(self.transform, self.data_x))
            #     self.data_x = Parallel(n_jobs=1, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)  #将读入的data_x数据提取特征，并转换为tensor的格式
            # torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            # print('Dataset saved to cache ', self.cache_fname)
        # if sample_size:
        #     select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)   #True表示可以取相同数字；False表示不可以取相同数字。随机选择samplesize个参数的 idx进行选出
        #     self.files_meta= [self.files_meta[x] for x in select_idx]
        #     self.data_x = [self.data_x[x] for x in select_idx]
        #     self.data_y = [self.data_y[x] for x in select_idx]
        #     self.data_sysid = [self.data_sysid[x] for x in select_idx]
        # self.length = len(self.data_x)
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
            return ASVFile(speaker_id=tokens[0],             #''
                file_name=tokens[1],                  #0                                #这个是比赛时读取测试集protocol文件的内容，现在测试的文件和比赛时不一样，
                path=os.path.join(self.files_dir, tokens[1] + '.flac'), #'0'              #现在19年发布的数据时测试集的protocol是和训练集、测试集的protocol时一样的。
                sys_id=self.sysid_dict[tokens[3]],                                               #0  3
                key=int(tokens[4] == 'bonafide'))                                                 #0
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):      ##解析protocol文件中每一行内容的细节。即每一行记录的元组 meta
        lines = open(protocols_fname).readlines()
        #lines=random.shuffle(lines)                 #lineShuffled  ZZY
        #lines=lines[:1000]             #lineShuffledTop1000    ZZY

        # lineShuffled=random.sample(lines, len(lines))       #ZZY
        # lineShuffledTop1000=lineShuffled[:20]             #ZZY
        lineShuffledTop1000 = lines[:]    #20
        files_meta = map(self._parse_line, lineShuffledTop1000)  #lines   lineShuffledTop1000
        return list(files_meta)

    def read_matlab_cache(self, filepath):
        f = h5py.File(filepath, 'r')
        # filename_index = f["filename"]
        # filename = []
        data_x_index = f["data_x"]
        sys_id_index = f["sys_id"]
        data_x = []
        data_y = f["data_y"][0]
        sys_id = []
        for i in range(0, data_x_index.shape[1]):
            idx = data_x_index[0][i]  # data_x
            temp = f[idx]
            data_x.append(np.array(temp).transpose())
            # idx = filename_index[0][i]  # filename
            # temp = list(f[idx])
            # temp_name = [chr(x[0]) for x in temp]
            # filename.append(''.join(temp_name))
            idx = sys_id_index[0][i]  # sys_id
            temp = f[idx]
            sys_id.append(int(list(temp)[0][0]))
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.astype(np.float32), data_y.astype(np.int64), sys_id








class ASVDataset2021(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None,
        is_train=True, sample_size=None,              #sample_size=None，1000
        is_logical=True, feature_name=None, is_eval=False,
        eval_part=0,eval_dirpath=None):

        self.transform = transform                          ###############################################数据封装 封装 封装
        self.x_data = [i[0] for i in self.files_meta]  #b = [i[0] for i in a]  从a中的每一行取第一个元素。


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
            return ASVFile(speaker_id=tokens[0],             #''
                file_name=tokens[1],                  #0                                #这个是比赛时读取测试集protocol文件的内容，现在测试的文件和比赛时不一样，
                path=os.path.join(self.files_dir, tokens[1] + '.flac'), #'0'              #现在19年发布的数据时测试集的protocol是和训练集、测试集的protocol时一样的。
                sys_id=self.sysid_dict[tokens[3]],                                               #0  3
                key=int(tokens[4] == 'bonafide'))                                                 #0
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):      ##解析protocol文件中每一行内容的细节。即每一行记录的元组 meta
        lines = open(protocols_fname).readlines()
        #lines=random.shuffle(lines)                 #lineShuffled  ZZY
        #lines=lines[:1000]             #lineShuffledTop1000    ZZY

        # lineShuffled=random.sample(lines, len(lines))       #ZZY
        # lineShuffledTop1000=lineShuffled[:20]             #ZZY
        lineShuffledTop1000 = lines[:]    #20
        files_meta = map(self._parse_line, lineShuffledTop1000)  #lines   lineShuffledTop1000
        return list(files_meta)

    def read_matlab_cache(self, filepath):
        f = h5py.File(filepath, 'r')
        # filename_index = f["filename"]
        # filename = []
        data_x_index = f["data_x"]
        sys_id_index = f["sys_id"]
        data_x = []
        data_y = f["data_y"][0]
        sys_id = []
        for i in range(0, data_x_index.shape[1]):
            idx = data_x_index[0][i]  # data_x
            temp = f[idx]
            data_x.append(np.array(temp).transpose())
            # idx = filename_index[0][i]  # filename
            # temp = list(f[idx])
            # temp_name = [chr(x[0]) for x in temp]
            # filename.append(''.join(temp_name))
            idx = sys_id_index[0][i]  # sys_id
            temp = f[idx]
            sys_id.append(int(list(temp)[0][0]))
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.astype(np.float32), data_y.astype(np.int64), sys_id




# if __name__ == '__main__':
#    train_loader = ASVDataset(LOGICAL_DATA_ROOT, is_train=True)
#    assert len(train_loader) == 25380, 'Incorrect size of training set.'
#    dev_loader = ASVDataset(LOGICAL_DATA_ROOT, is_train=False)
#    assert len(dev_loader) == 24844, 'Incorrect size of dev set.'

