# @Author :ZhenyuZhang
# @CrateTime   :2020/12/21 11:30
# @File   :dataloadfromfiles.py 
# @Email  :201718018670196

'''
从datasourcedirpath文件夹中读取real和fake datast文件夹的 wav文件;
'''

import os
from torch.utils.data import DataLoader, Dataset

''' 
    根据os.walk函数遍历文件夹下的所有文件
    root 表示当前正在访问的文件夹路径
    dirs 表示该文件夹下的子目录名list
    files 表示该文件夹下的文件list
'''
def findAllFile(dataset,datasourcedirpath,MaxNumUtterLabel):  #testing training validation D:\AudioForensics\Data\RicardoRemimaoFoR_Canada\for-norm\for-norm
    print(datasourcedirpath+'/'+dataset)
    labels = []
    wav_lists = []
    wav_name_lists=[]
    for root, ds, fs in os.walk(datasourcedirpath+'/'+dataset):
        for f2 in fs:  # 遍历文件
            wav_lists.append(root+'/'+f2)
            wav_name_lists.append(f2)
    wav_lists=wav_lists[:MaxNumUtterLabel]
    wav_name_lists=wav_name_lists[:MaxNumUtterLabel]
    return labels, wav_lists,wav_name_lists


class ASVDataset(Dataset):
    def __init__(self,  DataSetKind=None,DataRootDirPath=None,MaxNumUtterLabel=10000000):
        print(DataSetKind,DataRootDirPath)
        '''调用findAllFile()根据参数数据集类型testing training validation和数据集的根目录for-norm；返回数据集文件夹中的文件名列表 和 相应的标签 
           DataSetKind training testing validation D:\AudioForensics\Data\RicardoRemimaoFoR_Canada\for-norm\for-norm
        '''
        self.files_meta = findAllFile(DataSetKind,DataRootDirPath,MaxNumUtterLabel)
        self.x_data=self.files_meta[1]
        self.x_name=self.files_meta[2]
        self.length=self.x_data.__len__()
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        x = self.x_data[idx]
        return x

