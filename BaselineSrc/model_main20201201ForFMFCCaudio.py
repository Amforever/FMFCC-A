"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    Modified by: zhenyuzhang
"""
import argparse
import sys
import os
import data_utilsForASV2021
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
from startup_config import set_random_seed
import yaml
from ptflops import get_model_complexity_info
import logging
from datetime import datetime

from models import resnet34CNet,HGRes_TSSDNet
from extract_feature import compute_SdDCQCC_feats, get_wav




def logloss(label, predict):
    if predict>1-1e-12:
        predict=1-1e-12
    if predict<1e-12:
        predict=1e-12
    # if predict==1.0:
    #     predict=1-1e-12
    # if predict==0.0:
    #     predict=1e-12
    if label == 1.0:
        # print(predict)
        return -math.log(predict)
    elif label == 0.0:
        return -math.log(1 - predict)
    else:
        raise ValueError("The real Label didn't exist")

def pad(x, max_len=32000):  #64600
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x

def read_file(meta):
    data_x, sample_rate = sf.read(meta)  #根据 文件元路径 读取音频
    return data_x       #data_y也就是key 是0和1分别表示bonafide和spoof，sys_id是合成的类别，分为0~19

def evaluate_accuracy(data_loader, model, device,cache_path,networktype):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    score_pred=[]
    lab_real=[]
    sysmthod_id_x_real=[]
    lab_pred=[]
    ii_predInter = 0 #对于一个batch中的样本，作为一次迭代，计算次数；
    ii_loaddata = 0  #对于每一个样本的加载，计算次数
    logloss_sum = 0.0
    # for batch_x, batch_y, batch_meta in data_loader:
    for i, data in enumerate(data_loader, 0):
        inputs, labels, sysid = data
        ii_predInter += 1  #对于一个batch中的样本，作为一次迭代，计算次数；
        data_x = []
        labels_x = []
        sysmthod_id_x=[]
        '''提取特征保存npy文件，或加载已经提取好的特征的npy数据'''
        for indexx in range(len(inputs)):
            ii_loaddata+=1   #对于每一个样本的加载，计算次数
            cache_fname=cache_path+'/'+sysid[1][indexx]+'.npy'
            cache_fname_sysmthod_id =  sysid[3][indexx]
            # print(cache_fname)
            if os.path.exists(cache_fname):
                ##一旦训练过一次，将特征数据自动保存在，当前目录，从而再次运行时会自动调用。
                data_xx, labels_xx, sysid_x = torch.load(cache_fname)
                # if ii_loaddata % PrintIntervals == 0:   #对于已有的数据，npy文件直接加载，就不显示样本数量
                #     print('Dataset Dev loaded from cache ', cache_fname)
            else:
                # print("Nont the Train featrue file")
                path = (inputs[indexx],)  # 定义一个元祖，并赋值
                data = list(map(read_file, path))  # 按照read_file函数，根据file_meta数据，读取wav音频的数据
                data_xx = Parallel(n_jobs=1, prefer='threads')(
                    delayed(transforms)(x) for x in data)  # 将读入的data_x数据提取特征，并转换为tensor的格式
                torch.save((data_xx, labels[indexx], sysid[1][indexx]), cache_fname)
                if ii_loaddata % PrintIntervals == 0:
                    print('Dataset Eval/Dev saved to cache ', cache_fname)
                labels_xx = labels[indexx]
            data_x.append(data_xx)
            labels_x.append(labels_xx)
            sysmthod_id_x.append(cache_fname_sysmthod_id)

        batch_y=labels_x
        batch_x=data_x
        batch_sysmthod_id_x=sysmthod_id_x
        batch_size=batch_x.__len__()
        num_total += batch_x.__len__()

        '''对数据处理，变成tensor变量，并开始训练'''
        #对load的 list 型的data_x，每一行表示一个tensor是一个样本的。进行cat拼接，然后利用reshape进行转换为batch，，，等四维的tensor
        feats4Dim11=batch_x[0][0]
        for indexxx in range(1,batch_x.__len__()):
            # feats4Dim11 = torch.stack((feats4Dim11,data_x[indexxx]), dim=0)  #没办法for循环拼接
            feats4Dim11 = torch.cat((feats4Dim11, data_x[indexxx][0]), dim=0)  # 直接在后面拼接，不进行增加维度
        a=feats4Dim11
        if batch_size==1:
            if batch_x[0][0].shape.__len__()==2:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0].shape[0],batch_x[0].shape[1]))
            elif batch_x[0][0].shape.__len__()==3:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0].shape[0],batch_x[0].shape[1],batch_x[0].shape[2]))
            elif batch_x[0][0].shape.__len__()==1:
                a = torch.as_tensor(a.reshape(batch_x.__len__(), -1))
        else:
            if batch_x[0][0].shape.__len__()==2:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0][0].shape[0],batch_x[0][0].shape[1]))
            elif batch_x[0][0].shape.__len__()==3:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0][0].shape[0],batch_x[0][0].shape[1],batch_x[0][0].shape[2]))
            elif batch_x[0][0].shape.__len__()==1:
                a = torch.as_tensor(a.reshape(batch_x.__len__(), -1))
        #对load的 list 型的data_y，每一行表示一个tensor是一个样本的。进行cat拼接， 记得到一维的包含batch的0或1的toensor
        Lable4Dim11 = torch.tensor(batch_y)
        batch_x = a.to(device)
        batch_y = Lable4Dim11.view(-1).type(torch.int64).to(device)
        # batch_out = model(batch_x)

        if args.networkmodel == 'TCN':
            if batch_x.shape.__len__() == 3:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]])
            else:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[1] * batch_x.shape[2], batch_x.shape[3]])
            batch_out = model(batch_x_OneDim)
        elif args.networkmodel == 'RNN':
            # print(batch_x.shape.__len__())
            if batch_x.shape.__len__() == 3:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[2], batch_x.shape[
                    1]])  # 由于nn.LSTM()设置了batch_first=True,所以要让batch_size放到第一维。每句话的长度seq_len放到第二维，每个字的维度放到第三维为1
            else:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[3], batch_x.shape[1] * batch_x.shape[2]])
            batch_out = model(batch_x_OneDim)
        elif  args.networkmodel =='EncoderTrans_zzy_CNN' or args.networkmodel =='EncoderTrans_zzy_DirectFC' or args.networkmodel =='EncoderTrans_zzy_Resnet34':
            # print(batch_x.shape.__len__())
            if batch_x.shape.__len__()==3:
                batch_x_OneDim = batch_x.view([ batch_x.shape[0],batch_x.shape[2],batch_x.shape[1]])  #由于nn.LSTM()设置了batch_first=True,所以要让batch_size放到第一维。每句话的长度seq_len放到第二维，每个字的维度放到第三维为1
            else:
                batch_x_OneDim = batch_x.view([ batch_x.shape[0],batch_x.shape[3], batch_x.shape[1]*batch_x.shape[2]])
            EachSentence_input_lengths = [126] * batch_x.shape[0]
            batch_out = model(batch_x_OneDim,EachSentence_input_lengths)
        else:
            batch_out = model(batch_x)
            # print("卷积网络")

        mm=nn.Sigmoid()
        batch_out =mm(batch_out)  #nn.Softmax(dim=1)(batch_out)
        # batch_out = nn.Softmax(dim=1)(batch_out)
        _, batch_pred = batch_out.max(dim=1)
        batch_correct_num=(batch_pred == batch_y).sum(dim=0).item()
        num_correct += batch_correct_num
        if ii_predInter % PrintIntervals == 0:        #10
            # sys.stdout.write('\r \t The iterations times: {:.2f}'.format((num_correct/num_total)*100))             #每次eopch中，每隔10个batch进行一次 标准输出，一个 累加的 正确率数值。(num_correct)/(ii*batchsize+ii*batchsize+....+ii*batchsize)
            # print('\r \t The evalution iterations times:{:d}, and the batch accuracy:{:.2f}'.format(ii, (num_correct / num_total) * 100))
            sys.stdout.write('\r \t The evalution iterations times:{:d}, and the batch accuracy:{:.2f}\n'.format(ii_predInter,(num_correct/num_total)*100))
        #输出真实标签和网络预测的得分softamx之后的概率用于计算eer和logloss
        a=batch_out.data.cpu().numpy()
        b=a[:,0]
        lab=batch_y.data.cpu().numpy()
        score_pred = np.concatenate((score_pred, b), axis=0)
        lab_real = np.concatenate((lab_real, lab), axis=0)
        #输出合成方法id，网络预测标签 batch_out.max()，用于计算每个合成方法的检测正确率：
        lab_pred_cpu=batch_pred.data.cpu().numpy()
        lab_pred=np.concatenate((lab_pred, lab_pred_cpu), axis=0)
        sysmthod_id_x_real = np.concatenate((sysmthod_id_x_real, batch_sysmthod_id_x), axis=0)

        '''保存实验结果，正确率随着迭代次数的变化'''
        # with open(cache_path+'_'+networktype+'TestReslut.txt', 'a') as fh:
        #     fh.write('InterationNum:{},BatchAcc:{}\n'.format(ii, batch_correct_num/batch_size))


    #根据计算的标签和合成方法，计算每个合成方法的检测正确率：
    sysmethodid=np.unique(sysmthod_id_x_real)
    num_sysmethod = len(np.unique(sysmthod_id_x_real))
    SysmethodTrueCount=np.zeros(num_sysmethod)
    SysmethodFalseCount=np.zeros(num_sysmethod)
    SysmethodAllCount=np.zeros(num_sysmethod)
    SysmethodAccuracy=np.zeros(num_sysmethod)
    for ii in range(len(sysmthod_id_x_real)):
        for sysmethodcount in range(num_sysmethod):
            if sysmthod_id_x_real[ii]==sysmethodid[sysmethodcount] :
                SysmethodAllCount[sysmethodcount]=SysmethodAllCount[sysmethodcount]+1
                if lab_pred[ii]==lab_real[ii]:
                    SysmethodTrueCount[sysmethodcount]=SysmethodTrueCount[sysmethodcount]+1
                else:
                    SysmethodFalseCount[sysmethodcount] = SysmethodFalseCount[sysmethodcount] + 1

    print("************************************************************************************************************************************************************")
    for sysmethodcount in range(num_sysmethod):
        SysmethodAccuracy[sysmethodcount]=SysmethodTrueCount[sysmethodcount]/SysmethodAllCount[sysmethodcount]
        print("SysMethodId:%d, All:%d,True:%d,False:%d,Acc:%.10f " % (sysmethodid[sysmethodcount], SysmethodAllCount[sysmethodcount],SysmethodTrueCount[sysmethodcount], SysmethodFalseCount[sysmethodcount], SysmethodAccuracy[sysmethodcount]))

    #根据检测出的softmax的概率值和标签计算eer和logloss损失值：
    bona_cm=[]
    spoof_cm=[]
    cm_scores=list(zip(score_pred,lab_real))
    for cou in range(len(cm_scores)):
        # print(lab_pred[cou],lab_real[cou],len(cm_scores))
        if cm_scores[cou][1]==1:
            bona_cm.append(cm_scores[cou][0])
        else:
            spoof_cm.append(cm_scores[cou][0])
        logloss_ = logloss(lab_real[cou], 1-score_pred[cou])  #lab_pred需要是为真的概率，而b=model[:,0]为0，为假的概率。
        logloss_sum = logloss_sum + logloss_
    eer_asv_zz, asv_threshold = compute_eer(np.array(spoof_cm),np.array(bona_cm))
    Metric_logloss_avg = logloss_sum / len(cm_scores)
    # all_accuracy = (lab_real == lab_pred).sum().item()/num_total
    # print(100 * (num_correct / num_total), all_accuracy)
    return 100 * (num_correct / num_total),eer_asv_zz,Metric_logloss_avg



def train_epoch(data_loader, model, lr, device,cache_path,networktype):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii_predInter = 0 #对于一个batch中的样本，作为一次迭代，计算次数；
    ii_loaddata = 0  #对于每一个样本的加载，计算次数
    logloss_sum=0
    model.train()



    '''提取特征保存npy文件，或加载已经提取好的特征的npy数据'''
    # for batch_x, batch_y, batch_meta in data_loader:    #train_loader 为什么原始时的是 train_loader 和输入的data_loader名字不一样。  依然可以进行传递。
    for i, data in enumerate(data_loader, 0):
        inputs, labels, sysid = data
        ii_predInter+=1
        # print(len(inputs))
        data_x = []
        labels_x = []

        # for indexx in range
        for indexx in range(len(inputs)):
            ii_loaddata += 1
            cache_fname=cache_path+'/'+sysid[1][indexx]+'.npy'
            # print(cache_fname)
            if os.path.exists(cache_fname):
            ##一旦训练过一次，将特征数据自动保存在，当前目录，从而再次运行时会自动调用。
                data_xx, labels_xx, sysid_x = torch.load(cache_fname)
                # if ii_loaddata % PrintIntervals == 0:
                #     print('Dataset Train loaded from cache ', cache_fname)
            else:
                # print("Nont the Train featrue file")
                path = (inputs[indexx],) #定义一个元祖，并赋值
                # data, sample_rate = sf.read(path)
                data = list(map(read_file, path))  # 按照read_file函数，根据file_meta数据，读取wav音频的数据
                data_xx = Parallel(n_jobs=1, prefer='threads')(delayed(transforms)(x) for x in data)  # 将读入的data_x数据提取特征，并转换为tensor的格式
                # data_xx = Parallel(n_jobs=1, prefer='threads')(delayed(transforms)(data))
                torch.save((data_xx, labels[indexx], sysid[1][indexx]), cache_fname)
                if ii_loaddata % PrintIntervals == 0:
                    print('Dataset Train saved to cache ', cache_fname)
                labels_xx=labels[indexx]
            data_x.append(data_xx)
            labels_x.append(labels_xx)
        batch_y=labels_x
        batch_x=data_x
        batch_size=batch_x.__len__()
        num_total += batch_x.__len__()

        '''
        为了实现list数组的tensor转换为 增加一维度的tensor尝试的方法：
        # feats4Dim11=data_x[0]
        # for indexxx in range(1,batch_x.__len__()):
        #     # feats4Dim11 = torch.stack((feats4Dim11,data_x[indexxx]), dim=0)  #没办法for循环拼接
        #     feats4Dim11 = torch.cat((feats4Dim11, data_x[indexxx]), dim=0)  # 直接在后面拼接，不进行增加维度
            # feats4Dim11 = np.vstack((feats4Dim11, data_x[indexxx]))    #数组
            # feats4Dim11.append(data_x[indexxx])   #只对list可用
        # feats4Dim11 = torch.stack((data_x[0], data_x[1],data_x[2], data_x[3],data_x[4], data_x[5],data_x[6], data_x[7]), dim=0)
        # a=torch.cat(batch_x, dim=0)   #直接在后面拼接，不进行增加维度
        '''

        '''对数据处理，变成tensor变量，并开始训练'''
        #对load的 list 型的data_x，每一行表示一个tensor是一个样本的。进行cat拼接，然后利用reshape进行转换为batch，，，等四维的tensor
        feats4Dim11=batch_x[0][0]
        for indexxx in range(1,batch_x.__len__()):
            # feats4Dim11 = torch.stack((feats4Dim11,data_x[indexxx]), dim=0)  #没办法for循环拼接
            feats4Dim11 = torch.cat((feats4Dim11, data_x[indexxx][0]), dim=0)  # 直接在后面拼接，不进行增加维度
        a=feats4Dim11
        if batch_size==1:
            if batch_x[0][0].shape.__len__()==2:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0].shape[0],batch_x[0].shape[1]))
            elif batch_x[0][0].shape.__len__()==3:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0].shape[0],batch_x[0].shape[1],batch_x[0].shape[2]))
            elif batch_x[0][0].shape.__len__() == 1:
                a = torch.as_tensor(a.reshape(batch_x.__len__(), -1))
        else:
            if batch_x[0][0].shape.__len__()==2:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0][0].shape[0],batch_x[0][0].shape[1]))
            elif batch_x[0][0].shape.__len__()==3:
                a = torch.as_tensor(a.reshape(batch_x.__len__(),batch_x[0][0].shape[0],batch_x[0][0].shape[1],batch_x[0][0].shape[2]))
            elif batch_x[0][0].shape.__len__()==1:
                a = torch.as_tensor(a.reshape(batch_x.__len__(), -1))
        #对load的 list 型的data_y，每一行表示一个tensor是一个样本的。进行cat拼接， 记得到一维的包含batch的0或1的toensor
        Lable4Dim11 = torch.tensor(batch_y)
        batch_x = a.to(device)
        batch_y = Lable4Dim11.view(-1).type(torch.int64).to(device)


        if args.networkmodel == 'TCN':
            # 为了时间序列rnn和tcn模型 需要转换为1维   #(self, input_dim, hidden_dim, num_layers, output_dim, drop_out)
            # batch_x_OneDim = batch_x.view([batch_x.shape[0], 1, -1])  #batchsize )  # input should have dimension (N, C, L) (N,C_in,L_in)N为批次，C_in即为in_channels，即一批内输入一维数据个数，L_in是是一维数据基数
            # print(batch_x.shape.__len__())
            if batch_x.shape.__len__() == 3:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]])
            else:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[1] * batch_x.shape[2], batch_x.shape[3]])
            batch_out = model(batch_x_OneDim)
        elif args.networkmodel == 'RNN':
            # print(batch_x.shape.__len__())
            if batch_x.shape.__len__() == 3:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[2], batch_x.shape[
                    1]])  # 由于nn.LSTM()设置了batch_first=True,所以要让batch_size放到第一维。每句话的长度seq_len放到第二维，每个字的维度放到第三维为1
            else:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[3], batch_x.shape[1] * batch_x.shape[2]])
            batch_out = model(batch_x_OneDim)
        elif args.networkmodel == 'EncoderTrans_zzy_CNN' or args.networkmodel == 'EncoderTrans_zzy_DirectFC' or args.networkmodel == 'EncoderTrans_zzy_Resnet34':
            # print(batch_x.shape.__len__())
            if batch_x.shape.__len__() == 3:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[2], batch_x.shape[
                    1]])  # 由于nn.LSTM()设置了batch_first=True,所以要让batch_size放到第一维。每句话的长度seq_len放到第二维，每个字的维度放到第三维为1
                EachSentence_input_lengths = [batch_x.shape[2]] * batch_x.shape[0]
            else:
                batch_x_OneDim = batch_x.view([batch_x.shape[0], batch_x.shape[3], batch_x.shape[1] * batch_x.shape[2]])
                EachSentence_input_lengths = [batch_x.shape[3]] * batch_x.shape[0]
            batch_out = model(batch_x_OneDim, EachSentence_input_lengths)
        # elif args.networkmodel == 'RawNet2Baseline':

        else:
            batch_out = model(batch_x)

            # print("卷积网络")
        mm=nn.Sigmoid()
        batch_out =mm(batch_out)  #nn.Softmax(dim=1)(batch_out)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)  #dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值，返回预测的 负正样本的值。
        batch_correct_num=(batch_pred == batch_y).sum(dim=0).item()
        num_correct += batch_correct_num #不进行概率对比，只是进行标签对比
        running_loss += (batch_loss.item() * batch_size)

        a=batch_out.data.cpu().numpy()
        lab_pred=a[:,0]
        lab_real=batch_y.data.cpu().numpy()

        for cou in range(len(lab_pred)):
            logloss_ = logloss(lab_real[cou], 1 - lab_pred[cou])  # lab_pred需要是为真的概率，而b=model[:,0]为0，为假的概率。
            logloss_sum = logloss_sum + logloss_


        if ii_predInter % PrintIntervals == 0:        #10
            sys.stdout.write('\r \t The training iterations times:{:d}, and the batch accuracy:{:.2f}\n'.format(ii_predInter,(num_correct/num_total)*100))

        '''保存实验结果，正确率随着迭代次数的变化'''
        # with open(cache_path+'_'+networktype+'TrainReslut.txt', 'a') as fh:
        #     fh.write('InterationNum:{},Loss:{},BatchAcc:{}\n'.format(ii,running_loss, batch_correct_num/batch_size))
        # print('Result saved to {}'.format(cache_path))

        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    running_loss /= num_total
    logloss_sum/= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy,logloss_sum



def produce_evaluation_file(dataset, model, device, save_path):         #只用于eval为真 True时，分解文件，从而得到这些文件的检测。
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)        #32
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())

    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if not dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))  # fh.write('{} {}\n'.format(f, cm))
    print('Result saved to {}'.format(save_path))





if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser('FMFCC-A baseline model  By Zhenyu Zhang')
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')                 #默认训练时为False  True
    parser.add_argument('--model_path', type=str, default=None) #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--eval_output', type=str,default='.\TestResluts', help='Path to save the evaluation result')      #,default=None)#  #默认训练时为None   r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\TestResluts\Test1.txt'
    parser.add_argument('--batch_size', type=int, default=64)  #32  spec时最大为16，否则报错。  VGG16 设置为1 仍然报错OOM
    parser.add_argument('--num_epochs', type=int, default=2000)   #100
    parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='SdDCQCC')    # LFCCbaseline  get_wav_64600  get_wav  SingleMfccFilterHPFInter8diff SdDCQCC  SingleCQCCFilterHPFInter  SingleMfccFilterHPFInter   MfccFilterHPFInter  resnet34 SpectCRNN  mfcc SingleMfccFilterHPFInter MfccFilterHPFInter spect  TimeFilterSRM  TimeFilterHPFInter  MfccFilterSRM  MfccFilterKV  MfccFilterDCT MfccFilterGabor MfccFilterHPF  SpectLeNet5  SpectAlexNet SpectVGG16  SpectGoogleNetV1 SpectCRNN  MfccFilterHPFInter
    parser.add_argument('--networkmodel', type=str, default='resnet34')   #CNN_small Audio2_se_resnet34 HGRes_TSSDNet,HGRes_TSSDNet_2D,HGInc_TSSDNet LCNNbaseline RawNet2Baseline sincnetori ResNetFCNnet18 FCNet  resnext34varTanh  resnext50varTanh  resnext34varNonBN resnext50varNonBN resnetNesl vgg llcnn  cnn  resnet34  resnext50  resnext18 resnext34  SpectLeNet5 EfficientNet_zzy1_B0 SpectVGG16
    parser.add_argument('--is_eval', action='store_true', default=False, help='Training using eval dataset')                            #训练时默认为False   True
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--print_interval', type=int, default=200,help='load npy datafile and save npy datafile')  # 训练时默认为False   True

    ########FMFCC比赛数据集##############
    parser.add_argument('--DatasetName',  type=str, default='Fmfcc', help='Dataset For test: Fmfcc Asvspoof2019')  #StepLR50_09
    parser.add_argument('--protocoltrain', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_protocols/NameChangeTagsFile_FromTrain.txt')    #NameChangeTagsFile_FromTrain
    parser.add_argument('--protocoldev', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_protocols/NameChangeTagsFile_FromDev.txt')    # NameChangeTagsFile_FromDev  /home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_protocols/NameChangeTagsFile_FromDev.txt
    parser.add_argument('--protocoleval', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_protocols/NameChangeTagsFile_FromEval.txt')  #NameChangeTagsFile_FromEval_OnlyThePythonClips.txt  NameChangeTagsFile_FromEval  NameChangeTagsFile_FromEval_ContianDevReal
    parser.add_argument('--filedirtrain', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_train/wav')
    parser.add_argument('--filedirdev', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_dev/Dev')  #   IIE2021_LA_dev/wav    /home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_dev/wav
    parser.add_argument('--filedireval', type=str, default='/home1/zzy/Forensics/IIEForensicsDatasetV2/IIE2021_LA_eval')  #IIE2021_LA_eval


    #随机种子的初始化，用于实验的重复
    parser.add_argument('--seed', type=int, default=0,    help='random seed (default: 1234) Set 0 means not limiation;')
    parser.add_argument('--cudnn_deterministic_toggle', action='store_false',  default=True,   help='use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn_benchmark_toggle', action='store_true',  default=False, help='use cudnn-benchmark? (default false)')

    #优化器、权重、损失、学习率lr及其变换方式设置
    parser.add_argument('--optimizerDef', type=str, default='Adam',    help='the optimizer that be used')
    parser.add_argument('--weightDef',  type=str, default='weight19',    help='a manual rescaling weight given to each class')
    parser.add_argument('--criterionDef', type=str, default='CrossEntropyLoss',    help='the definition of loss function')
    parser.add_argument('--lr', type=float, default=0.00001,   help='the initial of lr')
    parser.add_argument('--schedulerDef',  type=str, default='ExponentialLR095', help='the adaptive strategies for lr')  #StepLR50_09

    #指定所使用的gpu卡的序号从0开始。
    parser.add_argument('--gpu_id_begin0',  type=str, default='3', help='the GpuCrd number')  #StepLR50_09



    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    IDgpuCard=args.gpu_id_begin0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] =  IDgpuCard   #export CUDA_VISIBLE_DEVICES=2,必须放在出现torch之前，否则将不会起效果

    set_random_seed(args.seed, args)  #    #make experiment reproducible
    print("cudnn_deterministic:",args.cudnn_deterministic_toggle )
    print("cudnn_benchmark:", args.cudnn_benchmark_toggle)

    track = args.track
    PrintIntervals = args.print_interval
    assert args.features in ['mfcc', 'spectLPC', 'cqcc','TimeFilterSRM','MfccFilterSRM','MfccFilterKV','MfccFilterDCT','MfccFilterGabor','MfccFilterHPF','SingleMfccFilterHPFInter','SpectLeNet5','SpectAlexNet','SpectVGG16','SpectGoogleNetV1','SpectCRNN','MfccFilterHPFInter','LogSpectFilterHPFInter','resnet34','EfficientNet_zzy1','SdDCQCC','SingleCQCCFilterHPFInter','SingleMfccFilterHPFInter8diff','SingleCQCCFilterHPFInter8diff','raw102462','raw1020588','SingleMfccFilterHPFInter8diffAddOri','SingleCQCCFilterHPFInter8diffAddOri','get_wav','get_wav_64600','LFCCbaseline'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}_{}_{}_{}'.format(track, args.features,args.networkmodel, args.num_epochs, args.batch_size, args.lr, args.is_eval)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)







    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats
        input_channels = 72
    elif args.features == 'SdDCQCC':  # SampleResidual
        feature_fn = compute_SdDCQCC_feats
        input_channels = 72
    elif args.features == 'get_wav':  # SampleResidual
        feature_fn = get_wav
    else:
        raise ValueError("the Audio Features  doesn't exist")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "resnetNesl" in args.networkmodel:
        if args.features == 'mfcc':
            model_cls = MFCCModel()
    elif args.networkmodel == 'resnext34':
        model_cls = resnext34()
    elif args.networkmodel == 'HGRes_TSSDNet':  # SampleResidual
        model_cls = HGRes_TSSDNet()  #和lstm层的输入有关，当用delta时维度为60；不用delta时默认的维度为20
    else:
        raise ValueError("The Neural Network doesn't exist")
    print(model_cls)

    nb_params = sum([param.view(-1).size()[0] for param in model_cls.parameters()])
    print('\nThe number of network params:',nb_params)



    '''添加日志'''
    timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S')
    # logger = logging.getLogger(os.path.join(model_save_path,timestamp+'atp.log'))

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    filehandle01 = logging.FileHandler(timestamp+'atp.log')
    formatter01 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    filehandle01.setFormatter(formatter01)
    logger.addHandler(filehandle01)

    logger.info('\n')
    logger.info(model_cls)
    logger.info('Trainable     parameters: {}    '.format(nb_params))

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail")
    logger.info("Finish")
    logger.error("error print log")


    transforms = transforms.Compose([
        lambda x: pad(x),                          #先对采样点进行pad，对折补数
        lambda x: librosa.util.normalize(x),       #对采样点进行归一化
        lambda x: feature_fn(x),                   #对归一化后的点进行提取特征。
        lambda x: Tensor(x)                        #提取的特征转换为tensor向量
    ])
    model = model_cls  #TCN 在定义时已经有括号了，所以不能用括号（）


    print('The number of GPUs: ',torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # 前提是model已经.cuda() 了
    model.to(device)
    print(args)


    '''加载数据并确认是直接拿eval做验证，还是用训练好的模型做测试，'''
    print('Data dev_set Load Begained!')
    if args.is_eval:  # 利用测试集eval进行边训练，边计算测试结果
        dev_set = data_utilsForASV2021.ASVDataset(is_train=False, is_logical=is_logical, transform=transforms,
                                                  feature_name=args.features, is_eval=args.is_eval,
                                                  protocols_dir=args.protocoleval, file_dir=args.filedireval)
    else:
        dev_set = data_utilsForASV2021.ASVDataset(is_train=False, is_logical=is_logical, transform=transforms,
                                                  feature_name=args.features, is_eval=args.is_eval,
                                                  protocols_dir=args.protocoldev, file_dir=args.filedirdev)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=True)#, num_workers=2)
    print('Data dev_set Load Finished!')


    if args.model_path:
        # model.load_state_dict(torch.load(args.model_path))
        model = torch.load(args.model_path)  # 加载模型
        print('Model loaded : {}'.format(args.model_path))

    if args.eval:                                                                              #如果进行eval 必须有输出 和模型的路径。
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'
        text_tag = '{}_{}_{}_{}_{}'.format(args.features, args.networkmodel, args.num_epochs, args.batch_size, args.lr)
        eval_output_file_path=args.eval_output+'/'+text_tag+'.txt'
        produce_evaluation_file(dev_set, model, device,eval_output_file_path)
        sys.exit(0)


    print('Data train_set Load Begained!')
    train_set = data_utilsForASV2021.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,
                                                feature_name=args.features,is_eval=args.is_eval,
                                                protocols_dir=args.protocoltrain,file_dir=args.filedirtrain)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    print('Data train_set Load Finished!')



    '''定义优化器和权重损失函数和lr优化器；通过参数的输入来选择'''
    if args.optimizerDef == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005) #weight_decay权重衰减的系数, 实际使用的是L2正则化;
    elif args.optimizerDef == 'Adam_NonWeightdecay':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.weightDef == 'weight19':
        # weight = torch.FloatTensor([0.5, 0.5]).to(device)
        weight = train_set.get_weights_FromTrainDataset().to(device)  # weight used for WCE
    if args.criterionDef == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=weight)  # baseline的损失函数  F.cross_entropy
    if args.schedulerDef == 'StepLR10_09':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.99) #等间隔地调整，调整倍数为 gamma， 调整的epoch 间隔为 step_size;即每隔 step_size个epoch，lr降为gamma*lr
    elif args.schedulerDef == 'StepLRLR1_09':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    elif args.schedulerDef == 'ExponentialLR095':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)    #每个epoch都做一次更新：lr降为gamma*lr
    elif args.schedulerDef == 'None':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=1)




    cache_path_train = 'cache_{}_{}_{}_{}'.format(track,args.DatasetName,'train_2s', args.features)
    cache_path_dev = 'cache_{}_{}_{}_{}'.format(track,args.DatasetName,'dev_2s', args.features)
    cache_path_eval = 'cache_{}_{}_{}_{}'.format(track,args.DatasetName,'eval_2s', args.features)
    if os.path.exists(cache_path_train) is False:
        os.makedirs(cache_path_train)
    if os.path.exists(cache_path_dev) is False:
        os.makedirs(cache_path_dev)
    if os.path.exists(cache_path_eval) is False:
        os.makedirs(cache_path_eval)

    '''开始训练模型'''
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_train_acc=0
    best_dev_acc=0
    best_dev_epoch=0
    best_dev_eer=100
    best_train_epoch=0
    best_eer_epoch=0
    numEpochsNotImproving=0
    best_Metric_logloss_avg_train=100000
    best_Metric_logloss_avg_dev=100000
    best_Metric_logloss_avg_train_epoch = 0
    best_Metric_logloss_avg_dev_epoch = 0
    for epoch in range(num_epochs):
        running_loss, train_accuracy, Metric_logloss_avg_train= train_epoch(train_loader, model, args.lr, device,cache_path_train,args.networkmodel)  #训练数据
        scheduler.step()
        # dev_accuracy, eer_asv_test = evaluate_accuracy(dev_loader, model, device, cache_path_dev)  # 验证数据
        if args.is_eval:  #利用测试集eval进行边训练，边计算测试结果
            dev_accuracy,dev_eer,Metric_logloss_avg_dev= evaluate_accuracy(dev_loader, model, device,cache_path_eval,args.networkmodel)  #验证数据
        else:
            dev_accuracy, dev_eer,Metric_logloss_avg_dev = evaluate_accuracy(dev_loader, model, device, cache_path_dev,args.networkmodel)  # 验证数据
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', dev_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)

        if (Metric_logloss_avg_train < best_Metric_logloss_avg_train):
            best_Metric_logloss_avg_train = Metric_logloss_avg_train
            best_Metric_logloss_avg_train_epoch=epoch
            torch.save(model, os.path.join(model_save_path, 'epochTrainLogloss_{}_{}.pth'.format(args.is_eval,epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。

        if (Metric_logloss_avg_dev < best_Metric_logloss_avg_dev):
            best_Metric_logloss_avg_dev = Metric_logloss_avg_dev
            best_Metric_logloss_avg_dev_epoch=epoch
            torch.save(model, os.path.join(model_save_path, 'epochDevLogloss_{}_{}.pth'.format(args.is_eval,epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。

        if (dev_eer < best_dev_eer):
            best_dev_eer = dev_eer
            best_eer_epoch = epoch
            torch.save(model, os.path.join(model_save_path, 'epochDevEER_{}_{}.pth'.format(args.is_eval,epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。

        if (train_accuracy > best_train_acc):
            best_train_acc = train_accuracy
            best_train_epoch = epoch
            torch.save(model, os.path.join(model_save_path, 'epochTrainACC_{}_{}.pth'.format(args.is_eval,epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。
        if (dev_accuracy > best_dev_acc):
            best_dev_acc = dev_accuracy
            best_dev_epoch = epoch
            # torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch))) #只保存
            torch.save(model, os.path.join(model_save_path, 'epochDevACC_{}_{}.pth'.format(args.is_eval,epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。
            numEpochsNotImproving = 0
        else:
            numEpochsNotImproving = numEpochsNotImproving + 1
        print('\nThe begining time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)),' The current running time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),' ModelName:',args.networkmodel, '  FeatureType:',args.features, ' Is_Eval:',args.is_eval)
        print("Epoch [%d/%d], Train Loss:%.10f, Train Acc:%.6f%%, Train_Logloss_avg:%.10f, Dev Acc:%.6f%%, Dev Logloss_avg:%.10f, LearningRate:%.6e" % ( epoch + 1, num_epochs, running_loss, train_accuracy,  Metric_logloss_avg_train ,dev_accuracy,Metric_logloss_avg_dev,   optim.param_groups[0]['lr'])) #optimizer.param_groups.lr()
        print("Train_Logloss_avg:%.10f,  Dev Logloss_avg:%.10f, Train_Logloss_avg_best:%.10f, corresponding epoch:%d, Dev_Logloss_avg_best:%.10f, corresponding epoch:%d" % (Metric_logloss_avg_train ,Metric_logloss_avg_dev,  best_Metric_logloss_avg_train, best_Metric_logloss_avg_train_epoch+1, best_Metric_logloss_avg_dev, best_Metric_logloss_avg_dev_epoch+1)) #optimizer.param_groups.lr()
        print('Best Train Accuracy:{:.6f}%, corresponding epoch:{}; Best Development EER :{:.6f}%, corresponding epoch:{}; Best Development Accuracy:{:.6f}%, corresponding epoch:{}; NumEpochsNotImproving:{:.0f}\n'.format(
                best_train_acc, best_train_epoch+1, best_dev_eer*100 ,best_eer_epoch+1,best_dev_acc, best_dev_epoch+1, numEpochsNotImproving))
