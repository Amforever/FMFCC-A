import math
from CQT import *
from Resample import *
from  dct import  *
import time
def cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD):
    nargin = cqcc.__code__.co_argcount   #可以通过fun.code.co_argcount来获取反射函数的参数个数
    if nargin <2:
        print('Not enough input arguments.')
        return
    if nargin < 3:
        B = 96
    if nargin < 4:
        fmax = int(fs/2)
    if nargin < 5:
        oct = math.ceil(math.log(fmax/20,2))
        fmin = fmax/(2**oct)
    if nargin < 6:
        d = 16
    if nargin < 7:
        cf = 19
    if nargin < 8:
        ZsdD = 'ZsdD'

    gamma = 228.7*(2**(1/B)-2**(-1/B))
    # %%% CQT COMPUTING, 调用CQT.py
    varargin=['rasterize', 'full', 'gamma', gamma]
    Xcq =cqt(x, B, fs, fmin, fmax, varargin)
    absCQT = abs(Xcq['c'])
    TimeVec = np.arange(1,absCQT.shape[1]+1)*Xcq['xlen']/absCQT.shape[1]/fs
    FreqVec = fmin*(2**((np.arange(absCQT.shape[0]))/B))
    LogP_absCQT =np.log((absCQT**2+2.2204e-16))
    kl = (B * math.log(1+1/d, 2))
    # var1 = 1 / (fmin * (2 ^ (kl / B) - 1))
    # arglist = np.arange([LogP_absCQT, FreqVec, 1 / (fmin * (2 ^ (kl / B) - 1)), 1, 1, 'spline'])
    arglist =[LogP_absCQT, FreqVec, 1 / (fmin * (2 **(kl / B) - 1)), 1, 1, 'spline']
    # %%% resample COMPUTING, 调用Resample.py脚本
    Ures_LogP_absCQT, Ures_FreqVec = resample(arglist)
    # CQcepstrum = dct(Ures_LogP_absCQT)
    # %%% DCT COMPUTING
    import  scipy.fftpack
    #这一步如果type=2, 输出结果与matlab一模一样,但是速度会慢很多, 如果type=1,速度会提升很多,但是结果略有偏差
    CQcepstrum = scipy.fftpack.dct(Ures_LogP_absCQT,type=2,axis=0 ,norm='ortho')

    if 'Z' in ZsdD:
        scoeff =1
    else:
        scoeff=2
    CQcepstrum_temp = CQcepstrum[scoeff-1:cf + 1,:]

    f_d = 1
    if ZsdD.replace('Z','')=='sdD':
        CQcc = np.concatenate((CQcepstrum_temp,Deltas(CQcepstrum_temp,f_d),Deltas(Deltas(CQcepstrum_temp,f_d),f_d)),axis=0)
    elif ZsdD.replace('Z','')=='sd':
        CQcc = np.concatenate((CQcepstrum_temp, Deltas(CQcepstrum_temp, f_d)), axis=0)
    elif ZsdD.replace('Z', '') == 'sD':
        CQcc = np.concatenate((CQcepstrum_temp, Deltas(Deltas(CQcepstrum_temp, f_d), f_d)), axis=0)
    elif ZsdD.replace('Z', '') == 's':
        CQcc = CQcepstrum_temp
    elif ZsdD.replace('Z', '') == 'd':
        CQcc =Deltas(CQcepstrum_temp,f_d)
    elif ZsdD.replace('Z', '') == 'D':
        CQcc= Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
    elif ZsdD.replace('Z', '') == 'dD':
        CQcc = np.concatenate(( Deltas(CQcepstrum_temp, f_d), Deltas(Deltas(CQcepstrum_temp, f_d), f_d)),axis=0)
    return CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec

import scipy.signal as signal
def Deltas(x,hlen):
    win =np.arange(hlen,-hlen-1,-1)
    # xx = [repmat(x(:, 1), 1, hlen), x, repmat(x(:, end), 1, hlen)];
    xx =np.concatenate(( np.tile(x[:,0],(1,hlen)).T,x, np.tile(x[:,-1],(1,hlen)).T ),axis=1)
    D = signal.lfilter(win,1,xx,axis=1)
    D = D[:,hlen*2:]
    D = D/(2*sum(np.arange(1,hlen+1)**2))
    return D
