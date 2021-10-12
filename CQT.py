import math
import numpy as np
from Nsgtf_real import *
from Nsgcqwin import *
from CqtCell2Sparse import  *

def cqt(x, B, fs, fmin, fmax,extact):
    rasterize = 'full'
    phasemode = 'global'
    outputFormat = 'sparse'
    normalize = 'sine'
    windowFct = 'hann'
    gamma = 0

    if cqt.__code__.co_argcount>=6:
        Larg = len(extact)
        for ii in range(0,Larg,2):
            if extact[ii]=='rasterize':
                rasterize = extact[ii + 1]
            if extact[ii]=='phasemode':
                phasemode = extact[ii + 1]
            if extact[ii]=='format':
                outputFormat = extact[ii + 1]
            if extact[ii]=='gamma':
                gamma = extact[ii + 1]
            if extact[ii]=='normalize':
                normalize = extact[ii + 1]
            if extact[ii]=='win':
                windowFct = extact[ii + 1]
    Var =[ 'winfun', windowFct, 'gamma', gamma, 'fractional', 0]

    g, shift, M = nsgcqwin(fmin, fmax, B, fs, len(x), Var)
    # print(len(g[555]),len(g[556]),len(g[557]))

    fbas = fs * np.cumsum(shift[1:])/ len(x)
    fbas = fbas[0: int((M.shape[0])/2)-1]


    # compute coefficients
    bins = int(M.shape[0]/2)-1
    if rasterize=='full':
        M[1: bins+1] = M[bins]
        M[bins + 2:] = M[bins: 0:-1]

    if rasterize=='piecewise':
        temp = M[bins]
        octs = math.ceil(math.log(fmax / fmin,2))
        temp = math.ceil(temp / 2 ** octs) * 2 ** octs
        mtemp = temp/np.linalg.inv(M)
        mtemp = 2** (math.ceil(math.log(mtemp,2)) - 1)
        mtemp = temp/ np.linalg.inv(mtemp);
        mtemp[bins + 1] = M[bins + 1]
        mtemp[0] = M[0]
        M = mtemp

    # nor=['sine', 'Sine', 'SINE', 'sin']
    if normalize in ['sine', 'Sine', 'SINE', 'sin']:
        normFacVec = 2 * M[0:bins + 2]/len(x)
    # nor1 = ['impulse', 'Impulse', 'IMPULSE', 'imp']

    elif normalize in ['impulse', 'Impulse', 'IMPULSE', 'imp']:
        L = 0
        for i in g:
            L += len(i)
        normFacVec = 2 * M[0:bins + 1]/float(L)

    # nor1 = ['none', 'None', 'NONE', 'no']
    elif normalize in ['none', 'None', 'NONE', 'no']:
        normFacVec = np.ones((bins + 2, 1))
    else:
        print('Unkown normalization method!')

    normFacVec = np.hstack((normFacVec, normFacVec[len(normFacVec)-1:1:-1]))
    for k in range(2*bins+2):
        g[k]=g[k]*normFacVec[k]

    # result =[]
    # for i in range((2*bins+2)-1):
    #     result.append(g[i]*normFacVec[i])
    # g =np.array(result)
    c = nsgtf_real(x, g, shift, M, phasemode)

    # print(c[859])
    if rasterize=='full':
        cDC = c[0].T
        cNyq = c[bins+1].T
        c = np.array(c[1:bins+1])
        c=c.reshape((-1,c.shape[2])).T

        # print(c[:,857])
    elif rasterize=='piecewise':
        cDC = c[0]
        cNyq = c[bins + 1]
        c = c[1:bins + 1].transpose()
        if outputFormat =='sparse':
            c = cqtCell2Sparse(c,M).transpose();
        else:
            c= c[1:len(c)-1]
    else:
        cDC = c[1]
        cNyq = c[-1]
        c = c[1:len(c)-1]
    # print(c[:,858])
    Xcq = {}
    Xcq['c']=c.T
    Xcq['g'] = g
    Xcq['shift'] = shift
    Xcq['M'] = M
    Xcq['xlen'] = len(x)
    Xcq['phasemode'] = phasemode
    Xcq['rast'] = rasterize
    Xcq['fmin'] = fmin
    Xcq['fmax'] = fmax
    Xcq['B'] = B
    Xcq['cDc'] = cDC
    Xcq['cNyq'] = cNyq
    Xcq['outputFormat'] = outputFormat
    Xcq['fbas'] = fbas
    return Xcq

