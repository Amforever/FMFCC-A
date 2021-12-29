import math
import numpy as np
from  Winfun import winfuns

def nsgcqwin(fmin, fmax, bins, sr, Ls, Var):
    #Constant-Q/Variable-Q dictionary generator
    bwfac = 1
    min_win = 4
    fractional = 0
    winfun = 'hann'
    gamma = 0
    if nsgcqwin.__code__.co_argcount < 5:
        print('Not enough input arguments')
    if nsgcqwin.__code__.co_argcount >= 6:
        Lvar = len(Var)
        if (Lvar%2):
           print('Invalid input argument')
        for kk in range(0,Lvar,2):
            if isinstance(Var[kk],str):

                if Var[kk]=='min_win':
                    min_win = Var[kk+1]
                elif Var[kk]=='gamma':
                    gamma = Var[kk+1]
                    # print(gamma)
                elif Var[kk]=='bwfac':
                    bwfac = Var[kk+1]
                elif Var[kk]=='fractional':
                    fractional = Var[kk+1]
                elif Var[kk]=='winfun':
                    winfun = Var[kk+1]
                else:
                    return ('Invalid input argument: '+Var[kk])
    nf = sr/2
    if fmax > nf:
        fmax = nf
    fftres = sr / Ls
    b = math.floor(bins * math.log(fmax/fmin,2))
    fbas = fmin*(2**((np.array(range(0,b+1)).T)/bins))

    Q = 2 ** (1/bins)-2 **(-1/bins)
    cqtbw = Q * fbas + gamma
    # print(cqtbw)
    cqtbw = cqtbw[:]
    # make sure the support of highest filter won't exceed nf
    temp = fbas+cqtbw/2
    # print(temp,nf)
    tmpId = -1
    for i in range(len(temp)):
        if temp[i] >nf:
            tmpId= i
            break
    if tmpId != -1:
        fbas = fbas[0:tmpId]
        cqtbw=cqtbw[0:tmpId]


    tmpId =-1
    temp = fbas + cqtbw/2
    for i in range(len(temp)-1,0,-1):
        if temp[i]<0:
            tmpId = i
            break
    # print(tmpId)
    if tmpId !=-1:
        fbas = fbas[tmpId:]
        cqtbw = cqtbw[tmpId:]


    Lfbas =len(fbas)

    fbas=np.insert(fbas,0,np.array([[0]]),axis=0)
    fbas=np.insert(fbas,Lfbas+1,np.array([[nf]]),axis=0)

    fbas = np.insert(fbas, Lfbas + 2, [0]*Lfbas, axis=0)
    fbas[Lfbas +2: 2 * (Lfbas + 1)] = sr - fbas[Lfbas: 0:-1]


    bw = np.insert(cqtbw, 0, 2*fmin, axis=0)
    bw = np.insert(bw, len(bw), fbas[Lfbas+2]-fbas[Lfbas], axis=0)
    bw = np.insert(bw, len(bw), cqtbw[len(cqtbw):0:-1], axis=0)
    bw = np.insert(bw, len(bw), cqtbw[0], axis=0)

    fftres = sr/Ls
    bw = bw/fftres
    fbas = fbas/fftres

    posit = np.zeros(fbas.shape)
    posit[0:Lfbas+2] = np.floor(fbas[0:Lfbas+2])
    posit[Lfbas+2:] = np.ceil(fbas[Lfbas+2:])

    # shift = [np.mod(-posit[-1],Ls), np.diff(posit,axis=1)]
    shift = np.insert(np.diff(posit), 0, np.mod(-posit[-1],Ls), axis=0)

    if (fractional):
        corr_shift = fbas-posit
        M = np.ceil(bw+1)
    else:
        bw = np.round(bw)
        M = bw

    for ii in range(0 ,2 * (Lfbas + 1)):
        if (bw[ii] < min_win):
            bw[ii] = min_win
            M[ii] = bw[ii]
    # print(bw,M)

    if(fractional):
       # 该部分还未转化
       g=0
    else:
        g =[]
        for i in bw:
            x = np.array([[i]])
            # print(x)
            g1 = winfuns(winfun,x)
            g1=g1.reshape(-1)
            g.append(g1)

    # print(g[555].shape,g[556].shape,g[557].shape)
    M = bwfac*np.ceil(M/bwfac)
    # Setup Tukey window
    for kk in [0,Lfbas+1]:
        if(M[kk] > M[kk+1]):
            # print(M[kk])
            # M[kk] = int(M[kk])
            g[kk] = np.ones((1,int(M[kk])))
            # print(g[kk][0][1:10])
            # print(g[kk][int(np.floor(int(M[kk])//2) - np.floor(int(M[kk+1])//2)+1): int(np.floor(int(M[kk])//2) + int(np.ceil(int(M[kk+1])//2)))+2])
            # print(len(g[kk][0]))
            # print(np.floor(int(M[kk])//2) - np.floor(int(M[kk+1])//2), int(np.floor(int(M[kk])//2)) + int(np.ceil(int(M[kk+1])//2)))
            # print(len(g[kk][0][int(np.floor(int(M[kk])/2) - np.floor(int(M[kk+1])/2)): int(np.floor(int(M[kk])/2) + int(np.ceil(int(M[kk+1])/2)))]),len(winfuns('hann',np.array([[int(M[kk+1])]]))))
            g[kk][0][int(np.floor(int(M[kk])/2) - np.floor(int(M[kk+1])/2)): int(np.floor(int(M[kk])/2) + int(np.ceil(int(M[kk+1])/2)))] = winfuns('hann',np.array([[int(M[kk+1])]]))
            g[kk] = g[kk]/math.sqrt(M[kk])
    return g, shift, M

