import numpy as np
import math

def resample(arglist):
    if len(arglist)<2 or len(arglist)>8:
        return -1
    method, arglist = getInterpMethod(arglist)
    if isinstance(arglist[1],(int,float,complex)):
        if method == ' ':
            print('signal:resample:UnexpectedInterpolation', method)
        y,h = uniformResample(arglist)
        return y,h
    else:
        if method == '':
            method = 'linear'
        y,ty,h=nonUniformResample(method,arglist)
        return y,ty

def nonUniformResample(method,varargin):
    x, tx = getSamples(varargin[0],varargin[1])
    if len(varargin)>2:
        fs = varargin[2]
        # validateFs(fs) 验证fs格式
        if isinstance(fs,float) == False:
            print('format error')
    else:
        fs = (len(tx)-1)/(tx[-1]-tx(1))

    if len(varargin)>4:
        p = varargin[3]
        q = varargin[4]
        validateResampleRatio(p, q)
    elif len(varargin)>3:
        print('signal:resample:MissingQ')
    else:
        p, q = getResampleRatio(tx, fs)

    tsGrid = p/(q * fs)
    # tGrid = np.linspace(tx[0],tx[-1], (tx[-1]-tx[0])/tsGrid+1)
    tGrid = np.arange(tx[0],tx[-1],tsGrid)
    # if isinstance(x,complex):
    #     realGrid = matInterp1(tx, x.real, tGrid, method)
    #     imagGrid = matInterp1(tx, x.imag, tGrid, method)
    #     xGrid = complex(realGrid, imagGrid)
    # else:
    #     xGrid = matInterp1(tx, x, tGrid, method)
    x_judge = np.isreal(x)

    if x_judge.all():
        import time
        s =time.time()
        xGrid = matInterp1(tx, x, tGrid, method)
        e = time.time()
        #print(e-s)         #注释byZZY便于显示插值所需要的时间
    else:
        realGrid = matInterp1(tx, x.real, tGrid, method)
        imagGrid = matInterp1(tx, x.imag, tGrid, method)
        xGrid = complex(realGrid, imagGrid)
    if len(varargin)==6:
        uniformResample_list =[xGrid, p, q, varargin[5:]]
        # y, h = uniformResample(xGrid, p, q, varargin[5:])
        y, h = uniformResample(uniformResample_list)
    else:
        uniformResample_list = [xGrid, p, q]
        y,h = uniformResample(uniformResample_list)
    if y.shape[0] ==1 or y.shape[1]==1:
        ty = tx[0]+ np.array([i for i in range(y.shape[0]*y.shape[1])])/fs
    else:
        ty = tx[0] + np.arange(y.shape[0])/fs

    tx = tx .reshape((1,len(tx)))
    if tx.shape[1]==1:
        ty = ty.T
    return y,ty,h


def uniformResample(uniformResample_list):
    if len(uniformResample_list)==5:
        bta =uniformResample_list[4]
        N = uniformResample_list[3]
    if len(uniformResample_list)==4:
        N = uniformResample_list[3]
    if len(uniformResample_list)<5:
        bta =5
    if len(uniformResample_list) < 4:
        N = 10
    x = uniformResample_list[0]
    p = uniformResample_list[1]
    q = uniformResample_list[2]

    # validateResampleRatio(p, q)
    # p,q=rat(p//q, 1e-12)
    if p == 1 and  q == 1:
        y = x
        h = 1
        return y,h
    pqmax = max(p, q)

    if len(N) > 1:
        L = len(N)
        h = N
    else:
        if N > 0:
            fc = 1 / 2 / pqmax
            L = 2 * N * pqmax + 1
            #滤波器实现
            # h= [0]*L
            import scipy.signal
            h= scipy.signal.lfilter(L-1,np.array([0, 2*fc,2*fc,1]),np.array([1,1,0,0]))*(scipy.signal.kaiser_atten(L,bta).T)
            # h = firls(L - 1, [0 2 * fc 2 * fc 1], [1 1 0 0]).* kaiser(L, bta)' ;
            h = p * h / sum(h)
        else:
            L = p
            h = np.ones(1, p)

    Lhalf = (L - 1) / 2
    if x.shape[0]==1 or x.shape[1]==1:
        isvect = 1
    else:
        isvect=0

    if isvect:
        Lx = len(x)
    else:
        Lx = x.shape[0]

    nz = np.floor(q - np.mod(Lhalf, q));
    z = np.zeros(1, nz)
    h = np.hstack((z,h.reshape(-1)))
    Lhalf = Lhalf + nz
    delay = np.floor(np.ceil(Lhalf)/q);
    nz1 = 0
    while np.ceil(((Lx - 1) * p + len(h) + nz1)/q) - delay < np.ceil(Lx * p / q):
        nz1 = nz1 + 1

    h = np.hstack((h, np.zeros(1, nz1)))

    y = upfirdn(x, h, p, q)
    Ly = np.ceil(Lx * p / q)

    if isvect:
        y[: delay] = []
        y[Ly + 1: ] = []
    else:
        y[:delay,:] = []
        y[Ly:,:] = []

    # h([1: nz(end - nz1 + 1):end]) = [];
    h[0: len(h):nz(len(nz)- nz1)]
    return y,h


def upfirdn(x, h, p, q):
    mx = x.shape[0]
    nx = x.shape[1]
    a = np.append(mx,nx)
    if np.min(np.where(a == 1))+1:
        for i in range(x.shape[1]):
            if i ==0:
                x1=x[:,i]
            else:
                x1 = np.hstack((x1,x[:,i]))
        x=x1
    Lx = x.shape[0]
    nChans =x.shape[1]

    if np.min(np.where(np.array(np.shape(h)) == 1))+1:
        for i in range(h.shape[1]):
            if i == 0:
                h1 = h[:, i]
            else:
                h1 = np.hstack((h1, h[:, i]))
        h = h1

    Lh  = h.shape[0]
    hCols = h.shape[1]
    varargin=[]
    varargin.append(p)
    varargin.append(q)
    p, q = validateinput(x, h,varargin)

    #Y = upfirdnmex(x, h, p, q, Lx, Lh, hCols, nChans)
    #这个方法不知道如何转换,没有经过
    Y =[1,2,3,4]

    if mx == 1 and  hCols == 1:
        for i in range(Y.shape[1]):
            if i == 0:
                Y1 = Y[:, i]
            else:
                Y1 = np.hstack((Y1, Y[:, i]))
        Y = Y1
        Y = Y.T
    return Y


def validateinput(x,h,opts):
    p = 1
    q = 1
    from collections import Counter
    if len(x)==0 or Counter(x)[0]>(x.shape[0]*x.shape[1])/2 or ~isinstance(list(x)[0],float):
        return 'error x'

    if len(h) == 0 or Counter(h)[0] > (h.shape[0] * h.shape[1]) / 2 or ~isinstance(list(h)[0], float):
        return 'error h'

    nChans = x.shape[1]
    hCols = h.shape[1]
    if (nChans > 1) and (hCols > 1) and (hCols != nChans):
        return 'signal:upfirdn:xNhSizemismatch X,H'

    nopts = len(opts)
    if (nopts >= 1):
        p = opts[1]
        if len(p)==0 or  ~isinstance(list(p)[0], float) or  p<1 or ~(np.round(p)==p).all():
            return 'signal:upfirdn:invalidP, P'
        elif (nopts == 2):
            q = opts[2]
            if len(q)==0 or  ~isinstance(list(q)[0], float) or  q < 1  or  ~(np.round(q)==q).all():
                return 'signal:upfirdn:invalidQ ,Q'
        if p * q > 2**31:
            return ('signal:upfirdn:ProdPNQTooLarge', 'Q', 'P')
    return p,q


def removeMissingTime(x,tx):
    idx = np.isnan(tx)
    idx = np.array(idx)
    # print(len(idx),type(idx))
    idx_index = np.where(idx ==True)
    if  len(idx)!=0:
        tx = np.delete(tx,idx_index)
        # tx[idx]=[]
        if x.shape[0]==1 or x.shape[0]==1:
            # x[idx] = []
            x=np.delete(x,idx_index)
        else:
            x = np.delete(x,idx_index,0)
            # x[idx, :] = []
    return x,tx


def  matInterp1(tin, xin, tout, method):
    tout=tout.reshape((len(tout),1))
    if xin.shape[0] == 1 or xin.shape[0] == 1:
        if xin.shape[0]==1:
            tout = tout.T
        idx = np.nonzero(~np.isnan(xin))[0]
        y = vecInterp1(tin[idx], xin[idx], tout, method)
    else:
        nRows = tout.shape[0]
        nCols = xin.shape[1]
        y = np.zeros((nRows, nCols))
        for col in range(nCols):
            idx = np.nonzero(~np.isnan(xin[:,col]))[0]
            interp1=vecInterp1(tin[idx], xin[idx,col], tout, method)
            y[:, col] = interp1
    return  y

def vecInterp1(tin, xin, tout, method):
    iDup = np.nonzero(np.diff(tin) == 0)[0]
    iRepeat = 1 + iDup
    # print(type(iDup),type(iRepeat))
    while len(list(iDup))!=0:
        Temp = (np.diff(iDup)!= 1)
        for i in len(Temp):
            if Temp[i] ==1:
                numEqual = i
                break
        if numEqual==None:
            numEqual = len(iDup)
        xSelect = xin[iDup[0] + np.arange(0,numEqual+1)]
        xMean = np.mean(xSelect[~np.isnan(xSelect)])
        xin[iDup[0]] = xMean
        iDup = iDup[1 + numEqual:]
    xin[iRepeat] = []
    tin[iRepeat] = []
    from scipy import interpolate
    xin = xin.reshape(-1)
    tin = tin.reshape(-1)
    tout = tout.reshape(-1)
    #线性插值,不太一致
    # p1d = interpolate.interp1d(tin,xin, kind='cubic',fill_value='extrapolate')
    # y =p1d(tout).reshape(-1)
    # y = interpolate.CubicSpline(tin,xin)(tout).reshape(-1)

    y = interpolate.interp1d(tin,xin,kind='cubic',fill_value='extrapolate')(tout).reshape(-1)
    # y = interpolate.CubicSpline(tin, xin,extrapolate=True)(tout).reshape(-1)
    # y =interpolate.spline(tin, xin, tout, order=3)
    # y = interpolate.InterpolatedUnivariateSpline(tin, xin.reshape(-1))(tout).reshape(-1)
    return y

def getSamples(x, tx):
    if len(x.shape) != 2:
        print('shape error')
    x_num = x.shape[0]*x.shape[1]
    tx_num = len(tx)
    if x.shape[0]==1 and x.shape[1]==1 and x_num != tx_num:
        print('error')
    if x.shape[0]!=1 and x.shape[1]!=1 and x.shape[0] != tx_num:
        print('error')
    # validateattributes(tx, {'numeric'}, {'real', 'vector'},'resample', 'Tx', 2)
    #该句为验证信息

    x, tx = removeMissingTime(x, tx)

    idx = np.argsort(tx)
    tx.sort()

    if x.shape[0]==1 and x.shape[1]==1:
        # x1=[]
        # for i in idx:
        #     x1.append(x[i])
        # x=x1
        x = x[idx]
    else:
        # x1= []
        # for i in idx:
        #     x1.append(x[i,:])
        # x = np.array(x1)
        x = x[idx,:]
    # validateattributes(tx, {'numeric'}, {'finite'}, ...
    # 'resample', 'Tx', 2)
    return x,tx

# def validateFs(fs):
#     # validateattributes(fs, {'numeric'}, {'real', 'finite', 'scalar', 'positive'}, ...
#     # 'resample', 'Fs', 3);
#     if isinstance(fs,(int,float))==False:
#         print('fs error')
#     return

def validateResampleRatio(p, q):
# validateattributes(p, {'numeric'},{'integer','positive','finite','scalar'}, ...
#     'resample','P');
# validateattributes(q, {'numeric'},{'integer','positive','finite','scalar'}, ...
#     'resample','Q');
    if isinstance(p,(int,float))==False:
        print('p errer')
    if isinstance(q,(int,float))==False:
        print('q.error')

def  getResampleRatio(t, fs):
    tsAvg = (t[-1] - t[0]) / (len(t) - 1)
    p, q = rat(tsAvg * fs, .01)
    if p<2:
        p = 1
        q = round(1 / (tsAvg * fs))

    return p,q

# def getInterpMethod(arglist):
#
#     method = ''
#     supportedMethods = ['linear','pchip','spline']
#
#     iFound = -1
#
#     for i in range(len(arglist)):
#
#         if isinstance(arglist[i],str):
#             # method = validatestring(arglist[i],supportedMethods,'resample','METHOD')
#             for arglist[i] in supportedMethods:
#                 iFound1 =i
#
#     if iFound !=-1:
#         del arglist[iFound]
#     return method, arglist


def rat(X,tol):
    if rat.__code__.co_argcount<2:
       tol = 1.e-6 * np.linalg.norm(X[math.isfinite(X)], ord =1)
    if not np.isreal(X):
        if np.linalg.norm(X.imag,ord=1)<= tol*np.linalg.norm(X.real,ord=1):
            X = X.real
        elif  rat.__code__.co_argcount > 1:
            NR, DR = rat(X.real)
            NI, DI = rat(X.imag)
            D = DR*DI/math.gcd(DR,DI)
            N = D/DR* NR + D/DI* NI * complex(0,1)
            return
        else:
            N = str(rat(X.real)+ ' +i* ...'+rat(X.imag))
            return
    if rat.__code__.co_argcount > 1:
        X = np.array([[X]])
        N = np.zeros((X.shape[0],X.shape[1]))
        D = np.zeros((X.shape[0],X.shape[1]))
    else:
        N = np.zeros((0, 0))

    for  j in range(X.size):
        x =X[j][j]

        if ~np.isfinite(x):

            if rat.__code__.co_varnames <= 1:
                s = str(x)
                k = len(s) - N.shape[1]
                N1 = []

                N1.append(np.array([str(i) for i in N*np.ones(j-1,k)]))
                N1.append( np.array([str(i) for i in s*np.ones]))
                N = np.array(N1)

            else:
                if ~np.isnan(x):
                    N[j] = x/abs(x)

                else:
                    N[j] = 0

                D[j] = 0
        else:
            k = 0
            C = np.array([[1,0],[0,1]])
            while True:
                k = k + 1
                neg = x < 0
                d = round(x)
                if ~np.isinf(x):
                    x = x - d
                    C = np.hstack((np.dot(C,np.array([[d],[1]])), C[:,0].reshape((C.shape[0],1))))
                    # print(C)
                else:
                    C = np.hstack(( np.array([[x],[0]]), C[:,0].reshape((C.shape[0],1))))

                if rat.__code__.co_argcount<=1:

                    d = str(abs(d))
                    if neg:
                        d  = '-'+d
                    if k == 1:
                        s = d
                    elif k == 2:
                        s = s + '+ 1/('+d+')'
                        # s = [s ' + 1/(' d ')'];
                    else:
                        for i in len(s):
                            if s[i] ==')':
                                p = i
                        # p = find(s == ')', 1);
                        s =s[:p-1]+ '+1/('+ d + ')' + s[p-1:p+k-3]
                        # s = [s(1:p - 1) ' + 1/('d')' s(p: p + k - 3)];
                    if (x == 0) or  (abs(C[0, 0] / C(1, 0) - X[j]) <= max(tol, X[j]* 2.2204e-16)):
                        break
                    x =1/x
            if rat.__code__.co_argcount > 1:
                N[j] = C[0, 0] / (C[1, 0] / abs(C[1, 0]))
                D[j] = abs(C[1,0])
            else:
                k = len(s)-N.shape[1]
                N = str(str(N)+' ')
                N1.append(np.array([str(i) for i in N * np.ones(j - 1, k)]))
                N1.append(np.array([str(i) for i in s * np.ones]))
                N = np.array(N1)

    return N,D




def getInterpMethod(arglist):
    method = ''
    supportedMethods = ['linear', 'pchip', 'spline']
    iFound = 0
    for i in range(len(arglist)):
        if isinstance(arglist[i],str):
            if arglist[i] in supportedMethods:
                method = arglist[i]
                iFound = i
    if iFound:
        del arglist[iFound]
    return method,arglist



