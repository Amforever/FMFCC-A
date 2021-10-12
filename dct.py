import numpy as np
import tensorflow as tf
def dct(a,n=-1):
    if dct.__code__.co_argcount ==0:
        return 'signal:dct:Nargchk'
    # sigcheckfloattype(a,'','dct','X')
    if len(a)==0:
        b = []
        return
    do_trans = (a.shape[0] == 1)
    if do_trans:
        # for i in range(len(a.shape[0])):
        #     if i ==0:
        #         a1=a[:,i]
        #     else:
        #         a1 = np.concatenate((a1,a[:,i]),axis=0)
        a=a.T

    if n==-1:
        n = a.shape[0]

    # n = sigcasttofloat(n, 'float', 'dct', 'N', 'allownumeric');
    # n =n
    m = a.shape[1]
    if a.shape[0]<n:
        aa = np.zeros((n,m))
        aa[:a.shape[0],:]=a
    else:
        aa = a[:int(n),:]

    ww =(np.exp(-complex(0,1)*np.arange(n)*np.pi/(2*n))/np.sqrt(2*n)).T
    # if isinstance(a[0], float):
    #     ww = float(ww)

    ww[0] = ww[0] / np.sqrt(2)
    if n%2==1 or not all(np.isreal(a)):
        y = np.zeros((2 * n, m))
        y[:n,:] = aa
        y[n: 2 * n,:] = np.flipud(aa)

        yy = np.fft.fft(y,axis=0)

        yy = yy[:n,:]

    else:
        y = np.concatenate((aa[:n:2,:],aa[n-1:2,-2,:]),axis=0)
        yy= np.fft.fft(y,axis=0)
        ww =2*ww
    ww = ww.reshape((len(ww),1))
    b = np.tile(ww,(1,m))*yy
    # b = ww[:,np.ones((1,m))]*yy
    if (np.isreal(a)).all():
        b = np.real(b)
    if do_trans:
        b=b.T

    return b


def sigcasttofloat(x, castType, fcnName, varName,datacheckflag):
    if sigcasttofloat.__code__.co_argcount < 3:
        fcnName = ''
        varName = ''
        datacheckflag = 'allowfloat'

    if sigcasttofloat.__code__.co_argcount < 4:
        varName = ''
        datacheckflag = 'allowfloat'
    if sigcasttofloat.__code__.co_argcount < 5:
        datacheckflag = 'allowfloat'

    sigcheckfloattype(x, '', fcnName, varName, datacheckflag)

    y = x.astype(castType)
    return  y



def sigcheckfloattype(x, dataType, fcnName, varName, datacheckflag=''):
    if sigcheckfloattype.__code__.co_argcount < 3:
        fcnName = ''
        varName = ''
        datacheckflag = 'allowfloat'
    if sigcheckfloattype.__code__.co_argcount < 4:
        varName = ''
        datacheckflag = 'allowfloat'
    if sigcheckfloattype.__code__.co_argcount < 5:
        datacheckflag = 'allowfloat'

    if datacheckflag=='allowfloat':

        typeCheck = isinstance(x[0,0],float)
        expType = 'double/single'
    elif datacheckflag=='allownumeric':
        typeCheck = x.isnumeric()
        expType = 'numeric'
    else:
        return 'signal:sigcheckfloattype:InvalidDataCheckFlag'
    if ~typeCheck:
        if (len(fcnName)!=0):
            if len(varName)!=0:
                return ( 'signal:sigcheckfloattype:InvalidInput'+' '+varName+' '+fcnName+' '+expType+' '+ type(x))
            else:
                return ('signal:sigcheckfloattype:InvalidInput1'+' '+fcnName+' '+ expType+' '+ type(x))
        else:
            if len(varName)!=0:
                return ('signal:sigcheckfloattype:InvalidInput2'+' '+varName+' '+expType+' '+type(x))
            else:
                return ('signal:sigcheckfloattype:InvalidInput3'+' '+expType+' '+type(x))

    flag = isinstance(x, type(dataType))

    return flag