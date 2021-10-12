import  numpy as np
def cqtCell2Sparse(c,M ):
    bins = M.shape[0]/2-1
    spLen = M[bins]
    cSparse = np.zeros(bins, spLen)
    M = M[1:bins + 1]
    step = 1
    distinctHops = np.log(M[bins]/M[0],2) + 1
    curNumCoef = M[bins]
    for ii in range(distinctHops):
        flag = (M==curNumCoef)
        idx = np.hstack((flag,False))
        temp = c[idx]
        step = step*2
        curNumCoef = curNumCoef/2
    cSparse1 ={}
    for i in range(cSparse.shape[0]):
        for j in range(cSparse.shape[1]):
            if cSparse[i][j]!=0:
                cSparse1[(i,j)]=cSparse[i][j]
    return cSparse1

