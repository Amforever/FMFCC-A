import numpy as np

def nsgtf_real(f,g,shift,M,phasemode):
    #Nonstationary Gabor filterbank for real signals
    if nsgtf_real.__code__.co_argcount <2:
        print('Not enough input arguments')
    f=f.reshape((f.shape[0],1))
    Ls = f.shape[0]
    CH = f.shape[1]
    if Ls == 1:
        f = f.transpose()
        Ls = CH
        CH = 1
    if CH>Ls:
        print('The number of signal channels '+ CH +' the number of samples per channel ' +Ls )
        replay = input("input Y or N:")
        if replay =='N':
            replay2= input("transpose single matrix?  input Y or N:")
            if (replay2=='N'):
                print('error')
            elif( replay2=='Y'):
                f = f.transpose()
                X = CH
                CH = Ls
                Ls = CH
            else:
                print('Invalid reply, terminating program')
        elif replay=='Y':
            print('Continuing program execution')
        else:
            print('Invalid reply, terminating program')
    N = len(shift)
    if nsgtf_real.__code__.co_argcount==3:
        M = np.zeros(N,1)
        for kk in range(N):
            M[kk]=len(g[kk])
    if max(M.shape)==1:
        M=M[0]*np.ones(N,1)
    # print(f,f.shape)
    # f = np.fft.fft(f)/len(f)
    #其中一种
    # f=np.fft.fft2

    import scipy.fftpack
    f = scipy.fftpack.fft(f,axis=0)
    # print(f[0])

    posit =np.cumsum(shift)-shift[0]
    fill = sum(shift) -Ls
    if fill>0:
        f= np.vstack((f,np.zeros(fill,CH)))
    Lg = []
    for i in g:
        L1 = max(i.shape)
        Lg .append(L1)
    Lg =np.array(Lg)
    # print(Lg[0],Lg.shape,type(Lg))
    L2 = posit - np.floor(Lg/2)
    L3 = int((Ls + fill)/2)
    Index=-1
    for i in range(len(L2)-1,-1,-1):
            if L2[i]<=L3:
                Index = i
                break
    N =Index+1
    c = []


    import time
    s=time.time()
    for ii in range(N):
        # if ii ==859:
        #     idx = [i for i in range(int(np.ceil(Lg[ii] / 2)), Lg[ii] - 1)] + [j for j in range(0, int(np.ceil(Lg[ii] / 2)))]
        #     idx = np.array(idx)
        #     # print(idx)
        #     win_range1 = np.array([i for i in range((-int(np.floor(Lg[ii] / 2))) + 1, int(np.ceil(Lg[ii] / 2)))])
        #     win_range = np.mod(posit[ii] + win_range1, Ls + fill) + 1
        # else:
        idx =[i for i in range(int(np.ceil(Lg[ii]/2)), Lg[ii])]+[j for j in range(0,int(np.ceil(Lg[ii]/2)))]
        idx = np.array(idx)
        # print(idx)
        win_range1=np.array([i for i in range((-int(np.floor(Lg[ii]/2))),int(np.ceil(Lg[ii]/2)))])
        win_range = np.mod(posit[ii]+win_range1, Ls+fill)+1

        #if 条件需要完善
        if M[ii]<Lg[ii]:
            col =np.ceil(Lg[ii]/M[ii])
            temp = np.zeros(col*M[ii],CH)

            temp1 = [i for i in range(Lg[ii]-np.floor(Lg[ii]/2)+1,len(Lg[ii]))]
            temp2=[j for j in range(np.ceil(Lg[ii]/2))]
            temp = np.vstack((temp1,temp2))
            temp = f[win_range]*g[ii][idx]
            temp = temp.reshape((M[ii], col,CH))
            c[ii] =np.squeeze(np.fft.ifft(np.sum(temp,axis=1)))
        else:
            B=[]
            A=[]
            for idx_index in idx:
                # print(g[ii].reshape(-1)[idx_index])
                B.append(g[ii].reshape(-1)[idx_index])
            for win_range_index in win_range:
                A.append(f[int(win_range_index-1)])
            B = np.array(B).reshape(-1)
            A = np.array(A).reshape(-1)
            # print(A.shape,B.shape)

            # temp = np.zeros((int(M[ii]), CH))
            # print(temp.shape,M.shape)
            temp1 = [i for i in range(int(M[ii]) - int(np.floor(Lg[ii]/2)), int(M[ii]))]
            temp2 = [j for j in range(int(np.ceil(Lg[ii]/2)))]
            # print(Lg[ii],temp1,temp2)
            temp3 = temp1+temp2
            # print(temp3)
            temp4=[complex(0,0)]*(int(M[ii]))
            # temp4 = np.array(temp4)
            A_B = A*B
            A_B_index = 0
            for temp3_index in temp3:
                temp4[temp3_index] = A_B[A_B_index]
                A_B_index+=1
            #两种都可以,时间差不多
            # temp4[temp3] = A_B[(list(range(len(temp3))))]
            # temp4  = list(temp4)



            if phasemode =='global':
                fsNewBins = M[ii]
                fkBins = posit[ii]
                displace = int(fkBins - int(np.floor(fkBins / fsNewBins)) * fsNewBins)
                # print(displace)
                if displace !=0:
                    temp4 = temp4[-int(displace):]+temp4[:len(temp4)-int(displace)]

                temp4 = np.array(temp4).reshape((len(temp4),1))
                # print(temp4.shape)

            # print(np.fft.ifft2(temp4))
            # c.append(np.fft.ifft2(temp4).T)

            c.append(scipy.fftpack.ifft(temp4,axis=0).T)

    e =time.time()
    # print(e-s)  #被zzy注释，不显示计算运行时间
    # print(len(c),c[1],c[1].shape)
    if int(max(M)) == int(min(M)):
        c = c
        c = np.reshape(c, N, CH)
    return  c






