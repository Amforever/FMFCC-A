import numpy as np
import math
def winfuns(name,x):
    if winfuns.__code__.co_argcount < 2:
        print('Not enough input arguments')

    Num = (x.shape[0])*(x.shape[1])
    if Num ==1:
        N=x
        if (winfuns.__code__.co_argcount<3):
            L=N
        if(L<N):
            print('Output length L must be larger than or equal to N')
        if np.mod(N,2)==0:
            x0 = (np.arange(0, 0.5-(1/N)+1/N-0.0001, 1/N))
            x2 = np.arange(-0.5, -1/N+1/N-0.0001, 1/N)
            if L > N:
                x1 = (-N * np.ones(1, L - N))
                x = np.hstack((x0, x1))
                x = np.hstack((x, x2))
            else:
                x = np.hstack((x0, x2))
            x = x.T

        else:
            x0 = np.arange(0, 0.5-(0.5/N)+1/N-0.0001, 1/N)
            x2 = np.arange(-0.5+0.5/N, -1/N +1/N-0.0001, 1/N)
            if L>N:
                x1 = (-N * np.ones(1, L - N))
                x = np.hstack((x0, x1))
                x = np.hstack((x, x2))
            else:
                x = np.hstack((x0,x2))
            x = x.T
    # print(x.size)
    # if len(x.shape) > 1:
    #     x = x.transpose()
    if name in ['Hann','hann','nuttall10','Nuttall10']:
        g = 0.5+0.5*np.cos(2*np.pi*x)
    elif name in ['Cosine','cosine','cos','Cos','sqrthann','Sqrthann']:
        g=math.cos(np.pi*x)
    elif name in ['hamming','nuttall01','Hamming','Nuttall01']:
        g=0.54 + 0.46*np.cos(2*np.pi*x)
    elif name in ['square','rec','Square','Rec']:
        g=float(np.abs(x)<.5)
    elif name in ['tri','triangular','bartlett','Tri','Triangular','Bartlett']:
        g=1-2*np.abs(x)
    elif name in ['blackman','Blackman']:
        g=.42 + .5*np.cos(2*np.pi*x) + 0.08*np.cos(4*np.pi*x)
    elif name in ['blackharr','Blackharr']:
        g=0.35875 + 0.48829*np.cos(2*np.pi*x) + 0.14128*np.cos(4*np.pi*x)+0.01168*np.cos(6*np.pi*x);
    elif name in ['modblackharr','Modblackharr']:
        g= .35872 + .48832*np.cos(2*np.pi*x) + .14128*np.cos(4*np.pi*x) + .01168*np.cos(6*np.pi*x);
    elif name in ['nuttall','nuttall12','Nuttall','Nuttall12']:
        g=.355768 + .487396*np.cos(2*np.pi*x) + .144232*np.cos(4*np.pi*x) + .012604*np.cos(6*np.pi*x);
    elif name in ['nuttall20','Nuttall20']:
        g=3/8 + 4/8*np.cos(2*np.pi*x) + 1/8*np.cos(4*np.pi*x)
    elif name in ['nuttall11','Nuttall11']:
        g = .40897 + .5*np.cos(2*np.pi*x) + .09103*np.cos(4*np.pi*x)
    elif name in ['nuttall02','Nuttall02']:
        g = .4243801 + .4973406*np.cos(2*np.pi*x) + .0782793*np.cos(4*np.pi*x)
    elif name in ['nuttall30','Nuttall30']:
        g = 10/32 + 15/32*np.cos(2*np.pi*x) + 6/32*np.cos(4*np.pi*x) + 1/32*np.cos(6*np.pi*x)
    elif name in ['nuttall21','Nuttall21']:
        g = .338946 + .481973*np.cos(2*np.pi*x) + .161054*np.cos(4*np.pi*x) + .018027*np.cos(6*np.pi*x)
    elif name in ['nuttall03','Nuttall03']:
        g = .3635819 + .4891775*np.cos(2*np.pi*x) + .1365995*np.cos(4*np.pi*x) + .0106411*np.cos(6*np.pi*x)
    elif name in ['gauss','truncgauss','Gauss','Truncgauss']:
        g = g = math.exp(-18*(x**2))
    elif name in ['wp2inp','Wp2inp']:
        g = math.exp(math.exp(-2 * x) * 25. * (1 + 2 * x));
        g = g / max(g)
    else:
        print('Unknown window function:'+ name)
    # print(np.abs(x))
    g = g*((np.abs(x) < .5))

    return g

