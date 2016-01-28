import numpy as np
import math as math
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def CalcAverage(theta, a, b):
    L = len(theta[0])
    res = [0] * L
    for i in xrange(L):
        res[i] = 0.0
        for j in xrange(a, b + 1):
            res[i] += theta[j][i]
        res[i] /= (b - a + 1)
    return res

# calculate diff between theta[a:b-1] and theta[b:c]
def CalcDiff(theta, a, b, c):
    A = CalcAverage(theta, a, b - 1)
    B = CalcAverage(theta, b, c)
    s = 0.0
    for i in xrange(len(A)):
        s += pow(A[i] - B[i], 2)
    return s

def TransferTheta(theta, K):
    T = len(theta)
    F = [[[-1] * T for _ in xrange(T)] for _ in xrange(K)]
    H = [[[-1] * T for _ in xrange(T)] for _ in xrange(K)]
    for i in xrange(T):
        F[0][i][0] = 0
        H[0][i][0] = -1
    for k in xrange(1, K):
        for i in xrange(k, T):
            for j in xrange(0, i + 1):
                for l in xrange(0, j):
                    if F[k - 1][j - 1][l] >= 0:
                        val = F[k - 1][j - 1][l] + CalcDiff(theta, l, j, i)
                        if val > F[k][i][j]:
                            F[k][i][j] = val
                            H[k][i][j] = l
    max_val = -1
    h = 0
    for i in xrange(T):
        if F[K - 1][T - 1][i] > max_val:
            max_val = F[k][T - 1][i]
            h = i
    t = T - 1
    k = K - 1
    _theta = []
    time_list = []
    while k >= 0:
        #print 't', k, ': [', h, t, ']'
        _theta = [CalcAverage(theta, h, t)] + _theta
        time_list = [h] + time_list
        _h = H[k][t][h]
        t = h - 1
        h = _h
        k -= 1
    return (_theta, time_list)

def TransferThetaByTime(theta, time_list):
    _theta = []
    h = 0
    if time_list[-1] != len(theta):
        time_list += [len(theta)]
    for t in time_list:
        for _ in xrange(t - h):
            _theta += [CalcAverage(theta, h, t - 1)]
        h = t
    return _theta

def Draw(y, _y, theta, rho, delta, T, f):
    rc("font", **{"family":"serif", "serif":["Times"]})
    rc("ytick", labelsize = 22)
    rc("xtick", labelsize = 22)
    rcParams['figure.figsize']= 10, 6
    #rcParams['pdf.fonttype'] = 42
    #rcParams['ps.fonttype'] = 42
    #rcParams['text.usetex'] = True
    
    t = len(_y)
    if t < T:
        T = t
    win = 2
    x = xrange(t)
    if t > 4:
        _y = smooth(_y, 2 * win)
        _y = _y[win : -win + 1]
    if f == 0:
        ax1 = plt.subplot(111)
    else:
        ax1 = plt.subplot(211)
    #xfit = array([amin(x), amax(x)])
    plt.xlim(0, t - 1)
    #plt.ylim(80, 100)
    data = plt.plot(x, y, 'o', markeredgecolor='black', markerfacecolor='None', mew=1, ms=13, label='Data')
    model = plt.plot(x, _y, '-', color='#E74C3C', lw=3, label='Model')
    ax1.legend(fontsize=22, numpoints=1)
    #ax1.get_xaxis().set_ticks([])
    plt.xlabel('time', size=22)
    plt.ylabel('popularity', size=22)
    file_dir = 'figs/res' + str(t) + '.pdf'

    if f > 0:
        ax2 = plt.subplot(212)
    k = len(theta[0])
    for i in xrange(len(theta)):
        for j in xrange(len(theta[i])):
            theta[i][j] = abs(theta[i][j])
    (Z, time_list) = TransferTheta(theta, T)
    print 'DP generated time list: ', time_list
    #time_list = [1, 6, 10, 17, 21]
    Z = TransferThetaByTime(theta, time_list)
    Z = np.asarray(theta).T
    #for i in xrange(k):
    #    for j in xrange(T):
    #        Z[i][j] = 0.0
    #        for k in xrange(j * t / T, (j + 1) * t / T):
    #            Z[i][j] += theta[k][i]
    #        Z[i][j] /= t / T
    print 'Z: ', Z
    for i in xrange(len(Z[0])):
        for j in xrange(len(Z) / 2):
            tmp = Z[j][i]
            Z[j][i] = Z[len(Z) - j - 1][i]
            Z[len(Z) - j - 1][i] = tmp

    print 'time intervals: ', time_list
    if f > 0:
        plt.xlim(0, len(Z[0]) - 1)
        c = plt.pcolor(Z, vmax = 1.2, edgecolors='none', cmap=plt.cm.gray_r, linewidth=0)
    for hy in xrange(0, len(Z)):
        plt.axhline(y = hy, linewidth = 2, ls='--', color='black')
    #for vx in time_list:
    #    if vx > 0 and vx < len(Z[0]) - 1:
    #        plt.axvline(x = vx, linewidth=2, ls='--', color='black')
    xticks = ['$\\theta_'+str(len(Z) - i - 1)+'$' for i in xrange(len(Z))]
    if f > 0:
        ax2.get_yaxis().set_ticks(np.array(xrange(0, len(Z))) + 0.5)
        ax2.get_yaxis().set_ticklabels(xticks, size=25)
        ax2.get_xaxis().set_ticks([])
    #plt.ylabel('$\\theta$', size=44)
    plt.savefig(file_dir, format='pdf', bbox_inches='tight')
    print 'Saving fig in ' + file_dir

