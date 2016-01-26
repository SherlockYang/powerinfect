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

def Draw(y, _y, t):
    rc("font", **{"family":"serif", "serif":["Times"]})
    rc("ytick", labelsize = 22)
    rc("xtick", labelsize = 22)
    rcParams['figure.figsize']= 10, 6
    #rcParams['pdf.fonttype'] = 42
    #rcParams['ps.fonttype'] = 42
    #rcParams['text.usetex'] = True
    
    win = 2
    x = xrange(1, t)
    _y = smooth(_y, 2 * win)
    _y = _y[win : -win + 1]
    fig, ax = plt.subplots()
    xfit = array([amin(x), amax(x)])
    plt.xlim(amin(x), amax(x))
    #plt.ylim(80, 100)
    data = plt.plot(x, y, 'o', markeredgecolor='black', markerfacecolor='None', mew=1, ms=10, label='Data')
    model = plt.plot(x, _y, '-', color='#E74C3C', lw=3, label='Model')
    ax.legend(fontsize=22, numpoints=1)
    plt.xlabel('time', size=22)
    plt.ylabel('popularity', size=22)
    plt.savefig('figs/res' + str(t) + '.pdf', format='pdf', bbox_inches='tight')

