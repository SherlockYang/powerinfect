import math as math
import time

# T: number of time intervals
# post_log[d][i] = t, the i-th retweet of post d happens at time t
def ReadWeixin(fileDir, max_line, post_id):
    file = open(fileDir)
    X = []
    cnt = 0
    min_t = -1
    max_t = -1
    N = 0
    for line in file:
        x = []
        tokens = line.split(' ')
        if str(tokens[1]) != post_id and post_id > 0:
            continue
        for i in xrange(len(tokens)):
            token = tokens[i]
            a = token.split(':')
            if (len(a) < 2):
                continue
            t = int(a[1])
            if t > max_t:
                max_t = t
            if t < min_t or min_t < 0:
                min_t = t
            x += [t]  
            N += 1
        X += [x]
        cnt += 1
        if cnt >= max_line and max_line > 0:
            break
    scale = []
    for log in X:
        val = len(log)
        for i in xrange(len(scale), val + 1):
            scale += [0]
        scale[val] += 1
    output = open('scale_wechat.dis', 'w') 
    for i in xrange(len(scale)):
        output.write(str(i) + ' ' + str(scale[i]) + '\n')
    mu = float(N) / (max_t - min_t + 1.0)
    print N, max_t - min_t + 1
    #print 'D:', len(X)
    #print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(min_t))
    #print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max_t))
    return [X, mu]

def pair_cmp(a, b):
    return cmp(a[0], b[0])

def DefineTimeInterval(post_log, T): 
    I = []
    # L[i]=(t, d): post d is retweeted once at time t
    L = []
    post_id = -1
    min_t = -1
    max_t = -1
    for log in post_log:
        post_id += 1
        for t in log:
            L += [(t, post_id)]
            if t < min_t or min_t < 0:
                min_t = t
            if t > max_t:
                max_t = t
    if T != -1:
        u = (max_t - min_t + 1) / T
        print max_t, min_t, T
        for i in xrange(min_t + u, max_t + u, u):
            I += [i]
        return I
    L.sort(pair_cmp)
    L += [(L[len(L) - 1][0] + 1, -1)]
    print '#retweets: ', len(L)

    s = set()
    for l in L:
        t = l[0]
        post = l[1]
        if post in s or post == -1:
            I += [t]
            s.clear()
        s.add(post)
    #print 'Time intervals: ', I
    return I

def FindTime(t, time_list):
    l = 0
    r = len(time_list) - 1
    mid = -1
    while l < r:
        mid = (l + r) / 2
        if time_list[mid] > t:
            r = mid
        else:
            l = mid + 1
    mid = (l + r) / 2
    if time_list[mid] <= t:
        print 'Error in binary search!'
    return mid

def TransferInput(post_log, time_list):
    # Y[d][t]: popularity of post d at time t
    Y = []
    T = len(time_list)
    for x in post_log:
        y = [0] * T
        for t in x:
            idx = FindTime(t, time_list)
            #print t, idx, time_list[idx] - t
            y[idx] += 1
        for t in xrange(1, T):
            y[t] += y[t - 1]
        Y += [y]
    # total number of retweets
    R = 0
    for y in Y:
        for val in y:
            if val > R:
                R = val
    R += 1
    X = [[0] * T for r in xrange(R)]
    for y in Y:
        for t in xrange(0, T):
            idx = y[t]
            X[idx][t] += 1
    #print 'X: ', X
    #print 'X^T: ', [[X[i][j] for i in xrange(R)] for j in range(T)]
    return (R, X)

def Load(file_dir, max_line, time_num, post_id):
    (post_log, mu) = ReadWeixin(file_dir, max_line, post_id)
    D = len(post_log)
    time_list = DefineTimeInterval(post_log, time_num)
    (R, X) = TransferInput(post_log, time_list)
    return (X, post_log, time_list, R, D, mu)
