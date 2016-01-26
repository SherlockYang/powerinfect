import numpy as np
import math as math
from pylab import * 
import matplotlib.pyplot as plt

# T: number of time intervals
def ReadWeixin(fileDir):
    file = open(fileDir)
    X = []
    max_line = 5
    cnt = 0
    for line in file:
        x = []
        tokens = line.split(' ')
        #print cnt
        if line.find('http') != -1:
            continue
        for token in tokens:
            a = token.split(':')
            if (len(a) < 2):
                continue
            t = int(a[1])
            x += [t]  
        X += [x]
        cnt += 1
        if cnt >= max_line and max_line > 0:
            break
    return X

def pair_cmp(a, b):
    return cmp(a[0], b[0])

def DefineTimeInterval(post_log): 
    I = []
    # L[i]=(t, d): post d is retweeted once at time t
    L = []
    post_id = -1
    for log in post_log:
        post_id += 1
        for t in log:
            L += [(t, post_id)]
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
    print 'R: ', R
    #print 'X: ', X
    #print 'X^T: ', [[X[i][j] for i in xrange(R)] for j in range(T)]
    return (R, X)

def Load(file_dir):
    post_log = ReadWeixin(file_dir)
    D = len(post_log)
    #print 'post_log: ', post_log
    time_list = DefineTimeInterval(post_log)
    print '#posts: ', D
    (R, X) = TransferInput(post_log, time_list)
    return (X, post_log, time_list, R, D)
