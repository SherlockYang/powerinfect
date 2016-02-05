import math as math
import random as random

class Model:
    def __init__(self, _X, _K, _D):
        self.X = _X
        self.K = _K
        self.D = _D
        self.R = len(_X)
        self.T = len(_X[0])
        self.mu = 0

    def InitParameter(self):
        self.beta = 1.0
        self.rho = [random.random() for _ in xrange(self.K)]
        self.delta = [random.random() for _ in xrange(self.K)]
        for i in xrange(self.K):
            self.rho[i] /= pow(2, i)
            self.delta[i] /= pow(2, i)
        self.delta[self.K - 1] = 0.0
        self.theta = [[0 for _ in xrange(self.K)] for _ in range(self.T)]
        self.theta[0][0] = 1.0

    def SaveParameter(self, model_dir):
        output = open(model_dir, 'w')
        output.write('rho:\n')
        output.write(str(self.rho) + '\n')
        output.write('delta:\n')
        output.write(str(self.delta) + '\n')
        output.write('theta:\n')
        for i in xrange(len(self.theta)):
            output.write(str(self.theta[i]) + '\n')
        output.close()

    def LoadParameter(self, data_dir):
        self.InitParameter()
        file = open(data_dir)
        cnt = 0
        for line in file:
            cnt += 1
            tokens = line[1:len(line) - 2].split(',')
            k = 0
            for token in tokens:
                if cnt == 2:
                    self.rho[k] = float(token)
                if cnt == 4:
                    self.delta[k] = float(token)
                if cnt > 5:
                    self.theta[cnt - 6][k] = float(token)
                k += 1

    def ShowRho(self):
        print 'rho: ', self.rho

    def ShowDelta(self):
        print 'delta: ', self.delta

    def ShowTheta(self):
        print 'theta: ', self.theta
    
    def CalcPhi(self, r, t):
        return float(r) / float(self.beta * (t + 1))

    def CalcTheta(self, t):
        if t == 0:
            return 
        C = 0.0
        for r in xrange(self.R):
            C += self.X[r][t] * self.CalcPhi(r, t)
        C /= self.D
        if self.K > 1:
            self.theta[t][0] = self.theta[t - 1][0] - self.rho[0] * self.theta[t - 1][0] * self.delta[0] * C
            self.theta[t][self.K - 1] = self.theta[t - 1][self.K - 1] + self.rho[self.K - 2] * self.theta[t - 1][self.K - 2] * self.delta[self.K - 2] * C
        for k in xrange(1, self.K - 1):
            self.theta[t][k] = self.theta[t - 1][k] + (self.rho[k - 1] * self.theta[t - 1][k - 1] * self.delta[k - 1] - self.rho[k] * self.theta[t - 1][k] * self.delta[k]) * C

    #####################################
    # Gradient calculation
    # t: base time, we are given X[t] and X[t + 1]
    # r: popularity
    # K: total number of user status 
    # D: total number of posts
    # time complexity: O(R + K)
    #####################################
    def CalcGradient(self, t, r, C2):
        # calculate the gradient of rho
        _rho = [0] * self.K
        C1 = -1 * self.X[r][t] * self.CalcPhi(r, t)
        if r > 0:
            C1 += self.X[r - 1][t] * self.CalcPhi(r - 1, t)
        C1 /= self.D
        if t > 0:
            C2 /= (self.beta * t)
        C2 /= self.D

        for k in xrange(self.K):
            if t > 0:
                _rho[k] = -2 * self.rho[k] * self.theta[t - 1][k] * self.delta[k]
                if k > 0:
                    _rho[k] += self.rho[k - 1] * self.theta[t - 1][k - 1] * self.delta[k - 1]
                _rho[k] = C1 * (self.theta[t - 1][k] + _rho[k] * C2)
            else:
                _rho[k] = C1
        # calculate the gradient of delta
        _delta = [0] * self.K
        C3 = -1 * self.X[r][t] * self.CalcPhi(r, t)
        if r > 0:
            C3 += self.X[r - 1][t] * self.CalcPhi(r - 1, t)
        for k in xrange(self.K):
            if t > 0:
                _delta[k] = 0.0
                if k < self.K - 1:
                    _delta[k] -= self.rho[k] * self.theta[t - 1][k]
                if k > 0:
                    _delta[k] += self.rho[k - 1] * self.theta[t - 1][k - 1]
                _delta[k] = (C3 * _delta[k] * self.rho[k] * C2) / (self.D * self.D)
        return (_rho, _delta)

    ################################
    # use X[t - 1] to predict X[t]
    # time complexity: O(R)
    ################################
    def Predict(self, t):
        if t == 0:
            return []
        _X = []
        C = 0.0
        for k in xrange(self.K):
            C += self.theta[t][k] * self.rho[k]
        for r in xrange(self.R):
            x = -1 * self.X[r][t - 1] * self.CalcPhi(r, t)
            if r > 0:
                x += self.X[r - 1][t - 1] * self.CalcPhi(r - 1, t - 1)
            x = abs(x * C / self.D + self.X[r][t - 1])
            _X += [x]
        return _X

    ################################
    # parameter estimation by L-M method
    # X: observed data
    # K: total number of user status
    # D: total number of posts
    # max_iter: maximum number of iterations
    # we print out the squares at each iteration
    ################################
    def Estimate(self, max_iter, dump):
        self.InitParameter()
        self.mu = 0.0
        for iteration in xrange(max_iter):
            print '########## Iter', iteration + 1, ' ##########'
            loss = 0.0
            for t in xrange(self.T - 1):
                #if t % 100 == 0:
                #    print t
                # update theta at time t+1
                self.CalcTheta(t + 1)
                _X = self.Predict(t + 1)
                #print 'pred res: ', _X
                C = [0] * self.R
                if t > 0:
                    for r in xrange(self.R):
                        C[r] = self.X[r][t - 1] * r
                for r in xrange(self.R):
                    error = self.X[r][t + 1] - _X[r]
                    loss += pow(error, 2)
                    (_rho, _delta) = self.CalcGradient(t, r, C[r])
                    for k in xrange(self.K):
                        self.rho[k] += dump * 2 * _rho[k] * error
                        self.delta[k] += dump * 2 * _delta[k] * error
                continue
                # adjust parameters to make them in [0, 1]
                for k in xrange(self.K):
                    self.rho[k] = abs(self.rho[k])
                    self.delta[k] = abs(self.delta[k])    
                    if self.rho[k] > 1:
                        p = pow(10, math.ceil(math.log10(self.rho[k])))
                        self.rho[k] /= p
                    if self.delta[k] > 1:
                        p = pow(10, math.ceil(math.log10(self.delta[k])))
                        self.delta[k] /= p
            self.ShowRho()
            self.ShowDelta()
            #self.ShowTheta()
            print 'square loss: ', loss 

        print '##############################'
        #self.ShowRho()
        #self.ShowDelta()
        #self.ShowTheta()
        #self.SaveParameter()

    def Fit(self, post_log, time_list, _T):
        Y = []
        _Y = []
        for t in xrange(self.T):
            if t > 0:
                _X = self.Predict(t)
            else:
                _X = [self.X[i][t] for i in xrange(self.R)]
            y = 0
            _y = 0.0
            for r in xrange(self.R):
                y += r * self.X[r][t]
                _y += r * _X[r]
                if t > 0:
                    y -= r * self.X[r][t - 1]
                    _y -= r * _last_X[r]
            _last_X = _X
            Y += [y]
            _Y += [_y]
        # adjust time intervals
        min_t = -1
        for log in post_log:
            for t in log:
                if t < min_t or min_t < 0:
                    min_t = t
        max_t = time_list[-1]
        unit_t = (max_t - min_t + 1) / _T
        y = 0
        _y = 0.0
        interval = 0
        Z = []
        _Z = []
        _Theta = []
        _theta = [0] * self.K
        cnt = 0
        for t in xrange(self.T):
            if t == 0:
                interval += time_list[t] - min_t
            else:
                interval += time_list[t] - time_list[t - 1]
            y += Y[t]
            _y += _Y[t]
            for k in xrange(self.K):
                _theta[k] += self.theta[t][k]
            cnt += 1
            if interval >= unit_t:# or t == self.T - 1:
                Z += [y]
                _Z += [_y]
                for k in xrange(self.K):
                    _theta[k] /= cnt
                _Theta += [_theta]
                _theta = [0] * self.K
                y = 0
                _y = 0.0
                interval = 0
                cnt = 0
        return (Z, _Z, _Theta)


