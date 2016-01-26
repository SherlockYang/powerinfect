import math as math
import random as random

class Model:
    def __init__(self, _X, _K, _D):
        self.X = _X
        self.K = _K
        self.D = _D
        self.R = len(_X)
        self.T = len(_X[0])

    def InitParameter(self):
        self.beta = 1.0
        self.rho = [random.random() for _ in xrange(self.K)]
        self.delta = [random.random() for _ in xrange(self.K)]
        self.theta = [[0 for _ in xrange(self.R)] for _ in range(self.T)]
        self.theta[0][0] = 1.0

    def SaveParameter(self):
        output = open('model', 'w')
        output.write('rho:\n')
        output.write(str(self.rho) + '\n')
        output.write('delta:\n')
        output.write(str(self.delta) + '\n')
        output.write('theta:\n')
        for i in xrange(len(self.theta)):
            output.write(str(self.theta[i]) + '\n')
        output.close()

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
        self.theta[t][0] = self.theta[t - 1][0] - self.rho[0] * self.theta[t - 1][0] * self.delta[0] * C
        for k in xrange(1, self.K):
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
                _delta[k] = -1 * self.rho[k] * self.theta[t - 1][k]
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
            x = x * C / self.D + self.X[r][t - 1]
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
    def Estimate(self, max_iter):
        self.InitParameter()
        for iteration in xrange(max_iter):
            print '##### Iter', iteration, ' #####'
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
                        self.rho[k] += -2 * _rho[k] * error
                        self.delta[k] += -2 * _delta[k] * error
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
            #self.ShowRho()
            #self.ShowDelta()
            #self.ShowTheta()
            print 'square loss: ', loss 
        self.SaveParameter()

    def Fit(self, time_list):
        Y = []
        _Y = []
        for t in xrange(1, self.T):
            _X = self.Predict(t)
            y = 0
            _y = 0.0
            for r in xrange(self.R):
                y += r * self.X[r][t]
                _y += r * _X[r]
            Y += [y]
            _Y += [_y]

        return (Y, _Y)


