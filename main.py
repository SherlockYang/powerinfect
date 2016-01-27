import data
import model
#import fit

max_line = 1000
(X, post_log, time_list, R, D) = data.Load('../data/url_share', max_line)
print 'T: ', len(time_list)
print 'R: ', R
print 'D: ', D
#print 'Time list: ', time_list
#print 'Post logs:', post_log
K = 4
max_iter = 10
time_interval = 100
dump = 0.02
pow_model = model.Model(X, K, D)
pow_model.Estimate(max_iter, dump)
#pow_model.LoadParameter('models/model_t_92_k_5')
(Y, _Y, theta) = pow_model.Fit(post_log, time_list, time_interval)
#print 'Data: ', Y
#print 'Model: ', _Y
#print 'Theta: ', theta
fit.Draw(Y, _Y, theta, pow_model.rho, pow_model.delta, 5)
