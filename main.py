import data
import model
import fit

(X, post_log, time_list, R, D) = data.Load('../data/url_share')
print 'T: ', len(time_list)
print 'R: ', R
print 'D: ', D
print 'Time list: ', time_list
print 'Post logs:', post_log
K = 2
max_iter = 10
pow_model = model.Model(X, K, D)
pow_model.Estimate(max_iter)
(Y, _Y) = pow_model.Fit(time_list)
print 'Data: ', Y
print 'Model: ', _Y
fit.Draw(Y, _Y, len(Y) + 1)
