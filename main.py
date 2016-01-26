import data
import model
import fit

(X, post_log, time_list, R, D) = data.Load('../data/url_share')
print 'T: ', len(time_list)
K = 2
max_iter = 3
pow_model = model.Model(X, K, D)
pow_model.Estimate(max_iter)
(Y, _Y) = pow_model.Fit()
print 'Data: ', Y
print 'Model: ', _Y
fit.Draw(Y, _Y, len(Y) + 1)
