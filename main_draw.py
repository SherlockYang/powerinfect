import data
import model
import fit

def Run(data_dir, post_id):
    max_line = -1
    (X, post_log, time_list, R, D, mu) = data.Load(data_dir, max_line, 100, post_id)
    print 'T: ', len(time_list)
    print 'R: ', R
    print 'D: ', D
    #print 'Time list: ', time_list
    #print 'Post logs:', post_log
    K = 5
    time_interval = 100
    pow_model = model.Model(X, K, D)
    #pow_model.Estimate(max_iter, dump)
    model_dir = 'models/160205/' + str(post_id) + '.model'
    pow_model.LoadParameter(model_dir)
    (Y, _Y, theta) = pow_model.Fit(post_log, time_list, time_interval)
    #print 'Data: ', Y
    #print 'Model: ', _Y
    #print 'Theta: ', theta
    fit.Draw(Y, _Y, theta, pow_model.rho, pow_model.delta, 5, 1, mu, post_id)

if __name__=="__main__":
    data_dir = '../data/wechat_cascade'
    data_dir = '../data/weibo.data'
    post_id_list = [3098668601, 3098663738, 3098668335, 3098663627, 3098663808, 3098660100, 3098664319, 3098666613, 3098664719]
    post_id_list = ['11669024264553', '100532008832545', '19642038467811']
    for post_id in post_id_list:
        Run(data_dir, post_id)
