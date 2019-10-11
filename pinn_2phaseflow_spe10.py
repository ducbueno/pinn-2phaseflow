#!/usr/bin/env python3

import time
import scipy.io
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pyDOE import lhs
from scipy.interpolate import griddata

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    def __init__(self, x0, p0, s0, p_lb, p_ub, s_lb, tb, 
                 X_f, layers, lb, ub, mu_o, mu_w, K, phi):
        X0 = np.concatenate((x0, 0*x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)
        
        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]
        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]
        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[: ,1:2]
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.p0 = p0
        self.s0 = s0
        self.p_lb = p_lb
        self.p_ub = p_ub
        self.s_lb = s_lb
        self.layers = layers
        self.lb = lb
        self.ub = ub
        self.mu_o = mu_o
        self.mu_w = mu_w
        self.K = K
        self.phi = phi

        self.weights, self.biases = self.initialize_NN(layers)

        self.x0_tf = tf.placeholder(tf.float32,
                                    shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32,
                                    shape=[None, self.t0.shape[1]])
        self.x_lb_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.t_lb.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.t_ub.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32,
                                     shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32,
                                     shape=[None, self.t_f.shape[1]])
        self.p0_tf = tf.placeholder(tf.float32,
                                    shape=[None, self.p0.shape[1]])
        self.s0_tf = tf.placeholder(tf.float32,
                                    shape=[None, self.s0.shape[1]])
        self.p_lb_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.p_lb.shape[1]])
        self.p_ub_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.p_ub.shape[1]])
        self.s_lb_tf = tf.placeholder(tf.float32,
                                      shape=[None, self.s_lb.shape[1]])
        self.K_tf = tf.placeholder(tf.float32, shape=[None])
        self.phi_tf = tf.placeholder(tf.float32, shape=[None])
        
        self.p0_pred, self.s0_pred = self.net_ps(self.x0_tf, self.t0_tf)
        self.p_lb_pred, self.s_lb_pred = self.net_ps(self.x_lb_tf, self.t_lb_tf)
        self.p_ub_pred, _ = self.net_f_ps(self.x_ub_tf, self.t_ub_tf)
        _, self.s_ub_pred = self.net_ps(self.x_ub_tf, self.t_ub_tf)
        self.f_p_pred, self.f_s_pred = self.net_f_ps(self.x_f_tf, self.t_f_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.p0_tf - self.p0_pred)) + \
                    tf.reduce_mean(tf.square(self.s0_tf - self.s0_pred)) + \
                    tf.reduce_mean(tf.square(self.p_lb_pred)) + \
                    tf.reduce_mean(tf.square(-1 - self.p_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.s_lb_tf - self.s_lb_pred)) + \
                    tf.reduce_mean(tf.square(self.s_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_p_pred)) + \
                    tf.reduce_mean(tf.square(self.f_s_pred)) 

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0*np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            
        return weights, biases


    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y


    def net_ps(self, x, t):
        ps = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        p = ps[:, 0:1]
        s = ps[:, 1:2]
        
        return p, s
        
    
    def net_s_lb(self, x, t):  # lower boundary condition on s
        p, s = self.net_ps(x, t)
        s_t = tf.gradients(s, t)[0]
        
        return self.por(x)*s_t + self.ff(s)
    
    
    def net_f_ps(self, x, t):
        p, s = self.net_ps(x, t)
        p_x = tf.gradients(p, x)[0]
        p_xx = tf.gradients(self.perm(x)*self.lbd(s)*p_x, x)[0]
        f_x = tf.gradients(self.ff(s), x)[0]
        s_t = tf.gradients(s, t)[0]
        
        fp = -p_xx
        fs = self.por(x)*s_t + f_x

        return fp, fs
    
    
    def perm(self, x):
        return tfp.math.interp_regular_1d_grid(x, x_ref_min=0., x_ref_max=1., 
                                               y_ref=self.K_tf)
        
        
    def por(self, x):
        return tfp.math.interp_regular_1d_grid(x, x_ref_min=0., x_ref_max=1., 
                                               y_ref=self.phi_tf)
        
    
    def lbd(self, s):  # mobilidade (lambda)
        lbd = s**2/self.mu_w + (1-s)**2/self.mu_o
        
        return lbd
    
    
    def ff(self, s):  # fractional flow -> f(s)
        ff = (s**2/self.mu_w)/self.lbd(s)
        
        return ff
    

    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0,
                   self.t0_tf: self.t0,
                   self.x_lb_tf: self.x_lb,
                   self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub,
                   self.t_ub_tf: self.t_lb,
                   self.x_f_tf: self.x_f,
                   self.t_f_tf: self.t_f,
                   self.p0_tf: self.p0,
                   self.s0_tf: self.s0,
                   self.p_lb_tf: self.p_lb,
                   self.p_ub_tf: self.p_ub,
                   self.s_lb_tf: self.s_lb,
                   self.K_tf: self.K,
                   self.phi_tf: self.phi}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)


    def predict(self, X_star):
        p_star = self.sess.run(self.p0_pred, {self.x0_tf: X_star[:, 0:1], 
                                              self.t0_tf: X_star[:, 1:2]})
        f_p_star = self.sess.run(self.f_p_pred, {self.x_f_tf: X_star[:, 0:1], 
                                                 self.t_f_tf: X_star[:, 1:2],
                                                 self.K_tf: self.K,
                                                 self.phi_tf: self.phi})
        s_star = self.sess.run(self.s0_pred, {self.x0_tf: X_star[:, 0:1], 
                                              self.t0_tf: X_star[:, 1:2]})
        f_s_star = self.sess.run(self.f_s_pred, {self.x_f_tf: X_star[:, 0:1], 
                                                 self.t_f_tf: X_star[:, 1:2],
                                                 self.K_tf: self.K,
                                                 self.phi_tf: self.phi})

        return p_star, f_p_star, s_star, f_s_star
    
    
if __name__ == "__main__":
    N0 = 100
    N_b = 50
    N_f = 10000
    layers =  [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat('data/2phaseflow_spe10_smooth.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact_p = np.real(data['P_history'])
    Exact_s = np.real(data['S_history'])    
    K = data['K'].flatten()
    phi = data['phi'].flatten()
    mu_w = 0.1
    mu_o = 1.0

    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    p_star = Exact_p.T.flatten()[:, None]
    s_star = Exact_s.T.flatten()[:, None]
    
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    p0 = Exact_p[idx_x, 0:1]
    s0 = 0.1 - 0.1*x0
        
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]
    
    p_lb = 0*tb
    p_ub = Exact_p[-1:, idx_t]
    s_lb = Exact_s[0:1, idx_t]
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    lbp = lb + np.array((1/x.size, 0))
    ubp = ub - np.array((1/x.size, 0))
    X_f = lbp + (ubp-lbp)*lhs(2, N_f)

    model = PhysicsInformedNN(x0, p0, s0, p_lb, p_ub, s_lb, tb, 
                              X_f, layers, lb, ub, mu_o, mu_w, K, phi)

    start_time = time.time()
    model.train(1000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    p_pred, f_p_pred, s_pred, f_s_pred = model.predict(X_star)

    error_p = np.linalg.norm(p_star - p_pred, 2)/np.linalg.norm(p_star, 2)
    print('Error p: %e' % (error_p))
    
    error_s = np.linalg.norm(s_star - s_pred, 2)/np.linalg.norm(s_star, 2)
    print('Error s: %e' % (error_s))

    P_pred = griddata(X_star, p_pred.flatten(), (X, T), method='cubic')
    S_pred = griddata(X_star, s_pred.flatten(), (X, T), method='cubic')

    np.savez('data/2phaseflow_spe10_smooth.npz', x0=x0, tb=tb, lb=lb, ub=ub, P_pred=P_pred,
             S_pred=S_pred, Exact_p=Exact_p, Exact_s=Exact_s, t=t, x=x)