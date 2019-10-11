#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:25:02 2019

@author: ducbueno
"""

import scipy.io
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def perm(x):
    return tfp.math.interp_regular_1d_grid(x, x_ref_min=0., x_ref_max=1., y_ref=K_tf)


def dperm(x):
    return tf.gradients(perm(x), x)[0]


def sq(x):
    return x**2


data = scipy.io.loadmat('2phaseflow_spe10.mat')
K = data['K'].flatten()
x = np.linspace(0, 1, 1000)[:, None]

K_tf = tf.placeholder(tf.float32, shape=[None])
x_tf = tf.placeholder(tf.float32, shape=[None, x.shape[1]])

sess = tf.Session()

K_interp = sess.run(perm(x_tf), {x_tf: x, K_tf: K})
dK_interp = sess.run(dperm(x_tf), {x_tf: x, K_tf: K})
x2 = sess.run(sq(x_tf), {x_tf: x})
