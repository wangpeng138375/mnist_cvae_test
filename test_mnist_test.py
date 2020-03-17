'''
Created on 2019年3月5日

@author: wangpeng
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data



mb_size = 64
z_dim = 100
X_dim = 784
y_dim = 10

print(X_dim,y_dim)
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================
c = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])





# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]),name="P_W1")
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]),name="P_b1")

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]),name="P_W2")
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]),name="P_b2")



def P(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    h = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits




# Sampling from random z
X_samples, _ = P(z, c)



sess = tf.Session()
chk_point="model_result/mnist_cvae-811000"
saver=tf.train.Saver()
saver.restore(sess,chk_point)



y = np.zeros(shape=[16, y_dim])
y[:, np.random.randint(0, y_dim)] = 1.
print(y)

samples = sess.run(X_samples,
                           feed_dict={z: np.random.randn(16, z_dim), c: y})

fig = plot(samples)
plt.savefig('out_test/{}.png'.format(str(1).zfill(3)), bbox_inches='tight')

plt.close(fig)