from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
from datetime import datetime
import os

import sphere_loss_v34 as sphere_loss
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def conv2d(x, W, b):
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)


def max_pool(x, size=2, stride=2):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def network(x, features_sz):
    with tf.name_scope('network'):
        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input', x_image, 10)

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            variable_summaries(W_conv1, 'W_conv1')
            b_conv1 = bias_variable([32])
            variable_summaries(b_conv1, 'b_conv1')
            h_conv1 = conv2d(x_image, W_conv1, b_conv1)

            # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            variable_summaries(W_conv2, 'W_conv2')
            b_conv2 = bias_variable([64])
            variable_summaries(b_conv2, 'b_conv2')
            h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)

            # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            variable_summaries(W_fc1, 'W_fc1')
            b_fc1 = bias_variable([1024])
            variable_summaries(b_fc1, 'b_fc1')

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, features_sz])
            variable_summaries(W_fc2, 'W_fc2')
            b_fc2 = bias_variable([features_sz])
            variable_summaries(b_fc2, 'b_fc2')

            out = tf.matmul(h_fc1, W_fc2) + b_fc2

    return tf.identity(out, name='features')


def main(args):
    batch_sz = 50
    images_per_class = batch_sz // 10
    # keep_prob_val=1

    features_sz = 16  # 256  # 1024

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

    features = network(x, features_sz)

    R = 0.5
    m = 0.6
    l = 1
    centers_decay = 0.9
    mx, sd, centers, embeddings = sphere_loss.get_sphere_loss(features, images_per_class, R, m, l)
    with tf.name_scope('classifier'):
        centers_var = weight_variable([features_sz, 10])
        variable_summaries(centers_var, 'W_fc2')
        variable_summaries(centers, 'centers_values')
        y_conv = tf.matmul(features, centers_var)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(mx)
        sm_loss= tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)

    with tf.name_scope('optimizer'):
        train_net = tf.train.AdamOptimizer(1e-4).minimize(loss)
        train_classifier = tf.train.AdamOptimizer(1e-4).minimize(sm_loss,var_list=[centers_var])

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    with tf.name_scope('stats'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('train_accuracy', accuracy)

    logdir = r'D:\models\mnist_conv_myloss_v3_4-' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '_features=' + str(
        features_sz) + '_m=' + str(m) + '_r=' + str(R) + '_l=' + str(l)

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def get_batch(training, unordered=True, size=50):
        if not training:
            return mnist.test.images, mnist.test.labels
        else:
            if unordered:
                b = mnist.train.next_batch(size)
                return b[0], b[1]
            else:
                sz = 0
                imgs = []
                lbls = []
                for i in range(10):
                    imgs.append([])
                    lbls.append([])
                images_p_class = size // 10
                while sz < images_p_class * 10:
                    b = mnist.train.next_batch(1)
                    if len(imgs[b[1][0]]) < images_p_class:
                        imgs[b[1][0]].append(b[0][0])
                        lbls[b[1][0]].append(b[1][0])
                        sz = sz + 1
                res_im,res_lb=np.asarray(imgs),np.asarray(lbls)
                res_im, res_lb=unison_shuffled_copies(res_im,res_lb)
                return res_im.reshape([-1, 784]), res_lb.reshape([-1])

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(20001):
            if i % 1000 == 0:
                tbat = get_batch(False)
                imgs = np.split(tbat[0], 200, 0)
                lbls = np.split(tbat[1], 200, 0)
                acc = 0.
                for j in range(200):
                    acc = acc + sess.run(accuracy, feed_dict={x: imgs[j], y_: lbls[j]})
                print('test accuracy %g' % (acc / 200))
                s = tf.Summary()
                s.value.add(tag='stats/test_accuracy', simple_value=acc / 200)
                train_writer.add_summary(s, i)
            batch = get_batch(True, False, batch_sz)
            if i % 100 == 0:
                train_accuracy, summ = sess.run([accuracy, merged], feed_dict={
                    x: batch[0], y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_writer.add_summary(summ, i)
            sess.run([train_net], feed_dict={x: batch[0], y_: batch[1]})
            sess.run([train_classifier], feed_dict={x: batch[0], y_: batch[1]})
        tbat = get_batch(False)
        imgs = np.split(tbat[0], 200, 0)
        lbls = np.split(tbat[1], 200, 0)
        acc = 0.
        for j in range(200):
            acc = acc + sess.run(accuracy, feed_dict={x: imgs[j], y_: lbls[j]})
        print('test accuracy %g' % (acc / 200))
        s = tf.Summary()
        s.value.add(tag='stats/test_accuracy', simple_value=acc / 200)
        train_writer.add_summary(s, i)
        train_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
