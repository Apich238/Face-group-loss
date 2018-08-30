import tensorflow as tf
import tensorflow.contrib.slim as slim


# def conv(input, ksz, str, out_sz):
#     return tf.layers.conv2d(input,out_sz,ksz,str,'same',activation=tf.nn.relu,use_bias=True)


# def maxpool(input, ksz=3, stride=2):
#     return tf.layers.max_pooling2d(input, ksz, stride, 'same')


def l2pool(input, ksz=3, stride=2,name='l2pool'):
    with tf.variable_scope(name):
        net=tf.square(input)
        net=ksz*ksz*slim.avg_pool2d(net,kernel_size=ksz,stride=stride)
        net=tf.sqrt(net)
    return net


# def inception(input, c1out, c3red, c3out, c5red, c5out,
#               pooltype='max', pool_red=0, out_stride=1, name='inception'):
#     branches = []
#     with tf.variable_scope(name):
#         if c1out > 0 and out_stride == 1:
#             with tf.variable_scope('1x1'):
#                 net = conv(input, 1, 1, c1out)
#                 branches.append(net)
#         if c3red > 0 and c3out > 0:
#             with tf.variable_scope('3x3'):
#                 net = conv(input, 1, 1, c3red)
#                 net = conv(net, 1, out_stride, c3out)
#                 branches.append(net)
#         if c5red > 0 and c5out > 0:
#             with tf.variable_scope('5x5'):
#                 net = conv(input, 1, 1, c5red)
#                 net = conv(net, 1, out_stride, c5out)
#                 branches.append(net)
#         if pooltype is not None:
#             with tf.variable_scope('pool_red'):
#                 if pooltype == 'max':
#                     net = maxpool(input, 3, out_stride)
#                 else:
#                     net = l2pool(input, 3, out_stride)
#                 if pool_red > 0:
#                     net = conv(net, 1, 1, pool_red)
#                 branches.append(net)
#         return tf.concat(branches, -1)
#

def inception(input, c1out, c3red, c3out, c5red, c5out,
              pooltype='max', pool_red=0, out_stride=1, name='inception'):
    branches = []
    with tf.variable_scope(name):
        if c1out > 0 and out_stride == 1:
            with tf.variable_scope('1x1'):
                net=slim.conv2d(input,c1out,kernel_size=1,stride=1,scope='reduce')
                branches.append(net)
        if c3red > 0 and c3out > 0:
            with tf.variable_scope('3x3'):
                net=slim.conv2d(input,c3red,kernel_size=1,stride=1,scope='reduce')
                net=slim.conv2d(net,c3out,kernel_size=3,stride=out_stride,scope='out')
                branches.append(net)
        if c5red > 0 and c5out > 0:
            with tf.variable_scope('5x5'):
                net=slim.conv2d(input,c5red,kernel_size=1,stride=1,scope='reduce')
                net=slim.conv2d(net,c5out,kernel_size=5,stride=out_stride,scope='out')
                branches.append(net)
        if pooltype is not None:
            with tf.variable_scope('pool_red'):
                if pooltype == 'max':
                    net = slim.max_pool2d(input,kernel_size=3,stride=out_stride,scope='pool')
                else:
                    net = l2pool(input, 3, out_stride)
                if pool_red > 0:
                    net = slim.conv2d(net,pool_red,kernel_size=1,stride=1)
                branches.append(net)
        return tf.concat(branches, -1)


#def norm(input):
    #return tf.nn.local_response_normalization(input,)



# def inference(images, keep_probability, phase_train=True,
#               bottleneck_layer_size=128, weight_decay=0.0, reuse=None, use_locrespnorm=False):
    # with tf.variable_scope('facenet_inception_model'):
    #     net = conv(images,7, 2, 64)
    #     net = maxpool(net, 3, 2)
    #     if use_locrespnorm:
    #         net = norm(net)
    #     # 2
    #     net = inception(net, 0, 64, 192, 0, 0, None, 0, 1, 'inception2')
    #     if use_locrespnorm:
    #         net = norm(net)
    #     net = maxpool(net, 3, 2)
    #     # 3
    #     net = inception(net, 64, 96, 128, 16, 32, 'max', 32, 1, 'inception3a')
    #     net = inception(net, 64, 96, 128, 32, 64, 'l2', 64, 1, 'inception3b')
    #     net = inception(net, 0, 128, 256, 32, 64, 'max', 0, 2, 'inception3c')
    #     # 4
    #     net = inception(net, 256, 96, 192, 32, 64, 'l2', 128, 1, 'inception4a')
    #     net = inception(net, 224, 112, 224, 32, 64, 'l2', 128, 1, 'inception4b')
    #     net = inception(net, 192, 128, 256, 32, 64, 'l2', 128, 1, 'inception4c')
    #     net = inception(net, 160, 144, 288, 32, 64, 'l2', 128, 1, 'inception4d')
    #     net = inception(net, 0, 160, 256, 64, 128, 'max', 0, 2, 'inception4e')
    #     # 5
    #     net = inception(net, 384, 192, 384, 48, 128, 'l2', 128, 1, 'inception5a')
    #     net = inception(net, 384, 192, 384, 48, 128, 'max', 128, 1, 'inception5b')
    #     # final
    #     net = tf.reduce_mean(net, axis=[1, 2], keep_dims=False, name='final_avg_pool')
    #     net = tf.layers.dense(net, bottleneck_layer_size, activation=None, use_bias=False, name='fully_conn')
    #     return net

def inference(images, keep_probability=1., phase_train=True,
                  bottleneck_layer_size=128, weight_decay=0.0, reuse=None,use_locrespnorm=False):
        """ Define an inference network for face recognition based
               on inception modules using batch normalization

        Args:
          images: The images to run inference on, dimensions batch_size x height x width x channels
          phase_train: True if batch normalization should operate in training mode
        """

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        }

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with tf.variable_scope('Inception', 'Inception', [images], reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=phase_train):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                        #1
                        net = slim.conv2d(images,64,kernel_size=7,stride=2,scope='Conv1_7x7')
                        net = slim.max_pool2d(net,kernel_size=3,stride=2,scope='MaxPool1')
                        # 2
                        net = inception(net, 0, 64, 192, 0, 0, None, 0, 1, 'inception2')
                        net = slim.max_pool2d(net,kernel_size=3,stride=2,scope='MaxPool1')
                        # 3
                        net = inception(net, 64, 96, 128, 16, 32, 'max', 32, 1, 'inception3a')
                        net = inception(net, 64, 96, 128, 32, 64, 'l2', 64, 1, 'inception3b')
                        net = inception(net, 0, 128, 256, 32, 64, 'max', 0, 2, 'inception3c')
                        # 4
                        net = inception(net, 256, 96, 192, 32, 64, 'l2', 128, 1, 'inception4a')
                        net = inception(net, 224, 112, 224, 32, 64, 'l2', 128, 1, 'inception4b')
                        net = inception(net, 192, 128, 256, 32, 64, 'l2', 128, 1, 'inception4c')
                        net = inception(net, 160, 144, 288, 32, 64, 'l2', 128, 1, 'inception4d')
                        net = inception(net, 0, 160, 256, 64, 128, 'max', 0, 2, 'inception4e')
                        # 5
                        net = inception(net, 384, 192, 384, 48, 128, 'l2', 128, 1, 'inception5a')
                        net = inception(net, 384, 192, 384, 48, 128, 'max', 128, 1, 'inception5b')
                        # final
                        net = tf.reduce_mean(net, axis=[1, 2], keepdims=False, name='final_avg_pool')
                        net=slim.fully_connected(net,bottleneck_layer_size,None,scope='Bottleneck', reuse=False,)
                        return net

# import numpy as np
#
# ph=tf.placeholder(tf.float32,shape=[None,120,120,3])
# rv=np.random.standard_normal([17,120,120,3])
# net=inference(ph,1.,True,128,0.1)
#
# s=tf.InteractiveSession()
# s.run(tf.global_variables_initializer())
#
# print(s.run(net,feed_dict={ph:rv}))
#
# s.close()
