import tensorflow as tf
import numpy as np
import math

init=np.asarray([[3,1]],dtype=float)
res1=np.asarray([[2,1]],dtype=float)
res2=np.asarray([[1,2]],dtype=float)
with tf.Session() as sess:
    var=tf.Variable(init,dtype=tf.float32,trainable=False)
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    ph=tf.placeholder(tf.float32,shape=[None,2])
    opt=var.assign(ph)
    with tf.control_dependencies([opt]):
        apply = ema.apply([var])
    averaged=ema.average(var)
    tf.global_variables_initializer().run()
    d, a = sess.run([var, averaged])
    print(d, a)
    for i in range(1000):
        if i<50:
            res=res1
        else:
            res=res2
        sess.run([apply],feed_dict={ph:np.asarray([[math.cos(i/100),math.sin(i/100)]],dtype=float)})
        d,a=sess.run([var,averaged])
        print(d,a)