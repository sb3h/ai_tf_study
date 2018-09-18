import tensorflow as tf
import numpy as np

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # result = sess.run(product)
    # print(result)
    sess.close()
