import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # result = sess.run(product)
    # print(result)
    sess.close()
