import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
"""
one_hot=True相当于于0到9的标签值
"""
# 在网上下载数据集，有时候比较慢
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定义批次的大小
"""
并非一张一张去加载
"""
batch_size = 100

"""
// 整除
计算一共有多少个批次
"""
n_batch = mnist.train.num_examples // batch_size

"""
#定义两个placeholder
784:像素积28*28
None:与批次有关
"""
# 点范围
x = tf.placeholder(tf.float32, [None, 784])
# 标签值范围
y = tf.placeholder(tf.float32, [None, 10])

# 简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""
信号值:tf.matmul(x, w) + b
概率值:tf.nn.softmax(tf.matmul(x, w) + b)
"""
prediction = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train_step = optimizer.minimize(loss)

"""
最大概率值的位置:tf.argmax(y, 1)
    --例如:0到9中哪个值中是1，就返回哪个下标
是否等于:tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    true or false
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
"""
转化类型值:tf.cast(correct_prediction,tf.float32)

"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):  # 周期
        for batch in range(n_batch):  # 一共多少的批次
            # 获取一个批次的数据
            """
            batch_xs:图片
            batch_ys:标签
            """
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        # 使用测试数据的求的准确率
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("lter " + str(epoch) + ", Testing Accuracy +" + str(acc))

    sess.close()


"""
优化方向:
    加大批次每次加载的大小
    添加神经网络的隐藏层        
    调整权重值和偏离值，含有初始化的值
    loss的算法:可更换"交叉熵"
    调整优化器的学习率
    加多训练次数
"""