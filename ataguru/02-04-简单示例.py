import tensorflow as tf
import numpy as np

# 使用numpy 生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# print(x_data)
# print()
# print(y_data)

# 通常 加数字 的变量都用 b
b = tf.Variable(0.)  # 截距 0.2
# 通常 乘数字 的变量都用 k
k = tf.Variable(0.)  # 斜率 0.1
"""
通过虚化参数，构造一个线性模型
上面的样本：y_data = x_data * 0.1 + 0.2
"""
y = x_data * k + b  # 模型

"""
通过tf中的函数来优化这个模型，让模型尽可能接近样本的数据"分布"
主要优化是 tf的变量 k 和 b
"""

# 二次代价函数()
"""
误差:     y_data - y
平方:     tf.square(y_data - y)
平均值:   tf.reduce_mean(tf.square(y_data - y))
note:平均值是除以2
"""
loss = tf.reduce_mean(tf.square(y_data - y))

# 下面的0.2是学习率
# 定义一个梯度下降法（最优化算法）(GradientDescentOptimizer)来进行训练的优化器
#梯度下降法应用在"损失函数的最小值时"
#梯度上升法应用在"损失函数的最大值时"
"""
优化器:不断改变k和b的值，从而让loss值越小 
"""
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
"""
训练目的就是最小化loss的值，越小就越接近真实值
k和b就越接近真实值
"""
train = optimizer.minimize(loss)



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        result = sess.run(train)
        if step%20==0:
            print("step:%s,sess.run([k,b]:%s" % (step,sess.run([k,b])))
            #?k和b是怎么变化的

    sess.close()
