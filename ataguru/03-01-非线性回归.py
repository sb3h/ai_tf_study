import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy 在-0.5到0.5这个范围生成200个随机点，
x_data_np = np.linspace(-0.5, 0.5, 200)
# 增加一个维度，转化为200百1列的矩阵
x_data = x_data_np[:, np.newaxis]
# 干扰数据
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

"""
变量结构
    随样本结构而定
    替换的数据，就是行数据
"""
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层,（1行10列）
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 偏离数据
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 矩阵乘法
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 中间层的输出
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 最后结果层的输出
prediction = tf.nn.tanh(Wx_plus_b_L2)
# ---------------------------------------------------
loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
# ---------------------------------------------------

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2001):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 获取预测值(只需要传入x值)
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    #画图
    plt.figure()
    #样本线
    plt.scatter(x_data,y_data)
    #预测线
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()

    sess.close()
