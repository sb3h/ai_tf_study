import tensorflow as tf

# fetch 运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input3, input2)
mul = tf.multiply(input1, add)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #当fetch时，传入就要是列表
    result = sess.run([mul, add])
    print(result)
    # [21.0, 7.0]
    sess.close()

#feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed 在使用时，再使用feed_dict字典进行赋值
    result = sess.run(output,feed_dict={input1:[7],input2:[2]})
    print(result)
    sess.close()
