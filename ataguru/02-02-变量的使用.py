import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

sub = tf.subtract(x, a)
add = tf.add(x, a)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run(sub)
    print(result)
    result = sess.run(add)
    print(result)
    sess.close()

state = tf.Variable(0, name="counter")
new_value = tf.add(state,1)
#assign赋值的意思
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        print("hhh:%s" % _)
        result = sess.run(update)
        print(result)

    sess.close()