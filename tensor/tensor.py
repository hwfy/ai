import tensorflow as tf

x = tf.constant(8)
y = tf.constant(9)
z = tf.mul(x, y)

sess = tf.Session()
sess.run(z)

