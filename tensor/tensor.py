# 2.0中不能直接使用Session、placeholder函数，因此引入compat.v1
import tensorflow.compat.v1 as tf
import numpy as np

# 保证sess.run()能够正常运行
tf.disable_eager_execution()


# 查看是否GPU
tf.test.is_gpu_available

# 基本运算
x = tf.constant(8)
y = tf.constant(9)
z = tf.multiply(x, y)
# 累加
# tf.add(x, y)
# 求平方
# x = tf.constant([1, 2, 3, 4])
# tf.square(x)
# 求和
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# tf.reduce_sum(x) => 6
# 纵向求和
# tf.reduce_sum(x, 0) => [2, 2, 2]
# 横向求和
# tf.reduce_sum(x, 1) => [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) => [[3], [3]]
# 矩阵相乘
# tf.matmul([[2,3], [1,3], [2,3]])

sess = tf.Session()
r = sess.run(z)
print(r)

# tensor和numpy转换
np.ones([3, 3])


# 在代码层面，每一个tensor值在graph上都是一个op，当我们将train数据分成一个个minibatch然后传入网络进行训练时，
# 每一个minibatch都将是一个op，这样的话，一副graph上的op未免太多，也会产生巨大的开销；
# 于是就有了tf.placeholder()，我们每次可以将 一个minibatch传入到x = tf.placeholder(tf.float32,[None,32])上，
# 下一次传入的x都替换掉上一次传入的x，这样就对于所有传入的minibatch x就只会产生一个op，不会产生其他多余的op，进而减少了graph的开销

# shape：数据形状，默认是None，就是一维值；也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
a = tf.placeholder(tf.float32, shape=(None, 2))
b = tf.constant(2.0)

output = tf.multiply(a, b)
with tf.Session() as sess:
    # 使用随机种子保证每次随机值一致
    rdm = np.random.RandomState(seed=23450)
    # 生产32行2列随机数矩阵（类型ndarray）, 这里的列2必须和shape中2一致
    X = rdm.rand(32, 2)
    # 转换成类型list
    Y = [[x1, x2] for (x1, x2) in X]
    # feed_dict 将minibatch传入到placeholder
    # minibatch定义：http://t.zoukankan.com/cation-p-11741740.html
    print(sess.run(output, feed_dict={a: Y}))
