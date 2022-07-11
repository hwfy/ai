import tensorflow as tf
import numpy as np

SEED = 23450

rdm = np.random.RandomState(SEED)
X = rdm.random(32, 2)
Y = [[x1, x2] for (x1, x2) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
