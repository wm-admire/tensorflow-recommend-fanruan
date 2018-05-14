# encoding: utf-8
# author: yaoh.wu

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA


def test():
    a = [[1, 2], [3, 4]]

    apd = pd.DataFrame(a)

    print(apd.head())

    b = [[1, 1, 1, 0, 0, 0, 2, 2, 2],
         [3, 3, 3, 6, 6, 6, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 2, 2, 2],
         [3, 3, 3, 6, 6, 6, 0, 0, 0]]

    tf_pca = PCA(n_components=0.9)
    c = tf_pca.fit_transform(np.array(b))
    print(np.shape(c))
    print(c)

    d = [[1, 2, 3], [4, 5, 6]]
    e = [1, 2, 3]
    f = [4, 5, 6]

    print(np.exp(d) / (np.exp(d) + 1))
    print(np.exp(e) / (np.exp(e) + 1))
    print(np.exp(f) / (np.exp(f) + 1))


def demo():
    a = [i for i in range(10)]
    print(a)

    b = tf.Variable([[1, 2], [1, 2], [2, 3]], name="a")
    print(b)

    t1 = [[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]]
    t4 = tf.reshape(t1, [-1, 3])
    print(t4)

    t2 = [[[1, 2, 3, 4]], [[1, 2, 3, 4]], [[1, 2, 3, 4]]]
    t3 = tf.concat([t1, t2], 2)
    print(t3)

    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]

    print(tf.shape(t1))
    print(tf.shape(t2))
    t3 = tf.concat([t1, t2], 0)
    print(t3)
    t4 = tf.concat([t1, t2], 1)
    print(t4)

    t1 = [[1], [2], [3], [4], [5]]
    t2 = [[9], [8], [7], [6], [5]]
    t3 = tf.concat([t1, t2], 1)
    print(t3)

    features = tf.range(-2, 3)
    print(features)

    sess = tf.Session()
    print(sess.run([features, tf.nn.relu(features)]))

    features2 = tf.to_float(tf.range(-1, 3))
    print(features2)
    print(sess.run([features2, tf.sigmoid(features2)]))

    print(sess.run([features2, tf.tanh(features2)]))

    features3 = tf.constant([-0.1, 0.0, 0.1, 0.2])
    print(features3)
    print(sess.run([features3, tf.nn.dropout(features3, keep_prob=0.5)]))

    batch_size = 1
    input_height = 3
    input_width = 3
    input_channels = 1
    layer_input = tf.constant([
        [
            [[1.0], [0.2], [1.5]],
            [[0.1], [1.2], [1.4]],
            [[1.1], [0.4], [0.4]]
        ]
    ])
    print(layer_input)

    kernel = [batch_size, input_height, input_width, input_channels]
    print(kernel)

    max_pool = tf.nn.max_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")
    print(max_pool)
    print(sess.run(max_pool))
    layer_input2 = tf.constant([
        [
            [[1.0], [1.0], [1.0]],
            [[1.0], [0.5], [0.0]],
            [[0.0], [0.0], [0.0]]
        ]
    ])
    print(layer_input2)

    avg_pool = tf.nn.avg_pool(layer_input2, kernel, [1, 1, 1, 1], "VALID")
    print(avg_pool)
    print(sess.run(avg_pool))
    layer_input3 = tf.constant([
        [
            [[1.], [2.], [3.]]
        ]
    ])
    print(layer_input3)
    lrn = tf.nn.local_response_normalization(layer_input3)
    print(lrn)
    print(sess.run([layer_input3, lrn]))
    image_input = tf.constant([
        [
            [[0., 0., 0.], [255., 255., 255.], [254., 0., 0.]],
            [[0., 191., 0.], [3., 108., 233.], [0., 191., 0.]],
            [[254., 0., 0.], [255., 255., 255.], [0., 0., 0.]]
        ]
    ])
    print(image_input)
    conv2d = tf.contrib.layers.convolution2d(
        image_input,
        num_outputs=4,
        kernel_size=(1, 1),
        activation_fn=tf.nn.relu,
        stride=(1, 1),
        trainable=True)
    print(conv2d)
    sess.run(tf.global_variables_initializer())
    print(sess.run(conv2d))

    features4 = tf.constant([
        [[1.2], [3.4]]
    ])
    print(features4)
    fc = tf.contrib.layers.fully_connected(features4, num_outputs=1)
    print(fc)
    sess.run(tf.global_variables_initializer())
    print(sess.run(fc))


def test_reshape():
    a = [[1, 2, 3], [4, 5, 6]]
    print(a)
    b = np.reshape(a[0], (-1, 1))
    print(b)
    c = np.reshape(b, (1, -1))[0]
    print(c)

    a[1] = c
    a[0] = [0, 0, 0]

    print(a)

    a = np.transpose(a)
    print(a)


def test_df_map():
    frame = pd.DataFrame(np.random.randn(4, 3), columns=list('abc'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    print(frame)

    f = lambda x:  if

    frame = frame.apply(f,axis=1)

    print(frame)

test_df_map()