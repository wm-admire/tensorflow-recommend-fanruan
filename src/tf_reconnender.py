# encoding: utf-8
# author: yaoh.wu

import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tfdbg

from src import utils


def load_data():
    history_title = ["id", "tname", "type", "ip", "username", "userrole", "time", "logtime", "memory"]
    # 从csv 中读取数据
    histories = pd.read_csv(filepath_or_buffer="../data/fr_data.csv", sep=",", header=0, usecols=history_title,
                            encoding="utf-8", engine="python")
    # 只要tname type username
    histories = histories.filter(regex='tname|type|username')
    histories_orig = histories.values
    user_map = {username: ii for ii, username in enumerate(set(histories["username"]))}
    users = {ii: username for ii, username in enumerate(set(histories["username"]))}
    tname_map = {tname: ii for ii, tname in enumerate(set(histories["tname"]))}
    tnames = {ii: tname for ii, tname in enumerate(set(histories["tname"]))}

    histories["username"] = histories["username"].map(user_map)
    histories["tname"] = histories["tname"].map(tname_map)

    # 调整后的历史数据
    print("调整后的历史数据：")
    print(histories.head())
    #    tname   type   username
    # 0   1104     2       557
    # 1    969     2       107
    # 2    498     2       557
    # 3    558     1       107
    # 4    762     1       107

    # 整理成浏览次数记录的数据，暨 某模板被某用户预览的次数
    ratings_table, tnames_features, users_features = __evaluate__(users, tnames, histories)

    # todo 使用对应的times 进行降维作为其对应的特征

    # tname 特征
    tnames_table = pd.DataFrame([[i, tnames_features[i]] for i in range(len(tnames_features))],
                                columns=["tname", "tname_feature"])
    users_table = pd.DataFrame([[i, users_features[i]] for i in range(len(users_features))],
                               columns=["user", "user_features"])

    # 预览次数列表
    ratings_table = pd.DataFrame(ratings_table, columns=["tname", "user", "rating"])

    # 某模板被某用户预览的次数
    print("某模板被某用户预览的次数：")
    print(ratings_table.head())
    #    tname  user  rating
    # 0   1104   557      35
    # 1   1104   709       5
    # 2   1104   509       6
    # 3    969   107       1
    # 4    969   768      11

    # 合并得到的数据
    data = pd.merge(pd.merge(tnames_table, ratings_table, on="tname"), users_table, on="user")
    print("合并得到的数据")
    print(data.columns)

    target_fields = ['rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets = targets_pd.values

    return histories, histories_orig, users, users_table, users_features, tnames, tnames_table, tnames_features, data, features, targets


def __evaluate__(users, tnames, histories):
    """
    计算出用户对模板的预览次数
    :param users:
    :param tnames:
    :param histories:
    :return:
    """
    # 预览次数 {key:tname value:{key:user value: times}}
    times = {}
    for index in histories.index:
        item = histories.loc[index]
        tname = item[0]
        username = item[2]

        if tname in times:
            if username in times[tname]:
                times[tname][username] += 1
            else:
                times[tname][username] = 1
        else:
            times[tname] = {}
            times[tname][username] = 1
        if index > 1000:
            break

    # 预览次数 row=tname，col=user
    tname_user_times = []
    for t in tnames.keys():
        t_list = []
        if t in times:
            item = times[t]
            for user in users.keys():
                if user in item:
                    t_list.append(item[user])
                else:
                    t_list.append(0.0)

        else:
            t_list = [0.0] * len(users)

        tname_user_times.append(t_list)

    tname_user_times = np.array(tname_user_times)

    # 矩阵转置 从 793（用户）*1262（模板）转换成 1262（模板）*793（用户）
    # 预览次数 row=user,col=tname
    user_tname_times = np.transpose(tname_user_times)

    # 次数列表
    # 预览次数 [[tname,user,times]]
    times_list = []
    for t in times.keys():
        t_item = times[t]
        for u in t_item.keys():
            u_item = t_item[u]
            times_list.append([t, u, u_item])

    return times_list, tname_user_times, user_tname_times


def get_inputs():
    uid_holder = tf.placeholder(tf.int32, [None, 1], name="uid")
    tid_holder = tf.placeholder(tf.int32, [None, 1], name="tid")
    target_holder = tf.placeholder(tf.int32, [None, 1], name="target")
    learning_rate_holder = tf.placeholder(tf.float32, name="learning-rate")
    dropout_keep_holder = tf.placeholder(tf.float32, name="dropout-keep")
    return uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder


def get_user_embedding(uid, user_features):
    with tf.name_scope("user-embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1, dtype="float64"), name="uid-embed-matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid-embed-layer")

        user_features_embed_matrix = tf.Variable(user_features, name="user-features-matrix")
        user_features_embed_layer = tf.nn.embedding_lookup(user_features_embed_matrix, uid, name="user-features-layer")

    return uid_embed_layer, user_features_embed_layer


def get_tname_embedding(tid, tname_features):
    with tf.name_scope("tname-embedding"):
        tid_embed_matrix = tf.Variable(tf.random_uniform([tid_max, embed_dim], -1, 1, dtype="float64"), name="tname-embed-matrix")
        tid_embed_layer = tf.nn.embedding_lookup(tid_embed_matrix, tid, name="tid-embed-layer")

        tname_features_embed_matrix = tf.Variable(tname_features, name="tname-features-matrix")
        tname_features_embed_layer = tf.nn.embedding_lookup(tname_features_embed_matrix, tid, name="tname-features-layer")

    return tid_embed_layer, tname_features_embed_layer


def get_user_feature_layer(uid_embed_layer, user_features_embed_layer):
    with tf.name_scope("user-fc"):
        # 第一层全连接
        uid_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="user-layer", activation=tf.nn.relu)
        uid_feature_layer = tf.layers.dense(user_features_embed_layer, u_feature_max, name="user-feature-layer", activation=tf.nn.relu)
        # 第二层全连接
        user_combined_layer = tf.concat([uid_layer, uid_feature_layer], 2)
        user_combined_layer = tf.contrib.layers.fully_connected(user_combined_layer, 1500, tf.tanh)

        user_combined_layer_flat = tf.reshape(user_combined_layer, [-1, 1500], name="user-combined-layer-flat")
    return user_combined_layer, user_combined_layer_flat


def get_tname_feature_layer(tid_embed_layer, tname_features_embed_layer):
    with tf.name_scope("tname-fc"):
        # 第一层全连接
        tid_layer = tf.layers.dense(tid_embed_layer, embed_dim, name="tname-layer", activation=tf.nn.relu)
        tid_feature_layer = tf.layers.dense(tname_features_embed_layer, t_feature_max, name="tname-feature-layer", activation=tf.nn.relu)
        # 第二层全连接
        tname_combined_layer = tf.concat([tid_layer, tid_feature_layer], 2)
        tname_combined_layer = tf.contrib.layers.fully_connected(tname_combined_layer, 1500, tf.tanh)

        tname_combined_layer_flat = tf.reshape(tname_combined_layer, [-1, 1500], name="tname-combined-layer-flat")
    return tname_combined_layer, tname_combined_layer_flat


def create_default_graph(user_features, tname_features):
    tf.reset_default_graph()
    train_graph = tf.Graph()
    with train_graph.as_default():
        uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder = get_inputs()

        uid_embed_layer, user_features_embed_layer = get_user_embedding(uid_holder, user_features)
        user_combined_layer, user_combined_layer_flat = get_user_feature_layer(uid_embed_layer, user_features_embed_layer)

        tid_embed_layer, tname_features_embed_layer = get_tname_embedding(tid_holder, tname_features)
        tname_combined_layer, tname_combined_layer_flat = get_tname_feature_layer(tid_embed_layer, tname_features_embed_layer)

        with tf.name_scope("inference"):
            inference = tf.reduce_sum(user_combined_layer_flat * tname_combined_layer_flat, axis=1)
            inference = tf.expand_dims(inference, axis=1)

        with tf.name_scope("loss"):
            # MSE损失，将计算值回归到评分
            cost = tf.losses.mean_squared_error(targets, inference)
            loss = tf.reduce_mean(cost)

        global_step = tf.Variable(0, name="global-step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate_holder)
        gradients = optimizer.compute_gradients(loss)  # cost
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    return train_graph, loss, global_step, gradients, train_op, uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder


def train(graph, loss, global_step, gradients, train_op, uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder):
    losses = {"train": [], "test": []}

    with tf.Session(graph=graph) as sess:
        # debug
        sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # 搜集数据给tensorBoard用
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Inference summaries
        inference_summary_op = tf.summary.merge([loss_summary])
        inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
        inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(num_epochs):

            # 将数据集分成训练集和测试集，随机种子不固定
            train_x, test_x, train_y, test_y = train_test_split(features,
                                                                targets,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = utils.get_batch(train_x, train_y, batch_size)
            test_batches = utils.get_batch(test_x, test_y, batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_x) // batch_size):
                x, y = next(train_batches)
                feed = {
                    uid_holder: np.reshape(x.take(2, 1), [batch_size, 1]),
                    tid_holder: np.reshape(x.take(0, 1), [batch_size, 1]),
                    target_holder: np.reshape(y, [batch_size, 1]),
                    dropout_keep_holder: dropout_keep,  # dropout_keep
                    learning_rate_holder: learning_rate
                }

                step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
                losses['train'].append(train_loss)
                train_summary_writer.add_summary(summaries, step)  #

                # Show every <show_every_n_batches> batches
                if (epoch_i * (len(train_x) // batch_size) + batch_i) % show_every_n_batches == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        batch_i,
                        (len(train_x) // batch_size),
                        train_loss))

            # 使用测试数据的迭代
            for batch_i in range(len(test_x) // batch_size):
                x, y = next(test_batches)

                feed = {
                    uid_holder: np.reshape(x.take(2, 1), [batch_size, 1]),
                    tid_holder: np.reshape(x.take(0, 1), [batch_size, 1]),
                    target_holder: np.reshape(y, [batch_size, 1]),
                    dropout_keep_holder: dropout_keep,  # dropout_keep
                    learning_rate_holder: learning_rate
                }

                step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

                # 保存测试损失
                losses['test'].append(test_loss)
                inference_summary_writer.add_summary(summaries, step)  #

                time_str = datetime.datetime.now().isoformat()
                if (epoch_i * (len(test_x) // batch_size) + batch_i) % show_every_n_batches == 0:
                    print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        batch_i,
                        (len(test_x) // batch_size),
                        test_loss))

        # Save Model
        saver.save(sess, save_dir)  # , global_step=epoch_i
        print('Model Trained and Saved')

    # 显示训练Loss
    plt.plot(losses['train'], label='Training loss')
    plt.legend()
    #  显示测试Loss
    plt.plot(losses['test'], label='Test loss')
    plt.legend()
    # 迭代次数再增加一些，下降的趋势会明显一些
    plt.show()


if __name__ == '__main__':
    # 读取数据
    histories, histories_orig, users, users_table, users_features, tnames, tnames_table, tnames_features, data, features, targets = load_data()
    # 数据存储到本地
    utils.save_params((histories, histories_orig, users, users_table, users_features, tnames, tnames_table, tnames_features, data, features, targets))
    # 从本地读取数据
    histories, histories_orig, users, users_table, users_features, tnames, tnames_table, tnames_features, data, features, targets = utils.load_params()

    # 嵌入矩阵的维度
    embed_dim = 32
    # 用户id个数
    # uid_max = max(features.take(2, 1)) + 1
    uid_max = len(users)
    # tname id 个数
    # tid_max = max(features.take(0, 1)) + 1
    tid_max = len(tnames)

    u_feature_max = tid_max

    t_feature_max = uid_max

    # 超参
    # number of epochs
    num_epochs = 5
    # batch size
    batch_size = 256

    dropout_keep = 0.5
    # learning rate
    learning_rate = 0.0001
    # show stats for every n number of batches
    show_every_n_batches = 20
    # save path
    save_dir = './save'

    graph, loss, global_step, gradients, train_op, uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder = create_default_graph(
        users_features, tnames_features)

    train(graph, loss, global_step, gradients, train_op, uid_holder, tid_holder, target_holder, learning_rate_holder, dropout_keep_holder)
