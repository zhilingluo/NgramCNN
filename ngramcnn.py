# -*- coding:utf-8 -*-
##########
# This script trains NgramCNN
# Author: Bruce Luo
# Date: 2017/12/20
##########

import pickle
import numpy as np
import sys
import time
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ngramcnn_utils import go, list_filter, load_data, weight_variable, bias_variable, preparenel_with_vali, \
    convertoutput
from ngramcnn_utils import load_graph_simple, exchangemax_simple

duplicate_limit = 0
max_node_limit = 200
cpu_parallel_num = 8

repeat_times = 1  # default is 10

train_percentage = 0.6
vali_percentage = 0.1
stop_acc_gap = 0.003
mult_k_num = (1, 2, 4, 8)
use_sigmoid = False
learning_rate = 1e-3


def load_data_to_list(graph_data):
    max_node_known = 0
    net_list = []
    pid = os.getpid()
    for i in range(len(graph_data)):
        if graph_data[i] is None:
            continue
        id, graph = graph_data[i]
        # print "pid {} doing No.{} graph, total {} graph".format(pid, i + 1, len(graph_data))
        # print "doing %d" % (id)
        A = load_graph_simple(graph, max_node_limit)
        node_num = len(A)
        edge_num = sum(sum(A)) / 2
        degree = edge_num * 1.0 / node_num
        if node_num > max_node_known:
            max_node_known = node_num
        A, others = exchangemax_simple(A, kernel_width, node_num, duplicate_limit)
        net_list.append((A, lbs[id], others, max_node_known, max_label_known, node_num, edge_num, degree))

    return net_list


if __name__ == "__main__":

    ds_name = sys.argv[1]  # dataset name
    operation = int(sys.argv[2])  # 1 means preparing , 2 means training
    kernel_width = int(sys.argv[3])  # kernel width

    print "Dataset: %s\nOperation: %s\nParameters:\nKernel width: %s" % (ds_name, operation, kernel_width)

    output_dir = "bufferdata"
    data_dir = "kdd_datasets"

    if operation == 1:

        max_label_known = 0
        max_node_known = 0

        start = time.time()
        graph_data, lbs = load_data(ds_name, data_dir)
        graph_data_list = []
        for i in range(len(graph_data)):
            graph_data_list.append((i, graph_data[i]))

        net_list = []
        other_list = []
        other_list_num = 0

        graph_data_array = np.asarray(graph_data_list)

        net_all_list = go(load_data_to_list, graph_data_array, cpu_parallel_num)
        # net_all_list = load_data_to_list(graph_data_array)
        print "graph num is %d" % (len(net_all_list))
        all_degree = 0
        max_edge_num = 0
        min_edge_num = 10000
        total_edge_num = 0
        max_node_num = 0
        min_node_num = 10000
        total_node_num = 0
        max_degree = 0
        min_degree = 10000
        total_degree = 0
        for A, label, others, max_node, max_label, node_num, edge_num, degree in net_all_list:
            all_degree += degree
            net_list.append((A, label, others))
            other_list_num += len(others)
            if max_node > max_node_known:
                max_node_known = max_node
            if label > max_label_known:
                max_label_known = label
            if node_num > max_node_num:
                max_node_num = node_num
            if node_num < min_node_num:
                min_node_num = node_num
            total_node_num += node_num
            if edge_num > max_edge_num:
                max_edge_num = edge_num
            if edge_num < min_edge_num:
                min_edge_num = edge_num
            total_edge_num += edge_num
            if degree > max_degree:
                max_degree = degree
            if degree < min_degree:
                min_degree = degree
            total_degree += degree

        max_label_known += 1
        end = time.time()

        print "max node: %d" % max_node_known
        print "max label: %d" % max_label_known
        print "max degree: %s" % max_degree
        print "min degree: %s" % min_degree
        print "avg degree: %s" % (total_degree / (len(net_list)))
        print "max node: %d" % max_node_num
        print "min node: %d" % min_node_num
        print "avg node: %d" % (total_node_num / len(net_list))
        print "max edge: %d" % max_edge_num
        print "min edge: %d" % min_edge_num
        print "avg edge: %d" % (total_edge_num / len(net_list))
        print "org graph number %d" % (len(net_list))
        print "extend graph number %d" % (len(other_list))
        print "start writing disk"
        with open(output_dir + "/%s_kwidth_%s" % (ds_name, kernel_width), 'w') as f:
            pickle.dump((net_list, max_node_known, max_label_known), f)
        print "done"

    elif operation == 2:

        with open(output_dir + "/%s_kwidth_%s" % (ds_name, kernel_width), 'r') as f:
            adj_list, max_node_known, max_label_known = pickle.load(f)
        print "load %d" % (len(adj_list))
        batch_size = int(sys.argv[4])
        kernel_num = int(sys.argv[5])
        kernel_size_other = int(sys.argv[6])
        kernel_num_other = int(sys.argv[7])
        epoch_num = int(sys.argv[8])
        dropout_ratio = float(sys.argv[9])
        altimes = repeat_times
        alres = []
        alval = []
        altim = []

        print "Start Train with Parameters:\nKernel width: %s\nKernel num: %s\nOther Kernel size: %s\nOther Kernel num: %s\nEpoch num: %s\nBatch size: %s\nDropout ratio: %s\n" % (
            kernel_width, kernel_num, kernel_size_other, kernel_num_other, epoch_num, batch_size, dropout_ratio)

        channel = 1
        node_num = max_node_known

        sys.stdout.flush()
        kernel_size = (kernel_width, mult_k_num[0] * kernel_size_other, mult_k_num[1] * kernel_size_other,
                       mult_k_num[2] * kernel_size_other, mult_k_num[3] * kernel_size_other)
        kernel_num_list = (kernel_num, kernel_num_other, kernel_num_other, kernel_num_other, kernel_num_other)

        class_num = int(max_label_known)
        type_num = 1
        # print type_num
        # print class_num
        # print kernel_num_list
        # print kernel_size

        # model here
        x = tf.placeholder(tf.float32, [None, kernel_size[0], kernel_size[0] * (node_num - kernel_size[0] + 1), 1])
        w_conv1 = weight_variable([kernel_size[0], kernel_size[0], type_num, kernel_num_list[0]])
        b_conv1 = bias_variable([kernel_num_list[0]])
        w_conv2 = weight_variable([kernel_size[1], kernel_size[1], kernel_num_list[0], kernel_num_list[1]])
        b_conv2 = bias_variable([kernel_num_list[1]])
        w_conv3 = weight_variable([kernel_size[2], kernel_size[2], kernel_num_list[1], kernel_num_list[2]])
        b_conv3 = bias_variable([kernel_num_list[2]])
        w_conv4 = weight_variable([kernel_size[3], kernel_size[3], kernel_num_list[2], kernel_num_list[3]])
        b_conv4 = bias_variable([kernel_num_list[3]])
        w_conv5 = weight_variable([kernel_size[4], kernel_size[4], kernel_num_list[3], kernel_num_list[4]])
        b_conv5 = bias_variable([kernel_num_list[4]])
        w_fc1 = weight_variable([kernel_num_list[4], kernel_num_list[4]])
        b_fc1 = bias_variable([kernel_num_list[4]])
        w_fc2 = weight_variable([kernel_num_list[4], class_num])
        b_fc2 = bias_variable([class_num])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, kernel_size[0], 1], padding='SAME') + b_conv1)
        n_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        n_pool1_mean, n_pool1_var = tf.nn.moments(n_pool1, [0])
        n_pool1 = tf.nn.batch_normalization(n_pool1, n_pool1_mean, n_pool1_var, None, None, 0.01)
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(n_pool1, w_conv2, strides=[1, 1, kernel_size[0], 1], padding='SAME') + b_conv2)
        n_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        n_pool2_mean, n_pool2_var = tf.nn.moments(n_pool2, [0])
        n_pool2 = tf.nn.batch_normalization(n_pool2, n_pool2_mean, n_pool2_var, None, None, 0.01)
        h_conv3 = tf.nn.relu(
            tf.nn.conv2d(n_pool2, w_conv3, strides=[1, 1, kernel_size[0], 1], padding='SAME') + b_conv3)
        n_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        n_pool3_mean, n_pool3_var = tf.nn.moments(n_pool3, [0])
        n_pool3 = tf.nn.batch_normalization(n_pool3, n_pool3_mean, n_pool3_var, None, None, 0.01)
        h_conv4 = tf.nn.relu(
            tf.nn.conv2d(n_pool3, w_conv4, strides=[1, 1, kernel_size[0], 1], padding='SAME') + b_conv4)
        n_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        n_pool4_mean, n_pool4_var = tf.nn.moments(n_pool4, [0])
        n_pool4 = tf.nn.batch_normalization(n_pool4, n_pool4_mean, n_pool4_var, None, None, 0.01)
        h_conv5 = tf.nn.relu(
            tf.nn.conv2d(n_pool4, w_conv5, strides=[1, 1, kernel_size[0], 1], padding='SAME') + b_conv5)
        n_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        n_pool5_flat = tf.reshape(n_pool5, [-1, kernel_num_list[4]])
        h_fc1 = tf.nn.sigmoid(tf.matmul(n_pool5_flat, w_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)  # using drop out to forbid overfitting
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        y_ = tf.placeholder(tf.float32, [None, class_num])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  # 损失函数，交叉熵
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        # model end

        for altime in range(altimes):
            sample_num = len(adj_list)
            train = int(sample_num * train_percentage)
            vali = int(sample_num * (train_percentage + vali_percentage))
            input_list, output_list, input_vali_list, output_vali_list, input_test_list, output_test_list = preparenel_with_vali(
                adj_list, kernel_width, node_num, train, vali)

            lens = len(input_list)
            input_list = np.reshape(input_list,
                                    [-1, kernel_size[0], kernel_size[0] * (node_num - kernel_size[0] + 1), type_num])
            output_list = convertoutput(output_list, class_num)
            input_vali_list = np.reshape(input_vali_list,
                                         [-1, kernel_size[0], kernel_size[0] * (node_num - kernel_size[0] + 1),
                                          type_num])
            output_vali_list = convertoutput(output_vali_list, class_num)
            input_test_list = np.reshape(input_test_list,
                                         [-1, kernel_size[0], kernel_size[0] * (node_num - kernel_size[0] + 1),
                                          type_num])
            output_test_list = convertoutput(output_test_list, class_num)

            print "%d to learn" % lens

            start = time.time()
            mean_acc = 0
            last_acc = 0
            last_loss = 100000
            last_vali_acc = 0
            loss_list = []
            acc_list = []
            acc_vali_list = []
            loss_vali_list = []
            sys.stdout.flush()

            merged = tf.summary.merge_all()
            sess = tf.InteractiveSession()
            train_writer = tf.summary.FileWriter('logs/summarize_train_logs/%s' % (altime), sess.graph)
            test_writer = tf.summary.FileWriter('logs/summarize_test_logs')

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            knownbestacc = 0
            for epoch in range(epoch_num):

                sum_loss = 0
                # print('epoch %d' % epoch)

                count = 0
                sum_acc = 0
                indexes = np.random.permutation(lens)

                for i in range(0, lens, batch_size):
                    input_batch = input_list[i:i + batch_size]
                    output_batch = output_list[i:i + batch_size]

                    real_batch_size = len(input_batch)
                    count += real_batch_size

                    train_accuracy = accuracy.eval(feed_dict={x: input_batch, y_: output_batch, keep_prob: 1.0})
                    train_loss = cross_entropy.eval(feed_dict={x: input_batch, y_: output_batch, keep_prob: 1.0})

                    train_step.run(feed_dict={x: input_batch, y_: output_batch, keep_prob: dropout_ratio})
                    acc = train_accuracy
                    loss = train_loss
                    sum_loss += loss * real_batch_size
                    sum_acc += acc * real_batch_size
                if count > 0:
                    mean_loss = sum_loss / count
                    mean_acc = sum_acc / count

                    loss_list.append(mean_loss.tolist())
                    acc_list.append(mean_acc.tolist())
                    # Validation

                    [summary_str, acc_vali, loss_vali] = sess.run([merged, accuracy, cross_entropy],
                                                                  feed_dict={x: input_vali_list, y_: output_vali_list,
                                                                             keep_prob: 1.0})
                    train_writer.add_summary(summary_str, epoch)

                    acc_vali_list.append(acc_vali.tolist())
                    loss_vali_list.append(loss_vali.tolist())
                    # validation Finished

                    if knownbestacc < acc_vali:
                        knownbestacc = acc_vali
                        saver.save(sess, output_dir + "/%s_kwidth_%s_Kn_%s_Kw2_%s_Kn2_%sEn_%s_Bs_%s_Dr_%s" % (
                            ds_name, kernel_width, kernel_num, kernel_size_other, kernel_num_other, epoch_num,
                            batch_size,
                            dropout_ratio))
                    sys.stdout.flush()
                    if abs(mean_acc - last_acc) < 0.001 and False:
                        epoch = epoch_num
                        conveva = last_conv
                        break
                    else:
                        last_loss = mean_loss
                        last_acc = mean_acc
                        last_vali_acc = acc_vali

            end = time.time()
            alval.append(acc_vali)
            print "vali loss: %f" % loss_vali
            print "vali acc: %f" % acc_vali
            print "Trained finished on Acc: %s, Time: %s" % (mean_acc, end - start)
            # saver.save(sess, output_dir + "/%s_kwidth_%s_Kn_%s_Kw2_%s_Kn2_%sEn_%s_Bs_%s_Dr_%s" % (
            #    ds_name, kernel_width, kernel_num, kernel_size_other, kernel_num_other, epoch_num, batch_size,
            #    dropout_ratio))

            print "start to test"
            saver.restore(sess, output_dir + "/%s_kwidth_%s_Kn_%s_Kw2_%s_Kn2_%sEn_%s_Bs_%s_Dr_%s" % (
                ds_name, kernel_width, kernel_num, kernel_size_other, kernel_num_other, epoch_num, batch_size,
                dropout_ratio))

            input_list = input_test_list
            output_list = output_test_list
            lens = len(input_list)

            indexes = np.random.permutation(lens)
            count = 0
            sum_acc = 0
            start = time.time()
            mean_acc = 0
            sum_time = 0
            for i in range(0, lens, batch_size):
                input_batch = input_list[indexes[i: i + batch_size]]
                output_batch = output_list[indexes[i: i + batch_size]]

                real_batch_size = len(input_batch.data)

                count += real_batch_size
                ne_timer = time.time()
                acc_test = accuracy.eval(feed_dict={x: input_batch, y_: output_batch, keep_prob: 1.0})
                loss_test = cross_entropy.eval(feed_dict={x: input_batch, y_: output_batch, keep_prob: 1.0})
                ne_timer2 = time.time()
                sum_time += ne_timer2 - ne_timer
                loss = loss_test
                acc = acc_test
                sum_acc += acc * real_batch_size

            if count > 0:
                mean_acc = sum_acc / count
                mean_acc = mean_acc.tolist()
            end = time.time()
            tim = end - start
            altim.append(tim)
            print "Tested finished on Acc: %s, Time: %s" % (mean_acc, tim)
            if mean_acc != 0:
                alres.append(mean_acc)
        if len(alres) == 0:
            alres.append(0)
        print alres
        alres = list_filter(alres)
        mean_acc = np.mean(alres)
        std_acc = np.std(alres)
        altim = list_filter(altim)
        mean_tim = np.mean(altim)
        print "Average Acc is %s, Std is %s, Avg Time is %s.\n" % (mean_acc, std_acc, mean_tim)
        print "done"
