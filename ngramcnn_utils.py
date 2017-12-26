# -*- coding:utf-8 -*-
##########
# This script provides some util functions in NgramCNN
# Author: Bruce Luo
# Date: 2017/12/20
##########

import numpy as np
import tensorflow as tf
import math
import pickle

from itertools_recipes import grouper
import concurrent.futures


def go(func, data, K):  # func is the function, returns res, data is the data in nparray, K is the multi thread number
    nk = np.ceil(float(len(data)) / K)
    nk = np.uint64(nk)
    executor = concurrent.futures.ProcessPoolExecutor(K)
    futures = [executor.submit(func, group)
               for group in grouper(nk, data)]
    concurrent.futures.wait(futures)
    result = []
    for fu in futures:
        try:
            res = fu.result()
            result = result + res
        except:
            continue
    return result


def list_filter(lis):
    lis = np.sort(lis).tolist()
    # lis = lis[1:-1]#for debug
    return lis


def convert2tensor(input_list, dtype=tf.float32):
    all_list = []
    for i in range(len(input_list)):
        all_list.append(tf.convert_to_tensor(input_list[i], dtype))
    return tf.convert_to_tensor(all_list, dtype)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convertoutput(output_list, class_num):
    out = []
    for i in range(len(output_list)):
        res = np.zeros(class_num)
        res[output_list[i]] = 1
        out.append(res)

    return np.asarray(out)


def convertA(A, kernel_width):
    node_num = len(A)
    x = np.zeros((kernel_width, kernel_width * (node_num - kernel_width + 1)), dtype=np.float32)
    for i in range(node_num - kernel_width + 1):
        x[:, (i * kernel_width):((i + 1) * kernel_width)] = A[i:(i + kernel_width), i:(i + kernel_width)]
    return x


def convertA_extend(A, kernel_width, max_node_num):
    input = []

    # extract first, then extend
    if len(A) < kernel_width:
        A = extend(A, kernel_width)
    x = convertA(A, kernel_width)
    temp = np.zeros((kernel_width, kernel_width * (max_node_num - kernel_width + 1)), dtype=np.float32)
    temp[:, : x.shape[1]] = x
    input.append(temp)

    return input


def preparenel_with_vali(adj_list, kernel_width=15, max_node_num=110, cut_line1=200, cut_line2=400):
    input_train_list = []
    output_train_list = []
    input_vali_list = []
    output_vali_list = []
    input_test_list = []
    output_test_list = []
    count = 0
    indexes = np.random.permutation(len(adj_list))

    for adj_i in range(len(adj_list)):
        A, x, others = adj_list[indexes[adj_i]]
        node_num = len(A)

        output = int(x)
        input = convertA_extend(A, kernel_width, max_node_num)

        input = np.asarray(input, dtype=np.float32)
        output = np.asarray(output, dtype=np.int32)
        if adj_i < cut_line1:
            input_train_list.append(input)
            output_train_list.append(output)
            for other in others:
                otherA = other
                input = convertA_extend(otherA, kernel_width, max_node_num)
                input = np.asarray(input, dtype=np.float32)
                output = np.asarray(output, dtype=np.int32)
                input_train_list.append(input)
                output_train_list.append(output)
        elif adj_i >= cut_line1 and adj_i < cut_line2:
            input_vali_list.append(input)
            output_vali_list.append(output)
            for other in others:
                otherA = other
                input = convertA_extend(otherA, kernel_width, max_node_num)
                input = np.asarray(input, dtype=np.float32)
                output = np.asarray(output, dtype=np.int32)
                input_vali_list.append(input)
                output_vali_list.append(output)
        else:
            input_test_list.append(input)
            output_test_list.append(output)
            for other in others:
                otherA = other
                input = convertA_extend(otherA, kernel_width, max_node_num)
                input = np.asarray(input, dtype=np.float32)
                output = np.asarray(output, dtype=np.int32)
                input_test_list.append(input)
                output_test_list.append(output)
    print "get train data %d" % (len(input_train_list))
    print "get validation data: %d" % (len(input_vali_list))
    print "get test data %d" % (len(input_test_list))
    return (input_train_list, output_train_list, input_vali_list, output_vali_list, input_test_list, output_test_list)


def extend(a, max_node_num):
    node_num = len(a)
    right_flag = np.zeros((node_num, max_node_num - node_num))
    a = np.concatenate((a, right_flag), axis=1)
    bottom_flag = np.zeros((max_node_num - node_num, max_node_num))
    a = np.concatenate((a, bottom_flag), axis=0)
    return a


def load_data(ds_name, data_dir):
    f = open(data_dir + "/%s.graph" % ds_name, "r")
    data = pickle.load(f)
    graph_data = data["graph"]
    labels = data["labels"]
    if len(labels) == 1:
        labels = labels[0]
    lbs = np.array(labels, dtype=np.float)
    all_lbs = np.unique(lbs)
    counter = 0
    lbs2 = np.zeros((len(lbs)))
    for al_lbs in all_lbs:
        lbs2[np.argwhere(lbs == al_lbs)] = counter
        counter += 1
    return graph_data, lbs2


def load_graph_simple(nodes, max_node_limit):
    node_list = {}
    counter = 0
    for nidx in nodes:
        node_list[nidx] = counter
        counter += 1
        if counter >= max_node_limit:
            break
    size = counter
    A = np.zeros((size, size), dtype=np.float32)  # A is the adjacency matrix
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            if (node_list.has_key(nidx)) and (node_list.has_key(neighbor)):
                source = node_list[nidx]
                to = node_list[neighbor]
                A[source, to] = 1
    return A


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def evaluate(A, kernel_width, node_num):
    loss = 0
    lis = np.argwhere(A == 1)
    for li in lis:
        dis = abs(li[1] - li[0])
        if dis > kernel_width:
            loss += 10000
        else:
            loss += 1
    return loss


def exchange(A, i, j, node_num):
    index = range(node_num)
    index[j] = i
    index[i] = j
    new_A = np.transpose(np.transpose(A[index])[index])
    return (new_A, index)


def exchangemax_simple(A, kernel_width, node_num, duplicate_limit):
    least_loss = 1000000

    loss = least_loss
    count = 0
    while loss != 0 and count < 1:
        count += 1
        for i in range(node_num):
            for j in range(i + 1, node_num):
                new_A, index = exchange(A, i, j, node_num)
                loss = evaluate(new_A, kernel_width, node_num)
                if loss < least_loss:  # this is more optimal
                    least_loss = loss
                    count = 0
                    A = new_A

                    j = node_num + 1
                    i = node_num + 1
                    break
    others = []

    counter = 0
    for i in range(node_num):
        if counter == duplicate_limit:
            break
        for j in range(i + 1, node_num):
            if counter == duplicate_limit:
                break
            new_A, index = exchange(A, i, j, node_num)

            loss = evaluate(new_A, kernel_width, node_num)
            if loss == least_loss:
                other_A = new_A
                others.append(other_A)
                counter += 1

    # loss = evalute(A, kernel_width, node_num)
    # print "loss is reduced from %d to %d"%(start_loss,loss)
    if len(others) > duplicate_limit:
        others = others[0:duplicate_limit]
    return A, others
