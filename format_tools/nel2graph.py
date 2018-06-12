# -*- coding:utf-8 -*-
##########
# This script can transfer the graph objects from format of .NEL to .graph, which can be used in NgramCNN
# In default the generated .graph file will be stored at kdd_datasets/
# Author: Bruce Luo
# Date: 2018/06/11
##########
import argparse
import numpy as np
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", '-d', default='mutag.nel', type=str, help='NEL data file to convert')
    parser.add_argument("--outputdir", '-o', default='kdd_datasets', type=str, help='the output directory')
    return parser.parse_args()


def NEL2Graph(nel_str):
    data = nel_str.split('\n\n')[:-1]
    graphs = []
    graph_labels = []
    for i in range(len(data)):
        graph = data[i].split('\n')
        node_labels = []
        flag = 0
        for line in graph:
            line = line.split(' ')
            if line[0] == 'n':
                node_labels.append(int(line[2]))
            elif line[0] == 'e':
                if flag == 0:
                    edge_labels = np.zeros((len(node_labels), len(node_labels)), dtype=np.int32)
                    flag = 1
                node1 = int(line[1]) - 1
                node2 = int(line[2]) - 1
                edge_labels[node1][node2] = int(line[3])

            elif line[0] == 'x':
                graph_label = eval(line[1])
                graph_labels.append(graph_label)
            # elif line[0] == 'g':
            #     graph_id = int(line[1])
        node_num = len(node_labels)
        # transform to .graph
        nodes = []
        for node_id in range(node_num):
            label = [node_labels[node_id]]
            neighbors = []
            for neighbor in range(node_num):
                if edge_labels[node_id][neighbor] != 0:
                    neighbors.append(neighbor)
                    label.append(edge_labels[node_id][neighbor])
            nodes.append({'neighbors': neighbors, 'label': tuple(label)})

        graphs.append(dict(zip(range(node_num), nodes)))

    g = dict(zip(range(len(graphs)), graphs))
    result = {'graph': g, 'labels': graph_labels}
    return result


if __name__ == '__main__':
    args = parse_args()
    nel_str = open(args.datafile, 'r').read()
    result = NEL2Graph(nel_str)
    dataset_name = args.datafile.split(".")[0]
    with open("../" + args.outputdir + "/" + dataset_name.lower() + '.graph', 'w') as f:
        pickle.dump(result, f)
