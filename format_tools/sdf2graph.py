# -*- coding:utf-8 -*-
##########
# This script can transfer the graph objects from format of .SDF to .graph, which can be used in NgramCNN
# In default the generated .graph file will be stored at kdd_datasets/
# Author: Bruce Luo
# Date: 2018/06/11
##########

import argparse
import pickle
from nel2graph import NEL2Graph


def SDFtoNEL(sdf_str):
    full_str = sdf_str
    full_str = full_str.replace("\r", "")
    full_str = full_str.replace("  ", " ")
    full_str = full_str.replace("  ", " ")
    full_str = "\n" + full_str
    full_str_list = full_str.split("$$$$")
    full_str_list = full_str_list[:-1]
    new_lines = ""
    for graph_str in full_str_list:
        line_list = graph_str.split("\n")
        a = 0
        start = False
        nexter = False
        for line in line_list:
            if len(line) > 6 and line[-5] == 'V' and not start:
                seg_list = line.split(" ")
                node_num = int(seg_list[1])
                if node_num > 1000:
                    a = 1
                edge_num = int(seg_list[2])
                counter = 0
                start = True
                continue
            if start:
                if len(line.split(" ")) == 11:
                    counter += 1
                    seg_list = line.split(" ")
                    new_lines += "n " + str(counter) + " " + seg_list[4] + "\n"
                elif len(line.split(" ")) == 7:
                    counter += 1
                    seg_list = line.split(" ")
                    new_lines += "e " + seg_list[1] + " " + seg_list[2] + " " + seg_list[3] + "\n"
                elif len(line) > 0:
                    if line[0] == ">":
                        nexter = True
                    elif nexter:
                        new_lines += "x " + line[0:] + "\n"
                        break
        new_lines += "\n"
    return new_lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", '-d', default='mutag.sdf', type=str, help='SDF data file to convert')
    parser.add_argument("--outputdir", '-o', default='kdd_datasets', type=str, help='the output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sdf_str = open(args.datafile, 'r').read()
    nel_str = SDFtoNEL(sdf_str)
    result = NEL2Graph(nel_str)
    dataset_name = args.datafile.split(".")[0]
    with open("../" + args.outputdir + "/" + dataset_name.lower() + '.graph', 'w') as f:
        pickle.dump(result, f)
