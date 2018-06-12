# NgramCNN
A Ngram based convolution neural network for graph objects classification.
The major idea of this approach is normalizing the graph objects and applying the specific convolutional neural layers to extract the subgraph structures from graph objects.
Those subgraph structures can be quite complicated while provide great contribution in classification task.

Here is the websize http://www.bruceluo.net/ngramcnn.html.

## FILES
NgramCNN source code package contains following files and folders.

* ngramcnn.py
* ngramcnn_utils.py
* kdd_datasets/
* format_tools/
* bufferdata/
* logdata/

## DEPENDENCY
Python 2.7 is required for NgramCNN.

Besides, following libs are needed:

* tensorflow
* numpy
* itertools-recipes
* futures
* six

Both these libs can be downloaded from their website (just google them) or installed by pip.
```bash
$pip install numpy futures itertools_recipes six tensorflow 
```

## DEMO

NgramCNN's demo contains two parts.

Preprocessing
-
The tamplate is as follows:
```bash
python ngramcnn.py <datasetname> 1 <kernel_width>
```
In this script, the first arg is the dataset name, see folder "kdd_datasets".

The second arg denotes current operation is pre-prcessing.

The third arg denotes the kernel size, namely the n in ngram.

This script will handle the data from dataset and write into the buffer dir.

A toy example is as follows:
```bash
python ngramcnn.py ptc 1 7
```

Training
-
```bash
python ngramcnn.py <dataset_name> 2 <kernel_width> <batch_size> <diag_kernel_num> <conv_kernel_size> <conv_kernel_num> <epoch_num> <dropout_ratio>
```

In training, the batch_size means the mini-batch size used in training.

diag_kernel_num means the number of kernels in diagonal convolution.

conv_kernel_size and conv_kernel_num denote the kernel size and number in rest convolution layers, resp.

epoch_num denotes the max epoch iteration number.

dropout_ratio is the parameter in dropout.

An example is:
```bash
python ngramcnn.py ptc 2 7 100 20 7 20 200 0.5
```

Note that we developed and tested these codes in MacOS and Ubuntu.

Windows OS may not support some OS commands and you can just remove those codes.

In default, the GPU is required.

Nvidia GeForce 1080 and titanX are suggested configuration.

## FORMAT

The default format for NgramCNN is .Graph which is a python data file dumped by pickle.
For this format, we prepared some tools to load data from other formats.
Currently, two formats are supported: SDF and NEL.
You can convert you own data by following steps:

step 1:
Copy your data file, either in sdf or nel format, in format_tools/
```bash
cp <datafile> format_tools/
```
Note that <datafile> should contain the suffix.
For example, mutag.nel instead of mutag.

step 2:
Move into format_tools folder and convert format:
```bash
cd format_tools/
python nel2graph.py -d <datafile>
```
The generated file will be stored at kdd_datasets/



## REFERENCE

Please cite our publication if you'd like to use our code (for comparison and promotion).

        @article{luo2017deep,
          title={Deep Learning of Graphs with Ngram Convolutional Neural Networks},
          author={Luo, Zhiling and Liu, Ling and Yin, Jianwei and Li, Ying and Wu, Zhaohui},
          journal={IEEE Transactions on Knowledge and Data Engineering},
          volume={29},
          number={10},
          pages={2125--2139},
          year={2017},
          publisher={IEEE}
        }


## CONTACT

Contact me if you have any questions about the code and its execution.

Dr. Bruce Luo

luozhiling@zju.edu.cn

The latest code version will be released in my homepage.

http://www.bruceluo.net

That's all, forks.
