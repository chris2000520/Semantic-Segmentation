# Semantic-Segmentation
语义分割自用实验模型，训练，预测代码

## configs

配置文件，json格式，对每一个新的模型进行一个记录，包含模型名称，数据集，输入尺寸，损失函数，训练轮次，学习率，批次大小等等。

## database

每一个数据集是单独一个文件夹，其中包含clsaa.csv文件，为分割图上色。

## results

每一个数据集单独一个文件夹，对实验结果进行保存，系统时间加随机数命名。在每一个结果中，包含了**最好训练权重**，**训练配置文件**，**日志**。

## run
包含日志输出

## toolbox

### datasets

相关的数据集`Camvid`和`Cityscapes`读取，处理

### loss

自定义损失函数

### models

深度学习神经网络模型


## train

模型训练

## predict

模型预测，输出分割后的预测图和指标

## evaluate

模型评估，包括**FPS**，**FLOPs**，**Parameters**
