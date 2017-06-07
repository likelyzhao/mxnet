#!/usr/bin/env bash
# 网络模型：resnet训练的第2轮epoch的结果
# 数据集：ILSVRC 2017 分类的val数据集
LOG=train_rcnn_imagenet_inceptionv3.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.train_rcnn --network inceptionv3                        \
                                  --prefix model/inception_split_rcnn          \
                                  --gpu 2,3                                   \
                                  >${LOG} 2>&1 &
#				  --pretrained model/imagenet_loc_2017   \
#				  --pretrained_epoch  4  \
