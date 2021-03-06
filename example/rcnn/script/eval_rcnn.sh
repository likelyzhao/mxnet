#!/usr/bin/env bash
# 网络模型：resnet训练的第2轮epoch的结果
# 数据集：ILSVRC 2017 分类的val数据集
LOG=rcnn_proposal_loc_baseline.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.test_rcnn --network resnet                        \
                                  --dataset imagenet           \
                                  --image_set val                           \
                                  --prefix model/resnet_split_rcnn         \
                                  --gpu 2                                   \
                                  --epoch 8                               \
                                  >${LOG} 2>&1 &
