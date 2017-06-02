#!/usr/bin/env bash
# 网络模型：resnet训练的第2轮epoch的结果
# 数据集：ILSVRC 2017 分类的val数据集
LOG=eval_proposal_imagenet_baseline.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.test_rpn --network resnet152                        \
                                  --dataset imagenet           \
                                  --image_set val                           \
                                  --prefix model/e2e          \
                                  --gpu 2                                   \
                                  --epoch 7                                \
                                  >${LOG} 2>&1 &
