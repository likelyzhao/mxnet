#!/usr/bin/env bash
# 网络模型：resnet训练的第2轮epoch的结果
# 数据集：ILSVRC 2017 分类的val数据集
LOG=draw_dets.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.draw_dets --dataset imagenet           \
                                  --image_set val                           \
                                  --dets_path data/cache/imagenet__val_detections.pkl         \
                                  --save_path /disk2/zhaozhijian/res         \
                                  --thresh 0.2 \
                                  >${LOG} 2>&1 &
