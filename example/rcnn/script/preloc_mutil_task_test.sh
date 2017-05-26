#!/usr/bin/env bash

LOG=loc_train_mutiltask_inceptionv3_v2.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python -m rcnn.tools.train_rpn_mutiltask --network inceptionv3              \
                                  --gpu 0,1,3                                 \
                                  --prefix model/mutiltask          \
				  --lr  0.01	                            \
				  --lr_step  2,5,10	                            \
                                  >${LOG} 2>&1 &


#                                  --dataset imagenet_loc_2017               \
#                                  --image_set train                         \
#                                  --root_path /disk2/data/imagenet_loc_2017 \
#                                  --dataset_path ILSVRC                     \
#                                  --prefix model/imagenet_loc_2017          \

