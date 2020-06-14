# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


python  ../tools/train.py kinetics RGB "../data/kinetics_train.csv" "../data/kinetics/val.csv" \
   --arch i3dresnet50 \
   --num_segments 4 \
   --data_length 8 \
   --gd 20 \
   --lr 0.01 \
   --lr_steps 70 110 140 \
   --epochs 150 \
   -b 64 \
   -j 24 \
   --dropout 0.5 \
   --snapshot_pref v4d_resnet50_kinetics