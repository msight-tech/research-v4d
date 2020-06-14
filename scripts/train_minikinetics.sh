# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


python ../tools/main.py mini_kinetics RGB "../data/mini_kinetics_train_list.txt" "../data/mini_kinetics_val_list.txt" \
   --arch i3dresnet50  \
   --num_segments 4 \
   --data_length 4 \
   --gd 20 \
   --lr 0.01 \
   --lr_steps 55 80 100 \
   --epochs 110 \
   -b 48 \
   -j 24 \
   --dropout 0.5 \
   --snapshot_pref v4d_resnet50_mini_kinetics