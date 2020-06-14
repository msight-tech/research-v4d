# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

python ../tools/test_models.py kinetics RGB "../data/kinetics_val.csv" "../pretrain_models/kinetics_models/_rgb_model_best.pth.tar" \ 
    --arch i3d_resnet50 \
    --fast_implementation 1 \
    --new_length 8 \