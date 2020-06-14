# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

python  ../tools/test_models.py mini_kinetics RGB "../data/mini_kinetics_val_list.txt" "../pertrain_models/mini_kinetics_models/_rgb_model_best.pth.tar" \
    --arch i3d_resnet50 \
    --fast_implementation 0 \
    --new_length 4 \
    --slow_testing 1