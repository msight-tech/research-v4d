# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
log_file="train_log.txt"
log_list=[x.strip() for x in open(log_file)]
train_list=[]
val_list=[]
for i,log in enumerate(log_list):
  if i%2==0:
    train_list.append(log)
  else:
    val_list.append(log)
train_loss_list=[]
val_loss_list=[]
train_top1_list=[]
val_top1_list=[]
for train_log in train_list:
  train_log=train_log.split(" ")
  train_loss=float(train_log[1].split(":")[1])
  train_top1=float(train_log[2].split(":")[1])
  train_loss_list.append(train_loss)
  train_top1_list.append(train_top1)
for val_log in val_list:
  val_log=val_log.split(" ")
  val_loss=float(val_log[3].split(":")[1])
  val_top1=float(val_log[1].split(":")[1])
  val_loss_list.append(val_loss)
  val_top1_list.append(val_top1)
print train_loss_list[:4]
print val_top1_list[:4]


x = range(len(train_loss_list))


plt.plot(x, train_top1_list, label='train_top1')
plt.plot(x, val_top1_list, label='val_top1')
plt.xlabel('Number Epoch')
plt.ylabel('top1')
plt.title('i2d res50\n maxpool3d early stage')
plt.legend()
plt.show()