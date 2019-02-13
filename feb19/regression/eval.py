from datasets import DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 32

network = ToyNet("eval_regression_1", project_dir="/root/blog/feb19/regression").cuda()
network.load_state_dict(torch.load("/root/blog/feb19/regression/training_logs/model_regression_1/checkpoints/model_regression_1_epoch_50.pth"))

val_dataset = DatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

network.eval() # (set in eval mode, this affects BatchNorm and dropout)
x_values = []
pred_y_values = []
for step, (x) in enumerate(val_loader):
    x = Variable(x).cuda().unsqueeze(1) # (shape: (batch_size, 1))

    outputs = network(x) # (shape: (batch_size, ))

    for i in range(x.size(0)):
        x_value = x[i].data.cpu().numpy()[0]

        pred_y_value = outputs[i].data.cpu().numpy()[0]

        x_values.append(x_value)
        pred_y_values.append(pred_y_value)

plt.figure(1)
plt.plot(x_values, pred_y_values, "r")
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.xlabel("x")
plt.title("predicted vs true f(x)")
plt.savefig("%s/f_pred_true.png" % network.model_dir)
plt.close(1)
