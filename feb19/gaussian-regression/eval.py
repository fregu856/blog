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

network = ToyNet("eval_gaussian-regression_1", project_dir="/root/blog/feb19/gaussian-regression").cuda()
network.load_state_dict(torch.load("/root/blog/feb19/gaussian-regression/training_logs/model_gaussian-regression_1/checkpoints/model_gaussian-regression_1_epoch_75.pth"))

val_dataset = DatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

network.eval() # (set in eval mode, this affects BatchNorm and dropout)
x_values = []
mean_values = []
sigma_alea_values = []
for step, (x) in enumerate(val_loader):
    x = Variable(x).cuda().unsqueeze(1) # (shape: (batch_size, 1))

    outputs = network(x)
    mean = outputs[0] # (shape: (batch_size, ))
    log_var = outputs[1] # (shape: (batch_size, )) (log(sigma^2))

    for i in range(x.size(0)):
        x_value = x[i].data.cpu().numpy()[0]

        mean_value = mean[i].data.cpu().numpy()[0]

        sigma_alea_value = torch.exp(log_var[i]).data.cpu().numpy()[0]

        x_values.append(x_value)
        mean_values.append(mean_value)
        sigma_alea_values.append(sigma_alea_value)

plt.figure(1)
plt.plot(x_values, mean_values, "r")
plt.fill_between(x_values, np.array(mean_values) - 2*np.sqrt(np.array(sigma_alea_values)), np.array(mean_values) + 2*np.sqrt(np.array(sigma_alea_values)), color="C3", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_pred_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("true mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, mean_values, "r")
plt.fill_between(x_values, np.array(mean_values) - 2*np.sqrt(np.array(sigma_alea_values)), np.array(mean_values) + 2*np.sqrt(np.array(sigma_alea_values)), color="C3", alpha=0.25)
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_pred.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, np.sqrt(np.array(sigma_alea_values)), "r")
plt.plot(x_values, 0.15*(1.0/(1 + np.exp(-np.array(x_values)))), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlabel("x")
plt.title("predicted vs true aleatoric uncertainty")
plt.savefig("%s/alea_pred_true.png" % network.model_dir)
plt.close(1)
