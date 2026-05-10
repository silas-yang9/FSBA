import torch
import matplotlib.pyplot as plt
import os
import argparse
import math
import numpy as np

os.system("mkdir -p ./trainlogs/")

#dataset = 'mnist'
data_path= 'breast.txt'
dataset = 'breast'
seed = 1
epoch = 2750
iterations = 10

model_path  = 'save_data_cleaning'


# FSBA algorithm (n  ew)
w_lr = 10.0
x_lr = 0.001
xhat_lr = 0.002
lmbd = 5.0
K=10
theta1=0.9
theta2=0.95
M = 10.0  # Add this new hyperparameter 'M'
m=1
iterations = 11
alg = "LFSBA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)
FSBA_time = np.array([x[0]  for x in stats])
FSBA_loss = np.array([x[1] for x in stats])

# LFSBA algorithm (new)
w_lr = 10.0
x_lr = 0.001
xhat_lr = 0.001
lmbd = 10.0
K= 10
theta1=0.85
theta2=0.75
M = 10.0  # Add this new hyperparameter 'M'
m=10
iterations = 10
alg = "LFSBA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)
LFSBA_time = np.array([x[0]  for x in stats])
LFSBA_loss = np.array([x[1] for x in stats])



w_lr = 10.0
x_lr =0.001
xhat_lr = 0.001
lmbd = 10.0
iterations = 10
alg = "F2BA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path) 
print(stats[0])       
print(len(stats))       
F2BA_time = np.array([x[0]  for x in stats])
F2BA_loss = np.array([x[1] for x in stats])


w_lr = 100.0
x_lr =0.001
xhat_lr = 0.001
alg = "ITD"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
ITD_time = np.array([x[0]  for x in stats])
ITD_loss = np.array([x[1] for x in stats])

w_lr = 100.0
x_lr =0.001
lmbd = 10.0
xhat_lr = 0.001
alg = "AID"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
AID_time = np.array([x[0]  for x in stats])
AID_loss = np.array([x[1] for x in stats])

# Normalize loss for comparison
# Calculate the minimum value in all loss arrays
min_loss = np.min([
    np.min(AID_loss),
    np.min(ITD_loss),
    np.min(F2BA_loss),
    np.min(LFSBA_loss),
    np.min(FSBA_loss)
])

# Normalize the loss values
AID_loss= AID_loss - min_loss
ITD_loss = ITD_loss - min_loss
F2BA_loss = F2BA_loss - min_loss
LFSBA_loss = LFSBA_loss - min_loss
FSBA_loss = FSBA_loss - min_loss


plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rc('font', size=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# AID
plt.plot(
    AID_time, AID_loss,
    color='green',
    linestyle='-',
    marker='x',
    markevery=100,          
    linewidth=2.5,
    markersize=6,
    label='AID-BiO'
)

# ITD
plt.plot(
    ITD_time, ITD_loss,
    'y-*',
    markevery=100,          
    linewidth=2.5,
    markersize=7,
    label='ITD-BiO'
)


# F2BA
plt.plot(
    F2BA_time, F2BA_loss,
    color='black',
    linestyle='-',
    marker='d',
    markevery=100,          
    linewidth=2.5,
    markersize=6,
    label=r'F${}^2$BA'
)


# LFSBA
plt.plot(
    LFSBA_time, LFSBA_loss,
    'm-.^',
    markevery=50,          
    linewidth=2.5,
    markersize=7,
    label='LFSBA'
)

#FSBA
plt.plot(
    FSBA_time, FSBA_loss,
    color='blue',
    linestyle='--',      
    marker='s',         
    markevery=100,
    linewidth=2.5,
    markersize=7,
    label='FSBA'
)

plt.xlabel('time(s)', fontsize=20)
plt.ylabel('gap', fontsize=20)

plt.xlim(0, 40)
plt.ylim(1e-4, 1)
plt.yscale('log')
plt.grid(True)

plt.legend(
    fontsize=16,
    framealpha=0.9,
    loc='lower left'      
)

plt.tight_layout()
plt.savefig(f"./{dataset}.png")
plt.savefig(f"./{dataset}.eps", format='eps')
plt.show()
