import torch
import matplotlib.pyplot as plt
import os
import argparse
import math
import numpy as np

os.system("mkdir -p ./trainlogs/")

dataset = 'mnist'

seed = 1
epoch = 10000
iterations = 10
model_path  = 'save_data_cleaning'

x_lr = 0.01
xhat_lr = 0.01
w_lr = 100.0
lmbd = 10.0
theta1 = 0.75 # AGD momentum for x
theta2 = 0.9  # AGD momentum for xhat
# Cubic regularization parameters
use_cubic = 1  # Enable cubic regularization
M = 1.0  # Cubic regularization parameter
cheb_K = 8 # Chebyshev polynomial order for Hessian approximation
l_est = 1.0  # Upper bound of Hessian eigenvalues
mu_est = 0.02  # Lower bound of Hessian eigenvalues
cubic_iters = 5 # Max iterations for cubic subproblem
alg = "IFSBA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} --theta1 {theta1} --theta2 {theta2} --use_cubic {use_cubic} --M {M} --cheb_K {cheb_K} --l_est {l_est} --mu_est {mu_est} --cubic_iters {cubic_iters} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)
IFSBA_time = np.array([x[0]  for x in stats])
IFSBA_loss = np.array([x[1] for x in stats])


w_lr = 100.0
x_lr =0.01
xhat_lr = 0.01
lmbd = 10.0
alg = "F2BA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)        
F2BA_time = np.array([x[0]  for x in stats])
F2BA_loss = np.array([x[1] for x in stats])


w_lr = 100.0
x_lr =0.1
xhat_lr = 0.1
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

 
w_lr = 100.0
x_lr = 0.01
xhat_lr = 0.01
alg = "RAHGD"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)
RAHGD_time = np.array([x[0] for x in stats])
RAHGD_loss = np.array([x[1] for x in stats])



# Normalize loss for comparison
# Calculate the minimum value in all loss arrays
min_loss = np.min([
    np.min(AID_loss),
    np.min(ITD_loss),
    np.min(F2BA_loss),
    np.min(IFSBA_loss),
    np.min(RAHGD_loss)
])

# Normalize the loss values
AID_loss= AID_loss - min_loss
ITD_loss = ITD_loss - min_loss
F2BA_loss = F2BA_loss - min_loss
IFSBA_loss = IFSBA_loss - min_loss
RAHGD_loss = RAHGD_loss - min_loss


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

# IFSBA
plt.plot(
    IFSBA_time, IFSBA_loss,
    'm-.^',
    markevery=50,          
    linewidth=2.5,
    markersize=7,
    label='IFSBA'
)

# RAHGD
plt.plot(
    RAHGD_time, RAHGD_loss,
    color='orange',
    linestyle='--',
    marker='p',
    markevery=100,
    linewidth=2.5,
    markersize=7,
    label='RAHGD'
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
