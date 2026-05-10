import os
import torch
import numpy as np
import matplotlib.pyplot as plt

os.system("mkdir -p ./trainlogs/")
os.system("mkdir -p ./save_data_cleaning/")

dataset = 'australian'
seed = 1
epoch = 1000
iterations = 10
model_path = 'save_data_cleaning'
alg = "LFSBA"

# ablation choices
m_list = [1, 5, 10]

# use the same hyperparameters as the old m=5 setting
w_lr = 100.0
x_lr = 0.1
xhat_lr = 0.1
lmbd = 1.0
K = 20
theta1 = 0.9
theta2 = 0.95
M = 10.0

results = {}

for m in m_list:
    log_name = (
        f"./trainlogs/{dataset}_{alg}_m{m}_k{iterations}"
        f"_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}.log"
    )

    cmd = (
        f"python data_cleaning.py "
        f"--dataset {dataset} "
        f"--alg {alg} "
        f"--epochs {epoch} "
        f"--seed {seed} "
        f"--iterations {iterations} "
        f"--x_lr {x_lr} "
        f"--w_lr {w_lr} "
        f"--xhat_lr {xhat_lr} "
        f"--lmbd {lmbd} "
        f"--m {m} "
        f"> {log_name}"
    )

    print("=" * 80)
    print(f"Running m={m}")
    print(cmd)
    ret = os.system(cmd)
    print(f"Return code: {ret}")

    save_path = (
        f"./{model_path}/{dataset}_{alg}_m{m}_k{iterations}"
        f"_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
    )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Result file not found for m={m}:\n{save_path}\n"
            f"Please check whether data_cleaning.py saves with '_m{{m}}' in the filename."
        )

    stats = torch.load(save_path)

    time_arr = np.array([x[0] for x in stats], dtype=float)
    loss_arr = np.array([x[1] for x in stats], dtype=float)

    results[m] = {
        "time": time_arr,
        "loss": loss_arr,
    }

# normalize jointly
min_loss = min(np.min(results[m]["loss"]) for m in m_list)
for m in m_list:
    results[m]["gap"] = results[m]["loss"] - min_loss

# plot
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rc('font', size=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

plt.plot(
    results[1]["time"], results[1]["gap"],
    'r-o',
    markevery=max(1, len(results[1]["time"]) // 20),
    linewidth=2.5,
    markersize=7,
    label='LFSBA (m=1)'
)

plt.plot(
    results[5]["time"], results[5]["gap"],
    'b-d',
    markevery=max(1, len(results[5]["time"]) // 20),
    linewidth=2.5,
    markersize=7,
    label='LFSBA (m=5)'
)

plt.plot(
    results[10]["time"], results[10]["gap"],
    'm-.^',
    markevery=max(1, len(results[10]["time"]) // 20),
    linewidth=2.5,
    markersize=7,
    label='LFSBA (m=10)'
)

plt.xlabel('time(s)', fontsize=20)
plt.ylabel('gap', fontsize=20)
plt.yscale('log')
plt.xlim(0, 60)
plt.ylim(1e-4, 1)
plt.grid(True)

plt.legend(
    fontsize=16,
    framealpha=0.9,
    loc='lower left'
)

plt.tight_layout()
plt.savefig(f"./{dataset}_ablation_m.png")
plt.savefig(f"./{dataset}_ablation_m.eps", format='eps')
plt.show()