import os
import torch
import numpy as np
import matplotlib.pyplot as plt

os.system("mkdir -p ./trainlogs/")
os.system("mkdir -p ./save_data_cleaning/")
os.system("mkdir -p ./save_data_cleaning/lfsba_M_sweep/")

# =========================
# Basic settings
# =========================
dataset = 'australian'
seed = 1
epoch = 150
iterations = 10
model_path = 'save_data_cleaning'

# =========================
# Fixed LFSBA settings
# (keep other settings unchanged)
# =========================
alg = "LFSBA"
w_lr = 100.0
x_lr = 0.008
xhat_lr = 0.01
lmbd = 1.0
m = 1

# Sweep over M
M_list = [0.01, 0.1, 1, 10, 100]

# Store results
results = {}

for M in M_list:
    print(f"\n===== Running {alg} with m={m}, M={M} =====")

    # log file name can contain M,m for convenience
    log_path = (
        f"./trainlogs/{dataset}_{alg}_M{M}_m{m}_k{iterations}"
        f"_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}.log"
    )

    # IMPORTANT:
    # We assume data_cleaning.py accepts --M and --m as arguments.
    # The saved stats file itself still follows your current save format
    # and DOES NOT contain M or m in the filename.
    cmd = (
        f"python data_cleaning.py "
        f"--dataset {dataset} "
        f"--alg {alg} "
        f"--lmbd {lmbd} "
        f"--epochs {epoch} "
        f"--seed {seed} "
        f"--iterations {iterations} "
        f"--x_lr {x_lr} "
        f"--w_lr {w_lr} "
        f"--xhat_lr {xhat_lr} "
        f"--m {m} "
        f"--M {M} "
        f"> {log_path}"
    )
    os.system(cmd)

    # Match CURRENT save format exactly
    save_path = (
        f"./{model_path}/{dataset}_{alg}_k{iterations}"
        f"_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
    )

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"stats file not found: {save_path}")

    stats = torch.load(save_path)

    # Extract time/loss
    time_arr = np.array([x[0] for x in stats])
    loss_arr = np.array([x[1] for x in stats])

    results[M] = {
        "stats": stats,
        "time": time_arr,
        "loss": loss_arr,
    }

    # Since original save_path has no M/m, save a copy to avoid overwriting
    tagged_save_path = (
        f"./{model_path}/lfsba_M_sweep/"
        f"{dataset}_{alg}_M{M}_m{m}_k{iterations}"
        f"_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}.pt"
    )
    torch.save(stats, tagged_save_path)
    print(f"Copied stats to: {tagged_save_path}")

# =========================
# Normalize loss for comparison
# =========================
min_loss = min(np.min(results[M]["loss"]) for M in M_list)

for M in M_list:
    results[M]["gap"] = results[M]["loss"] - min_loss

# =========================
# Plot
# =========================
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rc('font', size=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

markers = ['o', 's', '^', 'd', 'x']
linestyles = ['-', '--', '-.', ':', '-']

for i, M in enumerate(M_list):
    plt.plot(
        results[M]["time"],
        results[M]["gap"],
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        markevery=100,
        linewidth=2.5,
        markersize=7,
        label=f'LFSBA, M={M}'
    )

plt.xlabel('time(s)', fontsize=20)
plt.ylabel('gap', fontsize=20)
plt.xlim(0, 60)
plt.ylim(1e-4, 1)
plt.yscale('log')
plt.grid(True)
plt.legend(fontsize=14, framealpha=0.9, loc='lower left')
plt.tight_layout()

fig_name = f"./{dataset}_LFSBA_m{m}_M_sweep"
plt.savefig(fig_name + ".png")
plt.savefig(fig_name + ".eps", format='eps')
plt.show()