import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


STYLE = {
    'pzobo': dict(color='blue', marker='x', linestyle='-', label='PZOBO'),
    'qnbo': dict(color='red', marker='o', linestyle='-', label='qNBO'),
    'f2ba': dict(color='black', marker='D', linestyle='-', label='F2BA'),
    'ifsba': dict(color='purple', marker='^', linestyle='-', label='IFSBA'),
}


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)

    # Time
    if 'timea' in state and len(state['timea']) > 0:
        times = state['timea']
    elif 'run_timea' in state and len(state['run_timea']) > 0:
        times = state['run_timea']
    elif 'time' in state and len(state['time']) > 0:
        times = state['time']
    elif 'run_time' in state and len(state['run_time']) > 0:
        times = state['run_time']
    else:
        times = []

    # Accuracy
    if 'testaccuracy' in state and len(state['testaccuracy']) > 0:
        accs = state['testaccuracy']
    elif 'evals' in state and len(state['evals']) > 0:
        accs = state['evals']
    elif 'accs' in state and len(state['accs']) > 0:
        accs = state['accs']
    elif 'acc' in state and len(state['acc']) > 0:
        accs = state['acc']
    else:
        accs = []

    times = np.asarray(times, dtype=float)
    accs = np.asarray(accs, dtype=float)

    # If `evals` is a two-dimensional array, take the column containing the test, accuracy and mean values
    if accs.ndim == 2:
        if accs.shape[1] >= 3:
            accs = accs[:, 2]
        else:
            accs = accs.reshape(-1)

    times = np.asarray(times, dtype=float).reshape(-1)
    accs = np.asarray(accs, dtype=float).reshape(-1)

    n = min(len(times), len(accs))
    times = times[:n]
    accs = accs[:n]

    valid = np.isfinite(times) & np.isfinite(accs)
    times = times[valid]
    accs = accs[valid]

    if len(times) > 0:
        idx = np.argsort(times)
        times = times[idx]
        accs = accs[idx]

    return times, accs


def load_runs_by_pattern(pattern, algo_name):
    runs = []
    for i in range(1, 6):
        path = pattern.format(i=i)
        if not os.path.exists(path):
            print(f"[{algo_name}] missing: {path}")
            continue

        t, a = load_checkpoint(path)
        print(f"[{algo_name}] loaded: {path}")
        print(f"len(t)={len(t)}, len(a)={len(a)}")

        if len(t) == 0 or len(a) == 0:
            print(f"[{algo_name}] skipped empty file: {path}")
            continue

        runs.append((t, a))

    return runs


def compute_mean_std_raw(runs, tmax=None):
    """
    No interpolation, no smoothing:
    Truncate directly to the shortest length, then calculate the mean and standard deviation point by point
    """
    if len(runs) == 0:
        return None, None, None

    # Crop each run to tmax
    cropped_runs = []
    for t, a in runs:
        if tmax is not None:
            mask = (t <= tmax)
            t = t[mask]
            a = a[mask]

        if len(t) > 0 and len(a) > 0:
            cropped_runs.append((t, a))

    if len(cropped_runs) == 0:
        return None, None, None

    min_len = min(len(a) for _, a in cropped_runs)
    if min_len < 2:
        return None, None, None

    # Use the time of the first run as the x-axis
    ref_t = cropped_runs[0][0][:min_len]

    acc_matrix = []
    for t, a in cropped_runs:
        acc_matrix.append(a[:min_len])

    acc_matrix = np.asarray(acc_matrix, dtype=float)

    mean = acc_matrix.mean(axis=0)
    std = acc_matrix.std(axis=0)

    return ref_t, mean, std


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--metalogs', default='/root/autodl-tmp/metalogs')
    parser.add_argument('--output', default='compare_shadow_raw_miniimagenet.png')
    parser.add_argument('--tmax', type=float, default=1200)
    parser.add_argument('--shade_alpha', type=float, default=0.18)
    parser.add_argument('--linewidth', type=float, default=2.2)
    parser.add_argument('--markersize', type=float, default=6.5)
    parser.add_argument('--markevery', type=int, default=1)
    args = parser.parse_args()

    base = args.metalogs

    patterns = {
        'pzobo': os.path.join(base, 'pzobo_miniimagenet_run{i}.pt'),
        'qnbo': os.path.join(base, 'qNBO_miniimagenet_run{i}.pt'),
        'f2ba': os.path.join(base, 'F2BA_shots5_miniimagenet_T30_run{i}.pt'),
        'ifsba': os.path.join(base, 'IFSBA_miniimagenet_run{i}.pt'),
    }

    algo_order = ['pzobo', 'qnbo', 'f2ba', 'ifsba']
    all_stats = {}

    for algo_name in algo_order:
        runs = load_runs_by_pattern(patterns[algo_name], algo_name)

        if len(runs) == 0:
            print(f"{algo_name} skipped because no valid runs were found.")
            continue

        t, mean, std = compute_mean_std_raw(runs, tmax=args.tmax)

        if t is None:
            print(f"{algo_name} skipped because mean/std computation failed.")
            continue

        all_stats[algo_name] = (t, mean, std)

    if len(all_stats) == 0:
        print("No valid data found. Check your --metalogs path and filenames.")
        return

    plt.figure(figsize=(10, 8))

    for algo_name in algo_order:
        if algo_name not in all_stats:
            continue

        t, mean, std = all_stats[algo_name]
        style = STYLE[algo_name]

        # Shadow zone
        plt.fill_between(
            t,
            mean - std,
            mean + std,
            color=style['color'],
            alpha=args.shade_alpha,
            linewidth=0
        )

        # Original line chart + marker
        plt.plot(
            t,
            mean,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=args.linewidth,
            markersize=args.markersize,
            markevery=args.markevery,
            label=style['label']
        )

    plt.xlabel("Running time (s)", fontsize=24)
    plt.ylabel("Test accuracy (%)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, args.tmax)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print("Saved figure to", args.output)


if __name__ == "__main__":
    main()