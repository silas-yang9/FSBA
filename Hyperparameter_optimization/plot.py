import numpy as np
import matplotlib.pyplot as plt

# ========= Helper: Trim Invalid Rows ========= #
def trim_valid(data):
    """Remove trailing rows containing zeros (based on the time column)
    to prevent horizontal straight lines in plots."""
    valid = np.where(data[:, 2] > 0)[0]
    if len(valid) == 0:
        return data
    end = valid[-1] + 1
    return data[:end]

# Load all experimental results
data1 = np.load('./save_results/AID/results.npy')
data2 = np.load('./save_results/PAID/results.npy')
data3 = np.load('./save_results/F2BA/results.npy')
data4 = np.load('./save_results/ITD/results.npy')
data5 = np.load('./save_results/PRAHGD/results.npy')
data6 = np.load('./save_results/RAHGD/results.npy')
data7 = np.load('./save_results/IFSBA/results.npy')

# Clean invalid rows
data1 = trim_valid(data1)
data2 = trim_valid(data2)
data3 = trim_valid(data3)
data4 = trim_valid(data4)
data5 = trim_valid(data5)
data6 = trim_valid(data6)
data7 = trim_valid(data7)

# Global figure size settings
plt.rcParams['figure.figsize'] = (8.0, 6.0)

# ==============================
# Fig 1: Testing Accuracy vs Running Time
# ==============================
plt.figure()
plt.plot(data1[:, 2], data1[:, 1], 'g-x', label='AID-BiO')
plt.plot(data2[:, 2], data2[:, 1], 'b--s', label='PAID-BiO')
plt.plot(data3[:, 2], data3[:, 1], 'k-d', label='F2BA',markevery=100)
plt.plot(data4[:, 2], data4[:, 1], 'y-*', label='ITD-BiO')
plt.plot(data5[:, 2], data5[:, 1], 'r-o', label='PRAHGD')
plt.plot(data6[:, 2], data6[:, 1], color='orange', linestyle='--', marker='p', label='RAHGD')
plt.plot(data7[:, 2], data7[:, 1], color='purple', linestyle='-', marker='^', label='IFSBA', markevery=50)

plt.xlabel('running time / s', fontsize=17)
plt.ylabel('testing accuracy', fontsize=17)
plt.xlim(0, 150)
plt.legend()
plt.grid(True)
plt.savefig('accu_time.pdf', bbox_inches='tight')
plt.show()



# ==============================
# Fig 2: Testing Loss vs Running Time
# ==============================
plt.figure()
plt.plot(data1[:, 2], data1[:, 0], 'g-x', label='AID-BiO')
plt.plot(data2[:, 2], data2[:, 0], 'b--s', label='PAID-BiO')
plt.plot(data3[:, 2], data3[:, 0], 'k-d', label='F2BA',markevery=100)
plt.plot(data4[:, 2], data4[:, 0], 'y-*', label='ITD-BiO')
plt.plot(data5[:, 2], data5[:, 0], 'r-o', label='PRAHGD')
plt.plot(data6[:, 2], data6[:, 0], color='orange', linestyle='--', marker='p', label='RAHGD')
plt.plot(data7[:, 2], data7[:, 0], color='purple', linestyle='-', marker='^', label='IFSBA', markevery=50)

plt.xlabel('running time / s', fontsize=17)
plt.ylabel('testing loss', fontsize=17)
plt.xlim(0, 150)
plt.legend()
plt.grid(True)
plt.savefig('loss_time.pdf', bbox_inches='tight')
plt.show()