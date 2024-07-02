import matplotlib.pyplot as plt
import numpy as np

x1 = np.load('list_cost_split1_wise_trade_off_hmdb51_no_division.npy')
y1 = np.load('list_acc_split1_wise_trade_off_hmdb51_no_division.npy')

x2 = np.load('list_cost_split2_wise_trade_off_hmdb51_no_division.npy')
y2 = np.load('list_acc_split2_wise_trade_off_hmdb51_no_division.npy')

x3 = np.load('list_cost_split3_wise_trade_off_hmdb51_no_division.npy')
y3 = np.load('list_acc_split3_wise_trade_off_hmdb51_no_division.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no division', markersize=8)

x1 = np.load('list_cost_split1_wise_trade_off_hmdb51_no_scaling.npy')
y1 = np.load('list_acc_split1_wise_trade_off_hmdb51_no_scaling.npy')

x2 = np.load('list_cost_split2_wise_trade_off_hmdb51_no_scaling.npy')
y2 = np.load('list_acc_split2_wise_trade_off_hmdb51_no_scaling.npy')

x3 = np.load('list_cost_split3_wise_trade_off_hmdb51_no_scaling.npy')
y3 = np.load('list_acc_split3_wise_trade_off_hmdb51_no_scaling.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no division no scaling', markersize=8)

x1 = np.load('list_cost_split1_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')
y1 = np.load('list_acc_split1_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')

x2 = np.load('list_cost_split2_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')
y2 = np.load('list_acc_split2_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')

x3 = np.load('list_cost_split3_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')
y3 = np.load('list_acc_split3_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no lateral connections', markersize=8)

# SOTA WORKS
plt.plot(80172, 68.08, 'rD', label = 'CoViAR', markersize=8)
plt.plot(25214.4, 58.6, 'rd', label = 'MIMO', markersize=8)
plt.plot(130876.2, 56.2, 'rp', label = 'Wu et al.', markersize=8)

plt.legend()
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
plt.grid()
plt.title('HMDB-51')
plt.show()

