import matplotlib.pyplot as plt
import numpy as np

x1 = np.load('list_cost_split1_wise_trade_off_ucf101_no_division.npy')
y1 = np.load('list_acc_split1_wise_trade_off_ucf101_no_division.npy')

x2 = np.load('list_cost_split2_wise_trade_off_ucf101_no_division.npy')
y2 = np.load('list_acc_split2_wise_trade_off_ucf101_no_division.npy')

x3 = np.load('list_cost_split3_wise_trade_off_ucf101_no_division.npy')
y3 = np.load('list_acc_split3_wise_trade_off_ucf101_no_division.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no division', markersize=8)

x1 = np.load('list_cost_split1_wise_trade_off_ucf101_no_scaling_no_division.npy')
y1 = np.load('list_acc_split1_wise_trade_off_ucf101_no_scaling_no_division.npy')

x2 = np.load('list_cost_split2_wise_trade_off_ucf101_no_scaling_no_division.npy')
y2 = np.load('list_acc_split2_wise_trade_off_ucf101_no_scaling_no_division.npy')

x3 = np.load('list_cost_split3_wise_trade_off_ucf101_no_scaling_no_division.npy')
y3 = np.load('list_acc_split3_wise_trade_off_ucf101_no_scaling_no_division.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no division no scaling', markersize=8)

x1 = np.load('list_cost_split1_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')
y1 = np.load('list_acc_split1_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')

x2 = np.load('list_cost_split2_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')
y2 = np.load('list_acc_split2_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')

x3 = np.load('list_cost_split3_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')
y3 = np.load('list_acc_split3_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

idx_sorted = np.argsort(x)
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'KD-no lateral connections', markersize=8)

# SOTA WORKS
plt.plot(282743.18, 90.40, 'rD', label = 'CoViAR', markersize=8)
plt.plot(62310.88, 85.8, 'rd', label = 'MIMO', markersize=8)
plt.plot(323426.74, 88.5, 'rp', label = 'Wu et al.', markersize=8)

plt.legend()
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
plt.grid()
plt.title('UCF-101')
plt.show()

