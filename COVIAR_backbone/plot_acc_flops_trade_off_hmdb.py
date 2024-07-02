import matplotlib.pyplot as plt
import numpy as np
import scienceplots


# x1_wise = np.load('hmdb_list_cost_split1_wise_trade_off_v5.npy')
# y1_wise = np.load('hmdb_list_acc_split1_wise_trade_off_v5.npy')

# x2_wise = np.load('hmdb_list_cost_split2_wise_trade_off_v5.npy')
# y2_wise = np.load('hmdb_list_acc_split2_wise_trade_off_v5.npy')

# x3_wise = np.load('hmdb_list_cost_split3_wise_trade_off_v5.npy')
# y3_wise = np.load('hmdb_list_acc_split3_wise_trade_off_v5.npy')

# x1_wise = np.load('hmdb_list_cost_split1_wise_trade_off_v6.npy')
# y1_wise = np.load('hmdb_list_acc_split1_wise_trade_off_v6.npy')

# x2_wise = np.load('hmdb_list_cost_split2_wise_trade_off_v6.npy')
# y2_wise = np.load('hmdb_list_acc_split2_wise_trade_off_v6.npy')

# x3_wise = np.load('hmdb_list_cost_split3_wise_trade_off_v6.npy')
# y3_wise = np.load('hmdb_list_acc_split3_wise_trade_off_v6.npy')


# x1_wise = np.load('hmdb_list_cost_split1_wise_trade_off_v7.npy')
# y1_wise = np.load('hmdb_list_acc_split1_wise_trade_off_v7.npy')

# x2_wise = np.load('hmdb_list_cost_split2_wise_trade_off_v7.npy')
# y2_wise = np.load('hmdb_list_acc_split2_wise_trade_off_v7.npy')

# x3_wise = np.load('hmdb_list_cost_split3_wise_trade_off_v7.npy')
# y3_wise = np.load('hmdb_list_acc_split3_wise_trade_off_v7.npy')

x1 = np.load('list_cost_split1_wise_trade_off_hmdb51_no_division.npy')
y1 = np.load('list_acc_split1_wise_trade_off_hmdb51_no_division.npy')

x2 = np.load('list_cost_split2_wise_trade_off_hmdb51_no_division.npy')
y2 = np.load('list_acc_split2_wise_trade_off_hmdb51_no_division.npy')

x3 = np.load('list_cost_split3_wise_trade_off_hmdb51_no_division.npy')
y3 = np.load('list_acc_split3_wise_trade_off_hmdb51_no_division.npy')

x = (x1 + x2 + x3) / 3
y = (y1 + y2 + y3) / 3

# x_wise = (x1_wise + x2_wise + x3_wise) / 3
# y_wise = (y1_wise + y2_wise + y3_wise) / 3

# x1_wise = np.load('hmdb_list_cost_split1_wise_trade_off_v5.npy')
# y1_wise = np.load('hmdb_list_acc_split1_wise_trade_off_v5.npy')

# x2_wise = np.load('hmdb_list_cost_split2_wise_trade_off_v5.npy')
# y2_wise = np.load('hmdb_list_acc_split2_wise_trade_off_v5.npy')

# x3_wise = np.load('hmdb_list_cost_split3_wise_trade_off_v5.npy')
# y3_wise = np.load('hmdb_list_acc_split3_wise_trade_off_v5.npy')

# x_wise = x2_wise 
# y_wise = y2_wise

idx_sorted = np.argsort(x)

textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'Our', markersize=8)

# SOTA WORKS
plt.plot(80172, 68.08, 'rD', label = 'CoViAR', markersize=8)
plt.plot(25214.4, 58.6, 'rd', label = 'MIMO', markersize=8)
plt.plot(130876.2, 56.2, 'rp', label = 'Wu et al.', markersize=8)
plt.legend() 
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')

plt.show()

fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])
plt.plot(x[idx_sorted], y[idx_sorted], marker='o', label = 'Our', markersize=8)

# SOTA WORKS
plt.plot(80172, 68.08, 'rD', label = 'CoViAR', markersize=8)
plt.plot(25214.4, 58.6, 'rd', label = 'MIMO', markersize=8)
plt.plot(130876.2, 56.2, 'rp', label = 'Wu et al.', markersize=8)
plt.legend() 
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')

# Save the plot as an SVG file
plt.savefig('./figures/trade_off_hmdb51_v3.svg', format='svg')
plt.savefig('./figures/trade_off_hmdb51_v3.png', format='png')
plt.show()