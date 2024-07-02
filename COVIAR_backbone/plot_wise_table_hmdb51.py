import matplotlib.pyplot as plt
import numpy as np

split = 'split1'

x3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_hmdb51_no_division.npy')
y3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_hmdb51_no_division.npy')

idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted], y3[idx_sorted], marker='o', label = 'KD-no division', markersize=8)

x3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_hmdb51_no_scaling.npy')
y3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_hmdb51_no_scaling.npy')


idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted], y3[idx_sorted], marker='o', label = 'KD-no division no scaling', markersize=8)

x3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')
y3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_hmdb51_no_scaling_no_lateral_connections.npy')


idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted], y3[idx_sorted], marker='o', label = 'KD-no lateral connections', markersize=8)

# # SOTA WORKS
# plt.plot(80172, 68.08, 'rD', label = 'CoViAR', markersize=8)
# plt.plot(25214.4, 58.6, 'rd', label = 'MIMO', markersize=8)
# plt.plot(130876.2, 56.2, 'rp', label = 'Wu et al.', markersize=8)

plt.legend()
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
plt.grid()
plt.title('HMDB-51')
plt.show()

