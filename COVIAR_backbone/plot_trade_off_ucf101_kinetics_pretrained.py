import matplotlib.pyplot as plt
import numpy as np
import scienceplots

#################
split = 'split1'
x3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')
y3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')

for split in ['split2','split3']:
    x3 += np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')
    y3 += np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')

y3 = y3 / 3

idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted[1:5]], y3[idx_sorted[1:5]], marker='o', label = 'WISE', markersize=8)

#################

# SOTA WORKS
plt.plot(543903, 93.1, 'rD', label = 'CoViAR', markersize=8)
plt.plot(172587, 85.8, 'rd', label = 'MIMO', markersize=8)
# plt.plot(4717060, 88.5, 'rp', label = 'Wu et al.', markersize=8)


plt.legend()
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
plt.grid()
plt.title('UCF-101')

# Save the plot as an SVG file
# plt.savefig('./figures/trade_off_ucf101.svg', format='svg')
# plt.savefig('./figures/trade_off_ucf101.png', format='png')
plt.show()





