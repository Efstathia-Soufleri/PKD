import matplotlib.pyplot as plt
import numpy as np
import scienceplots

split = 'split1'

####################
x3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')
y3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')

idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted[1:5]], y3[idx_sorted[1:5]], marker='o', label = 'No Lateral Connections', markersize=8)

##################
x3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy')
y3 = np.load('./lists/acc_cost/ucf101_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy')

idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted[1:5]], y3[idx_sorted[1:5]], marker='o', label = 'No Scaling', markersize=8)

##################
x3 = np.load('./list_acc_cost/ucf101_kinetics_pretrained/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')
y3 = np.load('./list_acc_cost/ucf101_kinetics_pretrained/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')

idx_sorted = np.argsort(x3)
plt.plot(x3[idx_sorted[1:5]], y3[idx_sorted[1:5]], marker='o', label = 'WISE', markersize=8)

#################

plt.legend()
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
plt.grid()
plt.title('UCF-101')
plt.show()





