import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os

x1 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_split1_wise_trade_off_hmdb51_no_division.npy')
y1 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_split1_wise_trade_off_hmdb51_no_division.npy')

x2 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_split2_wise_trade_off_hmdb51_no_division.npy')
y2 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_split2_wise_trade_off_hmdb51_no_division.npy')

x3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_cost_split3_wise_trade_off_hmdb51_no_division.npy')
y3 = np.load('./lists/acc_cost/hmdb51_kinetics_pretrained/list_acc_split3_wise_trade_off_hmdb51_no_division.npy')

x = (x1 + x2 + x3)
y = (y1 + y2 + y3) / 3

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
plt.plot(222614.7, 68.08, 'rD', label = 'CoViAR', markersize=8)
plt.plot(70639.8, 58.6, 'rd', label = 'MIMO', markersize=8)
plt.plot(1930654.7, 56.2, 'rp', label = 'Wu et al.', markersize=8)
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
plt.plot(222614.7, 68.08, 'rD', label = 'CoViAR', markersize=8)
plt.plot(70639.8, 58.6, 'rd', label = 'MIMO', markersize=8)
# plt.plot(193065, 56.2, 'rp', label = 'Wu et al.', markersize=8)
plt.legend() 
plt.xlabel('Cost (GMACs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')

# Specify the folder path you want to check and create
folder_path = "./figures/"

if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

# Save the plot as an SVG file
plt.savefig('./figures/trade_off_hmdb51.svg', format='svg')
plt.savefig('./figures/trade_off_hmdb51.png', format='png')
plt.show()