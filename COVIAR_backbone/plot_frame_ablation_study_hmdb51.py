import matplotlib.pyplot as plt
import numpy as np
import scienceplots

x1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split1_hmdb51.npy')
y1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split1_hmdb51.npy')

x2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split2_hmdb51.npy')
y2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split2_hmdb51.npy')

x3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split3_hmdb51.npy')
y3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split3_hmdb51.npy')

x = [ x1[i] + x2[i] + x3[i] for i in range(len(x3))]
y = [ y1[i] + y2[i] + y3[i] for i in range(len(y3))]

# x = [x[i]/3 for i in range(len(x))]
y = [y[i]/3 for i in range(len(y))]

textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])

plt.plot(x, y, marker='o', label = 'mv')

text = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20 ]

for i in range(len(x)): 
    plt.annotate(text[i], (x[i], y[i]), xytext =(x[i]-0.1, y[i]-0.1)) 

plt.legend(loc='lower right') 
plt.xlabel('Cost (GFLOPs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')
plt.savefig('./figures/ablation_study_mv_hmdb51_kinetics_pretrained.png', format='png')
plt.show()

x1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split1_hmdb51.npy')
y1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split1_hmdb51.npy')

x2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split2_hmdb51.npy')
y2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split2_hmdb51.npy')

x3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_mv_split3_hmdb51.npy')
y3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_mv_split3_hmdb51.npy')

x = [ x1[i] + x2[i] + x3[i] for i in range(len(x3))]
y = [ y1[i] + y2[i] + y3[i] for i in range(len(y3))]

# x = [x[i]/3 for i in range(len(x))]
y = [y[i]/3 for i in range(len(y))]

textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])

plt.plot(x, y, marker='o', label = 'mv')

text = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20 ]

for i in range(len(x)): 
    plt.annotate(text[i], (x[i], y[i]), xytext =(x[i]-0.1, y[i]-0.1)) 

plt.legend(loc='lower right') 
plt.xlabel('Cost (GFLOPs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')
plt.savefig('./figures/ablation_study_mv_hmdb51_kinetics_pretrained.png', format='png')
plt.show()

x1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_residual_split1_hmdb51.npy')
y1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_residual_split1_hmdb51.npy')

x2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_residual_split2_hmdb51.npy')
y2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_residual_split2_hmdb51.npy')

x3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_residual_split3_hmdb51.npy')
y3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_residual_split3_hmdb51.npy')

x = [x1[i] + x3[i] + x2[i] for i in range(len(x3))]
y = [y1[i] + y3[i] + y2[i]  for i in range(len(y3))]

# x = [x[i]/3 for i in range(len(x))]
y = [y[i]/3 for i in range(len(y))]

textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])

plt.plot(x, y, marker='o', label = 'residual')

for i in range(len(x)): 
    plt.annotate(text[i], (x[i], y[i]), xytext =(x[i]-0.1, y[i]-0.1)) 

plt.legend(loc='lower right') 
plt.xlabel('Cost (GFLOPs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')
plt.savefig('./figures/ablation_study_residual_hmdb51_kinetics_pretrained.png', format='png')
plt.show()

x1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_iframe_split1_hmdb51.npy')
y1 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_iframe_split1_hmdb51.npy')

x2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_iframe_split2_hmdb51.npy')
y2 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_iframe_split2_hmdb51.npy')

x3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_cost_iframe_split3_hmdb51.npy')
y3 = np.load('./lists/ablation_study_frame_number/hmdb51_kinetics_pretrained/list_acc_iframe_split3_hmdb51.npy')

x = [x1[i] + x3[i] + x2[i]  for i in range(len(x3))]
y = [y1[i] + y3[i] + y2[i]  for i in range(len(y3))]

# x = [x[i]/3 for i in range(len(x))]
y = [y[i]/3 for i in range(len(y))]

textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
plt.style.use(['ieee', 'science', 'grid'])

plt.plot(x, y, marker='o', label = 'iframe')

for i in range(len(x)): 
    plt.annotate(text[i], (x[i], y[i]), xytext =(x[i]-0.1, y[i]-0.1)) 

plt.legend(loc='lower right') 
plt.xlabel('Cost (GFLOPs)')
plt.ylabel('Accuracy (\%)')
# plt.grid()
plt.title('HMDB-51')
plt.savefig('./figures/ablation_study_iframe_hmdb51_kinetics_pretrained.png', format='png')
plt.show()

