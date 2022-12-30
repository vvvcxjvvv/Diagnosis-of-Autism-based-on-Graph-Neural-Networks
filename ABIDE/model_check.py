# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0.1, 0.6, 6)
# y = np.linspace(0.1, 0.6, 6)
#
# plt.errorbar(x, y, fmt="bo:", yerr=0.5, xerr=0.02)
#
# plt.xlim(0, 0.7)
#
# plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

'''
AD: MCI_AD GCN
arr_mean:0.8798199999999999, arr_std:0.01063661600322206
arr_mean:0.8886, arr_std:0.032912429263121885
arr_mean:0.87722, arr_std:0.010640000000000009
arr_mean:0.8600199999999999, arr_std:0.011732757561630603
arr_mean:0.90876, arr_std:0.023597677851856548

AD: NC_SMC_AD GAT
arr_mean:0.8856999999999999, arr_std:0.025062082914235197
arr_mean:0.8277800000000001, arr_std:0.053926631639663894
arr_mean:0.9193999999999999, arr_std:0.020475644068014064
arr_mean:0.87346, arr_std:0.026441603582233807
arr_mean:0.9196799999999999, arr_std:0.024901277075684295

'''
labels = ['acc', 'sen', 'spe', 'f1', 'auc']
means1 = [0.8798, 0.8886, 0.8772, 0.86, 0.9088]
means2 = [0.8857, 0.8278, 0.9194, 0.8735, 0.9197]

std1 = [0.011, 0.033, 0.011, 0.012, 0.024]
std2 = [0.025, 0.054, 0.020, 0.026, 0.025]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means1, width, label='GCN')
rects2 = ax.bar(x + width/2, means2, width, label='GAT')



plt.errorbar(x - width/2,means1,yerr=std1,fmt='o',ecolor='hotpink',
			elinewidth=2,ms=5,mfc='wheat',mec='salmon',capsize=3)

plt.errorbar(x + width/2,means2,yerr=std2,fmt='o',ecolor='blue',
			elinewidth=2,ms=5,mfc='wheat',mec='salmon',capsize=3)


# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Scores')
ax.set_title('MCI/AD分类')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()
#y = np.linspace(0, 2, 10)
plt.ylim(0.4, 1.05)

plt.show()
