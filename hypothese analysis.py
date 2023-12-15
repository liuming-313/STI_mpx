import scipy.stats as stats
import numpy as np
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import math


calib_day=[83,89,108,135,142,149,169,177,213,239]
calib_week=[11,12,13,14,15,16,17,18,20,21]
# Define two data series
#actual data

##cummulative cases data
series1 = [89,161,359,750,799,839,892,905,924,943]
#currebt data
series2 = [286.5, 330.0, 484.0, 670.0, 704.5, 732.0, 789.0, 806.5, 853.0, 861.5]

##male ratio data
series1=[1,1,1,1,0.988349515,0.987012987,0.986398964,0.979681494,0.977307891,0.975634013]
series2=[0.9721668880035415,0.9702097620008068,0.9683803171406997,0.9656862745098039,0.9636966803229305, 0.961308144715737,0.9599235181644359,0.9580898496875527,0.9557806036987659,0.9544678281646415]



# Perform the Wilcoxon signed-rank test
statistic, p_value = stats.wilcoxon(series1, series2)

# Print the test statistic and p-value
print("Wilcoxon signed-rank test statistic: ", statistic)
print("P-value: ", p_value)

# Interpret the results
if p_value < 0.05:
    print("The Wilcoxon signed-rank test result is statistically significant at alpha = 0.05.")
    if statistic > 0:
        print("Series 1 is greater than Series 2.")
    else:
        print("Series 2 is greater than Series 1.")
else:
    print("The Wilcoxon signed-rank test result is not statistically significant at alpha = 0.05.")
    print("There is not enough evidence to conclude that there is a significant difference between the two data series.")

##load the result data from the five dicts
import pickle
with open('result_dict_0_3M.pkl', 'rb') as handle1:
    result_dict1=pickle.load(handle1)
with open('result_dict_0_5M.pkl', 'rb') as handle2:
    result_dict2=pickle.load(handle2)
with open('result_dict_1M.pkl', 'rb') as handle3:
    result_dict3=pickle.load(handle3)
with open('result_dict_2M.pkl', 'rb') as handle4:
    result_dict4=pickle.load(handle4)
with open('result_dict_5M.pkl', 'rb') as handle5:
    result_dict5 = pickle.load(handle5)
with open('result_dict_10M.pkl', 'rb') as handle6:
    result_dict6 = pickle.load(handle6)



##the the data
#a lasting duration
result_dict_all=[result_dict1,result_dict2,result_dict3,result_dict4,result_dict5,result_dict6]
#define an array
arr1_1,arr2_1,arr3_1,arr4_1,arr5_1,arr_6_1=[],[],[],[],[],[]
arr_all_1=[arr1_1,arr2_1,arr3_1,arr4_1,arr5_1,arr_6_1]

arr1_2,arr2_2,arr3_2,arr4_2,arr5_2,arr6_2=[],[],[],[],[],[]
arr_all_2=[arr1_2,arr2_2,arr3_2,arr4_2,arr5_2,arr6_2]

arr1_3,arr2_3,arr3_3,arr4_3,arr5_3,arr6_3=[],[],[],[],[],[]
arr_all_3=[arr1_3,arr2_3,arr3_3,arr4_3,arr5_3,arr6_3]

arr1_4,arr2_4,arr3_4,arr4_4,arr5_4,arr6_4=[],[],[],[],[],[]
arr_all_4=[arr1_4,arr2_4,arr3_4,arr4_4,arr5_4,arr6_4]

arr1_5,arr2_5,arr3_5,arr4_5,arr5_5,arr6_5=[],[],[],[],[],[]
arr_all_5=[arr1_5,arr2_5,arr3_5,arr4_5,arr5_5,arr6_5]

arr1_6,arr2_6,arr3_6,arr4_6,arr5_6,arr6_6=[],[],[],[],[],[]
arr_all_6=[arr1_6,arr2_6,arr3_6,arr4_6,arr5_6,arr6_6]

##find the index of the last non-zero element
for ind,result_dict in enumerate(result_dict_all):
    for i in range(500):
        ##1.get the duration: last non_zero index
        arr=result_dict[i]['dayn_all']
        #find the indeces of non-zero elements
        nonzero_indices=np.flatnonzero(arr)

        if len(nonzero_indices)>0:
            last_non_zero_index = np.flatnonzero(arr)[-1]
        else:
            last_non_zero_index = 0
        arr_all_1[ind].append(last_non_zero_index)

        ##2.get the cuminfections
        arr = result_dict[i]['cum_all']
        total_inf=arr[-1]
        arr_all_2[ind].append(total_inf)
        
        ##3.get the male ratio
        arr = result_dict[i]['sex_ratio_week']
        sex_ratio=arr[-1]
        if not math.isnan(sex_ratio):
            arr_all_3[ind].append(sex_ratio)
        
        ##4.get the homosexual proportion
        arr = result_dict[i]['weekn_ho_n']
        ho_pro=arr[-1]
        if not math.isnan(ho_pro):
            arr_all_4[ind].append(ho_pro)

        ##5.get the MSM porportion in males
        arr = result_dict[i]['MSM_ratio_week']
        msm_pro = arr[-1]
        if not math.isnan(msm_pro):
            arr_all_5[ind].append(msm_pro)
        ###6.get the sexual contact
        sexual_ratio=(result_dict[i]['cum_party'][-1]+result_dict[i]['cum_he'][-1]+result_dict[i]['cum_ho'][-1]+result_dict[i]['cum_bi'][-1])/result_dict[i]['cum_all'][-1]
        if not math.isnan(sexual_ratio):
            arr_all_6[ind].append(sexual_ratio)
        
       
        
        
        
        

arr1_1,arr2_1,arr3_1,arr4_1,arr5_1,arr_6_1=np.array(arr1_1),np.array(arr2_1),np.array(arr3_1),np.array(arr4_1),np.array(arr5_1),np.array(arr_6_1)
arr1_2,arr2_2,arr3_2,arr4_2,arr5_2,arr6_2=np.array(arr1_2),np.array(arr2_2),np.array(arr3_2),np.array(arr4_2),np.array(arr5_2),np.array(arr6_2)
arr1_3,arr2_3,arr3_3,arr4_3,arr5_3,arr6_3=np.array(arr1_3),np.array(arr2_3),np.array(arr3_3),np.array(arr4_3),np.array(arr5_3),np.array(arr6_3)
arr1_4,arr2_4,arr3_4,arr4_4,arr5_4,arr6_4=np.array(arr1_4),np.array(arr2_4),np.array(arr3_4),np.array(arr4_4),np.array(arr5_4),np.array(arr6_4)
arr1_5,arr2_5,arr3_5,arr4_5,arr5_5,arr6_5=np.array(arr1_5),np.array(arr2_5),np.array(arr3_5),np.array(arr4_5),np.array(arr5_5), np.array(arr6_5)
arr1_6,arr2_6,arr3_6,arr4_6,arr5_6,arr6_6=np.array(arr1_6),np.array(arr2_6),np.array(arr3_6),np.array(arr4_6),np.array(arr5_6), np.array(arr6_6)
# Create a figure with six subplots
fig, axs = plt.subplots(figsize=(15, 10))
plt.axis('off')

##first plot of outbreak duration

# Combine the data series into a list
data1 = [arr1_1,arr2_1,arr3_1,arr4_1,arr5_1,arr_6_1]
data2 = [arr1_2,arr2_2,arr3_2,arr4_2,arr5_2,arr6_2]
data3 = [arr1_3,arr2_3,arr3_3,arr4_3,arr5_3,arr6_3]
data4 = [arr1_4,arr2_4,arr3_4,arr4_4,arr5_4,arr6_4]
data5 = [arr1_5,arr2_5,arr3_5,arr4_5,arr5_5,arr6_5]
data6 = [arr1_6,arr2_6,arr3_6,arr4_6,arr5_6,arr6_6]

# Create a box plot
ax1 = fig.add_subplot(3, 2, 1)
ax1.boxplot(data1)
# Add labels and title
ax1.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax1.set_xlabel('Population size (million)')
ax1.set_ylabel('Outbreak duration (days)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax.set_title('Box plot of 5 data series')
delta=1e-12
# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data1]
medians = [np.median(d) for d in data1]
y_min, y_max = ax1.get_ylim()
ax1.set_title('Outbreak duration',y=1.18)
print(f'y={y_max + (y_max - y_min)* 0.05}')
for i in range(len(data1)):
    high, low = np.quantile(data1[i], [0.975, 0.025])
    ax1.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i]+delta,0):.0f}\nCI: [{round(low+delta,0):.0f}, {round(high+delta,0):.0f}]",
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=7)

# Show the plot
ax2 = fig.add_subplot(3, 2, 2)
ax2.boxplot(data2)
# Add labels and title
ax2.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax2.set_xlabel('Population size (million)')
ax2.set_ylabel('Cummulative infections')
#ax.set_title('Box plot of 5 data series')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data2]
medians = [np.median(d) for d in data2]

y_min, y_max = ax2.get_ylim()
ax2.set_title('Cummulative infections',y=1.18)
for i in range(len(data2)):
    high, low = np.quantile(data2[i], [0.975, 0.025])
    ax2.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i]+delta,0):.0f}\nCI: [{round(low,0)+delta:.0f}, {round(high,0)+delta:.0f}]",
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=7)

ax3 = fig.add_subplot(3, 2, 3)
ax3.boxplot(data3)
# Add labels and title
ax3.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax3.set_xlabel('Population size (million)')
ax3.set_ylabel('Percentage of cases')
ax3.set_title('Male ratio',y=1.18)
#set ax3 ticks labels to be  %
ax3.set_yticks(np.arange(0.00, 1.01, step=0.2))
#change the ticks to be %
ax3.set_yticklabels(['{:,.0%}'.format(x) for x in np.arange(0.00, 1.01, step=0.2)])
ax3.set_ylim(0,1.05)
#ax.set_title('Box plot of 5 data series')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.axhline(y=0.963,c='b',linestyle='--')
ax3.text(4.5,0.70,'Global male ratio: 0.963',c='b')
# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data3]
medians = [np.median(d) for d in data3]
y_min, y_max = ax3.get_ylim()
for i in range(len(data3)):
    high,low=np.quantile(data3[i],[0.975,0.025])
    ax3.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i],2):.2f}\nCI: [{round(low,2):.2f}, {round(high,2):.2f}]", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),fontsize=7)



ax4 = fig.add_subplot(3, 2, 4)
ax4.boxplot(data4)
# Add labels and title
ax4.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax4.set_xlabel('Population size (million)')
ax4.set_ylabel('Percentage of cases')
ax4.set_title('Homosexual ratio',y=1.18)
#set ax4 ticks labels to be  %
ax4.set_yticks(np.arange(0.00, 1.01, step=0.2))
#change the ticks to be %
ax4.set_yticklabels(['{:,.0%}'.format(x) for x in np.arange(0.00, 1.01, step=0.2)])
ax4.set_ylim(0,1.05)
#ax.set_title('Box plot of 5 data series')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.axhline(y=0.802,c='b',linestyle='--')
ax4.text(4.5,0.55,'Global homosexual ratio: 0.802',c='b')
# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data4]
medians = [np.median(d) for d in data4]
y_min, y_max = ax4.get_ylim()
for i in range(len(data4)):
    high, low = np.quantile(data4[i], [0.975, 0.025])
    ax4.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i],2):.2f}\nCI: [{round(low,2):.2f}, {round(high,2):.2f}]",
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=7)

ax5 = fig.add_subplot(3, 2, 5)
ax5.boxplot(data5)
# Add labels and title
ax5.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax5.set_xlabel('Population size (million)')
ax5.set_ylabel('Percentage of cases')
ax5.set_title('MSM ratio',y=1.18)
#ax.set_title('Box plot of 5 data series')
#set ax5 ticks labels to be  %
ax5.set_yticks(np.arange(0.00, 1.01, step=0.2))
#change the ticks to be %
ax5.set_yticklabels(['{:,.0%}'.format(x) for x in np.arange(0.00, 1.01, step=0.2)])
ax5.set_ylim(0,1.05)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.axhline(y=0.85,c='b',linestyle='--')
ax5.text(4.5,0.65,'Global MSM ratio: 0.850',c='b')
# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data5]
medians = [np.median(d) for d in data5]
y_min, y_max = ax5.get_ylim()
for i in range(len(data5)):
    high, low = np.quantile(data5[i], [0.975, 0.025])
    ax5.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i],2):.2f}\nCI: [{round(low,2):.2f}, {round(high,2):.2f}]",
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=7)

ax6 = fig.add_subplot(3, 2, 6)
ax6.boxplot(data6)
# Add labels and title
ax6.set_xticklabels(['0.3','0.5','1','2', '5','10'])
ax6.set_xlabel('Population size (million)')
ax6.set_ylabel('Percentage of cases')
ax6.set_title('Sexual contact ratio',y=1.18)
#set ax6 ticks labels to be  %
ax6.set_yticks(np.arange(0.00, 1.01, step=0.2))
#change the ticks to be %
ax6.set_yticklabels(['{:,.0%}'.format(x) for x in np.arange(0.00, 1.01, step=0.2)])
ax6.set_ylim(0,1.05)
#ax.set_title('Box plot of 5 data series')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.axhline(y=0.829,c='b',linestyle='--')
ax6.text(4.5,0.65,'Global sexual contact ratio: 0.829',c='b')

# Calculate and plot the mean, median, and 95% confidence intervals
means = [np.mean(d) for d in data6]
medians = [np.median(d) for d in data6]
y_min, y_max = ax6.get_ylim()
for i in range(len(data6)):
    high, low = np.quantile(data6[i], [0.975, 0.025])
    ax6.text(i + 1, y_max + (y_max - y_min) * 0.11, f"Median: {round(medians[i],2):.2f}\nCI: [{round(low,2):.2f}, {round(high,2):.2f}]",
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=7)

# Adjust the spacing between subplots
font_size=20
plt.subplots_adjust(wspace=0.2, hspace=0.6)
tx1, ty1 = 0.07, 0.93
tx2, ty2 = 0.51, 0.62
ty3 = 0.31
plt.figtext(tx1, ty1, 'a', fontsize=font_size)
plt.figtext(tx2, ty1, 'b', fontsize=font_size)
plt.figtext(tx1, ty2, 'c', fontsize=font_size)
plt.figtext(tx2, ty2, 'd', fontsize=font_size)
plt.figtext(tx1, ty3, 'e', fontsize=font_size)
plt.figtext(tx2, ty3, 'f', fontsize=font_size)


# Show the plot
plt.show()
fig_path = 'analysis of 5 scenarias.svg'
fig.savefig(fig_path, dpi=1000)
