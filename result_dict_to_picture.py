###merge it into the
import copy

####plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import covasim as cv
import sciris as sc
import pylab as pl
import pandas as pd
import datetime as dt
import matplotlib.ticker as ticker
import statistics

##load the data
import pickle
with open('result_dict_10M.pkl', 'rb') as handle:
    result_dict=pickle.load(handle)




####central function
day_stride = 21
week_stride = 3
def format_ax(ax, key=None):
    ''' Format the axes nicely '''
    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (dt.date(2022, 5, 1)+ dt.timedelta(days=x)).strftime('%b-%d')

    ax.xaxis.set_major_formatter(date_formatter)
    if key != 'r_eff':
        sc.commaticks()
    pl.xlim([14, 240])
    sc.boxoff()
    return

def format_ax_w(ax, key=None):
    ''' Format the axes nicely '''

    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (dt.date(2022, 5, 1) + dt.timedelta(weeks=x)).strftime('%b-%d')

    ax.xaxis.set_major_formatter(date_formatter)
    if key != 'r_eff':
        sc.commaticks()
    pl.xlim([2, 33])
    sc.boxoff()
    return


def plotter(key, ax, ys=None, calib=False, label='', ylabel='', low_q=0.025, high_q=0.975, flabel=True,subsample=2,stride=day_stride):
    ''' Plot a single time series with uncertainty '''


    if key == 'dayn_all' or key =='weekn_m':
        color = '#e33d3e'
        #print(1)
    elif key == 'cum_all' or key =='weekn_f':
        color = '#e6e600'
        #print(2)
    elif key == 'dayn_he' or key =="cum_he" or key == 'dayn_household' or key == 'weekn_household'or key=='cum_household' or key =="weekn_m_m" or key =="weekn_m_he" or key =="weekn_m_he_n" or key=='weekn_f_he' or key=="weekn_f_he_n" or key=="weekn_he" or key=="weekn_he_n":
        color= '#58D68D'
        #print(3)
    elif key == 'dayn_ho' or key =='cum_ho' or key =="weekn_m_f" or key =="weekn_m_ho" or key =="weekn_m_ho_n" or key=='weekn_f_ho' or key=="weekn_f_ho_n" or key=="weekn_ho" or key=="weekn_ho_n":
        color = '#FFFF00'
        #print(4)
    elif key == 'dayn_bi' or key =='cum_bi' or key =="weekn_f_m" or key =="weekn_m_bi" or key =="weekn_m_bi_n"or key=='weekn_f_bi' or key=="weekn_f_bi_n" or key=="weekn_bi" or key=="weekn_bi_n":
        color = '#FFA500'
    elif key == 'dayn_w' or key =='cum_w' or key =="weekn_f_f":
        color = '#CD853F'
    elif key == 'dayn_h' or key =='cum_h':
        color= '#40E0D0'
    elif key == 'dayn_s' or key =='cum_s':
        color = '#4169E1'
    elif key == 'dayn_c' or key =='cum_c':
        color = '#EE82EE'
    elif key == 'dayn_party' or key =='cum_party':
        color = '#DC143C'
    elif key == 'dayn_party_cc' or key =='cum_party_cc':
        color= '#FF1493'


    elif key == 'sex_ratio_week' or key =="MSM_ratio_week":
        color = '#000000'
    elif key == 'dayn_he':
        color = '#ff8000'
    elif key == 'dayn_ho':
        color = '#1f1f1f'

    # if key == 'new_infections' or 'cum_infections':
    #     color= '#e33d3e'
    # elif key == 'new_severe' or 'n_severe' or 'cum_severe':
    #     color= '#e6e600'
    # elif key == 'new_critical' or 'n_critical' or 'cum_critical':
    #     color= '#ff8000'
    # elif key == 'new_deaths' or 'cum_deaths':
    #     color = '#1f1f1f'
    if ys is None:
        ys = []
        for i in range(500):
            ys.append(result_dict[i][key])
    #print('ys=', ys)
    yarr = np.array(ys)

    best = pl.nanmedian(yarr, axis=0)  # Changed from median to mean for smoother plots
    low = pl.nanquantile(yarr, q=low_q, axis=0)
    high = pl.nanquantile(yarr, q=high_q, axis=0)
    global week
    #global
    # if key == 'weekn_m' :
    #     best_weekn_m=copy.deepcopy(best)
    # if key=='weekn_f':
    #     best_weekn_f=copy.deepcopy(best)

    # if key =="MSM_ratio_week" or key =="Male ratio" or key=="sex_ratio_week" or key=="weekn_m_he_n" or key=="weekn_m_ho_n" or key=="weekn_m_bi_n":
    #     valid = np.asarray([30 for i in range(len(best))])
    #     base = [[] for i in range(len(best))]
    #     base_all = np.asarray([0 for i in range(len(best))])
    #     for yarr_one in yarr:
    #         for ind,pos in enumerate(yarr_one):
    #             if np.isnan(pos)==True:
    #                 valid[ind]-=1
    #             else:
    #                 base[ind].append(pos)
    #
    #     for ind in range(len(yarr_one)):
    #         base_all[ind]=statistics.median(base[ind])



        # best=base_all
        # low =base_all
        # high = base_all
        # global yarr_test
        # yarr_test=yarr
        # print(best)

    tvec = np.arange(len(best))

    #data, data_t = None, None
    #if key in sim.data:
        #data_t = np.array((sim.data.index - sim['start_day']) / np.timedelta64(1, 'D'))
        #inds = np.arange(0, len(data_t), subsample)
        #data = sim.data[key][inds]
        #pl.plot(data_t[inds], data, 'd', c=color, markersize=10, alpha=0.5, label='Data')

    end = None
    if flabel:
        fill_label = None
        # if key == 'infections':
        #     fill_label = '95% CI'
        # else:
        #     fill_label = '95% CI'
    else:
        fill_label = None

    # Trim the beginning for r_eff and actually plot
    start = 2 if key == 'r_eff' else 0

    pl.fill_between(tvec[start:end], low[start:end], high[start:end], facecolor=color, alpha=0.2, label=fill_label)
    print('color is',color,'for', key)
    pl.plot(tvec[start:end], best[start:end], c=color, label=label, lw=4, alpha=1.0)

    #sc.setylim()
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(xmin, xmax, stride))

    #ax.set_yticks(np.arange(ymin, ymax))
    pl.ylabel(ylabel)

    plotres[key] = sc.objdict(dict(tvec=tvec, best=best, low=low, high=high))

    return  best,high,low

pl.rcParams['font.size'] = 15
plotres = sc.objdict()
fig_path = 'fig1.svg'
datafiles=sc.objdict()
plotres=sc.objdict()


fig = pl.figure(num='Mpox simulation', figsize=(30, 20))
tx1, ty1 = 0.05, 0.94
tx2, ty2 = 0.53, 0.79
ty3 = 0.64
ty4 = 0.49
ty5 = 0.34
ty6 = 0.19
font_size = 30


low_q = 0.025
high_q = 0.975


##data from WHO
calib_date=[11,12,13,14,15,16,17,18,20,21,23,24,29]
calib_day=[83,89,108,135,142,149,169,177,213,239]
Day_data=[89,161,359,750,799,839,892,905,924,943]
Ho_data=[0.806847545,0.801846154,0.795400943,0.794334278,0.789210234,0.777717976]
Bi_data=[0.127906977,0.128,0.129716981,0.131444759,0.130700779,0.130247578]
He_data=[0.020979021,0.024729521,0.030857143,0.041512915,0.046583851,0.065245478,0.070153846,0.074882075,0.074220963,0.080088988,0.092034446]
Sex_data=[1,1,1,1,0.988349515,0.987012987,0.986398964,0.979681494,0.977307891,0.975634013,0.975143403,0.974166275,0.968650613]
Case_data=[0]*18+[1,0,0,0,0,0,0,0,0,0,0,0,2,0,1,1,0,0,0,1,2,1,4,0,1,6,3,1,2,4,1,1,1,2,3,2,2,6,4,9,3,1,7,4,7,6,6,8,11,10,11,12,11,29,11,17,17,20,20,20,30,28,29,24,29,25,20,39,28,35,26,30,43,26,58,31,41,36,36,36,36,42,37,48,25,37,36,35,53,32,29,30,33,41,31,33,25,16,30,30,23,23,25,23,22,18,19,19,18,18,15,19,26,15,15,10,20,16,9,17,16,12,6,11,10,9,8,8,2,6,8,10,7,3,5,4,4,6,4,7,5,7,2,4,3,3,6,2,3,6,1,7,2,3,4,3,3,3,2,7,1,3,1,0,4,3,1,1,1,1,1,0,6,0,1,1,0,0,1,1,2,1,0,1,1,1,0,0,1,1,1,0,0,4,3,0,3,2,2,0,2,0,1,0,2,0,0,2,1,1,1,0,1,0,1,1,0,0,0,0,0,3,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,2,0,1,0,3,0,4,0,2,1,0,1,0,0,0,2,2,2,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,2,1,0,0,0,3,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0]
Cum_Case_data=np.cumsum(Case_data)
##plot a to j
pl.figtext(tx1, ty1, 'a', fontsize=font_size)
pl.figtext(tx2, ty1, 'b', fontsize=font_size)
pl.figtext(tx1, ty2, 'c', fontsize=font_size)
pl.figtext(tx2, ty2, 'd', fontsize=font_size)
pl.figtext(tx1, ty3, 'e', fontsize=font_size)
pl.figtext(tx2, ty3, 'f', fontsize=font_size)
pl.figtext(tx1, ty4, 'g', fontsize=font_size)
pl.figtext(tx2, ty4, 'h', fontsize=font_size)
pl.figtext(tx1, ty5, 'i', fontsize=font_size)
pl.figtext(tx2, ty5, 'j', fontsize=font_size)
pl.figtext(tx1, ty6, 'k', fontsize=font_size)
pl.figtext(tx2, ty6, 'l', fontsize=font_size)

##Fig a: Daily infections
x0, y0, dx, dy = 0.08, 0.81, 0.41, 0.13
ax1 = pl.axes([x0, y0, dx, dy])
format_ax(ax1)
best_daily,high_daily,low_daily=plotter('dayn_all', ax1, calib=True, label='Model', ylabel='Daily infections')
#pl.plot(range(len(Case_data)),Case_data,alpha=1,marker="o",label="Data",color='blue')
pl.ylim([0, 50])
pl.legend(loc='upper right', frameon=False)


# Fig b: cum infections:
x02 = 0.57
ax2 = pl.axes([x02, y0, dx, dy])
format_ax(ax2)
best_cuminf,high_cuminf,low_cuminf=plotter('cum_all',  ax2, calib=True, label='Model', ylabel='Cumulative infections')
pl.plot(range(len(Case_data)),Cum_Case_data,alpha=1,marker="o",label="Data",color='blue')
pl.ylim([0, 2500])
pl.legend(loc='upper left', frameon=False)



##fig C:Layer cases
y02 = 0.66
ax3 = pl.axes([x0, y02, dx, dy])
format_ax(ax3)
best_dayn_party,high_dayn_party,low_dayn_party=  plotter('dayn_party',  ax3, calib=True, label='Party with sexual contact', ylabel='Daily infections')
best_dayn_party_cc,high_dayn_party_cc,low_dayn_party_cc=plotter('dayn_party_cc',  ax3, calib=True, label='Party with no sexual contact', ylabel='Daily infections')
best_dayn_h,high_dayn_h,low_dayn_h=plotter('dayn_household',  ax3, calib=True, label='Household', ylabel='Daily infections')
best_dayn_w,high_dayn_w,low_dayn_w=plotter('dayn_w',  ax3, calib=True, label='Workplace', ylabel='Daily infections')
best_dayn_s,high_dayn_s,low_dayn_s=plotter('dayn_s',  ax3, calib=True, label='School', ylabel='Daily infections')
best_dayn_c,high_dayn_c,low_dayn_c=plotter('dayn_c',  ax3, calib=True, label='Community', ylabel='Daily infections')

pl.ylim([0, 25])
pl.legend(loc='upper right', frameon=False)


###fig D: accumulative layer cases with the transmission mode
ax4 = pl.axes([x02, y02, dx, dy])
format_ax(ax4)
best_cum_party,high_cum_party,low_cum_party=plotter('cum_party',  ax4, calib=True, label='Party with sexual contact', ylabel='Cummulative infections')
best_cum_party_cc,high_cum_party_cc,low_cum_party_cc=plotter('cum_party_cc',  ax4, calib=True, label='Party with no sexual contact', ylabel='Cummulative infections')
best_cum_household,high_cum_household,low_cum_household=plotter('cum_household',  ax4, calib=True, label='Household', ylabel='Cummulative infections')
best_cum_w,high_cum_w,low_cum_w=plotter('cum_w',  ax4, calib=True, label='Workplace', ylabel='Cummulative infections')
best_cum_s,high_cum_s,low_cum_s=plotter('cum_s',  ax4, calib=True, label='School', ylabel='Cummulative infections')
best_cum_c,high_cum_c,low_cum_c=plotter('cum_c',  ax4, calib=True, label='Community', ylabel='Cummulative infections')
pl.ylim([0, 2000])
pl.legend(loc='upper left', frameon=False)


###fig E: Male and female cases weekly
y03 = 0.51
ax5 = pl.axes([x0, y03, dx, dy])
format_ax_w(ax5)
pl.ylim([0, 200])
best_weekn_m,high_weekn_m,low_weekn_m=plotter('weekn_m',  ax5, calib=True, label='Male', ylabel='Weekly infections',stride=week_stride)
best_weekn_f,high_weekn_f,low_weekn_f=plotter('weekn_f',  ax5, calib=True, label='Female', ylabel='Weekly infections',stride=week_stride)
pl.legend(loc='upper right', frameon=False)



##fig:F: weekly male ratios(with intervals)
ax6 = pl.axes([x02, y03, dx, dy])
format_ax_w(ax6)
best_sex_ratio_week,high_sex_ratio_week,low_sex_ratio_week=plotter('sex_ratio_week',  ax6, calib=True, label='Male ratio', ylabel='Percentage of Cases',stride=week_stride)
pl.plot(calib_date,Sex_data,alpha=1,marker="o",label="Data",color='black')
pl.yticks(np.arange(0, 1.1, step=0.2),[str(i)+'%' for i in np.arange(0,110,step=20)])
pl.ylim([0, 1])
pl.legend(loc='lower right', frameon=False)

# fig G:Homosexual bisexual heterosexual in all genders
y04 = 0.36
ax7 = pl.axes([x0, y04, dx, dy])
format_ax_w(ax7)
best_weekn_ho,high_weekn_ho,low_weekn_ho=plotter('weekn_ho', ax7, calib=True, label='Homosexual', ylabel='Weekly infections',stride=week_stride)
best_weekn_bi,high_weekn_bi,low_weekn_bi=plotter('weekn_bi', ax7, calib=True, label='Bisexual', ylabel='Weekly infections',stride=week_stride)
best_weekn_he,high_weekn_he,low_weekn_he=plotter('weekn_he', ax7, calib=True, label='Heterosexual', ylabel='Weekly infections',stride=week_stride)

pl.ylim([0, 200])
pl.legend(loc='upper right', frameon=False)

# fig H: Homosexual bisexual heterosexual percentage in all genders
ax8 = pl.axes([x02, y04, dx, dy])
format_ax_w(ax8)

best_weekn_ho_n,high_weekn_ho_n,low_weekn_ho_n=plotter('weekn_ho_n', ax8, calib=True, label='Homosexual', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_bi_n,high_weekn_bi_n,low_weekn_bi_n=plotter('weekn_bi_n', ax8, calib=True, label='Bisexual', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_he_n,high_weekn_he_n,low_weekn_he_n=plotter('weekn_he_n', ax8, calib=True, label='Heterosexual', ylabel='Percentage of Cases',stride=week_stride)
pl.plot(calib_date[-len(Ho_data):],Ho_data,alpha=1, marker='o',label='Homosexual Data',color='#FFFF00')
pl.plot(calib_date[-len(Bi_data):],Bi_data,alpha=1, marker='o',label='Bisexual Data',color='#D68910')
pl.plot(calib_date[-len(He_data):],He_data,alpha=1, marker='o',label='Heterosexual Data',color='#2ECC71')
pl.yticks(np.arange(0, 1.1, step=0.2),[str(i)+'%' for i in np.arange(0,110,step=20)])
pl.ylim([0, 1])
pl.legend(loc='center right', frameon=False)

# fig I:Sex orientation in males with the current data
y05 = 0.21
ax9 = pl.axes([x0, y05, dx, dy])
format_ax_w(ax9)
best_weekn_m_ho,high_weekn_m_ho,low_weekn_m_ho=plotter('weekn_m_ho', ax9, calib=True, label='Homosexual in males', ylabel='Weekly infections',stride=week_stride)
best_weekn_m_bi,high_weekn_m_bi,low_weekn_m_bi=plotter('weekn_m_bi', ax9, calib=True, label='Bisexual in males', ylabel='Weekly infections',stride=week_stride)
best_weekn_m_he,high_weekn_m_he,low_weekn_m_he=plotter('weekn_m_he',ax9, calib=True, label='Heterosexual in males', ylabel='Weekly infections',stride=week_stride)
pl.legend(loc='upper right', frameon=False)

# fig J:Sex orientation in males with the current data
ax10 = pl.axes([x02, y05, dx, dy])
format_ax_w(ax10)
best_weekn_m_ho_n,high_weekn_m_ho_n,low_weekn_m_ho_n=plotter('weekn_m_ho_n', ax10, calib=True, label='Homosexual in males', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_m_bi_n,high_weekn_m_bi_n,low_weekn_m_bi_n=plotter('weekn_m_bi_n', ax10, calib=True, label='Bisexual in males', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_m_he_n,high_weekn_m_he_n,low_weekn_m_he_n=plotter('weekn_m_he_n', ax10, calib=True, label='Heterosexual in males', ylabel='Percentage of Cases',stride=week_stride)
pl.yticks(np.arange(0, 1.1, step=0.2),[str(i)+'%' for i in np.arange(0,110,step=20)])
pl.ylim([0, 1])
pl.legend(loc='center right', frameon=False)


##fig k:Sex orientation in females with the current data
y06 = 0.06
ax11 = pl.axes([x0, y06, dx, dy])
format_ax_w(ax11)
best_weekn_f_ho,high_weekn_f_ho,low_weekn_f_ho=plotter('weekn_f_ho', ax11, calib=True, label='Homosexual in females', ylabel='Weekly infections',stride=week_stride)
best_weekn_f_bi,high_weekn_f_bi,low_weekn_f_bi=plotter('weekn_f_bi', ax11, calib=True, label='Bisexual in females', ylabel='Weekly infections',stride=week_stride)
best_weekn_f_he,high_weekn_f_he,low_weekn_f_he=plotter('weekn_f_he',ax11, calib=True, label='Heterosexual in females', ylabel='Weekly infections',stride=week_stride)
pl.legend(loc='upper right', frameon=False)

##fig l:sex orientation in females
ax12 = pl.axes([x02, y06, dx, dy])
format_ax_w(ax12)
best_weekn_f_ho_n,high_weekn_f_ho_n,low_weekn_f_ho_n=plotter('weekn_f_ho_n', ax12, calib=True, label='Homosexual in females', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_f_bi_n,high_weekn_f_bi_n,low_weekn_f_bi_n=plotter('weekn_f_bi_n', ax12, calib=True, label='Bisexual in females', ylabel='Percentage of Cases',stride=week_stride)
best_weekn_f_he_n,high_weekn_f_he_n,low_weekn_f_he_n=plotter('weekn_f_he_n', ax12, calib=True, label='Heterosexual in females', ylabel='Percentage of Cases',stride=week_stride)
pl.yticks(np.arange(0, 1.1, step=0.2),[str(i)+'%' for i in np.arange(0,110,step=20)])
pl.ylim([0, 1])
pl.legend(loc='center right', frameon=False)


###
do_plot = 1
do_save = 1



if do_save:
    cv.savefig(fig_path, dpi=1000)

if do_plot:
    pl.show()

