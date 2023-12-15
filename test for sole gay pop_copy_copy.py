# install some basic packages including numpy,numba,pandas,scipy,statsmodels,matplotlib,sciris>=1.3.3

import numpy as np
import matplotlib.pyplot as plt
import covasim as cv
import sciris as sc
import pandas as pd
import random
import copy
import pickle



time_work,time_school,time_community,time_home, time_party,time_party_cc,time_he,time_he_step1,time_bi,time_ho=0,0,0,0,0,0,0,0,0,0


# test items
test_para = [1]  # 测试哪些内容，1，2，3，4，5,6
test_acu = 1  # *2 2 4 8 16  32
num_inf = 20##  # number of infections at the beggining

total_pop =2e6 # 3.29e6
factor = test_acu
group = 'ho_m'  # [ho:'homo',bi:'bisexual',he:'hetero',n:'normal']
change_relation=True
relation_factor=0.05##0.01
contact_ratio_school,contact_ratio_workplace,contact_ratio_community=0.3,0.5,1.1#adjusted value
##input para:
do_plot = 1  # 自动作图
do_save = 1  # 自动保存
save_sim = 1  # 是否存sim
plot_hist = 0  # whether to keep people
# paras of plotting
adj_fac=0.6

# actual_data = np.array([0, 0, 0, 0, 0, 0, 2.142857075, 2.428571463, 2, 2.428571463, 2.428571463,
#     2.571428537, 2.428571463, 1.857142806, 2, 2, 1.857142806, 2.428571463,
#     2.285714388, 3, 3.714285612, 3.857142925, 4.857142925, 6.428571224,
#     6.857142925, 7.571428776, 7.571428776, 9.285714149, 11.28571415,
#     11.71428585, 10.71428585, 13.14285755, 14, 16.7142849, 17.4285717,
#     18.4285717, 21.2857151, 24.1428566, 25.4285717, 27.5714283, 29.1428566,
#     31.5714283
# ])
#random select from postion of actual_data and follow the distribution of actual_data
# seed_position=np.random.choice(list(range(len(actual_data))),size=num_inf,p=actual_data/np.sum(actual_data))
# seed_position_order=[np.sum(seed_position==i) for i in range(len(actual_data))]
pop_size = int(total_pop / factor)
pop_scale = factor
pop_type = 'hybrid'
beta = 0.007 # previous value: 0.016  1/138=0.007246  calibration number for female ratio to be 3.56% for 0.015
verbose = 0
seed = 0  ##
asymp_factor = 0  # multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility
n_beds_hosp, n_beds_icu = 200, 20
party_num = 50

to_plot = sc.objdict({
    'Daily infections': ['new_infections'],
    'Cumulative infections': ['cum_infections'],
    'Daily hospitalisations': ['new_severe'],
    'Occupancy of hospitalisations': ['n_severe'],  # 'Cumulative hospitalisations': ['cum_severe']
    'Daily ICUs': ['new_critical'],
    'Occupancy of ICUs': ['n_critical'],
    'Cumulative hospitalisations': ['cum_severe'],
    'Cumulative ICUs': ['cum_critical'],
    # 'Cumulative ICUs': ['cum_critical'],
    # 'Daily quarantined':['new_quarantined'],
    # 'Number in quarantined':['n_quarantined'],
    'Daily deaths': ['new_deaths'],
    'Cumulative deaths': ['cum_deaths'],
    # 'R': ['r_eff'],
    # 'number vaccinated': ['n_vaccinated'],
    # 'proportion vaccinated': ['frac_vaccinated'],
    # 'Vaccinations ': ['cum_vaccinated'],
})
# init_time = ['2022-08-1' + str(x) for x in range(5, 10)] + ['2022-08-2' + str(x) for x in range(0, 10)]
# trace_time = init_time + ['2022-08-30', '2022-08-31', '2022-09-08',
#                           '2022-09-15', '2022-09-30', '2022-10-15', '2022-11-01', '2022-11-15', '2022-12-01',
#                           '2022-12-15', '2023-01-15']  # trace the age-distribution data

trace_state = ['infectious', 'severe', 'critical', 'dead']  # 文章中数据分析里面


def protect_reinfection(sim):
    sim.people.rel_sus[sim.people.recovered] = 0.0

vac_1_data=[0,0,0,0.00000,0.00000,0.00001,0.00001,0.00001,0.00012,0.00019,0.00046,0.00137,0.00317,0.00484,0.00719,0.00919,0.00742,0.00638,0.00496,0.00325,0.00316,0.00238,0.00190,0.00147,0.00118,0.00103,0.00081,0.00067,0.00056,0.00050,0.00025,0.00038,0.00029,0.00022,0.00000,0.00000]
vac_2_data=[0,0,0,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00001,0.00001,0.00005,0.00007,0.00012,0.00041,0.00109,0.00222,0.00326,0.00434,0.00502,0.00539,0.00393,0.00274,0.00216,0.00170,0.00133,0.00100,0.00087,0.00081,0.00037,0.00061,0.00043,0.00031,0.00000,0.00000]

def risk_target_vac(sim):
    if sim.t <= 56 or sim.t >= 244:
        return

    n_vac_1 = len(total_risk_pop_m) * vac_1_data[int(sim.t / 7)] / 0.915
    n_vac_2 = len(total_risk_pop_m) * vac_2_data[int(sim.t / 7)] / 0.94

    risk_target_m_1 = total_risk_pop_m[sim.people.rel_sus[total_risk_pop_m] == 1]
    risk_target_f_1 = total_risk_pop_f[sim.people.rel_sus[total_risk_pop_f] == 1]

    vac_m_1 = np.random.choice(risk_target_m_1, int(n_vac_1 * 0.915), False)
    vac_f_1 = np.random.choice(risk_target_f_1, int(n_vac_1 * 0.085), False)

    Vac_risk_pop_m[np.isin(total_risk_pop_m, vac_m_1)] = sim.t
    Vac_risk_pop_f[np.isin(total_risk_pop_f, vac_f_1)] = sim.t

    sim.people.rel_sus[vac_m_1] = 0.63
    sim.people.rel_sus[vac_f_1] = 0.63

    if sim.t <= 84:
        return

    risk_target_m_2 = total_risk_pop_m[Vac_risk_pop_m <= (sim.t - 28)]
    risk_target_f_2 = total_risk_pop_f[Vac_risk_pop_f <= (sim.t - 28)]

    vac_m_2 = np.random.choice(risk_target_m_2, int(n_vac_2 * 0.94), False)
    vac_f_2 = np.random.choice(risk_target_f_2, int(n_vac_2 * 0.06), False)

    Vac_risk_pop_m[np.isin(total_risk_pop_m, vac_m_2)] = 255
    Vac_risk_pop_f[np.isin(total_risk_pop_f, vac_f_2)] = 255

    sim.people.rel_sus[vac_m_2] = 0.31
    sim.people.rel_sus[vac_f_2] = 0.31







pars = sc.objdict(
    use_waning=False,
    pop_size=pop_size,
    pop_infected=0,
    pop_scale=pop_scale,
    pop_type=pop_type,
    beta=beta,
    asymp_factor=asymp_factor,
    rescale=True,
    rand_seed=seed,
    verbose=verbose,
    nab_boost=6,
    quar_period=21,
    start_day='2022-05-01',
    n_days = 244,
#+181,##152 to end of september  ##244 to end of this year 180 is the half year

    dur={'exp2inf': {'dist': 'lognormal_int', 'par1': 7.6, 'par2': 1.8},
         # https://www.medrxiv.org/content/10.1101/2022.06.22.22276713v1  For time from exposure to first symptom onset, we estimated a meanincubation period of 7.6days (95% credible interval(CrI): 6.2–9.7) (median 6.4, 95% CrI: 5.1 –7.9)and a standard deviation of 1.8days (95% CrI: 1.6–2.2)(Figure 1)
         'inf2sym': {'dist': 'uniform', 'par1': 0.0, 'par2': 5.0},
         # https://www.who.int/news-room/fact-sheets/detail/monkeypox  the invasion period (lasts between 0–5 days)
         'sym2sev': {'dist': 'uniform', 'par1': 1.0, 'par2': 2.0},
         # https://www.cdc.gov/poxvirus/monkeypox/clinicians/clinical-recognition.html each stage start 1-2 days
         'sev2crit': {'dist': 'uniform', 'par1': 3.0, 'par2': 7.0},  # assumption
         'mild2rec': {'dist': 'uniform', 'par1': 14.0, 'par2': 28.0},
         # https://www.who.int/news-room/fact-sheets/detail/monkeypox  The illness typically lasts 2-4 weeks.
         'asym2rec': {'dist': 'uniform', 'par1': 14.0, 'par2': 28.0},
         # https://www.who.int/news-room/fact-sheets/detail/monkeypox  The illness typically lasts 2-4 weeks.
         'sev2rec': {'dist': 'uniform', 'par1': 14.0, 'par2': 28.0},
         # https://www.who.int/news-room/fact-sheets/detail/monkeypox   The illness typically lasts 2-4 weeks.
         'crit2rec': {'dist': 'uniform', 'par1': 14.0, 'par2': 28.0},
         # https://www.who.int/news-room/fact-sheets/detail/monkeypox  The illness typically lasts 2-4 weeks.
         'crit2die': {'dist': 'uniform', 'par1': 1.0, 'par2': 2.0}},  # assumption


    ##severity parameters
    n_beds_hosp=int(n_beds_hosp / factor),
    n_beds_icu=int(n_beds_icu / factor),
    no_hosp_factor=2,
    no_icu_factor=2,  # without enough beds critically cases will 2 times likely to die
    contacts={'h':0.5,'s': 0, 'w':0, 'c': 0, 'homo': 0, 'bi': 0, 'hetero': 0,'party':0,'party_cc':0},#'h':0,
    # Number of contacts per person per day -- 'a' for 'all'
    #contacts={'h': 1.85 / 30, 's': 0.5 / 30, 'w': 0.5 / 30, 'c': 0.5 / 30, 'homo': 0, 'bi': 0, 'hetero': 0,'party':0},
    beta_layer={'h':0,'s': 1, 'w': 1, 'c': 1, 'homo': 1, 'bi': 1, 'hetero': 1,'party':1,'party_cc':1},#'h':0,
    # Per-population beta weights; relative
    dynam_layer=dict(h=0,s=1, w=1, c=1, homo=1, bi=1, hetero=1,party=1,party_cc=1),  # Which layers are dynamic -- none by default  h=0,
    # todo change it
    iso_factor=dict(h=0,s=0, w=0, c=0, homo=0, bi=0, hetero=0,party=0,party_cc=0),  #h=0,
    # Multiply beta by this factor for people in isolation
    quar_factor=dict(h=0,s=0, w=0, c=0, homo=0, bi=0, hetero=0,party=0,party_cc=0),  #h=0,
)
# test prob of symp_prob will test in two days,asymp_prob will test in seven days,wait for 3 days to know the result because conditional HK policy
# Define the testing and contact tracing interventions  contact tracing is stopped
tp_1 = cv.test_prob(symp_prob=0.1, asymp_prob=0, test_delay=9,do_plot=False,start_day='2022-05-16',end_day='2022-05-22')##week20 13-9=4
tp_2 = cv.test_prob(symp_prob=0.09, asymp_prob=0, test_delay=2,do_plot=False,start_day='2022-05-23',end_day='2022-06-30')##week21-23  7-3=5
tp_3 = cv.test_prob(symp_prob=0.1, asymp_prob=0,  test_delay=2,do_plot=False,start_day='2022-07-01')##week27  6-2=4
#
ct_1 = cv.contact_tracing(trace_probs=dict(h=1,s=0.1, w=0.1, c=0.05,homo=0.2, bi=0.2, hetero=0.2,party=0.1,party_cc=0.1),  quar_period=7,
                               do_plot=False, start_day='2022-06-01', end_day='2022-06-30')
ct_2 = cv.contact_tracing(trace_probs=dict(h=1,s=0.2, w=0.2, c=0.2,homo=0.5, bi=0.5, hetero=0.5,party=0.2,party_cc=0.2), quar_period=7,
                               do_plot=False, start_day='2022-07-01')


limit_sex = cv.change_beta(days=[20,45,70,90,110], changes=[0.95,0.90,0.85,0.80,0.75], layers=['homo','bi','hetero','party','party_cc'],
                                   do_plot=False)  # eliminate the sexual contacts  https://www.cdc.gov/mmwr/volumes/71/wr/mm7135e1.htm#T2_down

sim = cv.Sim(pars=pars, location='USA-California',
             # china, hong kong special administrative region',United States of America   USA-California
             # analyzers=[cv.age_histogram(days=trace_time, states=trace_state),
             # cv.daily_age_stats(states=trace_state)]
             interventions=[tp_1,tp_2,tp_3,ct_1,ct_2,risk_target_vac,
                            limit_sex,protect_reinfection])  # use age-distribution data

# params of the virus
rel_beta = 1.1  ##1.1  #Relative transmissibility varies by variant compared to wild type COVID-19 virus  2.43/2.2=1.1  #https://pubmed.ncbi.nlm.nih.gov/35994726/#:~:text=We%20analyzed%20the%20first%20255,%25%20CI%201.82%2D3.26).
rel_symp_prob = 0.831  # About 83.1% of the cases will have rash
rel_severe_prob = 0.035  ##about 2.9% of infected cases will developed severe symptoms  2.9/83.1=0.035
rel_crit_prob = 0.069  ##0.2% of  infected cases will develop critically ill symptoms 0.2/2.9=0.069
rel_death_prob = 0.2  ##The death rate is estimated to be 0.04%(28 confirmed death out of 72,198 cases) 0.04/0.2=0.2

Mpox = cv.variant(
    variant={'rel_beta': rel_beta, 'rel_symp_prob': rel_symp_prob, 'rel_severe_prob': rel_severe_prob,
             'rel_crit_prob': rel_crit_prob, 'rel_death_prob': rel_death_prob}, label='Mpox',
    days=sim.day('2022-05-10'), n_imports=0)  # np.array([16,15,125,98,97])





print(f'variant Mpox is {Mpox}')
sim['variants'] += [Mpox]
print(sim['variants'])
print(f'variant Mpox is {Mpox}')
##add layer###########
# Define the new layer, 'Sex_party Gay'

print(sim)
global infected_uids

age_ranges_common_male= {25: (3, 2), 35: (5, 3), 45: (8, 4), np.inf: (10, 5)}
age_ranges_common_male_ho={25: (3, 3), 35: (5, 5), 45: (8, 8), np.inf: (10, 10)}
age_ranges_common_female={25: (1, 8), 35: (2, 8), 45: (4, 10), np.inf: (6, 12)}
age_ranges_party_male={25: (5, 3), 35: (8, 4), 45: (10, 5), np.inf: (12, 7)}
age_ranges_party_male_ho={25: (5, 5), 35: (8, 8), 45: (10, 10), np.inf: (15, 15)}
age_ranges_party_female={25: (2, 8), 35: (4, 10), 45: (6, 12), np.inf: (8, 14)}


def sim_age(one_ind, search_group, bi_ho=False):
    """
    get the similar age group
    """
    one_age = sim.people.age[one_ind]
    one_sex = sim.people.sex[one_ind]

    # Create mapping of ages to search ranges
    age_ranges = age_ranges_common_male
    if (one_ind in Pop_ho_set) or bi_ho:
        age_ranges = age_ranges_common_male_ho
    elif one_sex == 0:  # Female
        age_ranges = age_ranges_common_female

    # Find applicable age range
    for max_age, (lower_diff, upper_diff) in age_ranges.items():
        if one_age < max_age:
            lower, upper = one_age - lower_diff, one_age + upper_diff
            break

    # Get indices of individuals in age range
    search_group_age = sim.people.age[search_group]
    join_index = (search_group_age>= lower) & (search_group_age <= upper)

    while not join_index.any():
        lower -= 1;
        upper += 1
        join_index = (search_group_age >= lower) & (search_group_age <= upper)

    return sim.people.uid[search_group][join_index]


def sim_age_party(one_ind, search_group, bi_ho=False):
    """
    get the similar age group
    """
    one_age = sim.people.age[one_ind]
    one_sex = sim.people.sex[one_ind]

    # Create mapping of ages to search ranges
    age_ranges = age_ranges_party_male
    if (one_ind in Pop_ho_set) or bi_ho:
        age_ranges = age_ranges_party_male_ho
    elif one_sex == 0:  # Female
        age_ranges = age_ranges_party_female

    # Find applicable age range
    for max_age, (lower_diff, upper_diff) in age_ranges.items():
        if one_age < max_age:
            lower, upper = one_age - lower_diff, one_age + upper_diff
            break

    # Get indices of individuals in age range
    search_group_age = sim.people.age[search_group]
    join_index = (search_group_age >= lower) & (search_group_age <= upper)

    while not join_index.any():
        lower -= 3;
        upper += 3
        join_index = (search_group_age>= lower) & (search_group_age <= upper)

    return sim.people.uid[search_group][join_index]


def lower_one(arr):
    """
    #give the lower than 1 number of 1 because they have sexual partners
    """
    arr[arr < 1] = 1
    return arr


class CustomLayer_workplace(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v


    def update(self, people):


        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids=np.concatenate((infectious_uids,np.random.choice(symptomatic_uids,int(0.5*len(symptomatic_uids)),False)))
        if len(infected_uids) != 0:
            self['p1'] = np.concatenate((np.random.choice(n_23_65_m,int(n_contact_workplace*0.55)),np.random.choice(n_23_65_f,int(n_contact_workplace*0.45))),dtype=cv.default_int)
            #print(f'this is {sim.t}')
            #print(f'the infectious people are {infected_uids}\n rash people are {symptomatic_uids}   \n severe people are {severe_uids} \n')
            self['p2'] = np.concatenate((np.random.choice(n_23_65_m,int(n_contact_workplace*0.45)),np.random.choice(n_23_65_f,int(n_contact_workplace*0.55))),dtype=cv.default_int)
            ##filter contacts in the infected_uids
            infect_pos = np.in1d(self['p1'], infected_uids) | np.in1d(self['p2'], infected_uids)
            self['p1']=self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)
        #filter based on infections
        else:
            pass
        #print(f'workplace successuful in {sim.t}')

        return


class CustomLayer_school(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v

    def update(self, people):

        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids = np.concatenate(
            (infectious_uids, np.random.choice(symptomatic_uids, int(0.5 * len(symptomatic_uids)), False)))
        if len(infected_uids) != 0:
            self['p1'] = np.concatenate((
                np.random.choice(n_5_10_m, int(n_contact_kind * 0.55)),
                np.random.choice(n_5_10_f, int(n_contact_kind * 0.45)),
                np.random.choice(n_11_13_m, int(n_contact_middle * 0.55)),
                np.random.choice(n_11_13_f, int(n_contact_middle * 0.45)),
                np.random.choice(n_14_18_m, int(n_contact_high * 0.55)),
                np.random.choice(n_14_18_f, int(n_contact_high * 0.45)),
                np.random.choice(n_19_22_m, int(n_contact_university * 0.55)),
                np.random.choice(n_19_22_f, int(n_contact_university * 0.45))
            ), dtype=cv.default_int)

            self['p2'] = np.concatenate((
                np.random.choice(n_5_10_m, int(n_contact_kind * 0.45)),
                np.random.choice(n_5_10_f, int(n_contact_kind * 0.55)),
                np.random.choice(n_11_13_m, int(n_contact_middle * 0.45)),
                np.random.choice(n_11_13_f, int(n_contact_middle * 0.55)),
                np.random.choice(n_14_18_m, int(n_contact_high * 0.45)),
                np.random.choice(n_14_18_f, int(n_contact_high * 0.55)),
                np.random.choice(n_19_22_m, int(n_contact_university * 0.45)),
                np.random.choice(n_19_22_f, int(n_contact_university * 0.55))
            ), dtype=cv.default_int)

            infect_pos = np.in1d(self['p1'], infected_uids) | np.in1d(self['p2'], infected_uids)
            self['p1'] = self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)

        return


class CustomLayer_community(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v


    def update(self, people):
        global time_community

        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids=np.concatenate((infectious_uids,np.random.choice(symptomatic_uids,int(0.5*len(symptomatic_uids)),False)))
        if len(infected_uids) != 0:
            self['p1'] = np.concatenate((np.random.choice(n_m,int(n_contact_community*0.55)),np.random.choice(n_f,int(n_contact_community*0.45))),dtype=cv.default_int)
            #print(f'this is {sim.t}')
            #print(f'the infectious people are {infected_uids}\n rash people are {symptomatic_uids}   \n severe people are {severe_uids} \n')
            self['p2'] = np.concatenate((np.random.choice(n_m,int(n_contact_community*0.45)),np.random.choice(n_f,int(n_contact_community*0.55))),dtype=cv.default_int)
            #infect_p1=np.concatenate(list(map(lambda ind:np.where(self['p1']==ind)[0],infected_uids)))
            #infect_p2=np.concatenate(list(map(lambda ind:np.where(self['p2']==ind)[0],infected_uids)))
            #infect_pos=np.union1d(infect_p1, infect_p2)
            infect_pos = np.in1d(self['p1'], infected_uids) | np.in1d(self['p2'], infected_uids)
            self['p1']=self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)
            #print(f'commuity has possible infection {len(self["p1"])}')
        else:
            pass
        #print(f'community successuful in {sim.t}')


        return


# subgroup in the MSG
class CustomLayer_he(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v

    def update(self, people):
        ''' Update the contacts '''

        global relation

        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids = np.concatenate(
            (infectious_uids, np.random.choice(symptomatic_uids, int(0.5 * len(symptomatic_uids)), False)))

        relation_he=relation[sim.t]['relation_he']



        if len(infected_uids) != 0:
            n_contacts_he_half_adj_fac = int(n_contacts_he / 2 * adj_fac)
            self['p1'] = np.array(
                np.random.choice(Pop_he_p1, n_contacts_he_half_adj_fac, replace=True, p=rand_num_he_prb),
                dtype=cv.default_int)
            self['p2'] = np.array([random.choice(relation_he[ind]) for ind in self['p1']], dtype=cv.default_int)

            infect_p1 = np.isin(self['p1'], infected_uids).nonzero()[0]
            infect_p2 = np.isin(self['p2'], infected_uids).nonzero()[0]
            infect_pos = np.union1d(infect_p1, infect_p2)

            self['p1'] = self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)

            for infected_id in infected_uids:
                infected_indices_p2 = np.where(self['p2'] == infected_id)[0]
                infected_indices_p1 = np.where(self['p1'] == infected_id)[0]

                if infected_indices_p2.size > 0:
                    self['beta'][infected_indices_p2] = 4
                    rand_mask = np.random.random(infected_indices_p2.size) < 3 / 20
                    self['beta'][infected_indices_p2][rand_mask] = 11
                    self['beta'][infected_indices_p2] *= np.array([condom_use(
                        int(sim.people[int(self['p1'][inf_ele])].uid), int(sim.people[int(self['p2'][inf_ele])].uid))
                                                                   for inf_ele in infected_indices_p2])

                if infected_indices_p1.size > 0:
                    self['beta'][infected_indices_p1] = 8
                    rand_mask = np.random.random(infected_indices_p1.size) < 3 / 20
                    self['beta'][infected_indices_p1][rand_mask] = 138
                    self['beta'][infected_indices_p1] *= np.array([condom_use(
                        int(sim.people[int(self['p1'][inf_ele])].uid), int(sim.people[int(self['p2'][inf_ele])].uid))
                                                                   for inf_ele in infected_indices_p1])
        else:
            pass


        return


class CustomLayer_bi(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v

        return

    def update(self, people):
        ''' Update the contacts '''

        global relation

        #provide a number of people who need to change their relations

        #get the infectious and symptomatic people index
        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        #get the people who are infectious but not symptomatic population and half of symptomatic population into the infected_uids
        infected_uids=np.concatenate((infectious_uids,np.random.choice(symptomatic_uids,int(0.5*len(symptomatic_uids)),False)))


        relation_bi_m_a=relation[sim.t]['relation_bi_m_a']
        relation_bi_m_v=relation[sim.t]['relation_bi_m_v']
        relation_bi_f_v=relation[sim.t]['relation_bi_f_v']
        relation_bi_f_o=relation[sim.t]['relation_bi_f_o']

        if len(infected_uids) != 0:

        # random choice
            cut1,cut2,cut3,cut4=int(n_contacts_bi_m_v/2*adj_fac),int(n_contacts_bi_m_v/2*adj_fac)+int(n_contacts_bi_m_a / 2 * adj_fac),int(n_contacts_bi_m_v/2*adj_fac)+int(n_contacts_bi_m_a / 2 * adj_fac)+int(n_contacts_bi_f_v / 2 * adj_fac),-int(n_contacts_bi_f_o / 2 * adj_fac)
            self['p1'] = np.concatenate((np.random.choice(Pop_bi_m, int(n_contacts_bi_m_v/2*adj_fac), replace=True, p=rand_num_bi_m_v_pro),
                                         np.random.choice(Pop_bi_m, int(n_contacts_bi_m_a / 2 * adj_fac), replace=True,p=rand_num_bi_m_a_pro),
                                         np.random.choice(Pop_bi_f, int(n_contacts_bi_f_v / 2 * adj_fac), replace=True,p=rand_num_bi_f_v_pro),
                                         np.random.choice(Pop_bi_f, int(n_contacts_bi_f_o / 2 * adj_fac), replace=True,p=rand_num_bi_f_o_pro)),dtype=cv.default_int)

            self['p2'] = np.array(list(map(lambda ind: np.random.choice(relation_bi_m_v[ind]), self['p1'][:cut1]))+list(map(lambda ind:np.random.choice(relation_bi_m_a[ind]),self['p1'][cut1:cut2]))+list(map(lambda ind:np.random.choice(relation_bi_f_v[ind]) ,self['p1'][cut2:cut3]))+list(map(lambda ind:np.random.choice(relation_bi_f_o[ind]),self['p1'][cut4:])),dtype=cv.default_int)


            infect_p1=np.concatenate(list(map(lambda ind:np.where(self['p1']==ind)[0],infected_uids)))
            infect_p2=np.concatenate(list(map(lambda ind:np.where(self['p2']==ind)[0],infected_uids)))
            infect_pos=np.union1d(infect_p1, infect_p2)
            self['p1']=self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)





            ## VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

            def update_beta(inf_ele, p1_ind, p2_ind, p1, p2, inf_eles_p1):
                if p1.sex == 1 and p2.sex == 1:
                    if int(inf_ele) in inf_eles_p1[0]:
                        if (p1_ind in Pop_ho_m_0_set) or (p1_ind in Pop_bi_m_0_set):
                            return 11
                        elif (p1_ind in Pop_ho_m_1_set) or (p1_ind in Pop_bi_m_1_set):
                            return 138
                        elif (p1_ind in Pop_ho_m_2_set) or (p1_ind in Pop_bi_m_2_set):
                            if (p2_ind in Pop_ho_m_1_set) or (p2_ind in Pop_bi_m_1_set):
                                return 11
                            else:
                                return 138
                elif p1.sex == 0 and p2.sex == 0:
                    return 2
                else:
                    temp_beta = 8 if p1.infectious or p2.infectious else 4
                    if np.random.random() < 3 / 20:
                        temp_beta = 138 if p1.infectious or p2.infectious else 11
                    return temp_beta * condom_use(int(p1.uid), int(p2.uid))

            p1_inds, p2_inds = np.where(np.in1d(self['p1'], infected_uids)), np.where(np.in1d(self['p2'], infected_uids))
            inf_eles = np.union1d(p1_inds, p2_inds)
            self['p1'], self['p2'] = self['p1'][inf_eles], self['p2'][inf_eles]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)

            for inf_ele in inf_eles:
                p1_ind, p2_ind = self['p1'][inf_ele], self['p2'][inf_ele]
                p1, p2 = sim.people[int(p1_ind)], sim.people[int(p2_ind)]
                self['beta'][inf_ele] = update_beta(inf_ele, p1_ind, p2_ind, p1, p2, p1_inds)
                self['beta'][inf_ele] = update_beta(inf_ele, p1_ind, p2_ind, p1, p2, p2_inds)
        else:
            pass

        #print(f'bi successuful in {sim.t}')
        return

class CustomLayer_ho(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v
        return

    def update(self, people):
        ''' Update the contacts '''
        global time_ho
        global relation
        # if sim.t < len(seed_position_order):
        #     if seed_position_order[sim.t]!=0 :
        #         importation_inds = np.random.choice(Pop_ho_m, int(seed_position_order[sim.t]/ factor),replace=False)  # Can't use cvu.choice() since sampling from indices
        #         print('importation_inds', importation_inds)
        #         sim.people.infect(inds=importation_inds, layer='homo', variant=1)

        # if sim.t==124:
        #     relation=relation2
        # update relation
        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids=np.concatenate((infectious_uids,np.random.choice(symptomatic_uids,int(0.5*len(symptomatic_uids)),False)))

        relation_ho_m=relation[sim.t]['relation_ho_m']
        relation_ho_f=relation[sim.t]['relation_ho_f']

        #print(f'infected_uids are {infected_uids}')
        if len(infected_uids) != 0:
            self['p1'] = np.concatenate((np.random.choice(Pop_ho_m, int(n_contacts_ho_m/2*adj_fac), replace=True, p=rand_num_ho_m_prb),
                                         np.random.choice(Pop_ho_f, int(n_contacts_ho_f / 2 * adj_fac), replace=True, p=rand_num_ho_f_prb)),
                                  dtype=cv.default_int)
            self['p2'] = np.array(list(map(lambda ind:np.random.choice(relation_ho_m[ind]),self['p1'][:int(n_contacts_ho_m/2*adj_fac)]))+list(map(lambda ind:np.random.choice(relation_ho_f[ind]), self['p1'][-int(n_contacts_ho_f/ 2 * adj_fac):])),dtype=cv.default_int)

            infect_p1=np.concatenate(list(map(lambda ind:np.where(self['p1']==ind)[0],infected_uids)))
            infect_p2=np.concatenate(list(map(lambda ind:np.where(self['p2']==ind)[0],infected_uids)))
            infect_pos=np.union1d(infect_p1, infect_p2)
            self['p1']=self['p1'][infect_pos]
            print(f'lens of p1 is {len(infect_pos)}')
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)
            for ele in infected_uids:
                if (ele in self['p1']) or (ele in self['p2']):
                    # the ele is in the homosexual group
                    inf_eles_p1 = np.where(self['p1'] == ele)
                    inf_eles_p2 = np.where(self['p2'] == ele)
                    inf_eles = np.union1d(inf_eles_p1, inf_eles_p1)  # union and get rid of the duplicate

                    for inf_ele in inf_eles:
                        # male
                        p1,p2=sim.people[int(self['p1'][inf_ele])] , sim.people[int(self['p2'][inf_ele])]
                        p1_ind, p2_ind = self['p1'][inf_ele], self['p2'][inf_ele]
                        if sim.people[int(self['p1'][inf_ele])].sex == 1:
                            # if the first is infectious
                            # print(f'{int(inf_ele)} in {inf_eles_p1}')
                            # print(f'{type(inf_ele)} in {type(inf_eles_p1)}')
                            if int(inf_ele) in inf_eles_p1[0]:
                                if (p1_ind in Pop_ho_m_0_set) or (p1_ind in Pop_bi_m_0_set):
                                    self['beta'][inf_ele] = 11
                                elif (p1_ind in Pop_ho_m_1_set) or (p1_ind in Pop_bi_m_1_set):
                                    self['beta'][inf_ele] = 138
                                elif (p1_ind in Pop_ho_m_2_set) or (p1_ind in Pop_bi_m_2_set):
                                    if (p2_ind in Pop_ho_m_1_set) or (p2_ind in Pop_bi_m_1_set):
                                        self['beta'][inf_ele] = 11
                                    else:
                                        self['beta'][inf_ele] = 138
                                self['beta'][inf_ele] = self['beta'][inf_ele] * condom_use(p1_ind,p2_ind)


                            # if the second is infectious
                            elif int(inf_ele) in inf_eles_p2[0]:
                                if (p2_ind in Pop_ho_m_0_set) or (p2_ind in Pop_bi_m_0_set):
                                    self['beta'][inf_ele] = 11
                                elif (p2_ind in Pop_ho_m_1_set) or (p2_ind in Pop_bi_m_1_set):
                                    self['beta'][inf_ele] = 138
                                elif (p2_ind in Pop_ho_m_2_set) or (p2_ind in Pop_bi_m_2_set):
                                    if (p1_ind in Pop_ho_m_1_set) or (p1_ind in Pop_bi_m_1_set):
                                        self['beta'][inf_ele] = 11
                                    else:
                                        self['beta'][inf_ele] = 138
                                self['beta'][inf_ele] = self['beta'][inf_ele] * condom_use(
                                    p1_ind, p2_ind)

        else:
            pass

        return
# change the contact day by day
global relation_need_change
global relation_need_change_pop

class CustomLayer_party(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v

        return

    def update(self, people):
        ''' Update the contacts '''
        global time_party

        import copy


        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids = np.concatenate(
            (infectious_uids, np.random.choice(symptomatic_uids, int(0.5 * len(symptomatic_uids)), False)))
        self['p1']= relation[sim.t]['party_p1']
        self['p2']= relation[sim.t]['party_p2']
        
        if len(infected_uids) != 0:


            # Filter out non-infected contacts
            infect_p1 = np.concatenate(list(map(lambda ind: np.where(self['p1'] == ind)[0], infected_uids)))
            infect_p2 = np.concatenate(list(map(lambda ind: np.where(self['p2'] == ind)[0], infected_uids)))
            infect_pos = np.union1d(infect_p1, infect_p2)
            self['p1'] = self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]
            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)


            print(f'party successful in {sim.t}')
            ## VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
            for ele in infected_uids:
                if (ele in self['p1']) or (ele in self['p2']):
                    # the ele is in the homosexual group
                    inf_eles_p1 = np.where(self['p1'] == ele)
                    inf_eles_p2 = np.where(self['p2'] == ele)
                    inf_eles = np.union1d(inf_eles_p1, inf_eles_p1)  # union and get rid of the duplicate
                    for inf_ele in inf_eles:
                        p1,p2=sim.people[int(self['p1'][inf_ele])],sim.people[int(self['p2'][inf_ele])]
                        infectious_p1,infectious_p2=sim.people[int(self['p1'][inf_ele])].infectious,sim.people[int(self['p2'][inf_ele])].infectious
                        # male
                        if p1.sex== 1 and p2.sex == 1:
                            # male to male
                            # if the first is infectious
                            if int(inf_ele) in inf_eles_p1[0]:
                                ## todo change there is bug in it
                                if (self['p1'][inf_ele] in Pop_ho_m_0_set) or (self['p1'][inf_ele] in Pop_bi_m_0_set):
                                    #insertive anal intercourse
                                    self['beta'][inf_ele] = 11
                                elif (self['p1'][inf_ele] in Pop_ho_m_1_set) or (self['p1'][inf_ele] in Pop_bi_m_1_set):
                                    #receptive anal intercourse
                                    self['beta'][inf_ele] = 138
                                elif (self['p1'][inf_ele] in Pop_ho_m_2_set) or (self['p1'][inf_ele] in Pop_bi_m_2_set):
                                    #versatile anal intercourse
                                    if (self['p2'][inf_ele] in Pop_ho_m_1_set) or (self['p2'][inf_ele] in Pop_bi_m_1_set):
                                        self['beta'][inf_ele] = 11
                                    else:
                                        self['beta'][inf_ele] = 138
                            # if the second is infectious
                            elif int(inf_ele) in inf_eles_p2[0]:
                                if (self['p2'][inf_ele] in Pop_ho_m_0_set) or (self['p2'][inf_ele] in Pop_bi_m_0_set):
                                    self['beta'][inf_ele] = 11
                                elif (self['p2'][inf_ele] in Pop_ho_m_1_set) or (self['p2'][inf_ele] in Pop_bi_m_1_set):
                                    self['beta'][inf_ele] = 138
                                elif (self['p2'][inf_ele] in Pop_ho_m_2_set) or (self['p2'][inf_ele] in Pop_bi_m_2_set):
                                    if (self['p1'][inf_ele] in Pop_ho_m_1_set) or (self['p1'][inf_ele] in Pop_bi_m_1_set):
                                        self['beta'][inf_ele] = 11
                                    else:
                                        self['beta'][inf_ele] = 138
                        ##female

                        elif p1.sex == 0 and p2.sex == 0:
                            ## female to female
                            pass
                        elif p2.sex == 1 and infectious_p1:
                            self['beta'][inf_ele] = 8
                        elif p2.sex== 1 and infectious_p2:
                            self['beta'][inf_ele] = 8
                        elif p1.sex == 1 and not infectious_p1:
                            self['beta'][inf_ele] = 4
                        elif p2.sex== 1 and not infectious_p2:
                            self['beta'][inf_ele] = 4
                        party_condom_use=condom_use(p1.uid,p2.uid)
                        if party_condom_use==0.2:
                            if np.random.random() < 0.5:
                                party_condom_use=1
                        self['beta'][inf_ele] = self['beta'][inf_ele] * party_condom_use
        else:
            pass


        return

class CustomLayer_party_cc(cv.Layer):
    ##close contact in party
    def __init__(self, layer):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k, v in layer.items():
            self[k] = v
        return

    def update(self, people):
        ''' Update the contacts '''
        infectious_uids = sim.people.uid[sim.people.infectious]
        symptomatic_uids = sim.people.uid[sim.people.symptomatic]
        infected_uids = np.concatenate(
            (infectious_uids, np.random.choice(symptomatic_uids, int(0.5 * len(symptomatic_uids)), False)))

        if len(infected_uids) != 0:

            self['p1'] = np.array(np.random.choice(party_attendee,int(2*len(party_attendee)),replace=True),dtype=cv.default_int)
            self['p2'] = np.ones(len(self['p1']), dtype=cv.default_int)
            #self['p2'] =  np.array(list(map(lambda ind: np.random.choice(relation_he[ind]), Pop_cc_he_m))+list(map(lambda ind:np.random.choice(relation_bi_m[ind]),Pop_cc_bi_m))+list(map(lambda ind:np.random.choice(relation_bi_f[ind]) ,Pop_cc_bi_f))+list(map(lambda ind:np.random.choice(relation_ho_m[ind]),Pop_cc_ho_m))+list(map(lambda ind:np.random.choice(relation_ho_f[ind]),Pop_cc_ho_f)),dtype=cv.default_int)        infect_p1=np.concatenate(list(map(lambda ind:np.where(self['p1']==ind)[0],infected_uids)))
            infect_p1 = np.concatenate(list(map(lambda ind: np.where(self['p1'] == ind)[0], infected_uids)))
            infect_p2=np.concatenate(list(map(lambda ind:np.where(self['p2']==ind)[0],infected_uids)))
            infect_pos=np.union1d(infect_p1, infect_p2)
            self['p1']=self['p1'][infect_pos]
            self['p2'] = self['p2'][infect_pos]


            self['beta'] = np.ones(len(self['p1']), dtype=cv.default_float)*2

            for ind,ind_cc in enumerate(self['p1'] ):
                if ind_cc in party_attendee_he_m_set:
                    self['p2'][ind]=np.random.choice(party_attendee_he_m_p2,1)
                if ind_cc in party_attendee_he_f_set:
                    self['p2'][ind]=np.random.choice(party_attendee_he_f_p2,1)
                elif ind_cc in party_attendee_bi_m_set:
                    if np.random.random()<0.5:
                        self['p2'][ind] = np.random.choice(party_attendee_bi_m_p2_m, 1)
                    else:
                        self['p2'][ind] = np.random.choice(party_attendee_bi_m_p2_f, 1)
                elif ind_cc in party_attendee_bi_f_set:
                    if np.random.random()<0.5:
                        self['p2'][ind] = np.random.choice(party_attendee_bi_f_p2_m, 1)
                    else:
                        self['p2'][ind] = np.random.choice(party_attendee_bi_f_p2_f, 1)
                elif ind_cc in party_attendee_ho_m_set:
                    self['p2'][ind] = np.random.choice(party_attendee_ho_m_p2, 1)
                elif ind_cc in party_attendee_ho_f_set:
                    self['p2'][ind] = np.random.choice(party_attendee_ho_f_p2, 1)
        else:
            pass

        #print(f'close contact in party !!{len(party_attendee_cc)} in {sim.t}')
        return
    #party close contact



dist_dict = {1: [0.70, 0.80, 0.70, 0.60], 2: [0.20, 0.60, 0.60, 0.80], 3: [0.20, 0.10, 0.50, 0.80],
             0: [1.0, 1.0, 1.0, 1.0]}

# update the initial parameters
# 1.update probalitities in all age groups
sim['prognoses']['age_cutoffs'] = np.array([0, 30, 40, 45, 50, 60, 70])  # Age cutoffs (lower limits)
sim['prognoses']['sus_ORs'] = np.array(
    [1, 1, 1, 0.2, 0.25, 0.30, 0.35])  # Odds ratios for relative susceptibility  [1, 1, 1, 0.2, 0.25, 0.30, 0.35])
sim['prognoses']['symp_probs'] = np.array(
    [0.85, 0.85, 0.85, 0.5, 0.55, 0.60, 0.65])  # Overall probability of developing symptoms
sim['prognoses']['severe_probs'] = np.array(
    [0.03, 0.03, 0.03, 0.02, 0.02, 0.03, 0.06])  # Overall probability of developing severe symptoms
sim['prognoses']['crit_probs'] = np.array(
    [0.0003, 0.0006, 0.0004, 0.0003, 0.0003, 0.006, 0.01])  # Overall probability of developing critical symptoms
sim['prognoses']['death_probs'] = np.array(
    [0.00005, 0.0001, 0.0001, 0.0001, 0.0001, 0.002, 0.005])  # Overall probability of dying


sim.initialize()
n_18_69 = cv.true((sim.people.age >= 18) * (sim.people.age < 70))
n_18_69_m = cv.true((sim.people.age >= 18) * (sim.people.age < 70) * (sim.people.sex == 1))
n_18_69_f = cv.true((sim.people.age >= 18) * (sim.people.age < 70) * (sim.people.sex == 0))
# according to the survey distribution 1.8% is the people who have multiple sexual parterners

##todo change MSG_fac
###concurrency factor
cur_fac=0.018
total_len_ho_m = len(n_18_69_m) * 0.063 *0.051
total_len_ho_f = len(n_18_69_f) * 0.15 * 0.027
total_len_bi_m = len(n_18_69_m) * 0.04 *0.039#assume 0.04 is half of the ho and he of male
total_len_bi_f = len(n_18_69_f) * 0.08 *0.076#assume 0.08 is half of the ho and he of female
total_len_he_m = len(n_18_69_m) * 0.019 *0.908
total_len_he_f = len(n_18_69_f) * 0.020 *0.892
total_n=total_len_ho_m+total_len_ho_f+total_len_bi_m+total_len_bi_f+total_len_he_m+total_len_he_f
# choose male




n_18_24_m = cv.true((sim.people.age >= 18) * (sim.people.age < 25) * (sim.people.sex == 1))
n_25_29_m = cv.true((sim.people.age >= 25) * (sim.people.age < 30) * (sim.people.sex == 1))
n_30_39_m = cv.true((sim.people.age >= 30) * (sim.people.age < 40) * (sim.people.sex == 1))
n_40_49_m = cv.true((sim.people.age >= 40) * (sim.people.age < 50) * (sim.people.sex == 1))
n_50_59_m = cv.true((sim.people.age >= 50) * (sim.people.age < 60) * (sim.people.sex == 1))
n_60_69_m = cv.true((sim.people.age >= 60) * (sim.people.age < 70) * (sim.people.sex == 1))
# n_18_24_m = cv.true((sim.people.age >= 18) * (sim.people.age < 70))

# choose female
n_18_24_f = cv.true((sim.people.age >= 18) * (sim.people.age < 25) * (sim.people.sex == 0))
n_25_29_f = cv.true((sim.people.age >= 25) * (sim.people.age < 30) * (sim.people.sex == 0))
n_30_39_f = cv.true((sim.people.age >= 30) * (sim.people.age < 40) * (sim.people.sex == 0))
n_40_49_f = cv.true((sim.people.age >= 40) * (sim.people.age < 50) * (sim.people.sex == 0))
n_50_59_f = cv.true((sim.people.age >= 50) * (sim.people.age < 60) * (sim.people.sex == 0))
n_60_69_f = cv.true((sim.people.age >= 60) * (sim.people.age < 70) * (sim.people.sex == 0))



##Homo  number * frequency were counted and normalised in our model
n_18_24_ho_m = n_18_24_m[:int(total_len_ho_m * 0.174)]
n_25_29_ho_m = n_25_29_m[:int(total_len_ho_m * 0.156)]
n_30_39_ho_m = n_30_39_m[:int(total_len_ho_m * 0.296)]
n_40_49_ho_m = n_40_49_m[:int(total_len_ho_m * 0.268)]  ###0.268
n_50_59_ho_m = n_50_59_m[:int(total_len_ho_m * 0.08)]
n_60_69_ho_m = n_60_69_m[:int(total_len_ho_m * 0.025)]

n_18_24_ho_f = n_18_24_f[:int(total_len_ho_f * 0.174)]
n_25_29_ho_f = n_25_29_f[:int(total_len_ho_f * 0.156)]
n_30_39_ho_f = n_30_39_f[:int(total_len_ho_f * 0.296)]
n_40_49_ho_f = n_40_49_f[:int(total_len_ho_f * 0.268)]
n_50_59_ho_f = n_50_59_f[:int(total_len_ho_f * 0.08)]
n_60_69_ho_f = n_60_69_f[:int(total_len_ho_f * 0.025)]

# bisexual
n_18_24_bi_m = n_18_24_m[int(total_len_ho_m * 0.174):int(total_len_ho_m * 0.174) + int(total_len_bi_m * 0.158)]
n_25_29_bi_m = n_25_29_m[int(total_len_ho_m * 0.156):int(total_len_ho_m * 0.156) + int(total_len_bi_m * 0.141)]
n_30_39_bi_m = n_30_39_m[int(total_len_ho_m * 0.296):int(total_len_ho_m * 0.296) + int(total_len_bi_m * 0.295)]
n_40_49_bi_m = n_40_49_m[int(total_len_ho_m * 0.268):int(total_len_ho_m * 0.268) + int(total_len_bi_m * 0.244)]
n_50_59_bi_m = n_50_59_m[int(total_len_ho_m * 0.08):int(total_len_ho_m * 0.08) + int(total_len_bi_m * 0.102)]
n_60_69_bi_m = n_60_69_m[int(total_len_ho_m * 0.025):int(total_len_ho_m * 0.025) + int(total_len_bi_m * 0.061)]

n_18_24_bi_f = n_18_24_f[int(total_len_ho_f * 0.174):int(total_len_ho_f * 0.174) + int(total_len_bi_f * 0.158)]
n_25_29_bi_f = n_25_29_f[int(total_len_ho_f * 0.156):int(total_len_ho_f * 0.156) + int(total_len_bi_f * 0.141)]
n_30_39_bi_f = n_30_39_f[int(total_len_ho_f * 0.296):int(total_len_ho_f * 0.296) + int(total_len_bi_f * 0.295)]
n_40_49_bi_f = n_40_49_f[int(total_len_ho_f * 0.268):int(total_len_ho_f * 0.268) + int(total_len_bi_f * 0.244)]
n_50_59_bi_f = n_50_59_f[int(total_len_ho_f * 0.08):int(total_len_ho_f * 0.08) + int(total_len_bi_f * 0.102)]
n_60_69_bi_f = n_60_69_f[int(total_len_ho_f * 0.025):int(total_len_ho_f * 0.025) + int(total_len_bi_f * 0.061)]


##he
n_18_24_he_m = n_18_24_m[-int(total_len_he_m * 0.141):]
n_25_29_he_m = n_25_29_m[-int(total_len_he_m * 0.125):]
n_30_39_he_m = n_30_39_m[-int(total_len_he_m * 0.294):]
n_40_49_he_m = n_40_49_m[-int(total_len_he_m * 0.219):]
n_50_59_he_m = n_50_59_m[-int(total_len_he_m * 0.125):]
n_60_69_he_m = n_60_69_m[-int(total_len_he_m * 0.097):]

n_18_24_he_f = n_18_24_f[-int(total_len_he_f * 0.141):]
n_25_29_he_f = n_25_29_f[-int(total_len_he_f * 0.125):]
n_30_39_he_f = n_30_39_f[-int(total_len_he_f * 0.294):]
n_40_49_he_f = n_40_49_f[-int(total_len_he_f * 0.219):]
n_50_59_he_f = n_50_59_f[-int(total_len_he_f * 0.125):]
n_60_69_he_f = n_60_69_f[-int(total_len_he_f * 0.097):]
total_risk_pop_m=np.concatenate((n_18_24_ho_m,n_18_24_bi_m,n_18_24_he_m,n_25_29_ho_m,n_25_29_bi_m,n_25_29_he_m,n_30_39_ho_m,n_30_39_bi_m,n_30_39_he_m,n_40_49_ho_m,n_40_49_bi_m,n_40_49_he_m))
total_risk_pop_f=np.concatenate((n_18_24_ho_f,n_18_24_bi_f,n_18_24_he_f,n_25_29_ho_f,n_25_29_bi_f,n_25_29_he_f,n_30_39_ho_f,n_30_39_bi_f,n_30_39_he_f,n_40_49_ho_f,n_40_49_bi_f,n_40_49_he_f))
total_risk_pop_m=np.intersect1d(cv.true((sim.people.age < 45)),total_risk_pop_m)
total_risk_pop_f=np.intersect1d(cv.true((sim.people.age < 45)),total_risk_pop_f)
Vac_risk_pop_m=np.ones(len(total_risk_pop_m))*255
Vac_risk_pop_f=np.ones(len(total_risk_pop_f))*255
global Pop_ho
global Pop_ho_m
global Pop_ho_f
global Pop_ho_m_0
global Pop_ho_m_1
global Pop_ho_m_2
Pop_ho_m = np.concatenate((n_18_24_ho_m, n_25_29_ho_m, n_30_39_ho_m, n_40_49_ho_m, n_50_59_ho_m, n_60_69_ho_m))
np.random.shuffle(Pop_ho_m)
Pop_ho_m_set=set(Pop_ho_m)
Pop_ho_m_0 = Pop_ho_m[:int(len(Pop_ho_m) * 0.199)]
Pop_ho_m_0_set=set(Pop_ho_m_0)
Pop_ho_m_1 = Pop_ho_m[int(len(Pop_ho_m) * 0.199):int(len(Pop_ho_m) * 0.511)]
Pop_ho_m_1_set=set(Pop_ho_m_1)
Pop_ho_m_2 = Pop_ho_m[int(len(Pop_ho_m) * 0.511):]
Pop_ho_m_2_set=set(Pop_ho_m_2)
Pop_ho_f = np.concatenate((n_18_24_ho_f, n_25_29_ho_f, n_30_39_ho_f, n_40_49_ho_f, n_50_59_ho_f, n_60_69_ho_f))
Pop_ho_f_set=set(Pop_ho_f)
np.random.shuffle(Pop_ho_f)

print(f'Pop_ho_m is {Pop_ho_m} and Pop_ho_f is {Pop_ho_f}')
Pop_ho = np.concatenate((Pop_ho_m, Pop_ho_f))
Pop_ho_set=set(Pop_ho)
Pop_bi_m = np.concatenate((n_18_24_bi_m, n_25_29_bi_m, n_30_39_bi_m, n_40_49_bi_m, n_50_59_bi_m, n_60_69_bi_m));np.random.shuffle(Pop_bi_m)
Pop_bi_set=set(Pop_bi_m)
Pop_bi_m_0 = Pop_bi_m[:int(len(Pop_bi_m) * 0.199)]
Pop_bi_m_0_set=set(Pop_bi_m_0)
Pop_bi_m_1 = Pop_bi_m[int(len(Pop_bi_m) * 0.199):int(len(Pop_bi_m) * 0.511)]
Pop_bi_m_1_set=set(Pop_bi_m_1)
Pop_bi_m_2 = Pop_bi_m[int(len(Pop_bi_m) * 0.511):]
Pop_bi_m_2_set=set(Pop_bi_m_2)
global Pop_he
global Pop_bi
Pop_bi_f = np.concatenate((n_18_24_bi_f, n_25_29_bi_f, n_30_39_bi_f, n_40_49_bi_f, n_50_59_bi_f, n_60_69_bi_f))
Pop_bi_f_set=set(Pop_bi_f)
np.random.shuffle(Pop_bi_f)
Pop_bi = np.concatenate((Pop_bi_m, Pop_bi_f))
Pop_he_m = np.concatenate((n_18_24_he_m, n_25_29_he_m, n_30_39_he_m, n_40_49_he_m, n_50_59_he_m, n_60_69_he_m))
Pop_he_m_set=set(Pop_he_m)
np.random.shuffle(Pop_he_m)
Pop_he_f = np.concatenate((n_18_24_he_f, n_25_29_he_f, n_30_39_he_f, n_40_49_he_f, n_50_59_he_f, n_60_69_he_f))
Pop_he_f_set=set(Pop_he_f)
np.random.shuffle(Pop_he_f)
Pop_he = np.concatenate((Pop_he_m, Pop_he_f))
Pop_he_set=set(Pop_he)









print(f'length of homo population are {len(Pop_ho_m) + len(Pop_ho_f)}')
print(f'Pop_he_m is {Pop_he_m} and Pop_ho_f is {Pop_he_f}')
print(f'length of hetero population are {len(Pop_he_m) + len(Pop_he_f)}')

n_people_ho = len(Pop_ho_m) + len(Pop_ho_f)
n_contacts_per_person_ho_m = 15 / 30
n_contacts_per_person_ho_f = 20 / 30

n_contacts_ho_m = int(n_contacts_per_person_ho_m * len(Pop_ho_m))
n_contacts_ho_f = int(n_contacts_per_person_ho_f * len(Pop_ho_f))
n_contacts_ho = n_contacts_ho_m + n_contacts_ho_f
print(f'total ho_m contact={n_contacts_ho_m} avg {n_contacts_per_person_ho_m}')
print(f'total ho_f contact={n_contacts_ho_f} avg {n_contacts_per_person_ho_f}')

n_people_bi = len(Pop_bi_m) + len(Pop_bi_f)
# bisexual male viginal sex
n_contacts_per_person_bi_m_v = 12/30
# bisexual male anal sex
n_contacts_per_person_bi_m_a = 9/30
# bisexual female anal sex
n_contacts_per_person_bi_f_v = 12/30
# bisexual female oral sex
n_contacts_per_person_bi_f_o = 10/30
n_contacts_bi_m_v = int(n_contacts_per_person_bi_m_v * len(Pop_bi_m))
n_contacts_bi_m_a = int(n_contacts_per_person_bi_m_a * len(Pop_bi_m))
n_contacts_bi_f_v = int(n_contacts_per_person_bi_f_v * len(Pop_bi_f))
n_contacts_bi_f_o = int(n_contacts_per_person_bi_f_o * len(Pop_bi_f))
n_contacts_bi = n_contacts_bi_m_v + n_contacts_bi_m_a + n_contacts_bi_f_v + n_contacts_bi_f_o
sex_ratio_bi_m = (n_contacts_bi_m_a) / (n_contacts_bi_m_a + n_contacts_bi_m_v)
sex_ratio_bi_f = (n_contacts_bi_f_v) / (n_contacts_bi_f_v + n_contacts_bi_f_o)
# print(f'total bi contact={n_contacts_bi} avg {n_contacts_per_person_bi}')

n_people_he = len(Pop_he_m) + len(Pop_he_f)
n_contacts_per_person_he = 20 / 30
n_contacts_he = int(n_contacts_per_person_he * n_people_he)
n_contacts_he_m = int(n_contacts_he * 90.8 / (90.8 + 89.2))
n_contacts_he_f = int(n_contacts_he * 89.2 / (90.8 + 89.2))
print(f'total he contact={n_contacts_he} avg {n_contacts_per_person_he}')

# get the contact data
contact_ho, contact_bi, contact_he = {}, {}, {}
global relation_he
relation_he = {}
Pop_he_p1 = Pop_he_m
Pop_he_p2 = Pop_he_f

rand_num_he = np.random.normal(2.4, 2.1, len(Pop_he_p1))
rand_num_he = lower_one(np.rint(rand_num_he).astype(int))
# relation less than 0 will become 1 relationship
num_relation_he = np.sum(rand_num_he)
# create the selection

for ind, ind_he_p1 in enumerate(Pop_he_p1):
    choice_he_p2 = sim_age(ind_he_p1, Pop_he_p2)
    relation_he[ind_he_p1] = list(np.random.choice(choice_he_p2, rand_num_he[ind], replace=False))

rand_num_he_prb = rand_num_he / np.sum(rand_num_he)

global he_p1_choose
global he_p2_choose
global contact_all_ho
contact_all_ho = []
# bi relationship
global relation_bi

relation_bi_m_v = {}
relation_bi_m_a = {}
relation_bi_f_v = {}
relation_bi_f_o = {}
def merge_dict(x,y):
    new_dict={}
    for k,v in x.items():
        new_dict[k] =v+y[k]
    return new_dict


def merge_dict_n(x,y):
    #all new
    new_dict={}
    for k1,v1 in x.items():
        new_dict[k1] =v1
    for k2,v2 in y.items():
        new_dict[k2] = v2
    return new_dict

Pop_bi_p1 = np.concatenate((Pop_bi_m, Pop_bi_f))
Pop_bi_p2_m_v = np.concatenate((Pop_he_f, Pop_bi_f))##[-int(len(Pop_he_f) * 0.05):]
Pop_bi_p2_m_a = np.concatenate((Pop_bi_m, Pop_ho_m))##[-int(len(Pop_ho_m) * 0.5):]
# one can only have sex with suitable people 0-1,2-0,2-1   versatile can have sex with multiple people including 0 1 and versatile
Pop_bi_p2_m_1_v = np.concatenate((Pop_he_f, Pop_bi_f))##[-int(len(Pop_he_f) * 0.05):]
Pop_bi_p2_m_1_a = np.concatenate((Pop_bi_m_0, Pop_bi_m_2, Pop_ho_m_2, Pop_ho_m_0))##[-int(len(Pop_ho_m_2) * 0.5):]  [-int(len(Pop_ho_m_0) * 0.5):]
Pop_bi_p2_m_0_v = Pop_bi_p2_m_1_v
Pop_bi_p2_m_0_a = np.concatenate((Pop_bi_m_1, Pop_bi_m_2, Pop_ho_m_2, Pop_ho_m_1)) ##[-int(len(Pop_ho_m_2) * 0.5):]  [-int(len(Pop_ho_m_1) * 0.5):]

Pop_bi_p2_f_v = np.concatenate((Pop_he_m, Pop_bi_m)) ##[-int(len(Pop_he_m) * 0.05):]
Pop_bi_p2_f_o = np.concatenate((Pop_bi_f, Pop_ho_f))  ##[-int(len(Pop_ho_f) * 0.5):]

rand_num_bi_m_v = lower_one(np.rint(np.random.normal(4.7, 3.5, len(Pop_bi_m))).astype(int))
rand_num_bi_m_v_pro=rand_num_bi_m_v/np.sum(rand_num_bi_m_v)
rand_num_bi_m_a = lower_one(np.rint(np.random.normal(5.5, 2.7, len(Pop_bi_m))).astype(int))
rand_num_bi_m_a_pro=rand_num_bi_m_a/np.sum(rand_num_bi_m_a)
rand_num_bi_f_v = lower_one(np.rint(np.random.normal(3.7, 3.5, len(Pop_bi_f))).astype(int))
rand_num_bi_f_v_pro=rand_num_bi_f_v/np.sum(rand_num_bi_f_v)
rand_num_bi_f_o = lower_one(np.rint(np.random.normal(4.4, 3.8, len(Pop_bi_f))).astype(int))
rand_num_bi_f_o_pro=rand_num_bi_f_o/np.sum(rand_num_bi_f_o)
rand_num_bi = np.concatenate((rand_num_bi_m_v + rand_num_bi_m_a, rand_num_bi_f_v + rand_num_bi_f_o))

# create the selection
# set the lowest value into 1
num_relation_bi_m_v = np.sum(rand_num_bi_m_v)

num_relation_bi_m_a = np.sum(rand_num_bi_m_a)

num_relation_bi_f_v = np.sum(rand_num_bi_f_v)
num_relation_bi_f_o = np.sum(rand_num_bi_f_o)
num_relation_bi = num_relation_bi_m_v + num_relation_bi_m_a + num_relation_bi_f_v + num_relation_bi_f_o

for ind, ind_bi_p1 in enumerate(Pop_bi_m):
    # let one have sex partners not less than 1
    if ind_bi_p1 in Pop_bi_m_0_set:
        # find similar age group
        # vaginal sex partner
        choice_bi_p2_m_0_v = sim_age(ind_bi_p1, Pop_bi_p2_m_0_v)
        choice_bi_p2_m_0_a = sim_age(ind_bi_p1, Pop_bi_p2_m_0_a)
        # in this group find sexual partners
        relation_bi_m_v[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_0_v, rand_num_bi_m_v[ind],replace=False))
        relation_bi_m_a[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_0_a[np.where(choice_bi_p2_m_0_a != ind_bi_p1)],rand_num_bi_m_a[ind], replace=False))

    elif ind_bi_p1 in Pop_bi_m_1_set:
        choice_bi_p2_m_1_v = sim_age(ind_bi_p1, Pop_bi_p2_m_1_v)
        choice_bi_p2_m_1_a = sim_age(ind_bi_p1, Pop_bi_p2_m_1_a)
        # in this group find sexual partners
        relation_bi_m_v[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_1_v,rand_num_bi_m_v[ind], replace=False))
        relation_bi_m_a[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_1_a[np.where(choice_bi_p2_m_1_a != ind_bi_p1)],rand_num_bi_m_a[ind], replace=False))
    elif ind_bi_p1 in Pop_bi_m_2_set:
        choice_bi_p2_m_v = sim_age(ind_bi_p1, Pop_bi_p2_m_v)
        choice_bi_p2_m_a = sim_age(ind_bi_p1, Pop_bi_p2_m_a)
        relation_bi_m_v[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_v,rand_num_bi_m_v[ind], replace=False))
        relation_bi_m_a[ind_bi_p1] = list(np.random.choice(choice_bi_p2_m_a[np.where(choice_bi_p2_m_a != ind_bi_p1)], rand_num_bi_m_a[ind],replace=False))

for ind, ind_bi_p1 in enumerate(Pop_bi_f):
    choice_bi_p2_f_v = sim_age(ind_bi_p1, Pop_bi_p2_f_v)
    choice_bi_p2_f_o = sim_age(ind_bi_p1, Pop_bi_p2_f_o)
    relation_bi_f_o[ind_bi_p1] = list(np.random.choice(choice_bi_p2_f_o[np.where(choice_bi_p2_f_o != ind_bi_p1)],rand_num_bi_f_o[ind], replace=False))
    relation_bi_f_v[ind_bi_p1] = list(np.random.choice(choice_bi_p2_f_v, rand_num_bi_f_v[ind], replace=False))

rand_num_bi_prb = rand_num_bi / np.sum(rand_num_bi)
relation_bi_m=merge_dict(relation_bi_m_v,relation_bi_m_a)
relation_bi_f=merge_dict(relation_bi_f_v,relation_bi_f_o)
# ho relationship
global relation_ho
relation_ho_m = {}
relation_ho_f = {}

Pop_ho_p1 = np.concatenate((Pop_ho_m, Pop_ho_f))
Pop_ho_p2_m = copy.deepcopy(Pop_ho_m)
Pop_ho_p2_f = copy.deepcopy(Pop_ho_f)
Pop_ho_p2_m_0 = np.concatenate((Pop_ho_m_1, Pop_ho_m_2))
Pop_ho_p2_m_1 = np.concatenate((Pop_ho_m_0, Pop_ho_m_2))
rand_num_ho_m = lower_one(np.rint(np.random.normal(4.96, 4.2, len(Pop_ho_m))).astype(int))
rand_num_ho_m_prb=rand_num_ho_m/np.sum(rand_num_ho_m)
rand_num_ho_f= lower_one(np.rint(np.random.normal(2.0, 3.9, len(Pop_ho_f))).astype(int))
rand_num_ho_f_prb=rand_num_ho_f/np.sum(rand_num_ho_f)
num_relation_ho=np.sum(rand_num_ho_m)+np.sum(rand_num_ho_f)
rand_num_ho=np.concatenate((rand_num_ho_m,rand_num_ho_f))
rand_num_ho_prb=rand_num_ho/np.sum(rand_num_ho)
# first male then female

# todo 可以不分开么
# print(f'num_relation_ho={num_relation_ho}')
# print(f'length 1 {len(rand_num_ho)}')
# create the selection

for ind, ind_ho_p1 in enumerate(Pop_ho_m):
    # let one have sex partners not less than 1

    if ind_ho_p1 in Pop_ho_m_0_set:
        choice_ho_p2_m_0 = sim_age(ind_ho_p1, Pop_ho_p2_m_0)
        relation_ho_m[ind_ho_p1] = list(np.random.choice(choice_ho_p2_m_0,rand_num_ho_m[ind], replace=False))
    elif ind_ho_p1 in Pop_ho_m_1_set:
        choice_ho_p2_m_1 = sim_age(ind_ho_p1, Pop_ho_p2_m_1)
        relation_ho_m[ind_ho_p1] = list(np.random.choice(choice_ho_p2_m_1,rand_num_ho_m[ind], replace=False))
    elif ind_ho_p1 in Pop_ho_m_2_set:
        choice_ho_p2_m = sim_age(ind_ho_p1, Pop_ho_p2_m)
        relation_ho_m[ind_ho_p1] = list(np.random.choice(choice_ho_p2_m[np.where(choice_ho_p2_m != ind_ho_p1)],rand_num_ho_m[ind], replace=False))

for ind,ind_ho_p1 in enumerate(Pop_ho_f):
    choice_ho_p2_f = sim_age(ind_ho_p1, Pop_ho_p2_f)
    relation_ho_f[ind_ho_p1] = list(np.random.choice(choice_ho_p2_f[np.where(choice_ho_p2_f != ind_ho_p1)],
                                              rand_num_ho_f[ind], replace=False))



# construct condom_use matrix
Pop_group = np.concatenate((Pop_ho, Pop_he, Pop_bi))
choose_array = np.array([1, 0.75, 0.5, 0.25, 0])
pop_age_array = np.zeros(len(Pop_group))
random_array = np.random.rand(len(Pop_group))
condom_dict = {}
import bisect

age_ranges = [15, 20, 25, 35, float('inf')]
probabilities_female = [(0.356, 0.586, 0.662, 0.842),
                        (0.179, 0.323, 0.376, 0.582),
                        (0.128, 0.208, 0.242, 0.373),
                        (0.109, 0.151, 0.178, 0.249)]
probabilities_male = [(0.535, 0.740, 0.823, 0.931),
                      (0.295, 0.569, 0.643, 0.797),
                      (0.162, 0.304, 0.363, 0.539),
                      (0.094, 0.169, 0.197, 0.301)]
condom_values = [1, 0.75, 0.5, 0.25, 0]

condom_dict = {}

for ind, pop_ind in enumerate(Pop_group):
    person = sim.people[int(pop_ind)]
    age = person.age
    sex = person.sex

    age_index = bisect.bisect_right(age_ranges, age) - 1
    probabilities = probabilities_female[age_index] if sex == 0 else probabilities_male[age_index]

    prob_index = bisect.bisect_right(probabilities, random_array[ind])

    condom_dict[int(pop_ind)] = condom_values[prob_index]

global importation_inds
###sex party population
party_attendee_he_m, party_attendee_he_f, party_attendee_bi_m_0,party_attendee_bi_m_1,party_attendee_bi_m_2,party_attendee_bi_f, party_attendee_ho_m_0,party_attendee_ho_m_1,party_attendee_ho_m_2, party_attendee_ho_f = np.random.choice(Pop_he_m, int(0.2*len(Pop_he_m)), False), np.random.choice(Pop_he_f, int(0.2*len(Pop_he_f)), False), np.random.choice(Pop_bi_m_0, int(0.2*len(Pop_bi_m_0)), False),np.random.choice(Pop_bi_m_1, int(0.2*len(Pop_bi_m_1)), False),np.random.choice(Pop_bi_m_2, int(0.2*len(Pop_bi_m_2)), False), np.random.choice(Pop_bi_f, int(0.2*len(Pop_bi_f)), False),np.random.choice(Pop_ho_m_0, int(0.2*len(Pop_ho_m_0)), False),np.random.choice(Pop_ho_m_1, int(0.2*len(Pop_ho_m_1)), False),np.random.choice(Pop_ho_m_2, int(0.2*len(Pop_ho_m_2)), False),np.random.choice(Pop_ho_f, int(0.2*len(Pop_ho_f)), False)

party_attendee_bi_m, party_attendee_ho_m = np.concatenate((party_attendee_bi_m_0, party_attendee_bi_m_1, party_attendee_bi_m_2)), np.concatenate((party_attendee_ho_m_0, party_attendee_ho_m_1, party_attendee_ho_m_2))
party_attendee = np.concatenate((party_attendee_he_m, party_attendee_he_f, party_attendee_bi_m, party_attendee_bi_f,party_attendee_ho_m,party_attendee_ho_f))

###people from sex party attendee to have sex in sex party
###assume that daily sexual is half
pro_party_attendee_sex = np.concatenate((np.ones(int(0.2*len(Pop_he_m))) * 0.60, np.ones(int(0.2*len(Pop_he_f))) * 0.36,
                                 np.ones(int(0.2*len(Pop_bi_m_0))) * 0.82, np.ones(int(0.2*len(Pop_bi_m_1))) * 0.82,np.ones(int(0.2*len(Pop_bi_m_2))) * 0.82,np.ones(int(0.2*len(Pop_bi_f))) * 0.86,
                                 np.ones(int(0.2*len(Pop_ho_m_0))) * 1,np.ones(int(0.2*len(Pop_ho_m_1))) * 1,np.ones(int(0.2*len(Pop_ho_m_2))) * 1, np.ones(int(0.2*len(Pop_ho_f))) * 1))

party_attendee_sex_num = np.sum(pro_party_attendee_sex)
pro_party_attendee_sex = pro_party_attendee_sex / np.sum(pro_party_attendee_sex)
##choose one as the first contractor

##party attendee:infected_uids = sim.people.uid[sim.people.infectious]
#symptomatic_uids = sim.people.uid[sim.people.symptomatic]
#severe_uids = sim.people.uid[sim.people.severe]
party_attendee_he_m_p2 = np.concatenate((party_attendee_he_f, party_attendee_bi_f))
party_attendee_he_f_p2 = np.concatenate((party_attendee_he_m, party_attendee_bi_m))
party_attendee_bi_m_p2_m = np.concatenate((party_attendee_ho_m, party_attendee_bi_m))
party_attendee_bi_m_p2_f = np.concatenate((party_attendee_bi_f, party_attendee_he_f))
party_attendee_bi_f_p2_m = np.concatenate((party_attendee_ho_m, party_attendee_bi_m))
party_attendee_bi_f_p2_f = np.concatenate((party_attendee_bi_f, party_attendee_he_f))
party_attendee_ho_m_p2 = np.concatenate((party_attendee_ho_m, party_attendee_bi_m))
party_attendee_ho_f_p2 = np.concatenate((party_attendee_ho_f, party_attendee_bi_f))

party_attendee_he_m_set = set(party_attendee_he_m)
party_attendee_he_f_set = set(party_attendee_he_f)
party_attendee_bi_m_set = set(party_attendee_bi_m)
party_attendee_bi_f_set = set(party_attendee_bi_f)
party_attendee_ho_m_set = set(party_attendee_ho_m)
party_attendee_ho_f_set = set(party_attendee_ho_f)



###school population

#In California, a child must be five years old BEFORE September 1 in order to enroll in kindergarten. Elementary school is kindergarten through 5th grade (ages 5-10), middle school is grades 6-8 (ages 11-13), and high school is grades 9-12 (ages 14-18).
#https://cardinalatwork.stanford.edu/benefits-rewards/worklife/children-family/school-age-resources
n_5_10_m=cv.true((sim.people.age >= 5) * (sim.people.age < 11) * (sim.people.sex == 1))
n_11_13_m=cv.true((sim.people.age >= 11) * (sim.people.age < 14) * (sim.people.sex == 1))
n_14_18_m=cv.true((sim.people.age >= 14) * (sim.people.age < 19) * (sim.people.sex == 1))
n_19_22_m = cv.true((sim.people.age >=19) * (sim.people.age < 23) * (sim.people.sex == 1))
n_5_10_f=cv.true((sim.people.age >= 5) * (sim.people.age < 11) * (sim.people.sex == 0))
n_11_13_f=cv.true((sim.people.age >= 11) * (sim.people.age < 14) * (sim.people.sex == 0))
n_14_18_f=cv.true((sim.people.age >= 14) * (sim.people.age < 19) * (sim.people.sex == 0))
n_19_22_f = cv.true((sim.people.age >=19) * (sim.people.age < 23) * (sim.people.sex == 0))
n_23_65_m = cv.true((sim.people.age >=23) * (sim.people.age < 65) * (sim.people.sex == 1))
n_23_65_f = cv.true((sim.people.age >=23) * (sim.people.age < 65) * (sim.people.sex == 0))
n_m = cv.true(sim.people.sex == 1)
n_f = cv.true(sim.people.sex == 0)

###workplace population
n_contact_kind = contact_ratio_school * (len(n_5_10_m) + len(n_5_10_f))
n_contact_middle = contact_ratio_school * (len(n_11_13_m) + len(n_11_13_f))
n_contact_high = contact_ratio_school * (len(n_14_18_m) + len(n_14_18_f))
n_contact_university = contact_ratio_school * (len(n_19_22_m) + len(n_19_22_f))
n_contact_workplace = contact_ratio_workplace * (len(n_23_65_m) + len(n_23_65_f))
n_contact_community = contact_ratio_community * (len(n_m) + len(n_f))
###community population


def infect():
    if group == 'ho':
        importation_inds = np.random.choice(Pop_ho, int(num_inf / factor),replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)
        print(sim.pars['variant_map'])
        sim.people.infect(inds=importation_inds, layer='homo', variant=1)
    elif group == 'ho_m':
        importation_inds = np.random.choice(Pop_ho_m, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)
        print(sim.pars['variant_map'])
        sim.people.infect(inds=importation_inds, layer='homo', variant=1)
    elif group == 'ho_f':
        importation_inds = np.random.choice(Pop_ho_f, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)
        print(sim.pars['variant_map'])
        sim.people.infect(inds=importation_inds, layer='homo', variant=1)
    elif group == 'bi':
        importation_inds = np.random.choice(Pop_bi, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='bi', variant=1)
    elif group == 'bi_m':
        importation_inds = np.random.choice(Pop_bi_m, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='bi', variant=1)
    elif group == 'bi_f':
        importation_inds = np.random.choice(Pop_bi_f, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='bi', variant=1)


    elif group == 'he':
        importation_inds = np.random.choice(Pop_he, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='hetero', variant=1)
    elif group == 'he_m':
        importation_inds = np.random.choice(Pop_he_m, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='hetero', variant=1)
    elif group == 'he_f':
        importation_inds = np.random.choice(Pop_he_f, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)

        sim.people.infect(inds=importation_inds, layer='hetero', variant=1)

    elif group == 'n':
        importation_inds = np.random.choice(sim.people.uid, int(num_inf / factor),
                                            replace=False)  # Can't use cvu.choice() since sampling from indices
        print('importation_inds', importation_inds)
        sim.people.infect(inds=importation_inds, variant=1)
    # print('sim.t=', sim.t)

    # sim.results['n_imports'][sim.t] += init_inf*factor
    # for key in sim.layer_keys():
    # print(f'key is {key}')
    # only use the homo first the check the whether the model runs quickly


def condom_use(u1, u2):
    if np.random.rand() <= (condom_dict[u1] + condom_dict[u2]) / 2:
        return 0.2
    return 1


# sim.update_pars(pars=pars)
# print('update sim successfully')
# print(sim)
# print('check people')
# print(sim.people)

###update and save relationship

pre_run = False
if pre_run:
    relation={}
    def update_relationship(ind, relation_dict, pop_set, sim_age_func):
        cho = sim_age_func(ind, pop_set)
        cho = cho[~np.isin(cho, relation_dict[ind])]
        relation_dict[ind].pop(0)
        relation_dict[ind].append(np.random.choice(cho, 1)[0])


    def update_bi_relationship(ind, relation_dict_1, relation_dict_2, pop_set_1, pop_set_2, sim_age_func):
        choice_ratio = len(relation_dict_1[ind]) / (len(relation_dict_1[ind]) + len(relation_dict_2[ind]))
        if np.random.rand() < choice_ratio:
            update_relationship(ind, relation_dict_1, pop_set_1, sim_age_func)
        else:
            update_relationship(ind, relation_dict_2, pop_set_2, sim_age_func)


    def choose_second_contractor(p1_array, p2_array, party_attendee_set, prob, sex_party, party_num):
        for ind, ind_sex_p1 in enumerate(p1_array):
            choose_group_party = np.random.choice(sex_party, party_num, replace=False, p=prob / np.sum(prob))
            choose_group = sim_age_party(ind_sex_p1, choose_group_party, True)
            p2_array[ind] = np.random.choice(choose_group, 1)
        return p2_array


    for i in range(0, pars['n_days']+184):
        if change_relation:
            # Homosexual relationship update
            for ind in np.random.choice(Pop_ho, int(relation_factor * num_relation_ho), replace=True,
                                        p=rand_num_ho_prb):
                if ind in Pop_ho_m_0_set:
                    update_relationship(ind, relation_ho_m, Pop_ho_p2_m_0, sim_age)
                elif ind in Pop_ho_m_1_set:
                    update_relationship(ind, relation_ho_m, Pop_ho_p2_m_1, sim_age)
                elif ind in Pop_ho_m_2_set:
                    update_relationship(ind, relation_ho_m, Pop_ho_p2_m, sim_age)
                elif ind in Pop_ho_f_set:
                    update_relationship(ind, relation_ho_f, Pop_ho_p2_f, sim_age)

            # Bisexual relationship update
            for ind in np.random.choice(Pop_bi, int(relation_factor * num_relation_bi), replace=True,
                                        p=rand_num_bi_prb):
                if ind in Pop_bi_m_0_set:
                    update_bi_relationship(ind, relation_bi_m_a, relation_bi_m_v, Pop_bi_p2_m_0_a, Pop_bi_p2_m_0_v,
                                           sim_age)
                elif ind in Pop_bi_m_1_set:
                    update_bi_relationship(ind, relation_bi_m_a, relation_bi_m_v, Pop_bi_p2_m_1_a, Pop_bi_p2_m_1_v,
                                           sim_age)
                elif ind in Pop_bi_m_2_set:
                    update_bi_relationship(ind, relation_bi_m_a, relation_bi_m_v, Pop_bi_p2_m_a, Pop_bi_p2_m_v, sim_age)
                elif ind in Pop_bi_f_set:
                    update_bi_relationship(ind, relation_bi_f_v, relation_bi_f_o, Pop_bi_p2_f_v, Pop_bi_p2_f_o, sim_age)

            # Heterosexual relationship update
            for ind in np.random.choice(list(relation_he.keys()), int(relation_factor * num_relation_he), replace=True,
                                        p=rand_num_he_prb):
                potential_new_relations = sim_age(ind, Pop_he_p2)
                for new_relation in potential_new_relations:
                    if new_relation not in relation_he[ind]:
                        update_relationship(ind, relation_he, potential_new_relations, sim_age)
                        break
            # Party relationship update
            # Splitting participants for the parties
            party_attendee_bi_m_num = int(0.5 * len(party_attendee_bi_m))
            party_attendee_bi_f_num = int(0.5 * len(party_attendee_bi_f))
            bi_in_gay = np.random.choice(party_attendee_bi_m, party_attendee_bi_m_num, replace=False)
            bi_m_in_mixed = np.random.choice(party_attendee_bi_m, party_attendee_bi_m_num, replace=False)
            bi_f_in_mixed = np.random.choice(party_attendee_bi_f, party_attendee_bi_f_num, replace=False)
            bi_in_female = np.random.choice(party_attendee_bi_f, party_attendee_bi_f_num, replace=False)

            # Define sex parties
            sex_party = [
                np.concatenate((party_attendee_ho_m, bi_in_gay)),
                np.concatenate((bi_m_in_mixed, party_attendee_he_m, party_attendee_he_f, bi_f_in_mixed)),
                np.concatenate((bi_in_female, party_attendee_ho_f))
            ]

            # Define probabilities for each party
            sex_gay_p1_pro = np.concatenate((np.ones(len(party_attendee_ho_m)), np.ones(party_attendee_bi_m_num) * 0.82))
            sex_mixed_p1_pro = np.concatenate((np.ones(party_attendee_bi_m_num) * 0.82,
                                               np.ones(len(party_attendee_he_m)) * 0.6,
                                               np.ones(len(party_attendee_he_f)) * 0.36,
                                               np.ones(int(0.5 * len(party_attendee_bi_f))) * 0.86))
            sex_female_p1_pro = np.concatenate(
                (np.ones(party_attendee_bi_f_num) * 0.86, np.ones(len(party_attendee_ho_f))))

            # Define p1 for each party
            sex_gay_p1 = np.random.choice(sex_party[0], int(np.sum(sex_gay_p1_pro)), replace=True,
                                          p=sex_gay_p1_pro / np.sum(sex_gay_p1_pro))
            sex_mixed_p1 = np.random.choice(sex_party[1], int(np.sum(sex_mixed_p1_pro)), replace=True,
                                            p=sex_mixed_p1_pro / np.sum(sex_mixed_p1_pro))
            sex_female_p1 = np.random.choice(sex_party[2], int(np.sum(sex_female_p1_pro)), replace=True,
                                             p=sex_female_p1_pro / np.sum(sex_female_p1_pro))

            # Define p2 for each party
            sex_gay_p2, sex_mixed_p2, sex_female_p2 = np.ones(len(sex_gay_p1), dtype=cv.default_int), np.ones(
                len(sex_mixed_p1), dtype=cv.default_int), np.ones(len(sex_female_p1), dtype=cv.default_int)

            # Define probabilities for choosing the second contractor
            pro_sex_he_m = np.concatenate((np.ones(len(party_attendee_he_f)) * 0.36, np.ones(len(bi_f_in_mixed)) * 0.86))
            pro_sex_he_f = np.concatenate((np.ones(len(party_attendee_he_m)) * 0.60, np.ones(len(bi_m_in_mixed)) * 0.82))
            pro_sex_bi_m = copy.deepcopy(pro_sex_he_m)
            pro_sex_bi_f = copy.deepcopy(pro_sex_he_f)

            # Choose another as the second contractor
            sex_gay_p2 = choose_second_contractor(sex_gay_p1, sex_gay_p2, sex_party[0], sex_gay_p1_pro, sex_party[0],
                                                  party_num)
            sex_female_p2 = choose_second_contractor(sex_female_p1, sex_female_p2, sex_party[2], sex_female_p1_pro,
                                                     sex_party[2], party_num)
            sex_mixed_p2 = choose_second_contractor(sex_mixed_p1, sex_mixed_p2, sex_party[1], sex_mixed_p1_pro,
                                                    sex_party[1], party_num)

            party_p1 = np.concatenate((sex_gay_p1, sex_mixed_p1, sex_female_p1), dtype=cv.default_int)
            party_p2 = np.concatenate((sex_gay_p2, sex_mixed_p2, sex_female_p2), dtype=cv.default_int)


        relation[i] = {'relation_ho_m': copy.deepcopy(relation_ho_m), 'relation_ho_f': copy.deepcopy(relation_ho_f),
                       'relation_bi_m_a': copy.deepcopy(relation_bi_m_a), 'relation_bi_m_v': copy.deepcopy(relation_bi_m_v),
                       'relation_bi_f_v': copy.deepcopy(relation_bi_f_v), 'relation_bi_f_o': copy.deepcopy(relation_bi_f_o),
                       'relation_he': copy.deepcopy(relation_he), 'party_p1': copy.deepcopy(party_p1), 'party_p2': copy.deepcopy(party_p2)}
        # print(i)
        # if i%124==123:
        #     f = open(f"relation_10m_{i//123}.pkl", "wb")
        #     pickle.dump(relation, f)
        #     f.close()
        #     relation={}

        #save the dict
#     if total_pop==0.3e6:
#         f = open("relation_0.3m.pkl", "wb")#save the relationship for 0.3m
#     elif total_pop==0.5e6:
#         f = open("relation_0.5m.pkl", "wb")#save the relationship for 0.5m
#     elif total_pop==1e6:
#         f = open("relation_1m.pkl", "wb")#save the relationship for 1m
#     elif total_pop==2e6:
#         f = open("relation_2m.pkl", "wb")#save the relationship for 2m
#     elif total_pop==5e6:
#         f = open("relation_5m.pkl", "wb")#save the relationship for 5m
#     elif total_pop==10e6:
#         f = open("relation_10m_1.pkl", "wb")#save the relationship for 10m
#     pickle.dump(relation, f)
#     f.close()
# ##read the dict
# if total_pop==0.3e6:
#     f = open("relation_0.3m.pkl", "rb")
# elif total_pop==0.5e6:
#     f = open("relation_0.5m.pkl", "rb")
# elif total_pop==1e6:
#     f = open("relation_1m.pkl", "rb")
# elif total_pop==2e6:
#     f = open("relation_2m.pkl", "rb")
# elif total_pop==5e6:
#     f = open("relation_5m.pkl", "rb")
# elif total_pop==10e6:
#     f = open("relation_10m.pkl", "rb")
# with open("relation_10m_1.pkl", 'rb') as f:
#     relation = pickle.load(f)



# print(f'sample total={total},average number={total_contact/total}')
#contact_data=np.random.randint(sim.pars['pop_size']*10,sim.pars['pop_size']*20,size=sim.pars['n_days']+1)
sim.people.contacts['w'] = CustomLayer_workplace(sim.people.contacts['w'])
sim.people.contacts['s'] = CustomLayer_school(sim.people.contacts['s'])
sim.people.contacts['c'] = CustomLayer_community(sim.people.contacts['c'])
sim.people.contacts['homo'] = CustomLayer_ho(sim.people.contacts['homo'])##including lambda in it
sim.people.contacts['bi'] = CustomLayer_bi(sim.people.contacts['bi'])##including lambda in it
sim.people.contacts['hetero'] = CustomLayer_he(sim.people.contacts['hetero'])


sim.people.contacts['party'] = CustomLayer_party(sim.people.contacts['party'])
sim.people.contacts['party_cc'] = CustomLayer_party_cc(sim.people.contacts['party_cc'])
#Create the simulation

# cv.options(jupyter=True, verbose=0)

##running with multisims


# todo change to accurage setting

# print('this is s0', s0)
# before the sim, get the pars
print(sim.pars)
#sim.save(filename='5million')
#sim.save(f'5million.sim', keep_people=True)



origin_sim=copy.deepcopy(sim)
# with open("relation_10m_1.pkl", 'rb') as f:
#     relation1 = pickle.load(f)
#
# with open("relation_10m_2.pkl", 'rb') as f:
#     relation2 = pickle.load(f)

with open("relation_2m.pkl", 'rb') as f:
    relation = pickle.load(f)

log_lis= {}
for i in range(100,200):
    print(f'this is the {i} simulation')
    sim=copy.deepcopy(origin_sim)
    relation=relation
    infect()
    seed=i
    sim['rand_seed']=seed
    sim.run()
    sim.summarize()
    # if  sum(sim.results['new_infections'])!=0:
    #     tt = sim.make_transtree()
    #     #change the log_history
    #     for log in tt.infection_log:
    #         target = log['target']
    #         log['date'] = int(sim.people.date_infectious[target])
    #     log_lis[i]=tt.infection_log
    # else:
    for log in sim.people.infection_log:
        target = log['target']
        log['date'] = int(sim.people.date_infectious[target])
    log_lis[i] = sim.people.infection_log
    f = open(f"full_log_list_2M_new_100_200.pkl", "wb")
    pickle.dump(log_lis, f)
    f.close()



# print('print new successfully')
# print('done')
# print(f'finishih time={time.time() - time1} with time intervel={time.time()-time_interval}')



# global age_sex
# age_sex = {i: [0, 0] for i in range(10)}
# infected_group = np.nonzero(sim.people.n_infections != 0)[0]
# for ind in infected_group:
#     # print(ind)
#     # break
#     ind = int(ind)
#     # age_cutoff=[10,18,30,40,50,60,70,80]
#     if sim.people[ind].age // 10 == 1:
#         if sim.people[ind].age < 18:
#             age_group = 1
#         else:
#             age_group = 2
#     else:
#         age_group = sim.people[ind].age // 10
#
#     if sim.people[ind].sex == 1:
#         # male
#         age_sex[age_group][0] = age_sex[age_group][0] + 1
#     elif sim.people[ind].sex == 0:
#         # female
#         age_sex[age_group][1] = age_sex[age_group][1] + 1
# print(age_sex)
#
# age_sex_detail = {'0-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0, '26-30': 0, '31-35': 0, '36-40': 0,
#                   '41-45': 0, '46-50': 0, '51-55': 0, '56-60': 0, '61-65': 0, '66-70': 0, '71-75': 0}
# age_inteval = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60',
#                '61-65', '66-70', '71-75']
# age_sex_list = [0 for i in range(15)]
# for ind in infected_group:
#     ind = int(ind)
#     age = sim.people[ind].age
#     if age < 75:
#         age_sex_list[int((age - 1) / 5)] += 1
#     else:
#         age_sex_list[14] += 1
# plt.figure(figsize=(12, 6), dpi=800)
# plt.bar(age_inteval, age_sex_list)
# plt.show()


