import matplotlib.pyplot as plt
##save dict and load

# #save
# import pickle
# f = open("log_list1.pkl","wb")
# pickle.dump(log_lis,f)
# f.close()
#
# #load
# import pickle
# log_lis_dict={}
# for i in range(1,14):
#     open_file_name='log_list'+str(i)+'.pkl'
#     with open(open_file_name, 'rb') as handle:
#         log_lis_dict[i] = pickle.load(handle)
#
# #merge toghter
# for i in range(2,14):
#     log_lis=merge_dict_n(log_lis,log_lis_dict[i])


# import pickle
# with open('log_list9.pkl', 'rb') as handle:
#     new_lis=pickle.load(handle)
#     with open('log_list16.pkl', 'rb') as handle1:
#         c_16 = pickle.load(handle1)
#         c16=dict((key, value) for key, value in c_16.items() if key>=353)
#     new_lis=merge_dict_n(new_lis,c16)
#     f = open("log_list9.pkl", "wb")
#     pickle.dump(new_lis, f)
#     f.close()

# import os
# os.chdir("D://cityu//OneDrive - City University of Hong Kong - Student//Paper//covasim-master//covasim//5e6 500simulationm")



# all_work,all_community,all_school,all_household,all_party,all_party_cc=[],[],[],[],[],[]
# for i in range(100):
#     all_work.append(sum(result_dict[i]['dayn_w']))
#     all_community.append(sum(result_dict[i]['dayn_c']))
#     all_school.append(sum(result_dict[i]['dayn_s']))
#     all_household.append(sum(result_dict[i]['dayn_household']))
#     all_party.append(sum(result_dict[i]['dayn_party']))
#     all_party_cc.append(sum(result_dict[i]['dayn_party_cc']))
# sum(all_work)
#再那一层
#性别
#men with men
import numpy as np
import random
result_dict={}
n_days=244+20
ppl = sim.people
for i in range(500):
    if i in log_lis:
        current_log=log_lis[i]
        result_dict[i]={}
        dayn_t=np.asarray([i for i in range(n_days)])
        [dayn_he,dayn_bi,dayn_ho,dayn_party,dayn_h,dayn_w,dayn_s,dayn_c,dayn_party_cc,dayn_m,dayn_f,dayn_all,dayn_m_m,dayn_m_f,dayn_f_f,dayn_f_m,dayn_msm]= [np.zeros(n_days) for i in range(17)]

        el_list_m=[]
        el_list_f=[]
        #get the else list
        m_he_num,m_bi_num,m_ho_num=0,0,0
        f_he_num,f_bi_num,f_ho_num=0,0,0
        m_list=[m_he_num,m_bi_num,m_ho_num]
        f_list=[f_he_num,f_bi_num,f_ho_num]
        c_Pop_he = copy.deepcopy(Pop_he)
        c_Pop_bi = copy.deepcopy(Pop_bi)
        c_Pop_ho = copy.deepcopy(Pop_ho)
        c_Pop_bi_m = copy.deepcopy(Pop_bi_m)
        c_Pop_ho_m = copy.deepcopy(Pop_ho_m)

        # reorganize the population
        for log in current_log:
            target = log['target']
            date = log['date']
            #some agent can not be classified into any group
            if ppl.sex[target] == 1:
                if target in Pop_he:
                    m_he_num+=1
                elif target in Pop_bi:
                    m_bi_num+=1
                elif target in Pop_ho:
                    m_ho_num+=1
                else:
                    rand_num=random.random()
                    if rand_num<m_he_num/(m_he_num+m_bi_num+m_ho_num):
                        m_he_num+=1
                        c_Pop_he = np.concatenate((c_Pop_he, np.asarray([target])))
                    elif rand_num<(m_he_num+m_bi_num)/(m_he_num+m_bi_num+m_ho_num):
                        m_bi_num+=1
                        c_Pop_bi = np.concatenate((c_Pop_bi, np.asarray([target])))
                        c_Pop_bi_m = np.concatenate((c_Pop_bi_m, np.asarray([target])))
                    else:
                        m_ho_num+=1
                        c_Pop_ho = np.concatenate((c_Pop_ho, np.asarray([target])))
                        c_Pop_ho_m = np.concatenate((c_Pop_ho_m, np.asarray([target])))
            else:

                if target in Pop_he:
                    f_he_num+=1
                elif target in Pop_bi:
                    f_bi_num+=1
                elif target in Pop_ho:
                    f_ho_num+=1
                else:
                    rand_num=random.random()
                    if f_he_num+f_bi_num+f_ho_num==0 or rand_num<f_he_num/(f_he_num+f_bi_num+f_ho_num):
                        f_he_num+=1
                        c_Pop_he = np.concatenate((c_Pop_he, np.asarray([target])))
                    elif rand_num<(f_he_num+f_bi_num)/(f_he_num+f_bi_num+f_ho_num):
                        f_bi_num+=1
                        c_Pop_bi = np.concatenate((c_Pop_bi, np.asarray([target])))
                    else:
                        f_ho_num+=1
                        c_Pop_ho = np.concatenate((c_Pop_ho, np.asarray([target])))




        for log in current_log:
            source=log['source']
            target=log['target']
            date=log['date']
            layer=log['layer']
            ##for sex
            if ppl.sex[target]==1:
                # if bisexual male and homosexual male
                if target in c_Pop_ho_m or target in c_Pop_bi_m:
                    dayn_msm[date] += 1
                dayn_m[date]+=1
            elif ppl.sex[target]==0:
                dayn_f[date] += 1

            ##in different setting
            if layer=="hetero":
                dayn_he[date]+=1
            elif layer=='bi':
                dayn_bi[date] += 1
            elif layer=='homo':
                dayn_ho[date]+=1
            elif layer=='party':
                dayn_party[date]+=1
            elif layer=='h':
                dayn_h[date] += 1
            elif layer=='s':
                dayn_s[date] += 1
            elif layer == 'c':
                dayn_c[date] += 1
            elif layer == 'w':
                dayn_w[date] += 1
            elif layer =="party_cc":
                dayn_party_cc[date]+=1
            if source is None or (ppl.sex[target]==1 and ppl.sex[source]==1):
                dayn_m_m[date]+=1
                # if layer in ['homo','bi','party']:
                #     dayn_msm[date]+=1
            elif ppl.sex[target]==1 and ppl.sex[source]==0:
                dayn_f_m[date] += 1
            elif ppl.sex[target]==0 and ppl.sex[source]==1:
                dayn_m_f[date] += 1
            elif ppl.sex[target] == 0 and ppl.sex[source] == 0:
                dayn_f_f[date] += 1
            dayn_all[date]+=1

        cum_m=np.cumsum(dayn_m)
        cum_f=np.cumsum(dayn_f)
        cum_he=np.cumsum(dayn_he)
        cum_bi=np.cumsum(dayn_bi)
        cum_ho = np.cumsum(dayn_ho)
        cum_h=np.cumsum(dayn_h)
        cum_w=np.cumsum(dayn_w)
        cum_s = np.cumsum(dayn_s)
        cum_c = np.cumsum(dayn_c)
        cum_party = np.cumsum(dayn_party)
        cum_party_cc = np.cumsum(dayn_party_cc)
        cum_all=np.cumsum(dayn_all)
        sex_ratio=cum_m/cum_all
        dayn_household = dayn_he + dayn_ho + dayn_bi + dayn_h
        cum_household=np.cumsum(dayn_household)
        result_dict[i]['cum_household'] = cum_household
        result_dict[i]['dayn_all']=dayn_all
        result_dict[i]['dayn_ho']=dayn_ho
        result_dict[i]['dayn_bi'] = dayn_bi
        result_dict[i]['dayn_he'] = dayn_he
        result_dict[i]['dayn_party']=dayn_party
        result_dict[i]['dayn_party_cc'] = dayn_party_cc
        result_dict[i]['dayn_h']=dayn_h
        result_dict[i]['dayn_c'] = dayn_c
        result_dict[i]['dayn_w']=dayn_w
        result_dict[i]['dayn_s']=dayn_s
        result_dict[i]['dayn_household'] = dayn_household

        result_dict[i]['cum_all'] = cum_all
        result_dict[i]['cum_ho'] = cum_ho
        result_dict[i]['cum_bi'] = cum_bi
        result_dict[i]['cum_he'] = cum_he
        result_dict[i]['cum_party'] = cum_party
        result_dict[i]['cum_party_cc'] = cum_party_cc
        result_dict[i]['cum_h'] = cum_h
        result_dict[i]['cum_c'] = cum_c
        result_dict[i]['cum_w'] = cum_w
        result_dict[i]['cum_s'] = cum_s






        cum_m_m=np.cumsum(dayn_m_m)
        cum_msm=np.cumsum(dayn_msm)

        MSM_ratio=cum_msm/cum_all        #plt


        n_weeks=int((sim['n_days']+1)/7)+4
        ###7days###
        weekn_t=np.asarray([i for i in range(n_weeks)])
        weekn_he = np.zeros(n_weeks)
        weekn_bi =np.zeros(n_weeks)
        weekn_ho= np.zeros(n_weeks)
        weekn_h= np.zeros(n_weeks)
        weekn_w= np.zeros(n_weeks)
        weekn_s= np.zeros(n_weeks)
        weekn_c= np.zeros(n_weeks)
        weekn_m= np.zeros(n_weeks)#male
        weekn_f = np.zeros(n_weeks)#female
        #all cases
        weekn_all=np.zeros(n_weeks)
        weekn_msm=np.zeros(n_weeks)
        weekn_m_m=np.zeros(n_weeks)
        weekn_m_f=np.zeros(n_weeks)
        weekn_f_f=np.zeros(n_weeks)
        weekn_f_m=np.zeros(n_weeks)
        weekn_m_ho=np.zeros(n_weeks)
        weekn_m_he=np.zeros(n_weeks)
        weekn_m_bi=np.zeros(n_weeks)
        weekn_f_he=np.zeros(n_weeks)
        weekn_f_bi=np.zeros(n_weeks)
        weekn_f_ho=np.zeros(n_weeks)


        week_cnt=0

        for log in current_log:
            source=log['source']
            target=log['target']
            date=log['date']
            layer=log['layer']


            week_cnt=int((date+1)/7)

            ##for sex
            if ppl.sex[target]==1:
                weekn_m[week_cnt]+=1
            elif ppl.sex[target]==0:
                weekn_f[week_cnt] += 1

            ##in different setting
            if layer=="hetero":
                weekn_he[week_cnt]+=1
            elif layer=='bi':
                weekn_bi[week_cnt] += 1
            elif layer=='homo':
                weekn_ho[week_cnt]+=1
            elif layer=='h':
                weekn_h[week_cnt] += 1
            elif layer=='s':
                weekn_s[week_cnt] += 1
            elif layer == 'c':
                weekn_c[week_cnt] += 1
            elif layer == 'w':
                weekn_w[week_cnt] += 1

            if source is None or (ppl.sex[target] == 1 and ppl.sex[source] == 1):
                weekn_m_m[week_cnt]+=1
                # if layer in ['homo','bi','party']:
                #     weekn_msm[week_cnt]+=1

            elif ppl.sex[target]==1 and ppl.sex[source]==0:
                weekn_f_m[week_cnt] += 1
            elif ppl.sex[target]==0 and ppl.sex[source]==1:
                weekn_m_f[week_cnt] += 1
            elif ppl.sex[target] == 0 and ppl.sex[source] == 0:
                weekn_f_f[week_cnt] += 1
            if ppl.sex[target] == 1:
                if target in c_Pop_bi_m or target in c_Pop_ho_m:
                    weekn_msm[week_cnt] += 1
                if target in c_Pop_he:
                    weekn_m_he[week_cnt] += 1
                elif target in c_Pop_bi:
                    weekn_m_bi[week_cnt] += 1
                elif target in c_Pop_ho:
                    weekn_m_ho[week_cnt] += 1
            if ppl.sex[target] ==0:
                if target in c_Pop_he:
                    weekn_f_he[week_cnt] += 1
                elif target in c_Pop_bi:
                    weekn_f_bi[week_cnt] += 1
                elif target in c_Pop_ho:
                    weekn_f_ho[week_cnt] += 1
            weekn_all[week_cnt]+=1
        weekn_household=weekn_he+weekn_ho+weekn_bi+weekn_h
        result_dict[i]['weekn_m'] = weekn_m
        result_dict[i]['weekn_f'] = weekn_f
        result_dict[i]['weekn_m_m'] = weekn_m_m
        result_dict[i]['weekn_f_f'] = weekn_f_f
        result_dict[i]['weekn_m_f'] = weekn_m_f
        result_dict[i]['weekn_f_m'] = weekn_f_m

        result_dict[i]['weekn_he']=weekn_m_he+weekn_f_he
        result_dict[i]['weekn_ho']=weekn_m_ho+weekn_f_ho
        result_dict[i]['weekn_bi']=weekn_m_bi+weekn_f_bi

        result_dict[i]['weekn_household'] = weekn_household
        sex_ratio_week=np.cumsum(weekn_m)/np.cumsum(weekn_all)
        result_dict[i]['sex_ratio_week'] = sex_ratio_week

        weekn_a_he=np.cumsum(weekn_m_he)+np.cumsum(weekn_f_he)
        weekn_a_ho= np.cumsum(weekn_m_ho) + np.cumsum(weekn_f_ho)
        weekn_a_bi = np.cumsum(weekn_m_bi) + np.cumsum(weekn_f_bi)
        #non heterosexual
        weekn_a_nhe=weekn_a_ho+weekn_a_bi
        result_dict[i]['weekn_he_n']=weekn_a_he/np.cumsum(weekn_all)
        result_dict[i]['weekn_ho_n']=weekn_a_ho/np.cumsum(weekn_all)
        result_dict[i]['weekn_bi_n']=weekn_a_bi/np.cumsum(weekn_all)


        result_dict[i]['weekn_a_he'] = weekn_a_he
        result_dict[i]['weekn_a_ho'] = weekn_a_ho
        result_dict[i]['weekn_a_bi'] = weekn_a_bi
        result_dict[i]['weekn_a_nhe'] = weekn_a_nhe
        result_dict[i]['weekn_all'] = weekn_all

        weekn_m_he_n=np.cumsum(weekn_m_he)/np.cumsum(weekn_m)
        weekn_m_ho_n=np.cumsum(weekn_m_ho)/np.cumsum(weekn_m)
        weekn_m_bi_n=np.cumsum(weekn_m_bi)/np.cumsum(weekn_m)
        weekn_f_he_n=np.cumsum(weekn_f_he)/np.cumsum(weekn_f)
        weekn_f_ho_n=np.cumsum(weekn_f_ho)/np.cumsum(weekn_f)
        weekn_f_bi_n=np.cumsum(weekn_f_bi)/np.cumsum(weekn_f)



        result_dict[i]['weekn_m_he'] = weekn_m_he
        result_dict[i]['weekn_m_ho'] = weekn_m_ho
        result_dict[i]['weekn_m_bi'] = weekn_m_bi
        result_dict[i]['weekn_m_he_n'] = weekn_m_he_n
        result_dict[i]['weekn_m_ho_n'] = weekn_m_ho_n
        result_dict[i]['weekn_m_bi_n'] = weekn_m_bi_n
        result_dict[i]['weekn_f_he'] = weekn_f_he
        result_dict[i]['weekn_f_ho'] = weekn_f_ho
        result_dict[i]['weekn_f_bi'] = weekn_f_bi
        result_dict[i]['weekn_f_ho_n'] = weekn_f_ho_n
        result_dict[i]['weekn_f_bi_n'] = weekn_f_bi_n
        result_dict[i]['weekn_f_he_n'] = weekn_f_he_n



        result_dict[i]['weekn_f_ho_n'] = weekn_f_ho_n
        result_dict[i]['weekn_f_bi_n'] = weekn_f_bi_n
        result_dict[i]['weekn_f_he_n'] = weekn_f_he_n
        MSM_ratio_week = np.cumsum(weekn_msm) / np.cumsum(weekn_all)
        result_dict[i]['MSM_ratio_week'] = MSM_ratio_week



