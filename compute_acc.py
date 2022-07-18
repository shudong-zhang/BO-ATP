import os
import numpy as np

path = '/data/zsd/attacks/black_box/BayesOpt_trans/imagenet_Resnet50_target_False_0.05_greedy_True_gauss.txt'

def read_file(path,queries,notrans):
    f = open(path,'r')
    lines = f.readlines()
    total_num = 0
    correct_num = 0
    res = []

    for line in lines:
        a = line.strip().split()
        if notrans:
            if int(a[-1][6:]) == 1:
                continue

        if a[1][8:] == 'True' and int(a[-1][6:]) < queries:
            res.append(int(a[-1][6:]))
            correct_num += 1
        total_num += 1
    res = np.asarray(res)
    
    print("success rate: {} average queries: {} at queries{} median query: {}".format(correct_num/total_num,np.mean(res),queries,np.median(res)))
    return correct_num/total_num,np.mean(res),np.median(res)

def read_file1(path,queries):
    f = open(path,'r')
    lines = f.readlines()
    total_num = 0
    correct_num = 0
    res = []

    for line in lines:
        a = line.strip().split()
        if a[-1] == 'True' and int(a[3]) < queries:
            res.append(int(a[3]))
            correct_num += 1
        total_num += 1
    res = np.asarray(res)
    print("success rate: {} average queries: {} median query: {}".format(correct_num/total_num,np.mean(res),np.median(res)))

s_array, a_array, m_array = [],[],[]
for i in range(100,1100,100):
    s,a,m = read_file(path,i,False)
    s_array.append(s)
    a_array.append(a)
    m_array.append(m)
f = open('boapt_results.txt','a')
f.write("boapt_imagenet_resnet50_gauss_untarget(0.05)={\nSucess: "+str(s_array)+',\n Avg_Q: '+str(a_array)+',\n Median:'+str(m_array)+'\n}\n\n')
# read_file1(path,20000)
