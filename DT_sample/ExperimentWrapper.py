#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tanaka (3/10/2021)
"""

### local library
import DecisionTree_own

### third-party library
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

### standard library
import collections
import sys
import random
import time
import os

def getFvTargetData(fv_filename, target_filename):
    """
    Given filenames to tables of feature vectors
    and target values, return plain python lists of these values.
    The lists will be sorted in non-decreasing order of target value.
    """
    
    # get the table of feature vectors
    fv_data = pd.read_csv(fv_filename)
    # fv_array = fv_data.to_numpy()
    fv_array = fv_data.values
    # let's get rid of the indices, i.e. the 0th column
    header = fv_data.columns.values.tolist()
    fv = fv_array[:, 1:]
    
    # next, a list of target data
    target_data = pd.read_csv(target_filename, index_col=["CID"])
    #target_array = target_data.to_numpy()
    target_array = target_data.values
    # getting rid of the indices

    td = list()
    for CID in fv_array[:, 0]:
        td.append(target_data.loc[CID].values)

    # td = target_array[:, 1:]
    
    # convert from numpy array to regular python lists
    rtd = [float(tt[0]) for tt in td]
    rfv = [list(ff) for ff in fv]
    
    # zip the lists to sort by target value
    zz = sorted(zip(rtd, rfv))
    
    # unzip the sorte lists
    sfv = [z[1] for z in zz]
    std = [z[0] for z in zz]
    
    return header, sfv, std

def writeDecisionTree(filename, ind_internal, ind_leaf, e, leaf_range, internal_data, leaf_data):
    ### DecisionTreeのグラフ構造g、葉ノードの予測関数uを出力

    l_inl = len(ind_internal)
    l_leaf = len(ind_leaf)
    l_edge = len(e)

    with open(filename, 'w') as f:
        f.write('# Decision Tree Structure Section\n')
        f.write(str(l_inl) + ' ' + ' '.join(map(str, ind_internal)) + '\n')
        f.write(str(l_leaf) + ' ' + ' '.join(map(str, ind_leaf)) + '\n')
        f.write('{}\n'.format(l_edge))
        for a, b, c in e:
            f.write('{} {} {}\n'.format(a, b, c))
        for ind, y_min, y_max in leaf_range:
            f.write('{} {} {}\n'.format(ind, y_min, y_max))
        if len(internal_data) != 0:
            f.write('{}\n'.format(internal_data[0][2]))
        else:
            f.write('{}\n'.format(len(leaf_data[0][3][0][0])))
        f.write('# Internal Node Data Section\n')
        for ind, kepa, K, W_tilde, M in internal_data:
            f.write('{}\n'.format(ind))
            f.write('{} {}\n'.format(kepa, K))
            for i in range(1):
                f.write(' '.join(map(str, W_tilde[0])) + ' ' + str(W_tilde[1]) + '\n')
            f.write('{}\n'.format(M))
        f.write('# Leaf Node Data Section\n')
        for ind, h, A, W, Lambda in leaf_data:
            f.write('{}\n'.format(ind))
            f.write('{} {}\n'.format(h, len(W[0][0])))
            f.write(' '.join(map(str, A)) + '\n')
            for i in range(h):
                f.write(' '.join(map(str, W[i][0])) + ' ' + str(W[i][1]) + '\n')
            f.write(' '.join(map(str, Lambda)) + '\n')

    return 0

def splitData(ind, fv, td):
    fv_train = []
    td_train = []

    for i in ind[0]:
        fv_train.append(fv[i])
        td_train.append(td[i])

    fv_test = []
    td_test = []

    for i in ind[1]:
        fv_test.append(fv[i])
        td_test.append(td[i])

    return fv_train, td_train, fv_test, td_test

def main(argv):

    if len(argv) < 8:
        print('''
        please input some paramaters as follows:
        python CalculateR2Score_v3.py chemical_property fv_filename td_filename h_star kappa rho_cost rho_miss solver_flag(0 is CPLEX, 1 is CBC)
        ''')
        exit()

    ### You can change parameters directly

    chemical_property = argv[1] # 'Hv'
    fv_name = argv[2] # 'data/' + chemical_property + '_desc_norm.csv'
    td_name = argv[3] #'data/' + chemical_property + '_values.txt'
    _, fv_data, td_data = getFvTargetData(fv_name, td_name)
    
    h_star = int(argv[4]) #10
    kappa = int(argv[5]) #1
    rho_rtn = 0.1 # automatically defined
    rho_miss = 0.3 # automatically defined
    rho_del = 0.3 # not used
    rho_cost = float(argv[6]) # 0.5
    rho_set = float(argv[7]) # 0.5
    eps = 0.001 # fixed
    limit_nodes = 20 # fixed
    solver_flag = int(argv[8]) # 1 # 0 is CPLEX, 1 is CBC
    rem_flag = int(argv[9])
    CV = 5 # fixed

    ###

    os.makedirs('result/' + chemical_property, exist_ok = True)
    os.makedirs('DT/' + chemical_property, exist_ok = True)

    for seed in range(1, 10+1):
        print('\n\n********** seed-{} STARTS **********'.format(seed))
        split_ind = []
        data_splitter = KFold(n_splits = CV, shuffle = True, random_state = seed)

        for train, test in data_splitter.split(fv_data):
            train_tmp = sorted(train)
            test_tmp = sorted(test)
            split_ind.append([train_tmp, test_tmp])

        for i in range(CV):
            
            print('\n\n********** round-{} STARTS **********'.format(i+1))
            train_fv, train_td, test_fv, test_td = splitData(split_ind[i], fv_data, td_data)
            
            st = time.perf_counter()

            r2_scores, DT_model, debug_infos, l_times, total_k = DecisionTree_own.DT(train_fv, train_td, test_fv, test_td, h_star, kappa, rho_rtn, rho_miss, rho_del, rho_cost, rho_set, eps, limit_nodes, solver_flag)

            et = time.perf_counter()

            print(l_times)
            print(total_k)

            
            print('\n\n********** round-{} ENDS **********'.format(i+1))
            #exit(1) ### only one round: by Haraguchi 3/18 21:00 ###
        
            calc_time = et - st

            if DT_model is not None:
                DT_filename = 'DT/' + chemical_property + '/' + chemical_property + '_DT_seed' + str(seed) + '_round' + str(i+1) + '.txt'
                writeDecisionTree(DT_filename, DT_model[0], DT_model[1], DT_model[2], DT_model[3], DT_model[4], DT_model[5])
            
            best_leaves = len(DT_model[1])
            print(best_leaves)

            for j in range(len(r2_scores)):
                print(j+1, r2_scores[j][0], r2_scores[j][1])

            if rem_flag == 0:
                fp = open('result/' + chemical_property + '/' + chemical_property + '_seed' + str(seed) + '_round' + str(i+1) + '.txt', 'w')

            fp.write('L-time(sec.) : {}\n\n'.format(calc_time))
            fp.write('DT, seed, round, L-time, total k, train R2, test R2\n')
            for j in range(len(r2_scores)):
                if r2_scores[j] is None:
                    fp.write('{}, {}, {}, _, _, infeasible, infeasible\n'.format(j+1, seed, i+1))
                else:
                    fp.write('{}, {}, {}, {}, {}, {}, {}\n'.format(j+1, seed, i+1, l_times[j+1] - l_times[0], total_k[j+1], r2_scores[j][0], r2_scores[j][1]))

            fp.close()
            
            if rem_flag == 0:
                fp = open('result/' + chemical_property + '/' + chemical_property + '_all.txt', 'a')

            fp.write('L-time(sec.) : {}\n\n'.format(calc_time))
            fp.write('DT, seed, round, L-time, total k, train R2, test R2\n')
            for j in range(len(r2_scores)):
                if r2_scores[j] is None:
                    fp.write('{}, {}, {}, _, _, infeasible, infeasible\n'.format(j+1, seed, i+1))
                else:
                    fp.write('{}, {}, {}, {}, {}, {}, {}\n'.format(j+1, seed, i+1, l_times[j+1] - l_times[0], total_k[j+1], r2_scores[j][0], r2_scores[j][1]))

            fp.close()
    
            if rem_flag == 0:
                fp = open('result/' + chemical_property + '_summary.txt', 'a')

            l_time1 = l_times[best_leaves] - l_times[0]
            l_time2 = l_times[-1] - l_times[0]

            print(l_times)

            k_leaves = total_k[best_leaves]
            fp.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(chemical_property, l_time1, l_time2, seed, i+1, best_leaves, k_leaves, r2_scores[best_leaves-1][1]))
                
            fp.close()

#[0, chemical_property_name, fv_filename, td_filename, h_star, kappa, rho_cost, rho_set, solver_flag(0 is CPLEX, 1 is CBC)]
#main([0, 'At', 'data/At_desc_norm.csv', 'data/At_values.txt', 15, 1, 0.5, 0.5, 1])
#main(sys.argv)
#exit(1)

#argv = sys.argv
#name = argv[1] #'RefIdx'
names = ['RefIdx', 'DC', 'BH']
for name in names:
    fv_name = 'data/' + name + '_desc_norm.csv' # fvrem_desc_norm.csv
    td_name = 'data/' + name + '_values.txt'
    main([0, name, fv_name, td_name, 15, 1, 0.5, 0.5, 0, 0])
