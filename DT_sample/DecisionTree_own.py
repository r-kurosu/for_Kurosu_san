#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tanaka (3/18/2021)
based on the paper on 3/15/2021
"""

### local library
import FindDCSet
import FindSeparator

### third-party library
import numpy as np
import pandas as pd

### standard library
import collections
import sys
import copy
import random
import time

EPS = 0.00001
random.seed(0)

### common functions

def _cdot(v1, v2):
    ### 2つのベクトルの内積を計算する

    ret = 0

    if len(v1) != len(v2):
        return 0
    else:
        for i in range(len(v1)):
            ret += v1[i] * v2[i]

    return ret


def chooseProper(a_i, h, flag = 0):
    V = {a for a in a_i}
    distinct_vals = sorted(list(V))
    n = len(distinct_vals)
            
    a_min = min(distinct_vals)
    a_max = max(distinct_vals)

    if n == 1:
        A = [a_min] * h
    else:
        A = []
        for i in range(h):
            pos = int((n-2)*(i+1)/(h+1))
            val = (distinct_vals[pos] + distinct_vals[pos+1]) / 2
            A.append(val)
        
    '''
    ### 数列を個数で等分する
    if flag == 0:
        a_max = max(a_i)
        a_min = min(a_i)
        q = len(a_i)

        A = []

        if len(a_i) >= 2 * h:
            for i in range(1, h+1):
                pos = (i * q) // (h+1)
                tmp = (a_i[pos] + a_i[pos+1])/2
                A.append(tmp)
        else:
            for i in range(1, h+1):
                pos = (i * q) // (h+1)
                tmp = a_i[pos]
                A.append(tmp)
    else:
        a_max = max(a_i)
        a_min = min(a_i)
        diff = (a_max - a_min)/(h+1)
        
        A = []
        td_set = set(a_i)
​
        ### 数列a_iを線形に分割する
        for i in range(1, h+1):
            if (a_min + i*diff) in td_set:
                A.append(a_min + i*diff + 0.00001)
            else:
                A.append(a_min + i*diff)
​
    ### DC-setの学習失敗を避けるためのアイデア(微小数で異なる値に無理やり変える)
    for i in range(len(A)):
        if A[i] == a_min:
            A[i] += random.uniform(0, 10**(-5))
        elif A[i] == a_max:
            A[i] += random.uniform(-10**(-5), 0)
        else:
            A[i] += random.uniform(-10**(-5), 10**(-5))
        
    A = sorted(A)
    '''
    
    return [a_min] + A + [a_max]


def findSeparator(A, h, D_i, a_i, eps, flag = 0):
    ### separator set Wを求める
    ### need to find W by MILP(Section 3)
    '''
    W = []

    for i in range(h):
        w = [0] * (len(D_i[0]))
        b = [0]
        W.append([w, b])
    '''

    W = FindSeparator.compute_wjbj(A, h, D_i, a_i, eps, flag)

    return W

def findRhoValue(W, D_i, A, a_i, h, M, eps, rho_del, data_size, flag = 0):
    ### 線形分離可能な最大のrho_rtn, rho_missを求める

    rho_rtn, rho_miss = FindRhoValue.findRhoValue(W, D_i, A, a_i, h, M, eps, rho_del, data_size, flag)

    return rho_rtn, rho_miss

def _computeMiss(W, D_i, A, a_i, h, flag = 0):
    ### missの集合の要素数を求める

    miss_Wg = []
    miss_Wl = []

    ### 条件にしたがい、Missに属する数を求める
    for j in range(h):
        w_j, b_j = W[j]

        if (a_i > A[j+1]) and (_cdot(w_j, D_i) - b_j < 0):
            miss_Wg.append(j)

        if (a_i < A[j+1]) and (_cdot(w_j, D_i) - b_j > 0):
            miss_Wl.append(j)

    if flag == 1:
        print(miss_Wg, miss_Wl)

    tmp = set(miss_Wg) | set(miss_Wl)

    cnt = len(tmp)

    return cnt

def computeSubsets(x_i, W, D_i, A, a_i, h, rho_rtn, rho_miss, eps = 0.001):
    ### D_rtn, D_delを求める

    D_rtn = []
    D_del = []

    ### 条件にしたがいD_rtn, D_delに属する頂点を求める
    for i in range(len(x_i)):
        x = x_i[i]

        if _computeMiss(W, D_i[i], A, a_i[i], h, 0) < rho_rtn * h - eps:
            D_rtn.append(x)

        if _computeMiss(W, D_i[i], A, a_i[i], h, 0) > rho_miss * h + eps:
            D_del.append(x)

    return D_rtn, D_del

def predictDCSet(W, D_i, A, a_i, h, flag = 0):

    L = FindDCSet.findDCSet(W, D_i, A, a_i, h, flag)

    return L

def _computeWTilde(kepa, fv, td, rand, W, D_rtn, D_del, M, rho_cost, rho_set, eps, flag = 0, flag2 = 0):
    ### W_tildeを求める
    ### need to find W_tilde by MILP(Seciton 7.1)

    '''
    kepa = 3
    W_tilde = []

    for i in range(kepa):
        w = [0] * (len(D_i[0]))
        b = [0]
        W_tilde.append([w, b])
    '''

    w_val, b_val, delta_val = ComputeWTilde.computeWTilde(kepa, fv, td, rand, W, set(D_rtn), set(D_del), M, rho_cost, rho_set, eps, flag, flag2)

    if w_val is None:
        return None, None

    W_tilde = []

    for i in range(len(b_val)):
        W_tilde.append([w_val[i], b_val[i]])

    return W_tilde, delta_val

def splitSet(kepa, D_i, a_i, D_i_test, a_i_test, A, W, D_rtn, D_del, x_i_train, x_i_test, M, rho_cost, rho_set, eps, flag = 0, flag2 = 0):
    ### W_tildeを元に2つの集合に分ける(Section 7)

    X_1_train = []
    X_1_test = []
    X_2_train = []
    X_2_test = []

    ### W_tilde, kepaを取得(kepaは定数なので外から与える)
    W_tilde, delta_val = _computeWTilde(kepa, D_i, a_i, A, W, D_rtn, D_del, M, rho_cost, rho_set, eps, flag, flag2)

    if W_tilde is None:
        return None, None, None

    ### 定義にしたがい、すべてのkepaについてw_tilde \cdot x - b_tilde <= 0となるものをX_1、そうでないものをX_2に分ける
    for i in range(len(x_i_train)):
        x = x_i_train[i]
        decision = 0

        for k in range(kepa):
            w_j, b_j = W_tilde[k]
            val = _cdot(w_j, D_i[i]) - b_j
            #print(i, k, w_j, b_j, D_i[i], val)
            if val > 0:
                decision = 1
                break
        if decision == 1:
            X_2_train.append(x)
        else:
            X_1_train.append(x)

    for i in range(len(x_i_test)):
        x = x_i_test[i]
        decision = 0

        for k in range(kepa):
            w_j, b_j = W_tilde[k]
            val = _cdot(w_j, D_i[i]) - b_j
            #print(i, k, w_j, b_j, D_i[i], val)
            if val > 0:
                decision = 1
                break
        if decision == 1:
            X_2_test.append(x)
        else:
            X_1_test.append(x)

    '''
    ### delta_valで分けてみる
    if flag == 0:        
        for i in range(len(x_i)):
            x = x_i[i]
            
            if delta_val[i] == 1.0:
                X_2.append(x)
            else:
                X_1.append(x)
    else:
        for i in range(len(x_i)):
            x = x_i[i]
            
            if delta_val[i] == 1.0:
                X_1.append(x)
            else:
                X_2.append(x)
    '''

    return W_tilde, [X_1_train, X_1_test], [X_2_train, X_2_test]

def findUpperBound(W):
    ret = 0

    for w, b in W:
        tmp = sum([abs(a) for a in w])  + abs(b)
        ret = max(ret, tmp)

    return ret

def judgeLeftRight(fv, DT, ind):
    # DT[4] = internal_data

    w_tilde = []
    b_tilde = []

    k = 0

    for i, kepa, K, W_tilde, M in DT[4]:
        if i != ind:
            continue
        else:
            for j in range(1): #range(kepa):
                w_tilde.append(W_tilde[0])
                b_tilde.append(W_tilde[1])
            k = 1 #kepa

    LEFT = 0
    RIGHT = 1

    for i in range(k):
        if _cdot(fv, w_tilde[i]) - b_tilde[i] > 0:
            return RIGHT
    
    return LEFT

def judgeGroup(fv, DT):
    # DT[0] = ind_inl
    # DT[1] = ind_leaf
    # DT[2] = edges

    pos = 1
    stop = set()
    for ind in DT[1]:
        stop.add(ind)

    max_node = max(DT[1])

    edges = [[[] for _ in range(2)] for _ in range(max_node+1)]

    for a, b, c in DT[2]:
        if a <= max_node and b <= max_node:
            edges[a][c] = b

    while pos not in stop:
        flag = judgeLeftRight(fv, DT, pos)
        if flag == 0:
            pos = edges[pos][0]
        else:
            pos = edges[pos][1]

    return pos

def predictValue(fv, DT, group):
    # DT[5] = leaf_data

    properSet = []
    hh = 0
    w = [0]
    b = [0]
    lambdas = []

    for ind, h, A, W, Lambda in DT[5]:
        if ind != group:
            continue
        else:
            properSet = A
            hh = h
            for i in range(h):                
                w.append(W[i][0])
                b.append(W[i][1])
            lambdas = Lambda

    k = 0
    for i in range(1, hh+1):
        if _cdot(fv, w[i]) - b[i] >= 0:
            k+=1
    
    pred = lambdas[k] * properSet[k] + (1-lambdas[k]) * properSet[k+1]

    #print('group : {}, k : {}, pred : {}'.format(group, k, pred))

    return pred

def calcPredictValue(fv, DT):
    group = judgeGroup(fv, DT)
    pred = predictValue(fv, DT, group)

    return pred

def calcR2Score(fv, td, avg, DT):
    bunsi = 0
    bunbo = 0

    for i in range(len(td)):
        pred = calcPredictValue(fv[i], DT)
        bunsi += (td[i] - pred) ** 2
        bunbo += (td[i] - avg) ** 2

    if bunbo > -EPS and bunbo < EPS:
        return 1-bunsi
    return 1 - bunsi/bunbo

def calcR2ScoreForDCSet(fv, td, avg, A, W, L):
    bunsi = 0
    bunbo = 0

    def predict(x, WW, AA, LL):
        h = len(AA) - 2

        k = 0

        for i in range(h):
            w = WW[i][0]
            b = WW[i][1]            
            if _cdot(w, x) - b >= 0:
                k += 1
        
        ret = LL[k] * AA[k] + (1-LL[k]) * AA[k+1]
        
        return ret

    for i in range(len(td)):
        pred = predict(fv[i], W, A, L)
        bunsi += (td[i] - pred) ** 2
        bunbo += (td[i] - avg) ** 2

    if bunbo > -EPS and bunbo < EPS:
        return 1-bunsi
    return 1 - bunsi/bunbo

def maxLeafSize(x, leaf_nodes, ignore_leaves):
    ret = 0
    for i in leaf_nodes:
        if i in ignore_leaves:
            continue
        ret = max(ret, len(x[i][0]))
        
    return ret

def findWorstNode(x, leaf_nodes, fv, td, DT_datas, limit_nodes, ignore_leaves):
    ret = [-1, 100]

    for i in range(1, len(x)):
        if i in leaf_nodes and i not in ignore_leaves and len(x[i][0]) >= limit_nodes:
            tmp_fv = []
            tmp_td = []
            for j in x[i][0]:
                tmp_fv.append(fv[j])
                tmp_td.append(td[j])
            tmp_r2 = calcR2Score(tmp_fv, tmp_td, sum(tmp_td)/len(tmp_td), DT_datas)
            if ret[1] > tmp_r2:
                ret = [i, tmp_r2]
    
    return ret[0]

def findBestDT(DT_datas, train_fv, train_td, test_fv, test_td):
    l = len(DT_datas)

    ret = [-1, -10**18]
    ret2 = [-1, -10**18]

    r2_scores = []

    for i in range(1, l):
        if DT_datas[i] is None:
            print('-------- DT model {} : result --------'.format(i))
            print(i, 'infeasible', 'infeasible')
            r2_scores.append(None)
            if ret2[0] == -1:
                return r2_scores, ret[0]
            else:
                return r2_scores, ret2[0]
        else:
            train = calcR2Score(train_fv, train_td, sum(train_td)/len(train_td), DT_datas[i])
            test = calcR2Score(test_fv, test_td, sum(test_td)/len(test_td), DT_datas[i])
            print('-------- DT model {} : result --------'.format(i))
            print(i, train, test)
            r2_scores.append([train, test])
            if ret[-1] < test:
                ret = [DT_datas[i], test]
            if ret2[-1] < test and train >= 0.9 * test:
                ret2 = [DT_datas[i], test]

    if ret2[0] == -1:
        return r2_scores, ret[0]
    else:
        return r2_scores, ret2[0]

### decision tree functions

def calcGini(D_1, D_2, X, fv, w, b):
    D_1_x, D_1_x_bar = [], []
    D_2_x, D_2_x_bar = [], []
    D_dash_x, D_dash_x_bar = [], []

    for i in D_1:
        if _cdot(w, fv[i]) - b >= 0:
            D_1_x.append(i)
        else:
            D_1_x_bar.append(i)
    
    for i in D_2:
        if _cdot(w, fv[i]) - b >= 0:
            D_2_x.append(i)
        else:
            D_2_x_bar.append(i)
        
    D_dash = sorted(list(set(D_1) | set(D_2)))


    for i in D_dash:
        if _cdot(w, fv[i]) - b >= 0:
            D_dash_x.append(i)
        else:
            D_dash_x_bar.append(i)

    if len(D_dash) == 0 or len(D_dash_x) == 0 or len(D_dash_x_bar) == 0:
        return 10**18

    gini = 1 
    gini -= (len(D_dash_x) / len(D_dash)) * ((len(D_1_x)/len(D_dash_x))**2 + (len(D_2_x)/len(D_dash_x))**2)
    gini -= (len(D_dash_x_bar) / len(D_dash)) * ((len(D_1_x_bar)/len(D_dash_x_bar))**2 + (len(D_2_x_bar)/len(D_dash_x_bar))**2)

    return gini

def prediction(x_i_train, x_i_test, fv, td, fv2, td2, h, eps, solver_flag = 0):

    print("\n----- prediction -----")
    
    a_i = []
    D_i = []

    for i in x_i_train:
        a_i.append(td[i])
        D_i.append(fv[i])

    a_i_test = []
    D_i_test = []

    for i in x_i_test:
        a_i_test.append(td2[i])
        D_i_test.append(fv2[i])

    ### Proper-Set, Separator-Setを計算し、DC-Setを計算する
    A = chooseProper(a_i, h)
    W = findSeparator(A, h, D_i, a_i, eps, solver_flag)

    best_DCSet = [[], [], [], -10**18]

    ks = set()
    ks.add((h+3)//4)
    ks.add((h+1)//2)
    ks.add((3*h+3)//4)
    ks.add(h)
    ks = sorted(list(ks))

    for i in range(len(ks)):
        k = ks[i]
        dif = (h+1) / (k+1)
        ind = []
        for j in range(1, k+1):
            ind.append(int(dif * j + 0.001))

        A_tmp = [A[0]]
        W_tmp = []
        for i in ind:
            A_tmp.append(A[i])
            W_tmp.append(W[i-1])
        A_tmp.append(A[-1])

        print("\tk={} ind={} A_tmp={} |TestSet|={}".format(k, ind, A_tmp, len(a_i_test)))

        L_tmp = predictDCSet(W_tmp, D_i, A_tmp, a_i, k, solver_flag)

        ##### Haraguchi: 2021/3/18 20:00 #####
        if len(a_i_test) == 0: # k最小 の場合の DCSet が採用される
            best_DCSet = [A_tmp, W_tmp, L_tmp, None] 
            break
        elif len(a_i_test) == 1:
            # R^2の計算で分母が1となるように処置しました
            # R^2 ではないですが, 唯一の test vector に対する二乗誤差 (を1から引いた値) になると思います. 
            test = calcR2ScoreForDCSet(D_i_test, a_i_test, a_i_test[0]-1, A_tmp, W_tmp, L_tmp)
        else:
            ##test = calcR2ScoreForDCSet(D_i_test, a_i_test, sum(td2)/len(td2), A_tmp, W_tmp, L_tmp)
            test = calcR2ScoreForDCSet(D_i_test, a_i_test, sum(a_i_test)/len(a_i_test), A_tmp, W_tmp, L_tmp)
        ######################################

        print("\tR^2 for train={}".format(calcR2ScoreForDCSet(D_i,a_i,sum(a_i)/len(a_i), A_tmp, W_tmp, L_tmp)))
        #print("\tR^2 for test={}, A_tmp={}, L_tmp={}".format(test, A_tmp, L_tmp))
        print("\tR^2 for test={}".format(test))

        
        if test > best_DCSet[-1]:
            best_DCSet = [A_tmp, W_tmp, L_tmp, test]

    A, W, L, _ = best_DCSet

    return A, W, L

def partition(l_time, k_leaves, total_k, fv, td, fv2, td2, h, kepa, rho_rtn, rho_miss, rho_del, rho_cost, rho_set, eps, limit_nodes, solver_flag):
    h_star = h

    ### データ管理用の情報
    l = 1
    p = 1

    x = [[]]
    tmp_train = list(range(len(td)))
    tmp_test = list(range(len(td2)))
    x.append([tmp_train, tmp_test])
    x_i_train = tmp_train
    x_i_test = tmp_test

    ignore_leaves = set()

    ### デバッグデータの情報
 
    debug_infos = [['-', '-', ['-', '-', '-'], ['-', '-']]]

    ### DT modelの情報
    internal_nodes = [[]]
    internal_node = set()
    internal_nodes.append(sorted(list(internal_node)))

    leaf_nodes = [[]]
    leaf_node = set()
    leaf_node.add(1)
    leaf_nodes.append(sorted(list(leaf_node)))

    edges = []

    lambdas = [[]]
    A, W, L = prediction(x_i_train, x_i_test, fv, td, fv2, td2, h_star, eps, solver_flag)
    if W is None or L is None:
        if W is None:
            print('----- in iteration 0 -----')
            print('----- cant find separator set -----')
            exit(1)
        if L is None:
            print('----- in iteration 0 -----')
            print('----- cant find dc-set -----')
            exit(1)
        DT_datas = None
        return DT_datas, debug_infos

    k_leaves[p] = len(W)
    total_k.append(len(W))

    lambdas.append(L)

    leaf_range = []
    leaf_range.append([1, min(td), max(td)])

    internal_data = []
    
    leaf_data = []
    leaf_data.append([1, len(W), A, W, L])

    DT_datas = [[]]
    DT_datas.append([internal_nodes[l], leaf_nodes[l], copy.deepcopy(edges), copy.deepcopy(leaf_range), copy.deepcopy(internal_data), copy.deepcopy(leaf_data)])

    ### 開始時のtrain R2を表示
    print("R^2 for training: {}".format(calcR2Score(fv, td, sum(td)/len(td), DT_datas[l])))
    print("R^2 for test: {}".format(calcR2Score(fv2, td2, sum(td2)/len(td2), DT_datas[l])))
    
    iteration = 0

    while 1:

        iteration += 1

        print("\n\n========== iteration: {} ==========".format(iteration))
        
        max_leaf_size = maxLeafSize(x, leaf_nodes[l], ignore_leaves)
        print(max_leaf_size)
        
        tt = time.perf_counter()
        l_time.append(tt)
        if max_leaf_size < limit_nodes:
            tmp = 0
            for val in k_leaves.values():
                tmp += val
            total_k.append(tmp)
            print('----- finish calculating -----')
            break

        ### leaf nodeの中で最も予測成績の悪いものを見つける
        ind = findWorstNode(x, leaf_nodes[l], fv, td, DT_datas[l], limit_nodes, ignore_leaves)
        print(ind)
        tmp = total_k[-1]
        tmp -= k_leaves[ind]
        del k_leaves[ind]
        total_k.append(tmp)

        x_i, x_i_test = x[ind]
        
        a_i = []
        D_i = []

        for i in x_i:
            a_i.append(td[i])
            D_i.append(fv[i])

        a_i_test = []
        D_i_test = []

        for i in x_i_test:
            a_i_test.append(td2[i])
            D_i_test.append(fv2[i])

        A = chooseProper(a_i, h)
        W = findSeparator(A, h, D_i, a_i, eps, solver_flag)

        ### D_plus, D_minusを求める
        D_plus = []
        D_minus = []
        X_j = []
        g_j = []
        for j in range(h):
            D_plus_tmp = []
            D_minus_tmp = []
            for i in x_i:
                if td[i] >= A[j+1]:
                    D_plus_tmp.append(i)
                else:
                    D_minus_tmp.append(i)
            D_plus.append(D_plus_tmp)
            D_minus.append(D_minus_tmp)

        ### X_j(j番目の分離平面の表側の空間)を求める
        X_j = []
        for j in range(h):
            X_j_tmp = []
            for i in x_i:
                if _cdot(W[j][0], fv[i]) - W[j][1] >= 0:
                    X_j_tmp.append(i)
            X_j.append(X_j_tmp)
        
        ### Gini係数を求める
        g_j = []
        for j in range(h):
            gini = calcGini(D_plus[j], D_minus[j], x_i, fv, W[j][0], W[j][1])
            g_j.append(gini)

        ### 特定の条件を満たすjのうち、Gini係数が最小となるものを見つける
        min_ind = -1
        tmp = 10**18
        for j in range(h):
            if min(len(X_j[j]), (len(x_i) - len(X_j[j]))) >= 1:
                if g_j[j] < tmp:
                    min_ind = j
                    tmp = g_j[j]
        
        if min_ind == -1:
            ### この葉ノードはこれ以上分割できないものとして扱う
            ignore_leaves.add(ind)
            continue

            print('----- in iteration {} -----'.format(iteration))
            print('----- cant find proper index j -----')
            exit(1)

            DT_datas.append(None)
            return DT_datas, debug_infos

        ### X_jを元にX_1, X_2を決定する

        X_1_train, X_1_test = [], []
        X_2_train, X_2_test = [], []

        for i in x_i:
            if _cdot(W[min_ind][0], fv[i]) - W[min_ind][1] >= 0:
                X_1_train.append(i)
            else:
                X_2_train.append(i)
        
        for i in x_i_test:
            if _cdot(W[min_ind][0], fv2[i]) - W[min_ind][1] >= 0:
                X_1_test.append(i)
            else:
                X_2_test.append(i)

        X_2 = [X_1_train, X_1_test]
        X_1 = [X_2_train, X_2_test]

        x.append(X_1)
        x.append(X_2)

        print(X_1, X_2)

        ### 分離した集合(leaf node)について予測関数を生成する
        if len(X_1) == 0:
            a_1, w_1, l_1 = [min(A), sum(A) / len(A), max(A)], [[[0] * len(W[0][0]), 0]], [0.5, 0.5]
        else:
            a_1, w_1, l_1 = prediction(X_1[0], X_1[1], fv, td, fv2, td2, h, eps, solver_flag)
            if w_1 is None or l_1 is None:
                if w_1 is None:
                    print('----- in iteration {} -----'.format(iteration))
                    print('----- cant find separator set -----')
                    exit(1)
                if l_1 is None:
                    print('----- in iteration {} -----'.format(iteration))
                    print('----- cant find dc-set -----')
                    exit(1)
                DT_datas.append(None)
                return DT_datas, debug_infos

        if len(X_2) == 0:
            a_2, w_2, l_2 = [min(A), sum(A) / len(A), max(A)], [[[0] * len(W[0][0]), 0]], [0.5, 0.5]
        else:
            a_2, w_2, l_2 = prediction(X_2[0], X_2[1], fv, td, fv2, td2, h, eps, solver_flag)
            if w_2 is None or l_2 is None:            
                if w_2 is None:
                    print('----- in iteration {} -----'.format(iteration))
                    print('----- cant find separator set -----')
                    exit(1)
                if l_2 is None:
                    print('----- in iteration {} -----'.format(iteration))
                    print('----- cant find dc-set -----')
                    exit(1)
                DT_datas.append(None)
                return DT_datas, debug_infos

        k_leaves[p+1] = len(w_1)
        k_leaves[p+2] = len(w_2)
        total_k[-1] += len(w_1)
        total_k[-1] += len(w_2)
            
        # debug_infos.append([max_leaf_size, condition_flag, [len(D_i), len(D_0), len(D_1)], [len(w_1), len(w_2)]])

        ### 新たなDT modelの情報を付け加える
        leaf_data.append([p+1, len(w_1), a_1, w_1, l_1])
        leaf_data.append([p+2, len(w_2), a_2, w_2, l_2])

        lambdas.append(l_1)
        lambdas.append(l_2)

        internal_node.add(ind)
        internal_nodes.append(sorted(list(internal_node)))

        M = findUpperBound(W)
        internal_data.append([ind, kepa, len(W[min_ind][0]), W[min_ind], M])

        leaf_range.append([p+1, min(a_1), max(a_1)])
        leaf_range.append([p+2, min(a_2), max(a_2)])

        leaf_node.remove(ind)
        leaf_node.add(p+1)
        leaf_node.add(p+2)
        leaf_nodes.append(sorted(list(leaf_node)))

        edges.append([ind, p+1, 0])
        edges.append([ind, p+2, 1])

        l += 1
        p += 2

        DT_datas.append([internal_nodes[l], leaf_nodes[l], copy.deepcopy(edges), copy.deepcopy(leaf_range), copy.deepcopy(internal_data), copy.deepcopy(leaf_data)])

        ### 新たに生成したDT modelでのtrain R2を表示する
        print(calcR2Score(fv, td, sum(td)/len(td), DT_datas[l]))
    
    return DT_datas, debug_infos

def DT(train_fv, train_td, test_fv, test_td, h_star, kepa, rho_rtn, rho_miss, rho_del, rho_cost, rho_set, eps, limit_nodes, solver_flag):
    l_time = []
    k_leaves = {}
    total_k = [0]

    st = time.perf_counter()
    l_time.append(st)

    ### 実行停止するまでに生成された決定木の情報のリストを取得
    DT_datas, debug_infos = partition(l_time, k_leaves, total_k, train_fv, train_td, test_fv, test_td, h_star, kepa, rho_rtn, rho_miss, rho_del, rho_cost, rho_set, eps, limit_nodes, solver_flag)
    if DT_datas is None:
        return ['infeasible', 'infeasible'], None, debug_infos, l_time, total_k

    print('-------- finished construct DT models --------')

    l = len(DT_datas)

    ### 複数の決定木からtest R2が最大となるものを取得
    r2_scores, best_DT_data = findBestDT(DT_datas, train_fv, train_td, test_fv, test_td)

    print('-------- finished finding best DT model --------')

    best_model = []

    internal_node = best_DT_data[0]

    leaf_node = best_DT_data[1]

    edges = best_DT_data[2]

    leaf_range = []
    for values in best_DT_data[3]:
        if values[0] in set(leaf_node):
            leaf_range.append(values)

    internal_data = []    
    for values in best_DT_data[4]:
        if values[0] in set(internal_node):
            internal_data.append(values)

    leaf_data = []
    for values in best_DT_data[5]:
        if values[0] in set(leaf_node):
            leaf_data.append(values)

    best_model = [internal_node, leaf_node, edges, leaf_range, internal_data, leaf_data]

    return r2_scores, best_model, debug_infos, l_time, total_k
