#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements procedures for computing 
- Separating hyperplanes
- Removing a convex region
- Retaining a convex region

@author: shurbevski, Azam (1/14/2021)
"""

import pulp
import CPLEX_Path

#cplex_path = "C:/Program Files/IBM/ILOG/CPLEX_Studio1210/cplex/bin/x64_win64/cplex.exe"
#cplex_path = "/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex"
cplex_path = CPLEX_Path.solver_path

def find_separator(A, B, c_A, c_B, eps, flag = 0):
    # A and B two disjoint sets
    # c_A, c_B cost values for A and B
    # esp tolerance
    model = pulp.LpProblem("Linear_Separator", pulp.LpMinimize)
    if len(A) != 0:
        a0 = A[0] 
    else:
        a0 = B[0]
    b = pulp.LpVariable("b", -2 * len(a0), 2 * len(a0), cat = pulp.LpContinuous)
    w = [pulp.LpVariable("w({})".format(i), -1, 1, cat = pulp.LpContinuous) for i in range(len(a0))]
    delta_A = [pulp.LpVariable("delta_A({})".format(i), 0) for i in range(len(A))] # using index of A
    delta_B = [pulp.LpVariable("delta_B({})".format(i), 0) for i in range(len(B))] # using index of B
    
    for i, v in enumerate(A):
        model += pulp.lpDot(w, v) - b + delta_A[i] >= eps
    for i, v in enumerate(B):
        model += pulp.lpDot(w, v) - b - delta_B[i] <= -eps
    
    model += pulp.lpSum(c_A[i]*delta_A[i] for i in range(len(A))) + \
        pulp.lpSum(c_B[i]*delta_B[i] for i in range(len(B)))
    model.writeLP("./model.lp")
    # print(model)
    if flag == 0:
        status = model.solve(pulp.CPLEX(path=cplex_path, msg=0))
    else:
        status = model.solve(pulp.PULP_CBC_CMD(msg = 0)) ## for CBC solver
    
    if status == pulp.LpStatusOptimal:
        #print('FindSeparator feasible')

        w_val = [var.value() for var in w]
        b_val = b.value()
        delta_A_val = [var.value() for var in delta_A]
        delta_B_val = [var.value() for var in delta_B]

        debug = False
        if debug == True:
            print(len(A), len(B))
            print(w_val, b_val)

        return w_val, b_val, delta_A_val, delta_B_val
    else:
        print('FindSeparator infeasible')
        return None, None, None, None


def split_AB(fv, td, ub, lb):
    """
    given two values ub and lb,
    put in listA all feature vectors in fv
    that have target value > lb,
    and in listB all with target value < ub
    """
    
    if lb < ub:
        # the sets are not separable
        return list (), list()
    
    listA = [f for f, v in zip(fv, td) if v >= lb]
    listB = [f for f, v in zip(fv, td) if v < ub]
    
    return listA, listB


def compute_wjbj(rand, h, fv, td, eps = 1, flag = 0):
    W = []
    
    for j in range(1, h+1):
       D_pos_j, D_neg_j = split_AB(fv, td, rand[j] , rand[j])
       d_D_pos_j = []
       d_D_neg_j = []
 
       for a in td:
           if a >= rand[j]:
               d_D_pos_j.append(abs(rand[j] - a)) 
           else:
               d_D_neg_j.append(abs(rand[j] - a))
              
       if ( len(d_D_neg_j) != len(D_neg_j) ):
           print("index error")

       # computing planes        
       w_j, b_j, dd_D_pos_j, dd_D_neg_j = \
           find_separator(D_pos_j, D_neg_j, d_D_pos_j, d_D_neg_j, eps, flag)
        
       if w_j is None:
           return None

       # when some w_j is none, we put 0.0
       if w_j:
           for i, w in enumerate(w_j):
               if not w:
                   w_j[i] = 0.0

       W.append([w_j, b_j])

    return W
