#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tanaka (2/19/2021)
based on the paper on 2/18/2021
"""

import pulp
import CPLEX_Path

#cplex_path = "C:/Program Files/IBM/ILOG/CPLEX_Studio1210/cplex/bin/x64_win64/cplex.exe"
#cplex_path = "/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex"
cplex_path = CPLEX_Path.solver_path

def _cdot(v1, v2):
    ret = 0

    if len(v1) != len(v2):
        return 0
    else:
        for i in range(len(v1)):
            ret += v1[i] * v2[i]
    
    return ret

def findDCSet(W, D_i, A, a_i, h, flag = 0):

    q = len(a_i)

    D = [[] for _ in range(h+1)]

    for i in range(q):
        k = 0

        for j in range(h):
            w_j, b_j = W[j]

            if _cdot(w_j, D_i[i]) - b_j >= 0:
                k += 1
        
        D[k].append(i)    

    MILP = pulp.LpProblem('Find_DC_Set', pulp.LpMinimize)

    ### variables
    # l = [1/2] * (h+1)
    l = {i : pulp.LpVariable(f'lambda_{i}', 0, 1, cat = pulp.LpContinuous) for i in range(h+1)}
    dif = {i : pulp.LpVariable(f'dif_{i}', lowBound = 0, cat = pulp.LpContinuous) for i in range(q)}

    ### object function
    MILP += pulp.lpSum(dif[i] for i in range(q))

    ### constraints
    for k in range(h+1):
        for i in D[k]:
            MILP += dif[i] >= l[k] * A[k] + (1 - l[k]) * A[k+1] - a_i[i]
            MILP += dif[i] >= a_i[i] - (l[k] * A[k] + (1 - l[k]) * A[k+1])

    ### solve
    MILP.writeLP("./model.lp")
    if flag == 0:
        status = MILP.solve(pulp.CPLEX(path=cplex_path, msg=0))
    else:
        status = MILP.solve(pulp.PULP_CBC_CMD(msg = 0))

    ### for showing the results
    if status == pulp.LpStatusOptimal:
        print('DC-set feasible')

        ret = []

        # print(pulp.value(MILP.objective))
        for i in range(h+1):
            if l[i].value() is None:
                ret.append(0)
            else:
                ret.append(l[i].value())
        
        return ret        
    else:
        print('DC-set infeasible')
        return None
