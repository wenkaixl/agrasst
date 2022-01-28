#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:14:32 2021

"""

import numpy as np
import matplotlib.pyplot as plt
import time
import gc

import sys
sys.path.append("../")

import kernel
import utils
import data
from data import DS_ERGM
import tests
import model

import networkx as nx
import igraph
import graphkernels as gk


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--d", type=int, default=20)
parser.add_argument("--n_sim", type=int, default=200)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--test", type=str, default="Approx")

parser.add_argument("--n_gen", type=int, default=1000)
args = parser.parse_args()

r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()


d = args.d
n_gen = args.n_gen
n_simulate = args.n_sim
n_test = args.n_test
test = args.test

# test = "ResampleB200"
print(d, n_gen, n_simulate, n_test)


# coef2 = np.array([-2, 0.])


#d=20  #d=30

coef3 = np.array([-2, 0, 0.01])
dat_e2st = data.DS_ERGM(d, r.construct_e2st_model, coef3)
dat_e2st.sample(100).X.mean()


if test == "exact":

    e2st_model = model.E2stModel(d, coef3)
    gkss = tests.GKSStest(e2st_model, n_simulate=n_simulate)

if test == "ER":

    er_app = model.ApproxEdgeStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(er_app, n_simulate=n_simulate)

if test == "Approx":
    e2st_app = model.ApproxE2StarStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app, n_simulate=n_simulate)


if "ResampleB" in test:
    B = int(test[-3:])
    e2st_app = model.ApproxE2StarStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app, n_simulate=n_simulate, resampled=True, B=B)

if test == "Cumulative":
    e2st_app_c = model.ApproxE2StarStatCumulative(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app_c, n_simulate=n_simulate)

if "CumulativeB" in test:
    B = int(test[-3:])
    e2st_app_c = model.ApproxE2StarStatCumulative(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app_c, n_simulate=n_simulate, resampled=True, B=B)

if "BiEdge" in test:
    e2st_app_b = model.ApproxBiDegStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app_b, n_simulate=n_simulate)
    

if "BiEdgeB" in test:
    B = int(test[-3:])
    e2st_app_b = model.ApproxBiDegStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.GKSStest(e2st_app_b, n_simulate=n_simulate, resampled=True, B=B)


if test == "Deg":
    e2st_app = model.ApproxEdgeStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.Degtest(e2st_app, n_simulate=n_simulate)


if test == "Param":
    e2st_app = model.ApproxEdgeStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.ParamTest(e2st_app, n_simulate=n_simulate)

if test == "MDdeg":
    e2st_app = model.ApproxEdgeStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.MDdegreeTest(e2st_app, n_simulate=n_simulate)

if "TV" in test:
    e2st_app = model.ApproxEdgeStat(dat_e2st, n_gen=n_gen, smooth=False)
    gkss = tests.TVdegreeTest(e2st_app, n_simulate=n_simulate)



scale = 0.03
c2 = np.arange(-20, 5)*scale

power = np.zeros(len(c2))

print("Start Assessment Test: " + test)
for j in range(len(c2)):
    coef3h1 = np.array([-2., c2[j], 0.1])
    # coef3h1 = np.array([-s, -0.25+ c2[j], s])
    dat_e2sth1 = data.DS_ERGM(d, r.construct_e2st_model, coef3h1)
    dat_dsh1 = dat_e2sth1.sample(n=n_test, return_adj=False)
    X1 = dat_dsh1.X

    # rej1h = np.zeros(len(gkss_test))
    rej1h = 0
    for i in range(len(X1)):
        # for t, test in enumerate(gkss_test):
            # res = test.perform_test(X1[i])
            # rej1h[t] += res["h0_rejected"]    
        res = gkss.perform_test(X1[i])
        rej1h += res["h0_rejected"]
        if i%50 == 0:
            print(i, rej1h)#, rej1h_exact, rej1h_r)
        np.savez(file="../res/e2st_per_2s_coef_temp.npz", rej = rej1h, c2 = c2)
    
    power[j] = rej1h/float(len(X1))
    
    print("Complete", j, c2[j], rej1h)
    # np.savez(file="../res/e2st_d"+str(d)+"_per_2s_coef_power"+test+".npz", power = power, c2 = c2) 
    np.savez(file="../res/e2st_d"+str(d)+"_per_2s_coef_nER_power"+test+".npz", power = power, c2 = c2) 
    del X1, dat_dsh1, dat_e2sth1
    gc.collect()

del gkss
gc.collect()
