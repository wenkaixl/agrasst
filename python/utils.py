#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:51 2021

"""
import numpy as np
import time 

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext


def sigmoid(X):
     return 1./(1+np.exp(-(X)))
 

def logit(X):
     return np.log(1./(1./(X) - 1.))
