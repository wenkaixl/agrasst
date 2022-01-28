#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:30:33 2021


This includes functions and model class based on Stein characterisations; 
with the explicit ERGM classes and approximated implicit model classes 
"""


from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy 
import scipy.stats as stats
from sklearn import kernel_ridge

import torch.autograd as autograd
import torch
import torch.distributions as dists

import utils 
import data

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

import warnings
import logging



def del_t2s(X):
    deg = X.sum(axis=1)[:,np.newaxis]
    deg_sum = deg + deg.T
    np.fill_diagonal(deg_sum, 0)
    return deg_sum


#The difference function for perturbing edge in triangle term
def del_tri(X):
    X2 = X@X.T
    np.fill_diagonal(X2, 0)
    return X2




class ErgmModel(with_metaclass(ABCMeta, object)):
    """
    An abstract class of an Exponential Random Graph Model(ERGM).  This is
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """

    @abstractmethod
    def t_fun(self, X):
        """
        The Delta_s t(x) for each edge s 
        """
        raise NotImplementedError()

    def log_normalized_prob(self, X):
        """
        Evaluate the exact normalised probability. The normalizer required. 
        This method is not essential. For sanity check when necessary
        Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this model.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    
    def cond_prob(self, X):
        '''
        Conditional probability of q(x^{(s,1)}|x_{-s}),  sigmoid(t_fun)
        '''
        return 1./(1+np.exp(-self.t_fun(X)))


    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

# end UnnormalizedDensity



class ErdosRenyi(ErgmModel):
    """
    explicit density model for Erdos-Renyi graph
    """
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "ERmodel"

    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is independent of other edges;
        returning the coeficient for edge statistics

        '''
        return self.coef * (X*0 + 1.)


    def get_datasource(self):
        r= ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_er_model, self.coef)

    def dim(self):
        return len(self.d)


class E2sModel(ErgmModel):
    """
    explicit density model for Edge-2Star ERGM
    """
    
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "E2Smodel"
        
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is the number of degrees associated with 
        both vertices of an edge;
        '''
        return self.coef[0] + self.coef[1]*del_t2s(X)


    def get_datasource(self):
        r = ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_e2s_model, self.coef)

    def dim(self):
        return len(self.d)
    
class E2stModel(ErgmModel):
    """
    explicit density model for Edge-2Star-Triangle ERGM
    """
    
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "E2STmodel"
        
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is the number of degrees associated with 
        both vertices of an edge and the number of common edges;
        '''
        return self.coef[0] + self.coef[1]*del_t2s(X) + self.coef[2]*del_tri(X)


    def get_datasource(self):
        r = ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_e2st_model, self.coef)

    def dim(self):
        return len(self.d)
    

    
###discarded previous approximation construction
# class ApproxModel(ErgmModel):
#     def __init__(self, d):
#         """
#         d: size of the network
#         """
#         self.d = d 
        
        
#     def t_fun(self, X):
#         '''
#         The approximated Delta_s t(x) (the hat version) for each edge s is independent of other edges;
#         returning the coeficient for edge statistics

#         '''
#         return self.coef[0] + self.coef[1]*del_t2s(X)


#     def get_datasource(self):
#         r = ro.r
#         r.source("../Rcode/utils.R")
#         return data.DSErgm(self.d, r.construct_e2s_model, self.coef)

#     def dim(self):
#         return len(self.d)
    
    
class ApproxModel(with_metaclass(ABCMeta, object)):
    """
    An abstract class of random graph models based on approximate conditional probability
    This is intended to be used to represent a model of the data for model validation
    """

    @abstractmethod    
    def cond_prob(self, X):
        '''
        The approximate conditional probability of \hat{q}(x^{(s,1)}|x_{-s})
        '''
        raise NotImplementedError()

    def t_fun(self, X):
        """
        The Delta_s t(x) for each edge s; logit q (x^{(s,1)}|x_{-s})
        """
        return np.log(1./(1./self.cond_prob(X)-1.))

    def log_normalized_prob(self, X):
        """
        Evaluate the exact normalised probability. The normalizer required. 
        This method is not essential. For sanity check when necessary
        Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this model.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    # @abstractmethod
    # def dim(self):
    #     """
    #     Return the dimension of the input.
    #     """
    #     raise NotImplementedError()


class ApproxEdgeStat(ApproxModel):
    """
    model with approximated conditional edge probability of ER graoph
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        # if smooth:
        #     self.EstEdgeProb(self.Xsample.X)
        # else:
        #     self.CountEdgeProb(self.Xsample.X)
        self.est_prob(self.Xsample.X)
        self._name="Approx_ER"

    def sample(self, n=100):
        return self.gen_model(n)        
    
    def est_prob(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        edge_count = X.sum()
        n = X.shape[0]
        d = X.shape[1]
        dc2 = d*(d-1)/2.
        #for undirected graph, average with removed diagonals
        prob = edge_count/(float(n) * 2.*dc2) 
        self.prob = prob
        self.coef = utils.logit(prob)
        
    
    def cond_prob(self, X):
        return self.prob * (X*0 + 1.)
    
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is independent of other edges;
        returning the coeficient for edge statistics
        '''
        return self.coef * (X*0 + 1.)


class ApproxE2StarStat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_E2S"

    def sample(self, n=100):
        return self.gen_model.sample(n)
    


    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        #remove diagonal terms
        # np.apply_along_axis(np.fill_diagonal, 0, deg_stat, val=0)
        # remove the presence of edge ij
        deg_stat -= 2*X
        return deg_stat

    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        idx, count = np.unique(deg_stat, return_counts = True)
        loc = np.searchsorted(idx, deg_stat)
        deg_count = np.bincount(loc.flatten(), X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--2(d-2)
        deg_list = np.arange(2*d)
        prob_list = np.ones(2*d) # * 0.5
        count_list = np.zeros(2*d)
        for i, x in enumerate(idx):
            prob_list[int(x)] = deg_prob[i]
            count_list[int(x)] = count[i]
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list
    
    def EstDegProb(self, X, method="krr", weighted=False):
        # prob_list, deg_list, count_list = self.CountDegProb(X)
        # prob_smooth, deg_smooth, _ = scipy.interpolate.splrep(deg_list,prob_list)
        deg_stat = self.DegStats(X)
        idx, count = np.unique(deg_stat, return_counts = True)
        loc = np.searchsorted(idx, deg_stat)
        deg_count = np.bincount(loc.flatten(), X.flatten())
        deg_prob = deg_count/count
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--2(d-2)
        deg_list = np.arange(2*d)
        
        if method == "krr":
            krr = kernel_ridge.KernelRidge(kernel="rbf", gamma=.3) #gaussian kernel default
            if weighted:
                krr.fit(idx[:,np.newaxis], deg_prob[:,np.newaxis], np.sqrt(deg_count+1e-6))
            else:
                krr.fit(idx[:,np.newaxis], deg_prob[:,np.newaxis])
            prob_list = krr.predict(deg_list[:,np.newaxis])
            prob_list = prob_list.clip(min=0.)
            prob_list = prob_list.clip(max=1.)

            krr_count = kernel_ridge.KernelRidge() #linear for count default
            krr_count.fit(idx[:,np.newaxis], deg_count[:,np.newaxis])
            count_list = krr_count.predict(deg_list[:,np.newaxis])
            count_list = count_list.clip(min=0)
            
        self.prob_list = prob_list
        self.deg_list = deg_list
        self.count_list = count_list
        self.prob_pred = krr
        return prob_list, deg_list, count_list

    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        prob = deg_stat * 0.

        for i, deg in enumerate(deg_list):
            prob[deg_stat == deg] = prob_list[i]
        
        return prob


    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
    


class ApproxE2StarStatCumulative(ApproxE2StarStat):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    The cumulative notion is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_E2S_cumulative"

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum degree between two vertices connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_stat = (deg_sum + abs(deg_diff))/2. - X
        return deg_stat
        # deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        # deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        # deg_stat = [deg_min_stat, deg_max_stat]
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        idx, count = np.unique(deg_stat, return_counts = True)
        loc = np.searchsorted(idx, deg_stat)
        deg_count = np.bincount(loc.flatten(), X.flatten())
        deg_prob = deg_count.cumsum()/count.cumsum()
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = np.arange(d)
        prob_list = np.zeros(d) # * 0.5
        count_list = np.zeros(d)
        count_list[0] = 1
        for i, x in enumerate(idx):
            prob_list[int(x)] = deg_prob[i]
            count_list[int(x)] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list



class ApproxBiDegStat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    The bi-variate degree is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_BiDeg"

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        deg_min_stat, deg_max_stat = deg_stat[0], deg_stat[1]
        
        deg_flat = np.array([deg_min_stat.flatten(), deg_max_stat.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = idx
        prob_list = np.zeros((d,d)) # * 0.5
        count_list = np.zeros((d,d))
        for i, x in enumerate(idx.T):
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list

    
    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.

        for i, deg in enumerate(deg_list.T):
            m1 = deg_stat[0] == deg[0]
            m2 = deg_stat[1] == deg[1]
            m = m1*m2
            prob[m] = prob_list[int(deg[0]), int(deg[1])]
        return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
