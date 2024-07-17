#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:02:56 2022

@author: sopsla
"""
import numpy as np
import scipy.stats as stats
from pyeeg.utils import lag_span
import mne
from meg import CH_NAMES


def cluster_2samp(trfs, info, tmin=-0.2, tmax=0.8, multiplication=1):
    """
    Computes 2-sample cluser-based permutation test
    
    trfs: list of lists of mne.Evoked
    """
    lags = lag_span(tmin, tmax, 200)
    
    # reading the adjacency
    idx = mne.channel_indices_by_type(info, picks=info.ch_names)
    adjacency, _ = mne.channels.read_ch_adjacency('ctf275', idx['mag'])
    
    nsubj = len(trfs[0])
    
    # preparing the data
    cond1 = np.zeros((nsubj, len(lags), len(CH_NAMES)))
    cond2 = np.zeros((nsubj, len(lags), len(CH_NAMES)))
    
    for cond, data in zip([cond1, cond2], trfs):
        for i,d in enumerate(data):
            cond[i, :, :] = d._data.T
    
    # setting the statfun
    statfun = lambda x, y: mne.stats.cluster_level.ttest_1samp_no_p(x-y) #None #
    threshold = stats.t.ppf(1-0.05/2, df=nsubj-1) * multiplication  #stats.f.ppf(1-0.05, dfd=nsubj-1, dfn=1)  # stats.t.ppf(1-0.05/2, df=nsubj-1) * multiplication
    tail = 0
    
    cluster_stats = mne.stats.spatio_temporal_cluster_test([cond1, cond2],
                                                           threshold = threshold,
                                                           n_permutations = 10000,
                                                           tail = tail,
                                                           stat_fun = statfun,
                                                           max_step = 1,
                                                           n_jobs = 1,
                                                           buffer_size = None,
                                                           adjacency = adjacency)
    
    print("There are %d significant clusters\n"%(cluster_stats[2]<0.05).sum())
    
    return cluster_stats

def cluster_1samp():
    raise NotImplementedError()
    pass