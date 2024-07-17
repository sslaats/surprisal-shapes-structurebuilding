#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:36:18 2022

@author: sopsla
"""
from collect import compile_trfs, grandaverage, compile_rvals
import mne
import numpy as np
import os
from meg import TRF, CH_NAMES
import pickle

surtype = 'GPT'
BANDPASS = 'delta'
resultsdir = f'/project/3027007.06/results/no-entropy-topdown-durmatch/{BANDPASS}/{surtype}' # f'/project/3027007.06/results/no-entropy/{BANDPASS}/{surtype}'
datapath = '/project/3027007.01/processed/'
conditions = ['trf']
save=True

# %%
# pilot 8/9/10
if surtype == 'GPT':
    srprs = 'surprisal_GPT'
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

# %% PREPARE MEG DATA ####
with open(f'/project/3027007.06/results/no-entropy-topdown/features-{surtype}.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

# %%
## ACTUAL COMMANDS
trfs = compile_trfs(resultsdir, conditions=conditions, FEATURES=FEATURES, models=FEATURES.keys(), save=save, source=False)
info = trfs['bottomup_split_surprisal']['trf'][4].info

GA = grandaverage(trfs, info=info, FEATURES=FEATURES, models=FEATURES.keys(), tmin=-0.1, tmax=1.0, conditions=conditions, savedir=resultsdir, save=save)
r_data = compile_rvals(resultsdir, FEATURES=FEATURES, info=info, conditions=['trf'], models=FEATURES.keys(), source=False, save=save)

# %%

# remove participant with index 5, 'sub-002'
#trfs_copy = {}
#for model in FEATURES.keys():
 #   trfs_copy[model]={}
  #  trfs_copy[model]['trf'] = [t for t in trfs[model]['trf'] if trfs[model]['trf'].index(t) != 5]
    
# compute grandaverage on those
#GA = grandaverage(trfs_copy, info=info, models=FEATURES.keys(), tmin=-0.2, tmax=1.0, conditions=conditions, savedir=resultsdir, save=False)
