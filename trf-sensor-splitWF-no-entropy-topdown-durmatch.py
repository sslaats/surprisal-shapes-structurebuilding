#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:00:06 2022

@author: sopsla
"""
# %%  import modules
import os
import sys
import pickle
import gc

import random
from pyeeg.utils import lag_matrix, lag_span
import pandas as pd
import numpy as np
import scipy.stats as stats
from statistics import mode
from utils import temporal_smoother
from meg import TRF
"""
Test-pipeline for full TRF-model.

Characteristics:
    - SCALING ONLY, NO CENTERING
    - surprisal only, but RESIDUALIZED against word length
    - bottom-up only

"""
# %% paths, stories & subjects #####
surtype = 'ngram'
datapath = '/project/3027007.01/processed/'
stories_path = '/home/lacnsg/sopsla/Documents/audiobook-transcripts/syntax/transcripts/'
subjects = [f for f in os.listdir(datapath) if len(f) == 7] # ['sub-009', 'sub-021', 'sub-025', 'sub-014', 'sub-003', 'sub-018', 'sub-033', 'sub-002', 'sub-010', 'sub-011'] 
subjects.remove('sub-002')

stories = os.listdir(stories_path) 
subject = subjects[int(sys.argv[1])] # for bash scripting

# %% settings
bp_dict = {'delta': [0.5, 4], 'theta': [4, 10], 'gamma': [30, 50], 'highgamma':[70,100]}

if surtype == 'GPT':
    srprs = 'surprisal_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'

# bandpass
BANDPASS = 'delta'
hp = bp_dict[BANDPASS][0] # delta TRF
lp = bp_dict[BANDPASS][1] # delta TRF
NUMBERGRAM = 3
quantile = 50 # for splitting the data
fs = 200
alpha = np.logspace(1,3, num=20, base=500)  #np.logspace(1,2, num=9, base=500) #np.logspace(-1,3, num=20, base=50000) # test value.
tmin = -0.1
tmax= 1.0
fit_intercept = True
include_nan = False

# %% features and folders
save_dir = f'/project/3027007.06/results/no-entropy-topdown-durmatch/{BANDPASS}/{surtype}'
predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')

with open(f'/project/3027007.06/results/no-entropy-topdown/features-{surtype}.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

durmatch_indices = pd.read_csv(f'/project/3027007.06/results/no-entropy-topdown-durmatch/indices-{srprs}.csv')
durmatch_indices['high_surprisal'].values.sort()
durmatch_indices['low_surprisal'].values.sort()

try:
    os.makedirs(save_dir)
except FileExistsError:
    # directory already exists
    pass

dirname = os.path.join(save_dir, str(subject))
try:
    os.makedirs(os.path.join(save_dir, str(subject)))
except FileExistsError:
    # directory already exists
    pass

# %% #### MEG #####
with open(f'/project/3027007.06/data/epochs/sensor/unfiltered/{subject}-epochs.pkl', 'rb') as f:
    epochs = pickle.load(f)

with open(f'/project/3027007.06/metadata/envelopes/{subject}-envs.pkl', 'rb') as f:
    envs = pickle.load(f)

for story in stories:
    story = story[:-12]

    # filter into appropriate frequency band    
    epoch = epochs[story]
    epoch = epoch.filter(hp, lp, fir_design='firwin')
    #epoch = epoch.resample(sfreq=fs)
    
    # take power
    if BANDPASS != 'delta':
        epoch.apply_hilbert(envelope=True)
        epoch = epoch.filter(0.5, 20, fir_design='firwin')
        
    epochs[story] = epoch
  
info = epoch.info # for future use
del epoch
gc.collect()

lags = lag_span(tmin, tmax, fs)

# %% take the 50% percentile, remove the occurrence
percentile_values = {}
percentile_values[srprs] = np.percentile(predictors[srprs], q=quantile)
surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]

predictors = predictors.drop(index=[surprisal_med.index[0]])
predictors.reset_index(inplace=True)

#%% ### create the models###########
for model, features in FEATURES.items():
    
    if os.path.isfile(os.path.join(save_dir, str(subject), f'TRF_{model}.pkl')) and os.path.isfile(os.path.join(save_dir, str(subject), f'R_{model}.pkl')):
        print(f'{model} done already. Moving on...')
        continue
    else:
        print(f'Estimating TRF model {model}...')
        
    # we pick all stories as test-story once
    all_betas = []
    all_rvals = []    
    
    for test_story in stories:
    
        # % model
        # estimate mu and sigma for the predictors in train
        sigma = {}
        for feature in ['wordonset','frequency', srprs, 'topdown', 'bottomup']:
            sigma[feature] = np.abs(predictors.loc[predictors['story'] != test_story, feature]).mean()
                   
        # set up lists for training and testing data
        test = []
        x_train = []
        y_train = []
        
        # set up the covariance matrix
        XtX = np.zeros((len(lags)*len(features) + int(fit_intercept), len(lags)*len(features) + int(fit_intercept)))
        Xty = np.zeros((len(lags)*len(features) + int(fit_intercept), epochs[story].info['nchan']))
        
        for story in stories:
            story = story[:-12]
            #print(f"Making predictor for {story}...")
            storystats = predictors.loc[predictors['story'] == story]
            
            min_idx = min(storystats.index)
            max_idx = max(storystats.index)
            
            durmatch_indices_high = durmatch_indices.loc[(durmatch_indices['high_surprisal'] >= min_idx) & (durmatch_indices['high_surprisal'] <= max_idx), 'high_surprisal'].values
            durmatch_indices_low = durmatch_indices.loc[(durmatch_indices['low_surprisal'] >= min_idx) & (durmatch_indices['low_surprisal'] <= max_idx), 'low_surprisal'].values

            storymeg = epochs[story]
            
            N = np.shape(storymeg[0]._data)[2]
            
            # initiate dataframe
            x = np.zeros((N,len(features)))
            
            # put in the envelope
            x[:,0] = envs[story] / np.abs(envs[story]).mean()
            
            # get the boundaries & put in the features
            x[storymeg.time_as_index(storystats['onset']),1] = 1
                
            if len(features) > 2:
                
                # get indices for random split - should NOT be done for each feature
                #ix = list(range(len(storystats)))
                #indices = np.random.choice(ix, size=int(len(ix)/2), replace=False)
                #indices.sort()
                
                # get index of indices
                #ix_of_indices = [np.where(ix == i) for i in indices]
                #op_indices = np.delete(ix, ix_of_indices)
                #op_indices.sort()
                                                 
                for i,feature in enumerate(features[2:]):
                    
                    # z-score with the overall feature means, not the split values
                    if feature in ['bottomup_high_surprisal', 'bottomup_low_surprisal', 'bottomup_high_random', 'bottomup_low_random',
                                   'frequency_high_surprisal','frequency_low_surprisal', 'frequency_high_random', 'frequency_low_random']:
                        
                        syntype, direction, distype = feature.split('_')
                        
                        if distype != 'random':
    
                            if surtype == 'GPT':
                                distype = distype + '_GPT'
                            
                            # split the predictor - have to do this here otherwise the z-scoring will turn zeros into another value (and they have to be zero!)
                            if direction == 'low':
                                storystats.loc[durmatch_indices_low, feature] = storystats.loc[durmatch_indices_low,f'{syntype}'] / sigma[syntype]
                            elif direction == 'high':
                                storystats.loc[durmatch_indices_high, feature] = storystats.loc[durmatch_indices_high,f'{syntype}'] / sigma[syntype]
                                
                            elif direction == 'lowvar':
                                raise NotImplementedError('Variance-related split not yet implemented')
                                # code here
                            elif direction == 'highvar':
                                # code here
                                raise NotImplementedError('Variance-related split not yet implemented')
                                
                            elif direction == 'int':
                                storystats.loc[:,feature] = (storystats[f'{distype}'] / sigma[f'{distype}']) * (storystats[f'{syntype}'] / sigma[f'{syntype}'])
        
                        elif distype == 'random':
                            all_indices = np.hstack([durmatch_indices_high, durmatch_indices_low])
                            ix_of_ix = random.sample(list(range(0, len(all_indices))), int(len(durmatch_indices_low)))
                            op_indices = np.delete(all_indices, ix_of_ix)
                            
                            indices=all_indices[ix_of_ix]
                            
                            if direction == 'low':
                                storystats.loc[:,feature] = storystats.loc[indices, f'{syntype}'] / sigma[syntype]
                            elif direction == 'high':
                                storystats.loc[:,feature] = storystats.loc[op_indices, f'{syntype}'] / sigma[syntype]
                            
                        else:
                            raise NotImplementedError(f'Distype {distype} not implemented, error betw line 187 and 213')
                
                        storystats = storystats.fillna(0)
                        
                        x[storymeg.time_as_index(storystats['onset']),i+2] = storystats[feature]
                        
                    else:
                        x[storymeg.time_as_index(storystats['onset']),i+2] = storystats[feature] / sigma[feature]
        
                # apply temporal smoothing for regularization
                if len(features) > 1:
                    x[:,1:] = temporal_smoother(x[:,1:], fs=fs, std_time=0.015)
                    
                # z-score the data too
                y = stats.zscore(np.squeeze(storymeg._data)).T
                
                # set up lag matrix
                X = lag_matrix(x, lags)
                
                nan_rows = np.isnan(X.mean(1))
                y = y[~nan_rows]
                X = X[~nan_rows]
            
                if story != test_story[:-12]:
                    x_train.append(X)
                    y_train.append(y)
                    
                    if fit_intercept:
                        X = np.hstack([np.ones((X.shape[0],1)),X])
                    
                    XtX += X.T @ X
                    Xty += X.T @ y
                
                else: 
                    test.append((X, y))

        # speed up computation for multiple alphas - compute TRFs
        u, s, v = np.linalg.svd(XtX) 
        betas = [(u @ np.diag(1/(s+a))) @ v @ Xty for a in alpha] 
        
        print(f"Done with test story {test_story}! Predicting...")
        
        X = test[0][0]
        y = test[0][1]
        
        X = np.hstack([np.ones((X.shape[0],1)), X])
        yhat = X @ betas # shape: len of (alphas, samples, channels)     
        
        scores = np.asarray([np.diag(np.corrcoef(yhat[a, :, :], y, rowvar=False), k=info['nchan'])
                             for a in range(len(alpha))])  # shape: alphas, channels
        
        all_betas.append(betas)
        all_rvals.append(scores)
    
    #  get the highest alpha
    scores = np.asarray(all_rvals)
    peaks = scores.mean(-1).argmax(1)  # Take the mean over sensors & maximum value over alphas
    best_alpha = alpha[mode(peaks)]
    
    # take the betas for the best alpha
    all_betas = np.asarray(all_betas)
    all_betas = all_betas[:,mode(peaks),1:,:] # the 1: removes the intercept
    
    # get the mean over the stories for both betas and rvalues
    coef = np.mean(all_betas, axis=0).reshape(len(lags), len(features), -1)
    R_model = np.mean(scores[:,mode(peaks),:], axis=0)
    
    # turn into TRF model for saving
    args = lags, info, features
    TRF_model = TRF(coef, *args)

# %%
    # save the average model
    with open(os.path.join(save_dir, str(subject), f'TRF_{model}.pkl'), 'wb') as f:
        trf = {}
        trf['alpha'] = best_alpha
        trf['trf'] = TRF_model
        pickle.dump(trf, f)
    
    with open(os.path.join(save_dir, str(subject), f'R_{model}.pkl'), 'wb') as fr:
        r_values = {}
        r_values['r'] = R_model
        pickle.dump(r_values, fr)

