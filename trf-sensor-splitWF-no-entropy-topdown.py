#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:00:06 2022

@author: sopsla
"""
# %%  import modules
import os
import mne
import sys
import pickle
import gc

from pyeeg.utils import lag_matrix, lag_span
import pandas as pd
import numpy as np
import scipy.stats as stats
from statistics import mode

#local modules
from meg import crossval, TRF, CH_NAMES
from utils import story_to_triggers, get_audio_envelope, temporal_smoother
"""
Test-pipeline for full TRF-model.

Characteristics:
    - No envelope to avoid need for regularization
    - No word idx
    - SCALING ONLY, NO CENTERING
    - surprisal and entropy, also for split

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
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

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
save_dir = f'/project/3027007.06/results/no-entropy-topdown/{BANDPASS}/{surtype}'
predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')

with open(f'/project/3027007.06/results/no-entropy-topdown/features-{surtype}.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

# predictor statistics for the variance
#with open(os.path.join(save_dir, 'sentence-variance.pkl'), 'rb') as f:
 #   sentence_variance = pickle.load(f)

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
percentile_values[entr] = np.percentile(predictors[entr], q=quantile)

surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]
entropy_med = predictors.loc[ (predictors[entr] == np.percentile(predictors[entr], q=quantile))]

predictors = predictors.drop(index=[entropy_med.index[0], surprisal_med.index[0]])
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
        for feature in ['wordonset','frequency', entr, srprs, 'topdown', 'bottomup']:
            sigma[feature] = np.abs(predictors[feature]).mean()
                   
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
                ix = list(range(len(storystats)))
                indices = np.random.choice(ix, size=int(len(ix)/2), replace=False)
                indices.sort()
                
                # get index of indices
                ix_of_indices = [np.where(ix == i) for i in indices]
                op_indices = np.delete(ix, ix_of_indices)
                op_indices.sort()
                                                 
                for i,feature in enumerate(features[2:]):
                    
                    # z-score with the overall feature means, not the split values
                    if feature in ['topdown_high_entropy', 'topdown_low_entropy', 'bottomup_high_entropy', 'bottomup_low_entropy', \
                                   'topdown_high_surprisal', 'topdown_low_surprisal', 'bottomup_high_surprisal', 'bottomup_low_surprisal',
                                   'topdown_high_random', 'topdown_low_random', 'bottomup_high_random', 'bottomup_low_random',
                                   'bottomup_int_surprisal', 'bottomup_int_entropy', 'topdown_int_surprisal', 'topdown_int_entropy',
                                   'frequency_high_surprisal','frequency_low_surprisal', 'frequency_high_random', 'frequency_low_random']:
                        
                        syntype, direction, distype = feature.split('_')
                        
                        if distype != 'random':
    
                            if surtype == 'GPT':
                                distype = distype + '_GPT'
                            
                            # split the predictor - have to do this here otherwise the z-scoring will turn zeros into another value (and they have to be zero!)
                            if direction == 'low':
                                storystats.loc[:,feature] = storystats.loc[ (storystats[f'{distype}'] < percentile_values[f'{distype}']), f'{syntype}'] / sigma[syntype]
                            elif direction == 'high':
                                storystats.loc[:,feature] = storystats.loc[ (storystats[f'{distype}'] > percentile_values[f'{distype}']), f'{syntype}'] / sigma[syntype]
                                
                            elif direction == 'lowvar':
                                raise NotImplementedError('Variance-related split not yet implemented')
                                # code here
                            elif direction == 'highvar':
                                # code here
                                raise NotImplementedError('Variance-related split not yet implemented')
                                
                            elif direction == 'int':
                                storystats.loc[:,feature] = (storystats[f'{distype}'] / sigma[f'{distype}']) * (storystats[f'{syntype}'] / sigma[f'{syntype}'])
        
                        elif distype == 'random':
                                               
                            if direction == 'low':
                                storystats.loc[:,feature] = storystats.loc[storystats['story'] == story, f'{syntype}'].iloc[indices] / sigma[syntype]
                            elif direction == 'high':
                                storystats.loc[:,feature] = storystats.loc[storystats['story'] == story, f'{syntype}'].iloc[op_indices] / sigma[syntype]
                            
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

