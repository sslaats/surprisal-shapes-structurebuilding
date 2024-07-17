#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:32:09 2023

@author: sopsla
"""
import pickle
import pandas as pd
import numpy as np

# %% load the predictors
surtype = 'GPT'

if surtype == 'GPT':
    srprs = 'surprisal_GPT'
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

NUMBERGRAM = 3

predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')

# %% take the 50% percentile, remove the occurrence
quantile = 50 

percentile_values = {}
percentile_values[srprs] = np.percentile(predictors[srprs], q=quantile)
percentile_values[entr] = np.percentile(predictors[entr], q=quantile)

surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]
entropy_med = predictors.loc[ (predictors[entr] == np.percentile(predictors[entr], q=quantile))]

predictors = predictors.drop(index=[entropy_med.index[0], surprisal_med.index[0]])

# %%
# for a model that has continuous surprisal: use main-effects/main_surprisal_bottomup

FEATURES = {
            'bottomup_split_random': ['envelope', 'wordonset', 'frequency', srprs,   'bottomup_low_random', 'bottomup_high_random'],
            'bottomup_split_surprisal': ['envelope', 'wordonset', 'frequency', srprs, 'bottomup_low_surprisal', 'bottomup_high_surprisal'],
            'bottomup_high_surprisal': ['envelope', 'wordonset', 'frequency', srprs,   'bottomup_high_surprisal'],
            'bottomup_low_surprisal': ['envelope', 'wordonset', 'frequency', srprs,   'bottomup_low_surprisal'],
            'bottomup_splitWF_surprisal':['envelope', 'wordonset', srprs, 'bottomup', 'frequency_low_surprisal', 'frequency_high_surprisal'],
            'bottomup_splitWF_random': ['envelope', 'wordonset', srprs, 'bottomup', 'frequency_low_random', 'frequency_high_random']
            }
with open(f'/project/3027007.06/results/no-entropy-topdown/features-{surtype}.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# %% get all the features separately
allfeats = set([f for mod,ft in FEATURES.items() for f in ft])
allfeats.remove('envelope')

# %%
for test_story in set(predictors['story']):
    print(f'Creating predictors for the test story {test_story}')
    save_predictors = predictors.copy()
    
    train_predictors = predictors.loc[predictors['story'] != test_story]
    test_predictors = predictors.loc[predictors['story'] == test_story]
    
     # % model
        # estimate mu and sigma for the predictors in train
    sigma = {}
    for feature in ['wordonset', 'frequency', entr, srprs, 'topdown', 'bottomup']:
        sigma[feature] = np.abs(train_predictors[feature]).mean()
               
    for feature in allfeats:
        if feature not in predictors.columns:
            # z-score with the overall feature means, not the split values   
            syntype, direction, distype = feature.split('_')

            if surtype == 'GPT':
                distype = distype + '_GPT'
            
            # split the predictor - have to do this here otherwise the z-scoring will turn zeros into another value (and they have to be zero!)
            if direction == 'low':
                save_predictors.loc[:,feature] = predictors.loc[ (predictors[f'{distype}'] < percentile_values[f'{distype}']), f'{syntype}'] / sigma[syntype]
           
            elif direction == 'high':
                save_predictors.loc[:,feature]= predictors.loc[ (predictors[f'{distype}'] > percentile_values[f'{distype}']), f'{syntype}'] / sigma[syntype]
                
            elif direction == 'int':
                # create the interaction feature
                save_predictors.loc[:,feature] = (predictors[syntype] / sigma[syntype]) * (predictors[distype] / sigma[distype])
                save_predictors = save_predictors.fillna(0)
                
            else:
                save_predictors[feature] = predictors[feature] / sigma[feature]
                
    save_predictors.to_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/{test_story}-test-predictors.csv')
