#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:40:13 2023

@author: sopsla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import random 
import statsmodels.api as sm
import statsmodels.formula.api as smf


# open the stimulus stats
NUMBERGRAM = 3
predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')

# calculate durations
predictors['duration'] = predictors['offset'] - predictors['onset']

"""
# %% OBTAIN RESIDUALS - NOT USED
# model trigram
model = smf.ols('surprisal ~ duration', data=predictors)
results = model.fit()

predictors['residual_surprisal'] = results.resid

# model GPT
model_GPT = smf.ols('surprisal_GPT ~ duration', data=predictors)
results_GPT = model_GPT.fit()

predictors['residual_surprisal_GPT'] = results_GPT.resid

# save this - NOT USED
predictors.to_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2-residuals.csv')
"""

# %% get median surprisal / NGRAM
srprs = 'surprisal'
quantile = 50

percentile_values = {}
percentile_values[srprs] = np.percentile(predictors[srprs], q=quantile)
surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]
predictors = predictors.drop(index=[surprisal_med.index[0]])
predictors.reset_index(inplace=True)

# %% get duration for high/low surprisal
duration_high = predictors.loc[predictors[srprs] > percentile_values[srprs], 'duration']
duration_low = predictors.loc[predictors[srprs] < percentile_values[srprs], 'duration']

fig,ax=plt.subplots(figsize=(4,3))
ax.hist(duration_high, alpha=0.7, bins=np.linspace(0,1.25,20))
ax.hist(duration_low, alpha=0.7, bins=np.linspace(0,1.25,20))
ax.legend(['High surprisal', 'Low surprisal'], frameon=False)
ax.set_xlabel('word duration (s)')
ax.set_ylabel('count')
sns.despine()
plt.tight_layout()

# %% t-test
result = stats.ttest_ind(duration_high, duration_low)
df = len(duration_low)+len(duration_high)-2

# %% print means
print(np.mean(duration_high), np.std(duration_high))
print(np.mean(duration_low), np.std(duration_low))

# %% histogram intersection
def return_overlap(hist_1, hist_2):
    zeros = np.zeros(len(hist_1))
    
    for idx in range(len(hist_1)):
        bin1 = hist_1[idx]
        bin2 = hist_2[idx]

        if bin1 == bin2:
            zeros[idx] = bin1
        elif bin1 > bin2:
            zeros[idx] = bin2
        elif bin2 > bin1:
            zeros[idx] = bin1
        
    return zeros
        
low_hist, bins = np.histogram(duration_low, bins=100, range=[0, 1.25])
high_hist, _ = np.histogram(duration_high, bins=100, range=[0, 1.25])

overlap = return_overlap(low_hist, high_hist)

fig,ax=plt.subplots(figsize=(4,3))
ax.plot(low_hist)
ax.plot(high_hist)
ax.plot(overlap)

# now we have found the overlapping distribution, so we must select from our data.
# %% selecting data
predictors[f'high_{srprs}'] = np.zeros((len(predictors)))
predictors[f'low_{srprs}'] = np.zeros((len(predictors)))

predictors[f'high_{srprs}'][np.where( predictors[srprs] > percentile_values[srprs])[0] ] =  predictors.loc[predictors[srprs] > percentile_values[srprs], srprs]
predictors[f'low_{srprs}'][np.where( predictors[srprs] < percentile_values[srprs])[0] ] =  predictors.loc[predictors[srprs] < percentile_values[srprs], srprs]

predictors[f'high_{srprs}_durmatch'] = np.zeros((len(predictors)))
predictors[f'low_{srprs}_durmatch'] = np.zeros((len(predictors)))

# %% function to get the values...

def get_indices_from_overlap(predictors, bins, overlap):
    
    row_indices = []
    
    for idx,(start,number) in enumerate(zip(bins,overlap)):
        end = bins[idx+1]
        duration_of_bin = predictors.loc[(predictors['duration'] >= start) & (predictors['duration'] < end)]   
        
        if len(duration_of_bin) > 0:
            selected_idx = random.sample(list(range(0, len(duration_of_bin))),int(number))
            row_indices.append(np.asarray(duration_of_bin.index)[selected_idx])
        
        else:
            continue
        
    return [value for bin_n in row_indices for value in bin_n]

# %% try it...
high_surprisal = predictors.loc[predictors[srprs] > percentile_values[srprs]]
low_surprisal = predictors.loc[predictors[srprs] < percentile_values[srprs]]

high_idx = get_indices_from_overlap(high_surprisal, bins, overlap)
low_idx = get_indices_from_overlap(low_surprisal, bins, overlap)

# %% let's plot the durations
plt.hist(predictors.iloc[high_idx]['duration'])
plt.hist(predictors.iloc[low_idx]['duration'])

# this is correct! Now let's try a quick t-test 
difference = stats.ttest_ind(predictors.iloc[high_idx]['duration'], predictors.iloc[low_idx]['duration'])
print(difference) # p = 0.96!

# %% next up: surprisal values
plt.hist(predictors.iloc[high_idx][srprs], alpha=0.5, bins=30, range=[0,30])
plt.hist(predictors.iloc[low_idx][srprs], alpha=0.5, bins=30, range=[0,30])
# still a nice split...

# %%

"""
# %% now let's do the same for residualized surprisal - NOT USED
predictors= pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2-residuals.csv')

# %% get median surprisal / NGRAM

srprs = 'residual_surprisal'
quantile = 50

percentile_values = {}
percentile_values[srprs] = np.percentile(predictors[srprs], q=quantile)
surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]
predictors = predictors.drop(index=[surprisal_med.index[0]])
predictors.reset_index(inplace=True)

# %% get duration for high/low surprisal
duration_high = predictors.loc[predictors[srprs] > percentile_values[srprs], 'duration']
duration_low = predictors.loc[predictors[srprs] < percentile_values[srprs], 'duration']

fig,ax=plt.subplots(figsize=(4,3))
ax.hist(duration_high, alpha=0.7)
ax.hist(duration_low, alpha=0.7)
ax.legend(['High surprisal', 'Low surprisal'], frameon=False)
ax.set_xlabel('word duration (s)')
ax.set_ylabel('count')
sns.despine()
plt.tight_layout()

# %% t-test
result = stats.ttest_ind(duration_high, duration_low)
df = len(duration_low)+len(duration_high)-2

# %% print means
print(np.mean(duration_high), np.std(duration_high))
print(np.mean(duration_low), np.std(duration_low))
"""
