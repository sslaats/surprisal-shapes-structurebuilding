#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:59:07 2023

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

# overlap between histograms
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

# function to get a selection of indices from overlap
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

# %% get median surprisal / NGRAM
figi, axi = plt.subplots(ncols=2, figsize=(7,3), sharey=True, sharex=True)

for i,srprs in enumerate(['surprisal', 'surprisal_GPT']):
    # open the predictors
    predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')
    
    # calculate durations
    predictors['duration'] = predictors['offset'] - predictors['onset']

    quantile = 50

    # remove the median (we will not use this word)
    percentile_values = {}
    percentile_values[srprs] = np.percentile(predictors[srprs], q=quantile)
    surprisal_med = predictors.loc[ (predictors[srprs] == np.percentile(predictors[srprs], q=quantile))]
    predictors = predictors.drop(index=[surprisal_med.index[0]])
    predictors.reset_index(inplace=True)

    # get duration for high/low surprisal
    duration_high = predictors.loc[predictors[srprs] > percentile_values[srprs], 'duration']
    duration_low = predictors.loc[predictors[srprs] < percentile_values[srprs], 'duration']
    
    #fig,ax=plt.subplots(figsize=(4,3))
    #ax.hist(duration_high, alpha=0.7, bins=np.linspace(0,1.25,20))
    #ax.hist(duration_low, alpha=0.7, bins=np.linspace(0,1.25,20))
    #ax.legend(['High surprisal', 'Low surprisal'], frameon=False)
    #ax.set_xlabel('word duration (s)')
    #ax.set_ylabel('count')
    #sns.despine(ax)
    #ax.tight_layout()
    #fig.savefig(f'/project/3027007.06/results/no-entropy-topdown-durmatch/{srprs}-durations.svg')

    # t-test
    result = stats.ttest_ind(duration_high, duration_low)
    df = len(duration_low)+len(duration_high)-2
    print(df, result)
    
    # print means
    print(np.mean(duration_high), np.std(duration_high))
    print(np.mean(duration_low), np.std(duration_low))
    
    #  histogram intersection
    low_hist, bins = np.histogram(duration_low, bins=100, range=[0, 1.25])
    high_hist, _ = np.histogram(duration_high, bins=100, range=[0, 1.25])
    overlap = return_overlap(low_hist, high_hist)
    
    #fig,ax=plt.subplots(figsize=(4,3))
    axi[i].plot(bins[:-1],low_hist, color=sns.color_palette('Blues_r', n_colors=5)[0], lw=2, alpha=0.8)
    axi[i].plot(bins[:-1],high_hist, color=sns.color_palette('Reds_r', n_colors=5)[0], lw=2, alpha=0.8)
    axi[i].fill_between(x=bins[:-1], y1=overlap, color='orange', alpha=0.6)
    axi[i].margins(x=0, y=0)
    
    #axi[i].plot(bins[:-1],overlap)
    
    axi[i].set_xlabel('Word duration (s)')
    if i == 0:
        axi[i].set_title('Trigram surprisal')
        axi[i].set_ylabel('Count')
    else:
        axi[i].set_title('GPT2 surprisal')
        axi[i].legend(['Low surprisal', 'High surprisal', 'Overlap'], frameon=False)
        axi[i].get_yaxis().set_visible(False)
   
   # sns.despine()
   # plt.tight_layout()
   # figi.savefig('/project/3027007.06/results/no-entropy-topdown-durmatch/overlapping_duration_distributions.svg')
        
   
    
    # now we have found the overlapping distribution, so we must select from our data.
    # selecting
    high_surprisal = predictors.loc[predictors[srprs] > percentile_values[srprs]]
    low_surprisal = predictors.loc[predictors[srprs] < percentile_values[srprs]]
    
    high_idx = get_indices_from_overlap(high_surprisal, bins, overlap)
    low_idx = get_indices_from_overlap(low_surprisal, bins, overlap)
    
    # let's plot the durations
    fig,ax=plt.subplots(figsize=(4,3))
    ax.hist(predictors.iloc[high_idx]['duration'])
    ax.hist(predictors.iloc[low_idx]['duration'])
    #fig.savefig()
    
    # appears correct! Now let's try a quick t-test 
    difference = stats.ttest_ind(predictors.iloc[high_idx]['duration'], predictors.iloc[low_idx]['duration'])
    print(difference) # p = 0.96! they do not overlap
    
    # next up: surprisal values
    fig,ax=plt.subplots(figsize=(4,3))
    ax.hist(predictors.iloc[high_idx][srprs], alpha=0.5, bins=30, range=[0,30])
    ax.hist(predictors.iloc[low_idx][srprs], alpha=0.5, bins=30, range=[0,30])
    # this has worked.
    
    total_idx=np.vstack([np.asarray(high_idx), np.asarray(low_idx)])
    
    # we save the indices for future use
    index_df = pd.DataFrame(columns=['high_surprisal', 'low_surprisal'], index=range(total_idx.shape[1]), data=total_idx.T)
    #index_df.to_csv(f'/project/3027007.06/results/no-entropy-topdown-durmatch/indices-{srprs}.csv')
    

    #empty_array = np.empty((len(predictors)))
    #empty_array[:] = np.nan
    #predictors[f'bottomup-durmatch-{srprs}'] = empty_array
    #predictors.iloc[high_idx][f'bottomup-durmatch-{srprs}'] = predictors.iloc[high_idx]['bottomup']
    #predictors.iloc[low_idx][f'bottomup-durmatch-{srprs}'] = predictors.iloc[low_idx]['bottomup']
    

