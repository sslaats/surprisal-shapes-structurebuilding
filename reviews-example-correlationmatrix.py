#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:35:56 2024

@author: sopsla
"""
import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from utils import temporal_smoother

# %% get the participants
datapath = '/project/3027007.01/processed/'
stories_path = '/home/lacnsg/sopsla/Documents/audiobook-transcripts/syntax/transcripts/'
subjects = [f for f in os.listdir(datapath) if len(f) == 7]
subjects.remove('sub-002')

stories = os.listdir(stories_path)

# we pick a random subject
subject = subjects[np.random.randint(0, len(subjects))]

# %% load the features, the envelopes and the epochs 
surtype = 'ngram'

if surtype == 'GPT':
    srprs = 'surprisal_GPT'
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

predictors = pd.read_csv('/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-3-gram-GPT2.csv')

with open(f'/project/3027007.06/results/main-effects/features-{surtype}.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

with open(f'/project/3027007.06/data/epochs/sensor/unfiltered/{subject}-epochs.pkl', 'rb') as f:
    epochs = pickle.load(f)

with open(f'/project/3027007.06/metadata/envelopes/{subject}-envs.pkl', 'rb') as f:
    envs = pickle.load(f)
    
    
# %% create the features
X = []
features = FEATURES['main_distributional_topdown_bottomup']

for story in stories:
    story = story[:-12]
    sigma = {}
    for feature in ['wordonset','frequency', entr, srprs, 'topdown', 'bottomup']:
        sigma[feature] = np.abs(predictors[feature]).mean()
            
    storystats = predictors.loc[predictors['story'] == story]
    storymeg = epochs[story]
    
    N = envs[story].shape[0]

    x = np.zeros((N, len(features)))

    x[:,0] = envs[story] / np.abs(envs[story]).mean()
    
    # get the boundaries & put in the features
    x[storymeg.time_as_index(storystats['onset']),1] = 1
    
    for i,feature in enumerate(features[2:]):
        x[storymeg.time_as_index(storystats['onset']),i+2] = storystats[feature] / sigma[feature]
        
    x[:,1:] = temporal_smoother(x[:,1:], fs=200, std_time=0.015)
    X.append(x)
    
X = np.concatenate(X)

# %%
features_df = pd.DataFrame(data=X, columns=features)
    
# %% create the correlation matrix
total_corr = features_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(total_corr, dtype=bool))
sns.set_theme(style='white')
sns.set_context('talk')

fig,ax = plt.subplots(figsize=(8,8))
heat = sns.heatmap(total_corr, vmin=-1, vmax=1, center = 0,
                 cmap = sns.diverging_palette(220,20, n=200),
                 square=True,
                 annot = True,
                 ax=ax,
                 mask=mask,
                 linewidths=.5,
                 cbar=True,
                 cbar_kws={'shrink':.5})

for item in heat.get_xticklabels():
    item.set_rotation(90)
    
ax.set_title("Correlation between predictors")    

total_inv = np.linalg.inv(total_corr)

fig,ax=plt.subplots(figsize=(6,3))

bars = sns.barplot(x=features, y=np.diagonal(total_inv), palette=sns.diverging_palette(20,220, n=200), ax=ax)
ax.axhline(y=5, xmin=0, xmax=1, linestyle='--')
ax.set_title("Variance Inflation Factor")
ax.set_ylabel('VIF')
for item in bars.get_xticklabels():
    item.set_rotation(45)
    
sns.despine()
fig.tight_layout()

