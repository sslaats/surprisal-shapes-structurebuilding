#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:07:58 2022

@author: sopsla
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import itertools
import os
from utils import get_word_frequency

# %%
NUMBERGRAM = 3
predictors = pd.read_csv(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/predictors/predictors-syntax-{NUMBERGRAM}-gram-GPT2.csv')

surtype = 'GPT'

# %%
cols = list(predictors.columns)
cols.remove('story')
cols.remove('sentenceidx')
cols.remove('word')
cols.remove('index')
cols.remove('onset')
cols.remove('offset')
cols.remove('TP')
cols.remove('wordonset')
#cols.remove('leftcorner')
#cols.remove('topdown')
cols.remove('bottomup')
cols.remove('Unnamed: 0')

cols.remove('wordidx')

if surtype == 'GPT':
    cols.remove('entropy')
    cols.remove('surprisal')
    
elif surtype == 'ngram':
    cols.remove('entropy_GPT')
    cols.remove('surprisal_GPT')
    
# zscore
included = predictors[cols]
included = included.apply(stats.zscore)

total_corr = included.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(total_corr, dtype=bool))
sns.set_theme(style='white')
sns.set_context('talk')

fig,ax = plt.subplots(nrows=2, figsize=(11,19),gridspec_kw={'height_ratios': [2, 1]})
heat = sns.heatmap(total_corr, vmin=-1, vmax=1, center = 0,
                 cmap = sns.diverging_palette(220,20, n=200),
                 square=True,
                 annot = True,
                 ax=ax[0],
                 mask=mask,
                 linewidths=.5,
                 cbar=True,
                 cbar_kws={'shrink':.5})

for item in heat.get_xticklabels():
    item.set_rotation(90)

ax[0].set_title(f"Correlation between predictors")

total_inv = np.linalg.inv(total_corr)

bars = sns.barplot(x=cols, y=np.diagonal(total_inv), palette=sns.diverging_palette(20,220, n=200), ax=ax[1])
ax[1].axhline(y=5, xmin=0, xmax=1, linestyle='--')
ax[1].set_title("Variance Inflation Factor")
ax[1].set_ylabel('VIF')
for item in bars.get_xticklabels():
    item.set_rotation(45)
    
sns.despine()
fig.tight_layout()

fig.savefig(f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/vif/corr-withLC-nobottomup-{surtype}.svg')

print('COMPLETE VIF:', np.diagonal(total_inv))

