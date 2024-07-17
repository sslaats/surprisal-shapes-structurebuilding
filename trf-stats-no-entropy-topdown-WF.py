#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:51:57 2022

@author: sopsla
"""
from stats import cluster_2samp
from viz import plot_2samp
import seaborn as sns

import pickle
import os
import numpy as np

tmin=-0.1
tmax=1.0

# %%
BANDPASS = 'delta'
surtype='ngram'
resultsdir = f'/project/3027007.06/results/no-entropy-topdown-durmatch/{BANDPASS}/{surtype}'

if surtype == 'GPT':
    srprs = 'surprisal_GPT'
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

#%%

with open(os.path.join(resultsdir, 'trfs.pkl'), 'rb') as f:
    trfs = pickle.load(f)

with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as f:
    GA = pickle.load(f)
    
# %% analyses

analyses = {}

model = 'bottomup_splitWF_surprisal' #stable_split_{stat}'
info = trfs[model]['trf'][0].info

highfeat = 'frequency_high_surprisal'
lowfeat = 'frequency_low_surprisal'

high = [i[highfeat] for i in trfs[model]['trf']]
low = [i[lowfeat] for i in trfs[model]['trf']]

clusterstat = cluster_2samp([high, low], info, tmin=tmin, tmax=tmax, multiplication = 1)
trf1 = GA[model]['trf'][highfeat].copy()
trf2 = GA[model]['trf'][lowfeat].copy()
        
analyses['frequency_surprisal'] = [clusterstat, trf1, trf2]
        
# %% save the results
trfstats = {}
for ft,reslist in analyses.items():
    trfstats[ft] = reslist[0]

with open(os.path.join(resultsdir, 'trf_stats_WF.pkl'), 'wb') as f:
    pickle.dump(trfstats, f)

# %% load results for plotting
with open(os.path.join(resultsdir, 'trf_stats_WF.pkl'), 'rb') as f:
    trfstats = pickle.load(f)
    
analyses['frequency_surprisal'] = [trfstats['frequency_surprisal'], 
                                  GA['bottomup_splitWF_surprisal']['trf']['frequency_high_surprisal'].copy(),
                                  GA['bottomup_splitWF_surprisal']['trf']['frequency_low_surprisal'].copy()]

# %% plot
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mne.viz.evoked import _handle_spatial_colors
import matplotlib.pyplot as plt
from viz import _rgb
import mne
from meg import CH_NAMES

info = trfs['bottomup_splitWF_surprisal']['trf'][6].info

idx = mne.channel_indices_by_type(info, picks=CH_NAMES)['mag']
        
chs = [info['chs'][i] for i in idx] 
locs3d = np.array([ch['loc'][:3] for ch in chs])
x,y,z = locs3d.T
colorz = _rgb(x,y,z)

plt.rcParams['axes.xmargin'] = 0

plot_style= 'sensors' #'clusters'
style = 'seaborn-paper'
plt.style.use(style)
vlim = 0.003

# %%
#colorz2 = [[int(i[0]*1.1), int(i[1]*1.1),int(i[2]*1.1)] for i in colorz]

colors =  [sns.color_palette("Blues_r",269), sns.color_palette("Reds_r", 269)]
#colors = [colorz, colorz]

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3),
                      gridspec_kw={'height_ratios':[1], 'width_ratios':[3,1]})

clusterstat, trf1, trf2 = analyses['frequency_surprisal']
ax_divider = make_axes_locatable(ax[1])
cax = ax_divider.append_axes("right", size='10%', pad=0.1)
axes = np.array([ax[0], ax[1], cax])
plot_2samp(trf=[trf1, trf2], cluster_stats=clusterstat, info=info, tmin=-0.05, tmax=0.9,
           lag_of_interest=[0.25],plot_style=plot_style, plot_diff=False,cluster_alpha=0.05,
           colors=colors,
           use_style=style, resultsdir=resultsdir, topomap=True, ax=axes,
           vlim=vlim)
ax[0].set_title('Word frequency for high vs. low surprisal')

plt.tight_layout()
fig.savefig(f'{resultsdir}/{BANDPASS}-WF-effects-sensors.svg')

# %% plots per participant [do not run if not necessary]
for participant in trfs['base']['trf']:
    fig,ax=plt.subplots(ncols=2, figsize=(8,3))
    participant['envelope'].plot(axes=ax[0],spatial_colors=True, show=False)
    participant['wordonset'].plot(axes=ax[1], spatial_colors=True)
    
# %% plotting the grand averages without the stats
fig,ax=plt.subplots(figsize=(4.5,3))
GA['full']['trf']['bottomup'].plot(axes=ax, spatial_colors=True, show=False)
plt.suptitle('Bottom-up response (Grand average)')
sns.despine()
plt.tight_layout()
fig.savefig(f'{resultsdir}/GA-bottomup.svg')
