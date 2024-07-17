#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:47:32 2022

@author: sopsla
"""
import re
import pandas as pd
from viz import plot_all_bar, plot_topomap_rvals
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from meg import CH_NAMES
import scipy.stats as stats

# %% settings
BANDPASS = 'delta'
surtype='ngram'
resultsdir = f'/project/3027007.06/results/no-entropy-topdown-durmatch/{BANDPASS}/{surtype}' # '/project/3027007.06/results/cas-comparison/delta-mean/' #
datapath = '/project/3027007.01/processed/'

#%% load data
r_data = pd.read_csv(os.path.join(resultsdir,'r_data.csv'))
r_data = r_data[r_data['subject'] != 'sub-024']
r_data = r_data.fillna('null')

with open(f'{resultsdir}/GA.pkl', 'rb') as f:
    GA = pickle.load(f)
    
#%% settings for permutation
info = GA['bottomup_split_surprisal']['trf'].info

# permutation settings
idx = mne.channel_indices_by_type(info, picks=CH_NAMES)
adjacency, _ = mne.channels.read_ch_adjacency('ctf275', idx['mag'])

statfun = lambda x,y: mne.stats.cluster_level.ttest_1samp_no_p(x-y) #, sigma=1e-3, method='relative') #None #
threshold = stats.t.ppf(1-0.05, df=len(set(r_data['subject']))-1) # stats.f.ppf(1-0.05, dfd=24-1, dfn=1)  # 
tail = 0


# %% let's get the sensors in which the TRFs differ from each other
with open(os.path.join(resultsdir, 'trf_stats_WF.pkl'), 'rb') as f:
    trfstats = pickle.load(f)

# %% get the chans
signi_chans = {}
for effectname,trfstat in trfstats.items():
    signi_chans[effectname] = []
    _, clusters, pvals, _ = trfstat
    for k_cluster, c in enumerate(clusters):
        if pvals[k_cluster] < 0.01: # p-value
            lags, chans = c
            
            for c in chans:
                signi_chans[effectname].append(c)
                
    signi_chans[effectname] = np.asarray(list(set(signi_chans[effectname])))
                
signi_masks = {}
for effectname, sensors in signi_chans.items():
    if len(sensors) != 0:
    
        mask = np.zeros((269))
       
        #sensi_sensors = np.asarray(list(set(sensors)))
        mask[sensors] = 1
        mask = np.array(mask, dtype=bool)
            
        signi_masks[effectname] = mask
        
        fig,ax=plt.subplots()
        _,_ = mne.viz.plot_topomap(data=np.zeros((269)), pos=info, mask=mask, axes=ax)
        ax.set_title(effectname)
                
# %% effects of split only in those sensors
for syntype in ['frequency']:
    for distype in ['surprisal']:
        effectname = f'{syntype}_{distype}'
        
        sensitive_sensors_idx = signi_chans[effectname]
        if len(sensitive_sensors_idx) > 0:
        
            #nosplit = np.asarray(r_data.loc[r_data['model'] == 'full', 'r_values']).reshape((len(set(r_data['subject'])),269))[:,sensitive_sensors_idx]
            randomsplit = np.asarray(r_data.loc[r_data['model'] == 'bottomup_splitWF_random', 'r_values']).reshape((len(set(r_data['subject'])),269))[:,sensitive_sensors_idx]
            onesplit = np.asarray(r_data.loc[r_data['model'] == f'bottomup_splitWF_{distype}', 'r_values']).reshape((len(set(r_data['subject'])),269))[:,sensitive_sensors_idx]
                    
            #print(f'{syntype} {distype} | No split (m {np.mean(nosplit)}) vs split (m {np.mean(onesplit)}):', stats.ttest_rel(np.mean(nosplit,axis=1), np.mean(onesplit,axis=1)))
            print(f'{syntype} {distype} | Random split (m {np.mean(randomsplit)}) vs split (m {np.mean(onesplit)}):', stats.ttest_rel(np.mean(randomsplit,axis=1), np.mean(onesplit,axis=1)))
    
#fullsplit = np.asarray(r_data.loc[r_data['model'] == 'full_split_surprisal', 'r_values']).reshape((len(set(r_data['subject'])),269))[:,sensitive_sensors_idx]
#topdownsplit = X2 = np.asarray(r_data.loc[r_data['model'] == 'full_topdown_split_surprisal', 'r_values'])

# %% WHOLE BRAIN - SURPRISAL SPLIT - BOTH SYNTYPES - compare against random
interactions = {}
for stattype in ['surprisal']:
    interactions[stattype] = {}
    
    for syntype in ['bottomup']:

        #  get data into shape
        X1 = np.asarray(r_data.loc[r_data['model'] == f'{syntype}_splitWF_random', 'r_values'])
        X1 = X1.reshape((len(set(r_data['subject'])),269))
        
        X2 = np.asarray(r_data.loc[r_data['model'] == f'{syntype}_splitWF_{stattype}', 'r_values'])
        X2 = X2.reshape((len(set(r_data['subject'])),269))
        
        # test
        surprisal_split = mne.stats.permutation_cluster_test([X2, X1], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)
        interactions[stattype][syntype] = surprisal_split

# %% plot these as well in one row
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(3,3), gridspec_kw={'width_ratios': [1,0.1]})

t_obs_bottomup, clusters_bottomup, pvals_bottomup, h0_bottomup = interactions['surprisal']['bottomup']
t_obs_bottomup = np.squeeze(t_obs_bottomup)

vmin = -4.5#min(t_obs_bottomup)
vmax = 4.5#max(t_obs_bottomup)

if abs(vmin) > vmax:
    vmax = abs(vmin)
elif vmax > abs(vmin):
    vmin = -vmax

mask = np.zeros((269))

for l,cluster in enumerate(clusters_bottomup):
    if pvals_bottomup[l] < 0.05:
        chans=cluster
        mask[chans] = 1
        
mask = np.array(mask, dtype=bool)
        
im,_ = mne.viz.plot_topomap(t_obs_bottomup, info, axes=ax[0], vmin=vmin, vmax=vmax, show=False, mask=mask)

#mne.viz.plot_brain_colorbar(ax=ax[3], clim={})
cbar = fig.colorbar(im, cax=ax[1])
ax[1].set_ylabel('t-value')
ax[0].set_title('Word frequency\nsplit by surprisal')

#fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4), gridspec_kw={'width_ratios': [1,0.1]})
#plot_topomap_rvals(surprisal_split, axes=ax, fig=fig, info=info, bar=True)
#ax[0].set_title(f'Split {syntype} by {stattype}')
plt.tight_layout()
fig.savefig(f'{resultsdir}/random_split-WF.svg')

# %% let's look at high- versus low {surprisal / entropy} {topdown/bottomup}
stattype = 'surprisal'
syntypes = 'bottomup'

# load the comparison from 'main-effects'
r_main = pd.read_csv(os.path.join(F'/project/3027007.06/results/main-effects/{BANDPASS}/{surtype}','r_data.csv'))
r_main = r_main[r_main['subject'] != 'sub-024']
r_main = r_main.fillna('null')

# base is a model that contains surprisal but not entropy or the syntactic variables
basemodel = 'main_surprisal'
base_rvals = r_main.loc[r_main['model'] == basemodel, 'r_values'].values

baseX = np.asarray(base_rvals)
baseX = baseX.reshape((len(set(r_main['subject'])),269))
del r_main
        
# improvement of low
lowmodel = f'{syntype}_low_{stattype}'
low_rvals = r_data.loc[r_data['model'] == lowmodel, 'r_values'].values
lowX = np.asarray(low_rvals)
lowX = lowX.reshape((len(set(r_data['subject'])),269))

# improvement of high
highmodel = f'{syntype}_high_{stattype}'
high_rvals = r_data.loc[r_data['model'] == highmodel, 'r_values'].values
highX = np.asarray(high_rvals)
highX = highX.reshape((len(set(r_data['subject'])),269))

# differences
lowdiff = mne.stats.permutation_cluster_test([lowX, baseX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)
highdiff = mne.stats.permutation_cluster_test([highX, baseX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)

# diff of diff
low_vs_high = mne.stats.permutation_cluster_test([highX-baseX, lowX-baseX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)

# %%
vlim = 5
fig,axes = plt.subplots(ncols=4, figsize=(10,3), gridspec_kw={'width_ratios':[1,1,1,0.1]})
for i,(efname,efstats,ax) in enumerate(zip(['low', 'high', 'difference'], [lowdiff, highdiff, low_vs_high], axes)):
    if i < 2:
        plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax], fig=fig, bar=False, vlim=vlim)
    if i == 2:
        plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax, axes[-1]], fig=fig, bar=True, vlim=vlim)             


    ax.set_title(efname)
plt.suptitle(f'{syntype} by {stattype}')
plt.tight_layout()

fig.savefig(f'{resultsdir}/scalpmaps-high-low-vs-base-{syntype}-{stattype}.svg')

# %% perspective of 'full' model instead - DOES NOT MAKE SENSE
stattypes = ['entropy', 'surprisal']
syntypes = ['topdown', 'bottomup']

full_rvals = r_data.loc[r_data['model'] == 'full', 'r_values'].values
fullX = np.asarray(full_rvals).reshape((len(set(r_data['subject'])),269))

for stattype in stattypes:
    print(stattype, syntype)
    
    for syntype in syntypes:
        
        # improvement of low
        lowmodel = f'{syntype}_low_{stattype}'
        low_rvals = r_data.loc[r_data['model'] == lowmodel, 'r_values'].values
        lowX = np.asarray(low_rvals)
        lowX = lowX.reshape((len(set(r_data['subject'])),269))
        
        # improvement of high
        highmodel = f'{syntype}_high_{stattype}'
        high_rvals = r_data.loc[r_data['model'] == highmodel, 'r_values'].values
        highX = np.asarray(high_rvals)
        highX = highX.reshape((len(set(r_data['subject'])),269))
        
        # differences
        lowdiff = mne.stats.permutation_cluster_test([fullX, highX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)
        highdiff = mne.stats.permutation_cluster_test([fullX, lowX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)
        
        # diff of diff
        low_vs_high = mne.stats.permutation_cluster_test([fullX-lowX, fullX-highX], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)

        vlim = 10
        fig,axes = plt.subplots(ncols=4, figsize=(10,3), gridspec_kw={'width_ratios':[1,1,1,0.1]})
        for i,(efname,efstats,ax) in enumerate(zip(['low', 'high', 'difference'], [lowdiff, highdiff, low_vs_high], axes)):
            if i < 2:
                plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax], fig=fig, bar=False, vlim=vlim)
            if i == 2:
                plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax, axes[-1]], fig=fig, bar=True, vlim=vlim)             
        
        
            ax.set_title(efname)
        plt.suptitle(f'{syntype} by {stattype}')
        plt.tight_layout()
        
        fig.savefig(f'{resultsdir}/scalpmaps-high-low-vs-full-{syntype}-{stattype}.svg')

# %% plot it
vlim = 5
fig,axes = plt.subplots(ncols=4, figsize=(10,4), gridspec_kw={'width_ratios':[1,1,1,0.1]})
for i,(efname,efstats,ax) in enumerate(zip(['low', 'high', 'difference'], [lowdiff, highdiff, low_vs_high], axes)):
    if i < 2:
        plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax], fig=fig, bar=False, vlim=vlim)
    if i == 2:
        plot_topomap_rvals(efstats, info=info, pv=0.05, axes=[ax, axes[-1]], fig=fig, bar=True, vlim=vlim)             


    ax.set_title(efname)

# %% save the effects
rastats = {}
rastats['interactions'] = interactions
rastats['maineffects'] = maineffects

with open(os.path.join(resultsdir, 'ra_stats.pkl'), 'wb') as f:
    pickle.dump(rastats, f)

# %% VIZ ONLY: interactions
minsmaxs = []
for name,effect in rastats['interactions'].items():
    t_obs = effect[0]
    minsmaxs.append(min(t_obs))
    minsmaxs.append(max(t_obs))

mx = max(minsmaxs)
mn = min(minsmaxs)

if abs(mn) > abs(mx):
    vlim = abs(mn)
elif mx > abs(mn):
    vlim = mx
    
fig,axes = plt.subplots(ncols=3, nrows=2, figsize=(6,6), gridspec_kw={'width_ratios':[1,1,0.1]})
for row,stat in zip(axes, ['surprisal', 'entropy']):
    for i,(axs, synt) in enumerate(zip(row[:4],['topdown', 'bottomup'])):
        effect = rastats['interactions'][f'{synt}_{stat}']
        
        if i < 1:
            plot_topomap_rvals(effect, info=info, pv=0.025, axes = [axs], fig=fig, bar=False, vlim=vlim)
        elif i == 1:
            plot_topomap_rvals(effect, info=info, pv=0.025, axes = [axs,row[2]], fig=fig, bar=True, vlim=vlim)
        axs.set_title(f'{synt}/{stat}')
        
fig.savefig(f'{resultsdir}/scalpmaps-spliteffect.svg')

# %% VIZ ONLY: simple effects
minsmaxs = []
for name,effect in rastats['simpleeffects'].items():
    t_obs = effect[0]
    minsmaxs.append(min(t_obs))
    minsmaxs.append(max(t_obs))

mx = max(minsmaxs)
mn = min(minsmaxs)

if abs(mn) > abs(mx):
    vlim = abs(mn)
elif mx > abs(mn):
    vlim = mx
    
fig,axes = plt.subplots(ncols=5, nrows=2, figsize=(12,6), gridspec_kw={'width_ratios':[1,1,1,1,0.1]})
for row,stat in zip(axes, ['surprisal', 'entropy']):
    print(stat)
    for i,(axs,(name,effect)) in enumerate(zip(row[:4],[item for item in rastats['simpleeffects'].items() if item[0].endswith(stat)])):
       # if name.endswith(stat):
        if i < 3:
            plot_topomap_rvals(effect, info=info, pv=0.025, axes = [axs], fig=fig, bar=False, vlim=vlim)
        elif i == 3:
            plot_topomap_rvals(effect, info=info, pv=0.025, axes = [axs,row[4]], fig=fig, bar=True, vlim=vlim)
        axs.set_title('/'.join(name.split('_')))

fig.savefig(f'{resultsdir}/scalpmaps-simple_effects.svg')

# %% the interactions
interaction_terms = {}
for interaction_model in ['interaction_bottomup_surprisal', 'interaction_bottomup_entropy', \
                         'interaction_topdown_surprisal', 'interaction_topdown_entropy']:
    X1 = np.asarray(r_data.loc[r_data['model'] == interaction_model, 'r_values'])
    X1 = X1.reshape((len(set(r_data['subject'])),269))
    
    X2 = np.asarray(r_data.loc[r_data['model'] == 'full', 'r_values'])
    X2 = X2.reshape((len(set(r_data['subject'])),269))

    result = mne.stats.permutation_cluster_test([X1, X2], adjacency=adjacency, stat_fun=statfun, threshold=threshold, tail=tail, n_permutations=10000)
    
    interaction_terms[interaction_model] = result

# %%
fig,ax=plt.subplots(nrows=1, ncols=4, figsize=(12,4))
for i,(name, results) in enumerate(interaction_terms.items()):
    plot_topomap_rvals(results, info=info, pv=0.05, axes=[ax[i]], bar=False, vlim=10)
    ax[i].set_title(name)
    
