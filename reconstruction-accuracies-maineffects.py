#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:24:21 2023

@author: sopsla
"""
# examining the top-down bottom-up effects
# reconstruction accuracies per region
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
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% settings
BANDPASS = 'delta'
surtype='ngram'
resultsdir = f'/project/3027007.06/results/main-effects/{BANDPASS}/{surtype}' # '/project/3027007.06/results/cas-comparison/delta-mean/' #
datapath = '/project/3027007.01/processed/'

#%% load data
r_data = pd.read_csv(os.path.join(resultsdir,'r_data.csv'))
r_data = r_data[r_data['subject'] != 'sub-024']
r_data = r_data.fillna('null')

with open(f'{resultsdir}/GA.pkl', 'rb') as f:
    GA = pickle.load(f)

info = GA['main_null']['trf'].info

#%% permutation settings
idx = mne.channel_indices_by_type(info, picks=CH_NAMES)
adjacency, _ = mne.channels.read_ch_adjacency('ctf275', idx['mag'])

statfun = lambda x,y: mne.stats.cluster_level.ttest_1samp_no_p(x-y) #, sigma=1e-3, method='relative') #None #
threshold = stats.t.ppf(1-0.05, df=len(set(r_data['subject']))-1) # stats.f.ppf(1-0.05, dfd=24-1, dfn=1)  # 
tail = 0
    
# %% we take the whole-brain average for (some) simplicity
r_data_means = r_data.groupby(by=['model', 'subject']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_means.reset_index(inplace=True)

for factor in ['surprisal', 'entropy', 'topdown', 'bottomup', 'leftcorner']:
    fdata = []
    for i in r_data_means['model']:
        subfactors = i.split('_')
        
        if factor in ['surprisal', 'entropy']:
            if factor in subfactors or 'distributional' in subfactors:
                fdata.append(1)
            else:
                fdata.append(0)
                
        elif factor in ['topdown','bottomup', 'leftcorner']:
            if factor in subfactors or 'syntax' in subfactors:
                fdata.append(1)
            else:
                fdata.append(0)
    r_data_means[factor] = fdata
            
#r_data_means.to_csv(f'{resultsdir}/r_data_wholebrain_{surtype}.csv')

# %% okay let's take the  difference from the null model
r_data_difference = r_data_means[r_data_means['model'] != 'main_null']
r_data_difference['difference'] = np.zeros(len(r_data_difference))
main_values = r_data_means.loc[r_data_means['model'] == 'main_null', 'r_values'].values

for model in r_data_difference['model'].unique():
    rvals = r_data_difference.loc[r_data_difference['model'] == model, 'r_values'].values
    diff = rvals - main_values
    r_data_difference.loc[r_data_difference['model'] == model, 'difference'] = diff

# %% let's run a linear mixed effects model
md = smf.mixedlm("r_values ~ entropy * surprisal * topdown * bottomup * leftcorner", data=r_data_means,
                 groups=r_data_means['subject'])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())

#%% and a model with only main effects
mds = smf.mixedlm("r_values ~ entropy + surprisal + topdown + bottomup + leftcorner", data=r_data_means,
                 groups=r_data_means['subject'])
mdfs = mds.fit(method="lbfgs")
print(mdfs.summary())

# %% okay, we know from the r-script what the effects are - lets viz them
factors = ['entropy', 'surprisal', 'topdown', 'bottomup', 'leftcorner']

for effect in factors:
    interactions = factors.copy()
    interactions.remove(effect)
    
    # main effect
    fig,axes=plt.subplots(nrows=5, sharex=True, sharey=True,
                          figsize=(4,12))

    sns.lineplot(x=effect, y='difference', data=r_data_difference,
                 err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                 markers=True, ax=axes[0])
    axes[0].set_title(f'Main effect of {effect}')
    axes[0].set_ylabel(r"$\Delta$ pearson's R")
    
    for ax,interaction in zip(axes[1:], interactions):
        # get the values without the interacting variable
        without = r_data_difference.loc[r_data_difference[interaction] == 0]
        
        # get the values with the interacting variable
        withit = r_data_difference.loc[r_data_difference[interaction] == 1]
        
        sns.lineplot(x=effect, y='difference', data=without,palette='Flare',
                     err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                     markers=True, ax=ax)

        sns.lineplot(x=effect, y='difference', data=withit,palette='Crest',
             err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
             markers=True, ax=ax)
        
        ax.set_ylabel(r"$\Delta$ pearson's R")
        
        ax.set_title(f'{effect} * {interaction}')
        ax.legend([f'Without {interaction}', f'With {interaction}'], 
                  loc=1, frameon=False)
        
    for ax in axes[1:-1]:
        ax.get_xaxis().set_visible(False)
    
    for ax in [axes[0], axes[-1]]:
        ax.set_xticks([0,1])
        ax.set_xticklabels([f'No {effect}', f'{effect}'])
        ax.set_xlabel(f'{effect}')
        ax.xaxis.label.set_visible(False)
        
    sns.despine()
    plt.tight_layout()
    
    fig.savefig(f'{resultsdir}/wholebrain_2way_interactions_{effect}.svg')

# %% we take only the models that do not have left-corner and repeat the above
sub_r_data_difference = r_data_difference.loc[r_data_difference['leftcorner'] == 0]


# %%
factors = ['entropy', 'surprisal', 'topdown', 'bottomup']

for effect in factors:
    interactions = factors.copy()
    interactions.remove(effect)
    
    # main effect
    fig,axes=plt.subplots(nrows=4, sharex=True, sharey=True,
                          figsize=(4,9.5))

    sns.lineplot(x=effect, y='difference', data=sub_r_data_difference,
                 err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                 markers=True, ax=axes[0])
    axes[0].set_title(f'Main effect of {effect}')
    axes[0].set_ylabel(r"$\Delta$ pearson's R")
    
    for ax,interaction in zip(axes[1:], interactions):
        # get the values without the interacting variable
        without = sub_r_data_difference.loc[sub_r_data_difference[interaction] == 0]
        
        # get the values with the interacting variable
        withit = sub_r_data_difference.loc[sub_r_data_difference[interaction] == 1]
        
        sns.lineplot(x=effect, y='difference', data=without,palette='Flare',
                     err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                     markers=True, ax=ax)

        sns.lineplot(x=effect, y='difference', data=withit,palette='Crest',
             err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
             markers=True, ax=ax)
        
        ax.set_ylabel(r"$\Delta$ pearson's R")
        
        ax.set_title(f'{effect} * {interaction}')
        ax.legend([f'Without {interaction}', f'With {interaction}'], 
                  loc=1, frameon=False)
        
    for ax in axes[1:-1]:
        ax.get_xaxis().set_visible(False)
    
    for ax in [axes[0], axes[-1]]:
        ax.set_xticks([0,1])
        ax.set_xticklabels([f'No {effect}', f'{effect}'])
        ax.set_xlabel(f'{effect}')
        ax.xaxis.label.set_visible(False)
        
    sns.despine()
    plt.tight_layout()
    
    fig.savefig(f'{resultsdir}/wholebrain_2way_interactions_{effect}_nolc.svg')

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
# %% ############ repeat all of the above on the ROIs
region_names = set([ch_name[1:3] for ch_name in info.ch_names])
regions = {r_name: [ch_name for ch_name in info.ch_names if ch_name[1:3]==r_name] for r_name in region_names}
regions['Z'] = [regions[x] for x in regions.keys() if x.startswith('Z')]
regions['Z'] = [i for u in regions['Z'] for i in u]
for x in ['ZP', 'ZF', 'ZC', 'ZO']:
    regions.pop(x)

# means per roi
r_data['region'] = [ch_name[1:3] for ch_name in r_data['sensor'].values]
r_data = r_data.replace(to_replace=['ZP', 'ZF', 'ZC', 'ZO'], value='Z')
    

# %%
r_data_rois = r_data.groupby(by=['region', 'model', 'subject']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_rois.reset_index(inplace=True)

for factor in ['surprisal', 'entropy', 'topdown', 'bottomup', 'leftcorner']:
    fdata = []
    for i in r_data_rois['model']:
        subfactors = i.split('_')
        
        if factor in ['surprisal', 'entropy']:
            if factor in subfactors or 'distributional' in subfactors:
                fdata.append(1)
            else:
                fdata.append(0)
                
        elif factor in ['topdown','bottomup', 'leftcorner']:
            if factor in subfactors or 'syntax' in subfactors:
                fdata.append(1)
            else:
                fdata.append(0)
    r_data_rois[factor] = fdata
            
r_data_rois.to_csv(f'{resultsdir}/r_data_roi_{surtype}.csv')

# %% difference from the null model
r_data_roi_difference = r_data_rois[r_data_rois['model'] != 'main_null']
r_data_roi_difference['difference'] = np.zeros(len(r_data_roi_difference))
main_values = r_data_rois.loc[r_data_rois['model'] == 'main_null', 'r_values'].values

for model in r_data_roi_difference['model'].unique():
    rvals = r_data_roi_difference.loc[r_data_roi_difference['model'] == model, 'r_values'].values
    diff = rvals - main_values
    r_data_roi_difference.loc[r_data_roi_difference['model'] == model, 'difference'] = diff

# %% let's run a linear mixed effects model
for roi in r_data_rois['region'].unique():
    print(roi)
    roidata = r_data_rois.loc[r_data_rois['region'] == roi]
    roidata_difference = r_data_roi_difference.loc[r_data_roi_difference['region'] == roi]

    #md = smf.mixedlm("r_values ~ entropy * surprisal * topdown * bottomup * leftcorner", data=roidata,
     #                groups=roidata['subject'])
    #mdf = md.fit(method=["lbfgs"])
    #print(mdf.summary())
    
    # and a model with only main effects
   # mds = smf.mixedlm("r_values ~ entropy + surprisal + topdown + bottomup + leftcorner", data=roidata,
    #                 groups=roidata['subject'])
    #mdfs = mds.fit(method="lbfgs")
    #print(mdfs.summary())
    
    # okay, we know from the r-script what the effects are - lets viz them
    factors = ['entropy', 'surprisal', 'topdown', 'bottomup', 'leftcorner']
    
    for effect in factors:
        interactions = factors.copy()
        interactions.remove(effect)
        
        # main effect
        fig,axes=plt.subplots(nrows=5, sharex=True, sharey=True,
                              figsize=(4,12))
    
        sns.lineplot(x=effect, y='difference', data=roidata_difference,
                     err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                     markers=True, ax=axes[0])
        axes[0].set_title(f'Main effect of {effect}')
        axes[0].set_ylabel(r"$\Delta$ pearson's R")
        
        for ax,interaction in zip(axes[1:], interactions):
            # get the values without the interacting variable
            without = roidata_difference.loc[roidata_difference[interaction] == 0]
            
            # get the values with the interacting variable
            withit = roidata_difference.loc[roidata_difference[interaction] == 1]
            
            sns.lineplot(x=effect, y='difference', data=without,palette='Flare',
                         err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                         markers=True, ax=ax)
    
            sns.lineplot(x=effect, y='difference', data=withit,palette='Crest',
                 err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                 markers=True, ax=ax)
            
            ax.set_ylabel(r"$\Delta$ pearson's R")
            
            ax.set_title(f'{effect} * {interaction}')
            ax.legend([f'Without {interaction}', f'With {interaction}'], 
                      loc=1, frameon=False)
            
        for ax in axes[1:-1]:
            ax.get_xaxis().set_visible(False)
        
        for ax in [axes[0], axes[-1]]:
            ax.set_xticks([0,1])
            ax.set_xticklabels([f'No {effect}', f'{effect}'])
            ax.set_xlabel(f'{effect}')
            ax.xaxis.label.set_visible(False)
            
        sns.despine()
        plt.tight_layout()
        
        fig.savefig(f'{resultsdir}/{roi}_2way_interactions_{effect}.svg')
    
    # we take only the models that do not have left-corner and repeat the above
    sub_roidata_difference = roidata_difference.loc[roidata_difference['leftcorner'] == 0]
    
    factors = ['entropy', 'surprisal', 'topdown', 'bottomup']
    
    for effect in factors:
        interactions = factors.copy()
        interactions.remove(effect)
        
        # main effect
        fig,axes=plt.subplots(nrows=4, sharex=True, sharey=True,
                              figsize=(4,9.5))
    
        sns.lineplot(x=effect, y='difference', data=sub_roidata_difference,
                     err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                     markers=True, ax=axes[0])
        axes[0].set_title(f'Main effect of {effect}')
        axes[0].set_ylabel(r"$\Delta$ pearson's R")
        
        for ax,interaction in zip(axes[1:], interactions):
            # get the values without the interacting variable
            without = sub_roidata_difference.loc[sub_roidata_difference[interaction] == 0]
            
            # get the values with the interacting variable
            withit = sub_roidata_difference.loc[sub_roidata_difference[interaction] == 1]
            
            sns.lineplot(x=effect, y='difference', data=without,palette='Flare',
                         err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                         markers=True, ax=ax)
    
            sns.lineplot(x=effect, y='difference', data=withit,palette='Crest',
                 err_style='bars', err_kws={'capsize':2, 'fmt':'', 'capthick' :1}, 
                 markers=True, ax=ax)
            
            ax.set_ylabel(r"$\Delta$ pearson's R")
            
            ax.set_title(f'{effect} * {interaction}')
            ax.legend([f'Without {interaction}', f'With {interaction}'], 
                      loc=1, frameon=False)
            
        for ax in axes[1:-1]:
            ax.get_xaxis().set_visible(False)
        
        for ax in [axes[0], axes[-1]]:
            ax.set_xticks([0,1])
            ax.set_xticklabels([f'No {effect}', f'{effect}'])
            ax.set_xlabel(f'{effect}')
            ax.xaxis.label.set_visible(False)
            
        sns.despine()
        plt.tight_layout()
        
        fig.savefig(f'{resultsdir}/{roi}_2way_interactions_{effect}_nolc.svg')
    
#%%   
from viz import plot_all_box

# lets plot both gpt and ngram in one
r_data_gpt = pd.read_csv(f'/project/3027007.06/results/main-effects/{BANDPASS}/GPT/r_data_wholebrain_GPT.csv')
r_data_ngram = pd.read_csv(f'/project/3027007.06/results/main-effects/{BANDPASS}/ngram/r_data_wholebrain_ngram.csv')

# calculate the differences, take out left corner
r_data_difference_gpt = r_data_gpt[r_data_gpt['model'] != 'main_null']
r_data_difference_gpt['difference'] = np.zeros(len(r_data_difference_gpt))
main_values = r_data_gpt.loc[r_data_gpt['model'] == 'main_null', 'r_values'].values

for model in r_data_difference_gpt['model'].unique():
    rvals = r_data_difference_gpt.loc[r_data_difference_gpt['model'] == model, 'r_values'].values
    diff = rvals - main_values
    r_data_difference_gpt.loc[r_data_difference_gpt['model'] == model, 'difference'] = diff

sub_r_data_difference_GPT = r_data_difference_gpt.loc[r_data_difference_gpt['leftcorner'] == 0]
sub_r_data_difference_GPT['r_values'] = sub_r_data_difference_GPT['difference'] 

r_data_difference_ngram = r_data_ngram[r_data_ngram['model'] != 'main_null']
r_data_difference_ngram['difference'] = np.zeros(len(r_data_difference_ngram))
main_values = r_data_ngram.loc[r_data_ngram['model'] == 'main_null', 'r_values'].values

for model in r_data_difference_ngram['model'].unique():
    rvals = r_data_difference_ngram.loc[r_data_difference_ngram['model'] == model, 'r_values'].values
    diff = rvals - main_values
    r_data_difference_ngram.loc[r_data_difference_ngram['model'] == model, 'difference'] = diff

sub_r_data_difference_ngram = r_data_difference_ngram.loc[r_data_difference_ngram['leftcorner'] == 0]
sub_r_data_difference_ngram['r_values'] = sub_r_data_difference_ngram['difference'] 

# %%
models = ['main_surprisal', 'main_entropy', 'main_distributional', 'main_topdown', 'main_bottomup', 'main_topdown_bottomup', 'main_surprisal_topdown',
          'main_surprisal_bottomup', 'main_surprisal_topdown_bottomup', 'main_entropy_topdown', 'main_entropy_bottomup', 'main_entropy_topdown_bottomup', 
          'main_distributional_topdown', 'main_distributional_bottomup', 'main_distributional_topdown_bottomup']
abbrev = ['surp.', 'entr.', 'surp./entr.', 't.d.', 'b.u.', 't.d./b.u.', 'surp./t.d.', 'surp./b.u.', 'surp./t.d./b.u.', 'entr./t.d.', 'entr./b.u.', 
          'entr./t.d./b.u.', 'surp./entr./t.d.', 'surp./entr./b.u.', 'surp/entr./t.d./b.u.']

fig,axes=plt.subplots(figsize=(10,4), sharey=True, ncols=2)

GPT_colors = sns.color_palette('flare', n_colors=len(models))
ngram_colors = sns.color_palette('crest', n_colors=len(models))

plot_all_box(sub_r_data_difference_ngram, colors=ngram_colors, ax=axes[0], models=models, abbrev=abbrev, save=False)
plot_all_box(sub_r_data_difference_GPT, colors=GPT_colors, ax=axes[1], models=models, abbrev=abbrev, save=False)

axes[1].get_yaxis.set_visibility(False)

plt.tight_layout()
sns.despine()

#fig.savefig(f'{resultsdir}/main-effects-{surtype}.svg')

# %% cluster-based permutation tests for our main effects

full = 'main_distributional_topdown_bottomup' # we do not want to include left-corner

effects = {'entropy': 'main_surprisal_topdown_bottomup',
           'surprisal': 'main_entropy_topdown_bottomup',
           'topdown':  'main_distributional_bottomup',
           'bottomup': 'main_distributional_topdown'}

results = {}

for effname, model in effects.items():
    X1 = np.asarray(r_data.loc[r_data['model'] == full, 'r_values'])
    X1 = X1.reshape((len(set(r_data['subject'])),269))

    X2 = np.asarray(r_data.loc[r_data['model'] == model, 'r_values'])
    X2 = X2.reshape((len(set(r_data['subject'])),269))

    outstats = mne.stats.permutation_cluster_test([X1, X2], adjacency=adjacency, 
                                               stat_fun=statfun, 
                                               threshold=threshold, 
                                               tail=tail, n_permutations=10000)
    
    print(effname)
    _, clusters, pvals, _ = outstats
    
    print(f'{sum(pvals <= 0.05)} significant clusters')
    results[effname] = outstats
    
    

# %%
fig,axes=plt.subplots(ncols=5, figsize=(16,4), gridspec_kw={'width_ratios': [1,1,1,1,0.1]})
vmax = 7
vmin = -7

for ax, effname in zip(axes[:-1],results.keys()):
    
    t_obs, clusters, pvals, h0 = results[effname]
    
    mask = np.zeros((269))
    for i,cluster in enumerate(clusters):
        if pvals[i] < 0.0125:
            chans=cluster
            mask[chans] = 1
            
    mask = np.array(mask, dtype=bool)

    im,_ = mne.viz.plot_topomap(t_obs, info, axes=ax, vmin=vmin, vmax=vmax, show=False, mask=mask,
                                cmap='coolwarm')
    ax.set_title(effname)

cbar = fig.colorbar(im, cax=axes[-1])



# %% null
null = 'main_null' # we do not want to include left-corner

null_effects = {'entropy': 'main_entropy',
               'surprisal': 'main_surprisal',
               'topdown':  'main_topdown',
               'bottomup': 'main_bottomup'}

null_results = {}

for effname, model in null_effects.items():
    X1 = np.asarray(r_data.loc[r_data['model'] == null, 'r_values'])
    X1 = X1.reshape((len(set(r_data['subject'])),269))

    X2 = np.asarray(r_data.loc[r_data['model'] == model, 'r_values'])
    X2 = X2.reshape((len(set(r_data['subject'])),269))

    outstats = mne.stats.permutation_cluster_test([X2, X1], adjacency=adjacency, 
                                               stat_fun=statfun, 
                                               threshold=threshold, 
                                               tail=tail, n_permutations=10000)
    
    print(effname)
    _, clusters, pvals, _ = outstats
    
    print(f'{sum(pvals <= 0.05)} significant clusters')
    null_results[effname] = outstats
    
# %%

fig,axes=plt.subplots(ncols=5, figsize=(16,4), gridspec_kw={'width_ratios': [1,1,1,1,0.1]})
vmax = 7
vmin = -7

for ax, effname in zip(axes[:-1],null_results.keys()):
    
    t_obs, clusters, pvals, h0 = null_results[effname]
    
    mask = np.zeros((269))
    for i,cluster in enumerate(clusters):
        if pvals[i] < 0.0125:
            chans=cluster
            mask[chans] = 1
            
    mask = np.array(mask, dtype=bool)

    im,_ = mne.viz.plot_topomap(t_obs, info, axes=ax, vmin=vmin, vmax=vmax, show=False, mask=mask)
    ax.set_title(effname)

cbar = fig.colorbar(im, cax=axes[-1])








#%% add the predictors
#for factor in ['surprisal', 'entropy', 'topdown', 'bottomup', 'leftcorner']:
 #   fdata = []
  #  for ix,row in r_data.iterrows():
   #     subfactors = row['model'].split('_')
        
    #    if factor in ['surprisal', 'entropy']:
     #       if factor in subfactors or 'distributional' in subfactors:
      #          fdata.append(1)
       #     else:
        #        fdata.append(0)
                
        #elif factor in ['topdown','bottomup', 'leftcorner']:
         #   if factor in subfactors or 'syntax' in subfactors:
          #      fdata.append(1)
           # else:
            #    fdata.append(0)
    #r_data[factor] = fdata
    
#r_data.to_csv(os.path.join(resultsdir,'r_data_factors.csv'))

r_data = pd.read_csv(os.path.join(resultsdir,'r_data_factors.csv'))

# %%
r_data_nolc = r_data.loc[r_data['leftcorner'] == 0]

for effect in ['entropy', 'surprisal', 'topdown', 'bottomup']:
    print(f'No {effect}: {np.mean(r_data_nolc.loc[r_data_nolc[effect] == 0, "r_values"].values)}')
    print(f'Yes {effect}: {np.mean(r_data_nolc.loc[r_data_nolc[effect] == 1, "r_values"].values)}')
    
# %%
r_data_topdown = r_data_nolc.groupby(by=['subject', 'sensor', 'topdown']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_topdown.reset_index(inplace=True)

r_data_bottomup = r_data_nolc.groupby(by=['subject', 'sensor', 'bottomup']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_bottomup.reset_index(inplace=True)

r_data_surprisal = r_data_nolc.groupby(by=['subject', 'sensor', 'surprisal']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_surprisal.reset_index(inplace=True)

r_data_entropy = r_data_nolc.groupby(by=['subject', 'sensor', 'entropy']).aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_entropy.reset_index(inplace=True)

# %%
contrast_results = {}

for name,data in zip(['topdown', 'bottomup', 'surprisal', 'entropy'],[r_data_topdown, r_data_bottomup, r_data_surprisal, r_data_entropy]):
    
    X1 = np.asarray( data.loc[ data[name] == 1, 'r_values'])
    X1 = X1.reshape((len(set( data['subject'])),269))

    X2 =  np.asarray( data.loc[ data[name] == 0, 'r_values'])
    X2 = X2.reshape((len(set( data['subject'])),269))

    outstats = mne.stats.permutation_cluster_test([X1, X2], adjacency=adjacency, 
                                               stat_fun=statfun, 
                                               threshold=threshold, 
                                               tail=tail, n_permutations=10000)
    
    print(name)
    _, clusters, pvals, _ = outstats
    
    print(f'{sum(pvals <= 0.05)} significant clusters')
    contrast_results[name] = outstats

#%%
fig,axes=plt.subplots(ncols=5, figsize=(10,3), gridspec_kw={'width_ratios': [1,1,1,1,0.1]})
vmax = 7
vmin = -7

for ax, effname in zip(axes[:-1],contrast_results.keys()):
    
    t_obs, clusters, pvals, h0 = contrast_results[effname]
    
    mask = np.zeros((269))
    for i,cluster in enumerate(clusters):
        if pvals[i] < 0.05:
            chans=cluster
            mask[chans] = 1
            
    mask = np.array(mask, dtype=bool)

    im,_ = mne.viz.plot_topomap(t_obs, info, axes=ax, vmin=vmin, vmax=vmax, show=False, mask=mask)
    ax.set_title(effname)

cbar = fig.colorbar(im, cax=axes[-1])

plt.tight_layout()

#fig.savefig(os.path.join(resultsdir, 'maineffect-means.svg'))

# %% sensitive sensors
sign_clusters = []
for name,stat in contrast_results.items():
    if name == 'entropy':
        continue
    else:
        sign_clusters.append([stat[1][i] for i in range(len(stat[1])) if stat[2][i] <= 0.05])
    
sensitive_sensors_idx = np.asarray(list(set(sign_clusters[0][0][0]) & set(sign_clusters[1][0][0])), dtype=int)

mask = np.zeros((269))
mask[sensitive_sensors_idx] = 1
_,_ = mne.viz.plot_topomap(data=np.zeros((269)), pos=info, mask=mask)

sensitive_sensors_idx = sign_clusters[0]
for cluster in sign_clusters[1:]:
    for sensor in cluster:
        if sensor in sensitive_sensors_idx:
            continue
        else:
            sensitive_sensors_idx.append(sensor)

mask = np.zeros((269))
mask[sensitive_sensors_idx] = 1
_,_ = mne.viz.plot_topomap(data=np.zeros((269)), pos=info, mask=mask)

# %%
sns.histplot(r_data_means, x='r_values', hue='model')
plt.legend()

































# %% take the difference with the null model and plot this
r_data['difference'] = [r_data.loc[r_data['model'] == m, 'r_values'].values - r_data.loc[r_data['model'] == 'main_null', 'r_values'].values for m in r_data['model'].unique()]
# %% let's quickly plot them
fig,ax=plt.subplots(figsize=(12,12))
ax.axvline(x=np.median(r_data_means.loc[r_data_means['model'] == 'main_null', 'r_values']))
sns.boxplot(x='r_values', y='model', data=r_data_means, ax=ax, order=['main_null', 'main_entropy', 'main_surprisal', 'main_distributional',
                                                                'main_topdown', 'main_leftcorner', 'main_bottomup', 
                                                                'main_topdown_leftcorner', 'main_bottomup_leftcorner', 'main_topdown_bottomup', 'main_syntax',
                                                                      'main_entropy_topdown', 'main_entropy_leftcorner', 'main_entropy_bottomup',
                                                                      'main_entropy_topdown_leftcorner', 'main_entropy_bottomup_leftcorner', 'main_entropy_topdown_bottomup',
                                                                      'main_entropy_syntax',
                                                                      'main_surprisal_topdown', 'main_surprisal_leftcorner', 'main_surprisal_bottomup',
                                                                      'main_surprisal_topdown_leftcorner', 'main_surprisal_bottomup_leftcorner', 'main_surprisal_topdown_bottomup',
                                                                      'main_surprisal_syntax',
                                                                      'main_distributional_topdown', 'main_distributional_leftcorner', 'main_distributional_bottomup',
                                                                      'main_distributional_topdown_leftcorner', 'main_distributional_bottomup_leftcorner', 'main_distributional_topdown_bottomup',
                                                                      'main_distributional_syntax'])

# %% full means
r_data_fullmeans = r_data.groupby(by='model').aggregate(r_values=pd.NamedAgg('r_values', np.mean))
r_data_fullmeans.reset_index(inplace=True)

# %% print the differences of the means
print('effect of top-down')
print('simple models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_topdown', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional', 'r_values'].values)
print('full models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_syntax', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_bottomup_leftcorner', 'r_values'].values)

print('effect of bottom-up')
print('simple models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_bottomup', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional', 'r_values'].values)
print('full models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_syntax', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_topdown_leftcorner', 'r_values'].values)

print('effect of left-corner')
print('simple models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_leftcorner', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional', 'r_values'].values)
print('full models', r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_syntax', 'r_values'].values - r_data_fullmeans.loc[r_data_fullmeans['model'] == 'main_distributional_topdown_bottomup', 'r_values'].values)

# %% let's check the effects of left corner, top down and bottom up in different ways
results = pd.DataFrame(data=np.zeros((6,4)), columns=['effect','contrast','t', 'pvalue'])

index = 0
    
# effect of top-down, distributional model vs topdown model
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_topdown', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional', 'r_values'].values)
results.iloc[index,:] = ['topdown', 'distributional_topdown', res[0], res[1]]
index += 1                            

# effect of top-down, full model vs bottom-up/left corner
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_syntax', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional_bottomup_leftcorner', 'r_values'].values)
results.iloc[index,:] = ['topdown', 'syntax_bottomupleftcorner', res[0], res[1]]
index += 1                            

# effect of bottom-up, distributional model vs bottom-up model
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_bottomup', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional', 'r_values'].values)
results.iloc[index,:] = ['bottomup', 'distributional_bottomup', res[0], res[1]]
index += 1                            

# effect of bottom-up, full model vs bottom-up/left corner
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_syntax', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional_topdown_leftcorner', 'r_values'].values)
results.iloc[index,:] = ['bottomup', 'syntax_topdownleftcorner', res[0], res[1]]
index += 1               

# effect of left corner, distributional model vs left corner model
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_leftcorner', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional', 'r_values'].values)
results.iloc[index,:] = ['topdown', 'distributional_topdown', res[0], res[1]]
index += 1                            

# effect of left corner, full model vs bottom-up/topdown
res = stats.ttest_rel(r_data_means.loc[r_data_means['model'] == 'main_distributional_syntax', 'r_values'].values, 
                      r_data_means.loc[r_data_means['model'] == 'main_distributional_topdown_bottomup', 'r_values'].values)
results.iloc[index,:] = ['leftcorner', 'syntax_topdownbottomup', res[0], res[1]]
