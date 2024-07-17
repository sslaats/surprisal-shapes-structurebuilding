#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:21:23 2023

@author: sopsla
"""
import os
import pickle
import numpy as np
from pyeeg.utils import lag_span
import mne

# local modules
from meg import CH_NAMES
from viz import _rgb

# correlation
from scipy.signal import correlate, correlation_lags, find_peaks
from itertools import chain
from statistics import mode

# plotting
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mne.viz.evoked import _handle_spatial_colors
import matplotlib.pyplot as plt
import seaborn as sns

# %%
surtype = 'GPT'
BANDPASS = 'delta'
resultsdir = f'/project/3027007.06/results/no-entropy-topdown-durmatch/{BANDPASS}/{surtype}' # f'/project/3027007.06/results/no-entropy/{BANDPASS}/{surtype}'
datapath = '/project/3027007.01/processed/'
conditions = ['trf']
save=True

# %%
# pilot 8/9/10
if surtype == 'GPT':
    srprs = 'surprisal_GPT'
    entr = 'entropy_GPT'
elif surtype == 'ngram':
    srprs = 'surprisal'
    entr = 'entropy'

# %% load the TRFs and the permutations
with open(os.path.join(resultsdir, 'GA.pkl'), 'rb') as f:
    GA = pickle.load(f)
    
info = GA['bottomup_split_surprisal']['trf'].info
lags = lag_span(-0.1, 1.0, info['sfreq'])

# load the cluster-based permutation test
with open(os.path.join(resultsdir, 'trf_stats_interactions.pkl'), 'rb') as f:
    perm = pickle.load(f)

_, clusters, pvals, _ = perm['bottomup_surprisal']

signi_clusters = []
for cl,pv in zip(clusters, pvals):
    if pv <= 0.05:
        signi_clusters.append(cl)

# %% get the shared sensors
ch_idx = [clu[1] for clu in signi_clusters]
ch_idx = list(set(chain(*ch_idx)))

# copy the GA and take away the edge artifact
high = GA['bottomup_split_surprisal']['trf']['bottomup_high_surprisal'].copy()
high = high.crop(tmin=-0.05, tmax=0.95)

low = GA['bottomup_split_surprisal']['trf']['bottomup_low_surprisal'].copy()
low = low.crop(tmin=-0.05, tmax=0.95)

high_channels = high.data[ch_idx]
low_channels = low.data[ch_idx]

# %% perform cross correlation
lags_xcorr = correlation_lags(len(high_channels[0, :]), len(low_channels[0, :]), 'same')

xcorr = []
for ch in range(len(ch_idx)):
    x1 = high_channels[ch, :] # / np.std(high_channels[ch, :])
    x2 = low_channels[ch,:] #/ np.std(low_channels[ch, :])
    
    c = correlate(x1, x2, mode='same') / (np.std(x1) * np.std(x2)) / len(x1)
    #c /= np.max(c) # normalize the correlation
    xcorr.append(c)

xcorr = np.asarray(xcorr)

# %% Get the peaks & roll each signal by its correlation peak
#peaks = [find_peaks(xcorr[i,:])[0] for i in list(range(len(ch_idx)))]
#peaks = [i[0] if len(i) != 0 else lags_xcorr[-1] for i in peaks]
peaks = []
for i in list(range(len(ch_idx))):
    try: 
        pk = find_peaks(xcorr[i,:], height = 0)
        
        if len(pk[0]) > 1:
            max_idx = list(pk[1]['peak_heights']).index(max(pk[1]['peak_heights']))
        
        else:
            max_idx == 0
            
        print(pk[0][max_idx], pk[1]['peak_heights'][max_idx])
        plag = lags_xcorr[pk[0][max_idx]]
        peaks.append(plag)
    
    except IndexError: # no positive peaks
        continue
        
#peaks = [lags_xcorr[find_peaks(xcorr[i,:], height = 0)[0][0]] for i in list(range(len(ch_idx)))]

rolled_low = np.zeros(np.shape(low_channels))

for ch in list(range(np.shape(low_channels)[0])):
    rolled_low[ch,:] = np.roll(low_channels[ch,:], mode(peaks), axis=0) # peaks[ch]

pearson = np.diag(np.corrcoef(rolled_low, high_channels), k=len(ch_idx))

# %% pick random channels + random lags for comparison (=> many times = distribution for comp.)
iterations = 10000
rdm_results = []

for i in range(iterations):
    rdm_idx = np.random.choice(range(269), replace=False, size=(len(ch_idx),))
    rdm_channels = low.data[rdm_idx]
    contrast_channels = high.data[rdm_idx]
    
    rdm_peaks_mean = np.random.randint(info['sfreq'])
    rdm_peaks = np.random.normal(loc=rdm_peaks_mean, scale=np.std(peaks), size=len(ch_idx))
    #rdm_peaks = peaks.copy()
    rdm_rolled = np.zeros(np.shape(low_channels))
    for ch in range(len(ch_idx)):
        rdm_rolled[ch,:] = np.roll(rdm_channels[ch,:], np.int(mode(rdm_peaks)), axis=0) # np.int(np.round(rdm_peaks[ch]))

    rdm_pearson = np.diag(np.corrcoef(rdm_rolled, contrast_channels), k=len(ch_idx))
    rdm_results.append(rdm_pearson)

# %%
xcorr_clean = xcorr
clean_idx = ch_idx

# %%
# remove outliers from correlation values (for viz only)
#xcorr_clean = []
#clean_idx = []
#xcorr_outliers = [np.mean(xcorr)-2*np.std(xcorr), np.mean(xcorr)+2*np.std(xcorr)]
#for sensor_idx,sensor in enumerate(ch_idx):
 #   if np.mean(xcorr[sensor_idx,:]) <= xcorr_outliers[0] or np.mean(xcorr[sensor_idx,:]) >= xcorr_outliers[1]:
  #      continue
  #  else:
   #     xcorr_clean.append(xcorr[sensor_idx,:])
    #    clean_idx.append(sensor)
        
#xcorr_clean = np.asarray(xcorr_clean)
#peaks = [lags_xcorr[find_peaks(xcorr_clean[i,:])[0][0]] for i in list(range(len(clean_idx)))]

# %% ## PLOT THE RESULTS ###
# channel colors
from matplotlib.lines import Line2D
plt.rcParams['axes.xmargin'] = 0

tmp_info = high.info

style = 'seaborn-paper'
plt.style.use(style)

idx = mne.channel_indices_by_type(tmp_info, picks=CH_NAMES)['mag']
trf_times = lag_span(-0.1, 1.0, info['sfreq']) / info['sfreq']
chs = [tmp_info['chs'][i] for i in idx]

colors =  [sns.color_palette("Blues_r",269), sns.color_palette("Reds_r", 269)]

chs = tmp_info['chs']
locs3d = np.array([ch['loc'][:3] for ch in chs])
x, y, z = locs3d.T
spatial_colors = _rgb(x, y, z)

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(6,4))

# plot the original sensors
for i,ch in enumerate(ch_idx):
    axes[0,0].plot(low.times, low.data[ch,:].T,
                  color=colors[0][ch], linewidth=1, alpha=0.8, linestyle='solid')
    axes[0,0].plot(high.times, high.data[ch,:].T,
              color=colors[1][ch], linewidth=1, alpha=0.8, linestyle='solid')

# set legend 
#custom_lines = [Line2D([0], [0], lw=1, color='black', linestyle='solid'), Line2D([0], [0], lw=1, color='black', linestyle='dashed')]
axes[0,0].legend(['Low surprisal', 'High surprisal'],loc='upper left', frameon=False, fontsize=6)
#axes[0,0].xaxis.set_visible(False)
axes[0,0].set_ylabel('Coeff (a.u.)')
axes[0,0].set_title('Bottom-up response')
axes[0,0].set_xlabel('Time (s)')

for i,ch in enumerate(ch_idx):
 #   print(idx)
    axes[1,0].plot(low.times, np.roll(low_channels[i,:].T, mode(peaks), axis=0), color=colors[0][ch], linewidth=1, alpha=0.8)
    axes[1,0].plot(high.times, high_channels[i,:].T, color=colors[1][ch], linewidth=1, alpha=0.8)
    
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Coeff (a.u.)')
axes[1,0].set_title(f'Shifted response by {round(mode(peaks)/(info["sfreq"]/1000))}ms')

# Plot the correlations
sub_colors = sns.color_palette('crest', n_colors=len(ch_idx))
for i,ch in enumerate(clean_idx):
    axes[0,1].plot(lags_xcorr/info['sfreq'], xcorr_clean.T[:,i], color=sub_colors[i], linewidth=1, alpha=0.6)

axes[0,1].set_xlabel('Relative lag (low vs high) (s)')
axes[0,1].axvline(0, c='black',linestyle='dashed')
axes[0,1].axvline(mode(peaks)/info['sfreq'], c='red',linestyle='dashed')
axes[0,1].set_title('Cross-correlation')
axes[0,1].set_ylabel('Corr. coeff (a.u.)')

axes[1,1].set_title('Permutation', x=1.1)
# Plot the Pearson's values against the random ones
# make two axes out of the last one
ax_divider = make_axes_locatable(axes[1,1])
cax = ax_divider.append_axes("right", size='100%', pad='20%', sharey=axes[1,1])
std = [np.std(r) for r in rdm_results]
mu = [np.mean(r) for r in rdm_results]

for name,value,actual_result,ax,title in zip(['mean', 'standard deviation'],[mu, std], [np.mean(pearson), np.std(pearson)], 
                                             [axes[1,1], cax],['mean', 'std of Pearsons R']):
    sns.kdeplot(x=value, ax=ax, fill=True)
    ax.axvline(x=actual_result, color='red')
    ax.set_xlabel(name)
    ax.set_ylabel('')
    #ax.set_title(title)
    if name == 'standard deviation':
      ax.yaxis.set_visible(False)  
    elif name == 'mean':
        ax.legend(['Random', 'Observed'], frameon=False, fontsize=6)
        ax.set_ylabel('Density' )
    ax.margins(x=0.1)

sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(resultsdir, 'correlation-analysis.svg'))

#%% poster plot

tmp_info = low.info

style = 'seaborn-paper'
plt.style.use(style)

idx = mne.channel_indices_by_type(tmp_info, picks=CH_NAMES)['mag']
trf_times = lag_span(-0.2, 0.8, info['sfreq']) / info['sfreq']
chs = [tmp_info['chs'][i] for i in idx]

#chs = tmp_info['chs']
locs3d = np.array([ch['loc'][:3] for ch in chs])
x, y, z = locs3d.T
colors = _rgb(x, y, z)

#plt.style.use('seaborn-paper')
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

# plot the original sensors
for i,ch in enumerate(ch_idx):
    axes[0].plot(low.times, low.data[ch,:].T,
                  color=colors[ch], linewidth=1, alpha=1, linestyle='solid')
    axes[0].plot(high.times, high.data[ch,:].T,
              color=colors[ch], linewidth=1, alpha=0.8, linestyle='dashed')

_handle_spatial_colors(colors=colors[np.asarray(ch_idx)], info=tmp_info, 
                       idx=np.asarray(ch_idx), ch_type='mag', psd=False, ax=axes[0], sphere=None)
#axes[0].legend(['sentence', 'word list'],loc='upper right')
axes[0].set_title('Shared sensors')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Coeff (a.u.)')

# Plot the correlations
sub_colors = sns.color_palette(palette=[colors[ch] for ch in ch_idx])
for i,ch in enumerate(ch_idx):
    axes[1].plot(lags_xcorr/info['sfreq'], xcorr.T[:,i], color=sub_colors[i])

#axes[0,1].plot(lags_xcorr/info['sfreq'], xcorr.T, cmap=my_cmap)
axes[1].set_title("Cross-correlation")
axes[1].set_xlabel('Lag (s)')
axes[1].axvline(0, c='black',linestyle='dashed')
sns.despine()
plt.tight_layout()

#fig.savefig(os.path.join(resultsdir, 'correlation-analysis-POSTER.svg'))