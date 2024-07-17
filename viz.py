#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:09:19 2022

@author: sopsla
"""
import os
import numpy as np
import warnings
import mne

# plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from sklearn.decomposition import PCA
from scipy.stats import sem

# local modules
from pyeeg.utils import lag_span
from meg import CH_NAMES

listcolors = sns.color_palette('flare')[2:5]
sentcolors = sns.color_palette('crest')[2:5]

colors={'list': sns.color_palette('flare'),
        'sentence': sns.color_palette('crest')}

def plot_topomap_rvals(clusterstats, info, pv=0.0125, bar=True, fig=None, axes=None, vlim=None):
    if axes is None:
        if bar == True:
            fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4), gridspec_kw={'width_ratios': [1,0.1]})
        elif bar == False:
            fig,axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
            axes = [axes]
            
    if bar == True and len(axes) < 2:
        raise UserWarning('Supplied too few axes, bar will be omitted')
        bar = False
        
    t_obs, clusters, pvals, h0 = clusterstats
    
    if vlim == None:
        vmin = min(t_obs)
        vmax = max(t_obs)
    
        if abs(vmin) > vmax:
            vmax = abs(vmin)
        elif vmax > abs(vmin):
            vmin = -vmax
    else:
        vmin = -vlim
        vmax = vlim
        
    mask = np.zeros((269))
    for i,cluster in enumerate(clusters):
        if pvals[i] < pv:
            chans=cluster
            mask[chans] = 1
                
    mask = np.array(mask, dtype=bool)
    
    if sum(pvals < pv) == len(pvals):
        mask = None
    
    im,_ = mne.viz.plot_topomap(t_obs, info, axes=axes[0], vmin=vmin, vmax=vmax, show=False, mask=mask)
    
    if bar:
        _ = fig.colorbar(im, cax=axes[1])
    

# figure for single line evoked plots
def plot_2samp(trf, cluster_stats, info, use_style, resultsdir, cluster_alpha=0.01, tmin=-0.2, tmax=0.8,
                lag_of_interest=None, plot_diff=False, plot_style='clusters', topomap=False, ax=None, 
                colors=None,  legend=True, save=False, flip=True, vlim=0.006):
    """
    Plots TRFs and clusters in ERP-style.
    
    INPUT
    -----
    trf : list of mne.Evoked (grand average trf)
    analysis : str | description of analysis, ie large or small
    out : tuple | output of mne.stats.spatio_temporal_cluster_test
    feature : str
    use_style : str
    resultsdir : directory to save
    lag_of_interest : list of float | time points in seconds for topomaps
    plot_diff : bool | plot difference between time courses
    plot_style : str | options: 'clusters', 'gfp', 'butterfly'
    colors : list of color palettes | listcolors, sentcolors 
    topomap : bool 
    save : bool
    
    OUTPUT
    -----
    matplotlib.figure
    
    """
    
    plt.style.use(use_style)

    if lag_of_interest == None and topomap == True:
        raise ValueError('To plot topomaps, supply lags of interest')
    if lag_of_interest != None and topomap == False:
        warnings.warn(
            'Lags of interest have been supplied, topomaps will be added')
        topomap = True
    if ax is None and topomap == True:
        topomap = False
        raise UserWarning('Only one ax supplied, topomap will be omitted')

    # selecting the right feature stats
    t_obs, clusters, pvals, h0 = cluster_stats
    orig_times = trf[0].times
    
    # getting the data & removing the edge effect
    trf1 = trf[0].copy().crop(tmin=tmin+0.04, tmax=tmax-0.04)
    trf2 = trf[1].copy().crop(tmin=tmin+0.04, tmax=tmax-0.04)    
    
    trf_times = trf1.times
    trf1 = trf1.data.T
    trf2 = trf2.data.T
    
    diff_trf = trf1 - trf2

    # no idea what is happening here
    nan_trf = np.ones_like(trf1) * np.nan
    nan_trf_list = np.ones_like(trf1) * np.nan
    nan_diff = nan_trf.copy()

    allchans = []
    for k_cluster, c in enumerate(clusters):
        if pvals[k_cluster] < cluster_alpha:
            lags, chans = c
            
            for cx in list(set(chans)):
                allchans.append(cx)
            
            if max(lags) > (max(trf_times)*200)+40:
                lagidx = np.where(lags >= (max(trf_times)*200)+(abs(tmin)*200))[0]
                lags = np.delete(lags, lagidx)
                chans = np.delete(chans, lagidx)
                
            if min(lags) < (min(trf_times)*200)+40:
                lagidx = np.where(lags <= (min(trf_times)*200)+(abs(tmin)*200))[0]
                lags = np.delete(lags, lagidx)
                chans = np.delete(chans, lagidx)
            
            lags = lags - (len(orig_times) - len(trf_times))
            
            #print(f"{len(np.unique(chans))} channels in this cluster, time-span: {max(lags)-min(lags)}")
            if abs(trf1[lags, chans].mean()) > abs(trf2[lags, chans].mean()):
                nan_trf[lags, chans] = trf1[lags, chans]
            else:
                nan_trf_list[lags, chans] = trf2[lags, chans]
            nan_diff[lags, chans] = diff_trf[lags, chans]

    signi_times = (~np.all(np.isnan(nan_diff), axis=1)).astype(np.float)
    signi_times[signi_times == 0.] = np.nan
    signi_chans_bool = (~np.all(np.isnan(nan_diff), axis=0))
    # print(np.asarray(CH_NAMES)[signi_chans_bool])

    if sum(signi_chans_bool) == 0 or sum(signi_chans_bool == 1):
        signi_chans_bool = (np.all(np.isnan(nan_diff), axis=0))

    ### PLOT ####
    if topomap:
        if ax is None:
            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), gridspec_kw={
                                 'width_ratios': [4, 1.5, 1.5, 0.15]})
        plotax = ax[0]
    else:
        if ax is None:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        plotax = ax
    
    # TRFs
    if plot_diff:
        plotax.plot(trf_times, diff_trf, lw=0.5,
                    alpha=0.4, color='k', zorder=1)
        plotax.plot(trf_times, nan_diff, lw=2, color='k', zorder=2)
    else:
        
        if plot_style == 'sensors':
            # plot only the sensors that belong to clusters
            channels = np.asarray(CH_NAMES)[np.asarray(allchans, dtype=int)]
            
            if colors == None:
                raise NotImplementedError()
                
            else:
                lcolors = colors[0]
                scolors = colors[1]
                
                idx = mne.channel_indices_by_type(info, picks=channels)['mag']
                #chs = [info['chs'][i] for i in idx]
    
                for ch in idx:
                    plotax.plot(trf_times, trf2[:,ch], color=lcolors[ch],alpha=0.6, lw=2)
                    plotax.plot(trf_times, trf1[:,ch], color=scolors[ch],alpha=0.6, lw=2)
                    
                  #  plotax.plot(trf_times, nan_trf_list[:,ch], color=lcolors[ch],alpha=0.8, lw=3)
                  #  plotax.plot(trf_times, nan_trf[:,ch], color=scolors[ch],alpha=0.8, lw=3)
            
            
        if plot_style == 'butterfly':
            # Plot all sensor TRFs
            if colors == None:
                plotax.plot(trf_times, trf2, lw=2,
                            color=listcolors[-1], alpha=0.6, zorder=1)
                plotax.plot(trf_times, trf1, lw=2, alpha=0.6,
                            color=sentcolors[-1], zorder=3)
                
            else:
                lcolors = colors[0]
                scolors = colors[1]
                
                idx = mne.channel_indices_by_type(info, picks=info.ch_names)['mag']
                #chs = [info['chs'][i] for i in idx]
    
                for ch in idx:
                    plotax.plot(trf_times, trf2[:,ch], color=lcolors[ch],alpha=0.6, lw=2)
                    plotax.plot(trf_times, trf1[:,ch], color=scolors[ch],alpha=0.6, lw=2)
                    
                    plotax.plot(trf_times, nan_trf_list[:,ch], color=lcolors[ch],alpha=0.8, lw=3)
                    plotax.plot(trf_times, nan_trf[:,ch], color=scolors[ch],alpha=0.8, lw=3)
                    
            #plotax.plot(trf_times, nan_trf, lw=3, color='r', zorder=4);
            #plotax.plot(trf_times, nan_trf_list, lw=3, color='k', zorder=2);
        # Global "field power"
        elif plot_style == 'gfp':
            plotax.plot(trf_times, np.sqrt(np.std(trf2**2, 1)),
                        lw=2.5, color=listcolors[-1], zorder=1)
            plotax.plot(trf_times, np.sqrt(np.std(trf1**2, 1)),
                        lw=2.5, color=sentcolors[-1], zorder=3)
        # Average significant sensors
        elif plot_style == 'clusters':
            if flip:
                # We flip signed based on projection to the greatest PC
                proj = PCA(1).fit_transform(trf2[:, signi_chans_bool].T)
                plotax.plot(trf_times, np.mean(trf2[:, signi_chans_bool] *
                                               np.sign(proj).squeeze(), 1), color=listcolors[-1], linestyle='dashed')
                plotax.fill_between(trf_times, np.mean(trf2[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) -
                                    np.std(trf2[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    np.mean(trf2[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) +
                                    np.std(trf2[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    alpha=0.5, color=listcolors[-1], label='_nolegend_')
                proj = PCA(1).fit_transform(trf1[:, signi_chans_bool].T)
                plotax.plot(trf_times, np.mean(trf1[:, signi_chans_bool]
                                               * np.sign(proj).squeeze(), 1), color=sentcolors[-1])
                plotax.fill_between(trf_times, np.mean(trf1[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) -
                                    np.std(trf1[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    np.mean(trf1[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) +
                                    np.std(trf1[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    alpha=0.5, color=sentcolors[-1], label='_nolegend_')
            else:
                # here we use the PCA from one condition to inform the flip of
                # the other condition
                proj = PCA(1).fit_transform(trf2[:, signi_chans_bool].T)
                #proj = PCA(1).fit_transform(trf1[:, signi_chans_bool].T)
                plotax.plot(trf_times, np.mean(trf2[:, signi_chans_bool] *
                                               np.sign(proj).squeeze(), 1), color=listcolors[-1], linestyle='dashed')
                plotax.fill_between(trf_times, np.mean(trf2[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) -
                                    np.std(trf2[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    np.mean(trf2[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) +
                                    np.std(trf2[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    alpha=0.5, color=listcolors[-1], label='_nolegend_')
                plotax.plot(trf_times, np.mean(trf1[:, signi_chans_bool]
                                               * np.sign(proj).squeeze(), 1), color=sentcolors[-1])
                plotax.fill_between(trf_times, np.mean(trf1[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) -
                                    np.std(trf1[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    np.mean(trf1[:, signi_chans_bool] * np.sign(proj).squeeze(), 1) +
                                    np.std(trf1[:, signi_chans_bool]
                                           * np.sign(proj).squeeze(), 1),
                                    alpha=0.5, color=sentcolors[-1], label='_nolegend_')
                   
        # Plot significant times as a line at the bottom
        plotax.plot(trf_times, signi_times*0+0.02,
                    transform=plotax.get_xaxis_transform(), color='k', lw=4)
        plotax.set_ylabel('Coefficient (a.u.)')
        
    if legend:
        leg = plotax.legend(['Low', 'High'], loc='best', frameon=False)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
            line.set_alpha(1)
            
    sns.despine()

    if topomap:
        for k, l in enumerate(lag_of_interest):
            l = np.argmin(abs(l-trf_times))
            plotax.axvspan(trf_times[l]-0.01, trf_times[l]+0.01,
                           facecolor='grey', alpha=0.3, zorder=-1)
            
            # Topos
            im, _ = mne.viz.topomap.plot_topomap((trf1-trf2)[l, :], info,
                                                 mask=(~np.isnan(nan_trf_list[l, :]) | ~np.isnan(
                                                 nan_trf[l, :])),
                                                 show=False, axes=ax[k+1], vmin=-vlim, vmax=vlim)
                
        plt.colorbar(im, cax=ax[-1])
        plotax.title.set_text('Contrast - reset manually')

    plotax.set_xlabel('Time (s)')
    
    if ax is None: #type(ax) != np.ndarray or type(ax) != matplotlib.axes._subplots.AxesSubplot:
        f.tight_layout()
    
     #   if save:
      #      f.savefig(os.path.join(resultsdir, 'TRF_stats_{0}_{1}_{2}_ttest.png'.format(
       #         feature, analysis, use_style[8:])))
        return f

# figure for one-sample TRF plots
def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors.
    CODE FROM MNE.
    """
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb

def plot_1samp(GA, clusters, condition, use_style, resultsdir, plot_type='clusters', color=sentcolors[-1], 
               colors=None,feature='word frequency', title=True, topomap=False, colorbar=True, topolags=None,
               info=None, ax=None, save=False):
    """
    Plots one-sample cluster-based permutation tests
    
    INPUT
    -----
    GA : mne.evoked (sorry)
    clusters : tuple | output of mne.stats.permutation_cluster_test
    condition : str | 'list' or 'sentence'
    use_style : str | one of matplotlibs styles
    resultsdir : str | full pathname, where plot will be saved
    color : list of RGB values
    topomap : bool
    topolags : bool
    info : mne.info object
    ax : matplotlib.pyplot.Axes
    save : bool
    
    OUTPUT
    -----
    matplotlib.figure
    
    """
    plt.style.use(use_style)
    t_obs, clusters, pvals, _ = clusters
    
   # trf_times = lag_span(-0.2, 0.8, 120) / 120

    if ax is None:
        
        if topomap:
            if colorbar:
                f, tmpax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), gridspec_kw={
                                     'width_ratios': [4, 1.2, 1.2, 0.1]})
            else:
                f, tmpax = plt.subplots(nrows=1, ncols=3, figsize=(15,4), gridspec_kw={
                    'width_ratios': [4,1.2,1.2]})
            plotax = tmpax[0]
        else:
            f, plotax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    
    else:
        if topomap:
            tmpax = ax
            plotax = ax[0]
        else:
            plotax = ax
        
    
    #trf = GA.data.copy().T
    # crop off the edge effects
    tmp = GA.copy().crop(tmin=-0.17, tmax=0.77)
    trf_times = tmp.times
    
    trf = tmp.data.copy().T
    del tmp
    
    nan_trf = np.ones_like(trf) * np.nan
    for k_cluster, c in enumerate(clusters):
        if pvals[k_cluster] < 0.05: # p-value
            lags, chans = c
            try:
                nan_trf[lags,chans] = trf[lags,chans]
            except IndexError:
                nan_trf[lags[:-1],chans] = trf[lags[:-1],chans]

    signi_times = (~np.all(np.isnan(nan_trf), axis=1)).astype(np.float)
    signi_times[signi_times == 0.] = np.nan
    signi_chans_bool = (~np.all(np.isnan(nan_trf), axis=0))
    # print(np.asarray(CH_NAMES)[signi_chans_bool])

    if sum(signi_chans_bool) == 0 or sum(signi_chans_bool == 1):
        signi_chans_bool = (np.all(np.isnan(nan_trf), axis=0))

    trf_sign = trf[:,signi_chans_bool]
    
    if plot_type == 'clusters':
        plotax.plot(trf_times, np.mean(trf_sign, axis=1), color=color)
        plotax.fill_between(trf_times, np.mean(trf_sign,axis=1) - np.std(trf_sign,axis=1),
                            np.mean(trf_sign, axis=1) + np.std(trf_sign,axis=1), color=color,alpha=0.5)
    
    elif plot_type == 'butterfly':
        if info == None:
            raise ValueError('supply mne info structure')
        
        idx = mne.channel_indices_by_type(info, picks=info.ch_names)['mag']
        
        chs = [info['chs'][i] for i in idx]
        locs3d = np.array([ch['loc'][:3] for ch in chs])
        
        if colors == None:
            x, y, z = locs3d.T
            colors = _rgb(x, y, z)
        
        for ch in idx:
            plotax.plot(trf_times, trf[:,ch], color=colors[ch],alpha=0.3, linewidth=1)
            plotax.plot(trf_times, nan_trf[:,ch], color=colors[ch],alpha=0.8, linewidth=3)
        
    if topomap:
        for k, l in enumerate(topolags):
            l = np.argmin(abs(l-trf_times))
            plotax.axvspan(trf_times[l]-0.01, trf_times[l]+0.01,
                           facecolor='grey', alpha=0.3, zorder=-1)
            # Topos
            im, _ = mne.viz.topomap.plot_topomap(trf[l, :], info,
                                                 mask=(~np.isnan(nan_trf[l, :]) | ~np.isnan(
                                                     nan_trf[l, :])),
                                                 show=False, axes=tmpax[k+1], vmin=-0.01, vmax=0.01)
            if colorbar:
                plt.colorbar(im, cax=tmpax[-1])

    # black bar at the bottom
    plotax.plot(trf_times, signi_times*0+0.02,
            transform=plotax.get_xaxis_transform(), color='k', lw=4)
    plotax.set_ylim((-0.01,0.01))

    # labels etc
    plotax.set_ylabel('Coefficient (a.u.)')
    plotax.set_xlabel('Time (s)')
    sns.despine()
    
    if title:
        plotax.title.set_text(f"{''.join([feature[0].upper(), feature[1:]])} in {condition} condition")
    #plotax.legend(labels=['sentence','word list'], loc='upper right')

    if ax is None:
        f.tight_layout()

        if save:
            plt.savefig(os.path.join(resultsdir, f'trf-{condition}-stats.png'))

        return f

def plot_all_bar(r_values, use_style=None, axes=None, models=None, abbrev=None, save=False):
    """
    Creates a bar plot with bars per condition and model.

    INPUT
    -----
    r_data : pd.DataFrame
        output of collect.compile_rvals()
    axes : matplotlib axes object | None
    save : Boolean
    
    OUTPUT
    ------
    bar_means : matplotlib figure
    
    """
    if use_style != None:
        plt.style.use(use_style)

    if models is None:
        models = ['onset', 'entropy', 'surprisal', 'frequency','entropy_surprisal', 'frequency_entropy', 'frequency_surprisal', 'frequency_entropy_surprisal']
    if abbrev is None:
        abbrev = ['Onset', 'Entr.', 'Surp.', 'Freq.', 'Entr/surp.', 'Freq/entr.', 'Freq/surp.','Full']
    colors={'list': sns.color_palette('flare', n_colors=len(models)),
        'sentence': sns.color_palette('crest', n_colors=len(models)),
        'r': sns.color_palette('mako', n_colors=len(models))}
        
    sems = dict.fromkeys(set(r_values['condition']))
    means = dict.fromkeys(set(r_values['condition']))
    
    # calculate the standard error of the mean (SEM)
    for condt in set(r_values['condition']):
        sems[condt] = dict.fromkeys(models)
        means[condt] = dict.fromkeys(models)
        
        for mdl in models:
            sems[condt][mdl] = sem(r_values.loc[(r_values['condition'] == condt) & (r_values['model'] == mdl), 'r_values'])
            means[condt][mdl] = np.mean(r_values.loc[(r_values['condition'] == condt) & (r_values['model'] == mdl), 'r_values'])
            
    if axes is None:
        fig,axes = plt.subplots(figsize=(6,3), ncols=len(set(r_values['condition'])), sharey=True)
    
    if len(set(r_values['condition'])) == 1:
        axes = [axes]
        
    for i,(ax,condition) in enumerate(zip(axes, set(r_values['condition']))):

        sns.barplot(data=r_values.loc[r_values['condition'] == condition], y='r_values',
                    x='model', palette=colors[condition], ax=ax,
                    dodge=False, order=models,
                    ci=None)
        if i == 0:
            ax.set_title(list(set(r_values['condition']))[0])
            ax.set_ylabel(r"$\Delta$ pearson's R")
        elif i == 1:
            ax.set_title(list(set(r_values['condition']))[1])
            ax.get_yaxis().set_visible(False)
        
        x_values = list(range(len(set(r_values['model']))))
        y_values = list(means[condition].values())
        y_error = list(sems[condition].values())
        ax.errorbar(x_values, y_values, yerr=y_error, fmt='', capsize=2, ls='', color='black', linewidth=1, capthick=1)
        ax.set_xticklabels(abbrev)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.xaxis.label.set_visible(False)
    
    if axes is None:
        plt.tight_layout()
        plt.suptitle('Absolute accuracy increase from Envelope model', fontweight='bold')

        return fig
    
def plot_all_box(r_values, colors=None, use_style=None, ax=None, models=None, abbrev=None, save=False):
    """
    Creates a bar plot with bars per condition and model.

    INPUT
    -----
    r_data : pd.DataFrame
        output of collect.compile_rvals()
    axes : matplotlib axes object | None
    save : Boolean
    
    OUTPUT
    ------
    bar_means : matplotlib figure
    
    """
    if use_style != None:
        plt.style.use(use_style)

    if colors == None:
        colors=sns.color_palette('crest', n_colors=len(models))

    # calculate the standard error of the mean (SEM)
    sems = dict.fromkeys(models)
    means = dict.fromkeys(models)
    
    for mdl in models:
        sems[mdl] = sem(r_values.loc[r_values['model'] == mdl, 'r_values'])
        means[mdl] = np.mean(r_values.loc[r_values['model'] == mdl, 'r_values'])
        
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,3), ncols=1, sharey=True)
    
    sns.boxplot(data=r_values, y='r_values',
                x='model', palette=colors, ax=ax,
                dodge=False, order=models,
                showmeans=True,
                showcaps=False,
                fliersize=0,
                meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black", "markersize":4})
    
    ax.set_ylabel(r"$\Delta$ pearson's R")
   
    ax.set_xticklabels(abbrev)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.xaxis.label.set_visible(False)

    if ax is None:
        plt.tight_layout()
        plt.suptitle('Absolute accuracy increase from Envelope model', fontweight='bold')

        return fig