#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:42:36 2022

@author: sopsla
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import KFold
import mne
from mne.beamformer import apply_lcmv_epochs

with open(os.path.join('/project/3027007.06/metadata', '269_chnames.csv')) as f:
    CH_NAMES = f.read().split()

class TRF(object):
    "A quick wrapper class to use TRF as Evoked mne data"
    def __init__(self, beta, lags, info, features=["Envelope"], intercept=None):
        self.coef_ = beta
        self.intercept_ = intercept
        self.info = info
        self.lags = lags
        self.times = lags/info['sfreq']
        self.trf = {}
        self.feat_names = features
        for k, f in enumerate(features):
            self.trf[f] = mne.EvokedArray(beta[:, k, :].T, info, tmin=lags[0]/info['sfreq'])
        self.nfeats = len(features)
        
    def __getitem__(self, f):
        return self.trf[f]
    
    def plot_single(self, f, *args, **kwargs):
        self.trf[f].plot(*args, **kwargs)
        
    def plot_all(self, spatial_colors=True, sharey=True, **kwargs):
        f, ax = plt.subplots(1, self.nfeats, figsize=(6*self.nfeats, 4), squeeze=False, sharey=sharey)
        for k, aax in enumerate(ax[0, :]):
            #aax.plot(self.times, self.coef_[:, k, :], 'k', alpha=0.3)
            self.trf[self.feat_names[k]].plot(axes=aax, spatial_colors=spatial_colors, **kwargs, show=False)
            aax.set_title(self.feat_names[k])
        if spatial_colors:
            while len(f.get_axes()) > self.nfeats + 1: # remove all but one color legend
                f.get_axes()[self.nfeats].remove()
        return f
            
    def plot_joint_single(self, f, *args, **kwargs):
        self.trf[f].plot_joint(*args, **kwargs)

def epoch_variable(raw, prestim=0, poststim=0):
    """
    Epochs mne.Raw into variable duration epochs on the basis of
    the event information in the raw data.
    
    input
    -----
    raw :   mne.Raw object
    prestim : float | time in seconds before stimulus starts
    poststim : float | time in seconds after stimulus has ended
    
    output
    -----
    epochs : list of mne.EpochsArray | the epochs
    onsets : np.array | onset times
    offsets : np.array | offset times
    """
    print('\nReading events...')
    events = mne.find_events(raw, stim_channel='UPPT001')
    
    onsets = events[::2]
    offsets = events[1::2]
        
    epochs = []
    for onset, offset in zip(onsets, offsets):
        t_idx = raw.time_as_index([onset[0]/1200-prestim, offset[0]/1200+poststim])
        epoch, times = raw[:, t_idx[0]:t_idx[1]]
        epochs.append(mne.EpochsArray(np.expand_dims(epoch, 0), raw.info, tmin=-prestim, verbose=False))
        
    print('Epoching done: {0} trials found.'.format(len(epochs)))
    
    return epochs, onsets[:np.shape(offsets)[0],2], offsets[:,2]

def crossval(xlist, ylist, lags, fs, alpha, features, info, 
             n_splits=5, fit_intercept=True, plot=True):
    """
    Use cross-validation to find the best lambda values (regularization) among
    the list given for one subject.
    Works for multiple "epochs" or stories.
    Mean is computed over sensors. Best lambda is
    determined as the most frequent one over all folds.
    
    Input
    -----
    x : list of np.array
        features
    y : list of np.array 
        meg
    lags : np.array
        Output of lag_span()
    fs : int
        Sampling frequency
    alpha : list of int
        Regularization parameters to test
    features : list of str
        Features to include in the model.
        Options: 'envelope', 'word onset', 'word frequency'
    info : mne info struct
    n_splits : int
        Number of folds. Default = 5
    fit_intercept : bool
        Whether to add a column of 1s to the design matrix
        to fit the intercept.
    
    Returns
    -------
    alpha : int
        best regularization parameter for subject
    scores : np.array
        R-values
    betas_new :
        betas
    TRF_list : class TRF
        Instance of class TRF for the best alpha for both conditions
    """
    # make sure we have several alphas
    if np.ndim(alpha) < 1 or len(alpha) <= 1:
        raise ValueError("Supply more than one alpha to use this cross-validation method.")   
    
    info["sfreq"] = int(fs)
    
    # create KFold object for splitting...
    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = np.zeros((n_splits, len(alpha), info['nchan']))
    
    # allocate memory for covmat
    if fit_intercept:
        e = 1
    else:
        e = 0
    
    array_size = len(features)
        
    XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
    Xty = np.zeros((len(lags)*array_size + e, info['nchan']))   

    # start cross-validation
    for kfold, (train, test) in enumerate(kf.split(xlist)):
        print("Training/Evaluating fold {0}/{1}".format(kfold+1, n_splits))
       
        # reset covmat!
        XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
        Xty = np.zeros((len(lags)*array_size + e, info['nchan']))    
        
        x_train, x_test = np.array(xlist,dtype=object)[np.array(train)], np.array(xlist,dtype=object)[np.array(test)]
        y_train, y_test = np.array(ylist,dtype=object)[np.array(train)], np.array(ylist,dtype=object)[np.array(test)]
       
        # accumulate covmat for this fold
        for X,y in zip(x_train,y_train):        
            if fit_intercept:
                X = np.hstack([np.ones((X.shape[0],1)),X])
                
            XtX += X.T @ X
            Xty += X.T @ y
        
        # split into u,s,v
        u,s,v = np.linalg.svd(XtX)
        
        # compute betas
        if np.ndim(alpha) > 0:
            betas = [(u @ np.diag(1/(s+a))) @ v @ Xty for a in alpha]
        else:
            betas = np.linalg.inv(XtX + alpha * np.eye(len(lags))) @ Xty
        
        del XtX, Xty
        
        # test the model 
        r_ = []  # list accumulates r scores per trial
        for X,y in zip(x_test, y_test):
            X = np.hstack([np.ones((X.shape[0],1)), X])
                
            yhat = X @ betas  # shape: len of (alphas, samples, channels)     
            r_.append(np.asarray([np.diag(np.corrcoef(yhat[a, :, :], y, rowvar=False), k=info['nchan'])
                             for a in range(len(alpha))]))  # shape: alphas, channels
        
        r_means = (np.asarray(r_)).mean(0)  # first dimension of array from list = trials/epochs. Shape: (alpha, channel)
        scores[kfold, :, :] = r_means
        print(r_means.mean(-1).mean(0))

   # plt.plot(scores)
    # Get the best alpha 
    # Take the mean over sensors & maximum value over alphas
    peaks = scores.mean(-1).argmax(1)
    
    # catch function: if reconstruction accuracy never peaks, take value
    # BEFORE reconstruction accuracy takes a steep dive.
    catch_r = scores.mean(-1)

    for kf in list(range(n_splits)):
        if all([catch_r[kf,i+1] < catch_r[kf,i] for i in list(range(len(alpha)-1))]):
            deriv = [catch_r[kf,i+1] - catch_r[kf,i] for i in list(range(len(alpha)-1))]
            peaks[kf] = deriv.index(min(deriv))
                
    best_alpha = alpha[mode(peaks)]     
    
    # plotting
    if plot:
        # TODO: change colors
        scores_plot = scores.mean(-1).T  # .mean(-1)
        plt.semilogx(alpha, scores_plot)
        plt.semilogx(alpha[peaks], [scores.mean(-1)[i,peak] for i,peak in enumerate(peaks)], '*k')
    
    print("Returning the best alpha, r-values, and highest alpha per fold.")
    return best_alpha, scores, peaks

def crossval_trf(xlist, ylist, lags, fs, alpha, features, info, 
                 n_splits=5, fit_intercept=True, plot=True):
    """
    Use cross-validation to find the best lambda values (regularization) among
    the list given for one subject.
    Works for multiple "epochs" or stories.
    Mean is computed over sensors. Best lambda is
    determined as the most frequent one over all folds.
    
    Returns average of best betas.
    
    Input
    -----
    x : list of np.array
        features
    y : list of np.array 
        meg
    lags : np.array
        Output of lag_span()
    fs : int
        Sampling frequency
    alpha : list of int
        Regularization parameters to test
    features : list of str
        Features to include in the model.
        Options: 'envelope', 'word onset', 'word frequency'
    info : mne info struct
    n_splits : int
        Number of folds. Default = 5
    fit_intercept : bool
        Whether to add a column of 1s to the design matrix
        to fit the intercept.
    
    Returns
    -------
    alpha : int
        best regularization parameter for subject
    scores : np.array
        R-values
    betas_new :
        betas
    TRF_list : class TRF
        Instance of class TRF for the best alpha for both conditions
    """
    # make sure we have several alphas
    if np.ndim(alpha) < 1 or len(alpha) <= 1:
        raise ValueError("Supply more than one alpha to use this cross-validation method.")   
    
    info["sfreq"] = int(fs)
    
    # create KFold object for splitting...
    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = np.zeros((n_splits, len(alpha), info['nchan']))
    all_betas = []
    
    # allocate memory for covmat
    if fit_intercept:
        e = 1
    else:
        e = 0
    
    array_size = len(features)
        
    XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
    Xty = np.zeros((len(lags)*array_size + e, info['nchan']))   

    all_betas = []
    
    # start cross-validation
    for kfold, (train, test) in enumerate(kf.split(xlist)):
        print("Training/Evaluating fold {0}/{1}".format(kfold+1, n_splits))
       
        # reset covmat!
        XtX = np.zeros((len(lags)*array_size + e, len(lags)*array_size + e))
        Xty = np.zeros((len(lags)*array_size + e, info['nchan']))    
        
        x_train, x_test = np.array(xlist,dtype=object)[np.array(train)], np.array(xlist,dtype=object)[np.array(test)]
        y_train, y_test = np.array(ylist,dtype=object)[np.array(train)], np.array(ylist,dtype=object)[np.array(test)]
       
        # accumulate covmat for this fold
        for X,y in zip(x_train,y_train):        
            if fit_intercept:
                X = np.hstack([np.ones((X.shape[0],1)),X])
                
            XtX += X.T @ X
            Xty += X.T @ y
        
        # split into u,s,v
        u,s,v = np.linalg.svd(XtX)
        
        # compute betas
        if np.ndim(alpha) > 0:
            betas = [(u @ np.diag(1/(s+a))) @ v @ Xty for a in alpha]
        else:
            betas = np.linalg.inv(XtX + alpha * np.eye(len(lags))) @ Xty
        
        del XtX, Xty
        
        # test the model 
        r_ = []  # list accumulates r scores per trial
        for X,y in zip(x_test, y_test):
            X = np.hstack([np.ones((X.shape[0],1)), X])
                
            yhat = X @ betas  # shape: len of (alphas, samples, channels)     
            r_.append(np.asarray([np.diag(np.corrcoef(yhat[a, :, :], y, rowvar=False), k=info['nchan'])
                             for a in range(len(alpha))]))  # shape: alphas, channels
        
        r_means = (np.asarray(r_)).mean(0)  # first dimension of array from list = trials/epochs. Shape: (alpha, channel)
        scores[kfold, :, :] = r_means
        print(r_means.mean(-1).mean(0))

   # plt.plot(scores)
    # Get the best alpha 
    # Take the mean over sensors & maximum value over alphas
    peaks = scores.mean(-1).argmax(1)
    
    # catch function: if reconstruction accuracy never peaks, take value
    # BEFORE reconstruction accuracy takes a steep dive.
    catch_r = scores.mean(-1)

    for kf in list(range(n_splits)):
        if all([catch_r[kf,i+1] < catch_r[kf,i] for i in list(range(len(alpha)-1))]):
            deriv = [catch_r[kf,i+1] - catch_r[kf,i] for i in list(range(len(alpha)-1))]
            peaks[kf] = deriv.index(min(deriv))
                
    best_alpha = alpha[mode(peaks)]     
    
    # select the best betas
    all_betas = np.asarray([b[mode(peaks), :, :] for b in all_betas])
    
    # average over the folds
    mn_betas = np.mean(all_betas, axis=1)
    
    # plotting
    if plot:
        # TODO: change colors
        scores_plot = scores.mean(-1).T  # .mean(-1)
        plt.semilogx(alpha, scores_plot)
        plt.semilogx(alpha[peaks], [scores.mean(-1)[i,peak] for i,peak in enumerate(peaks)], '*k')
    
    print("Returning the best alpha, r-values, and highest alpha per fold.")
    
    
    return best_alpha, scores, peaks, mn_betas

def lcmv_apply(epochs, filters):
    for epoch in epochs:
        yield apply_lcmv_epochs(epoch, filters, max_ori_out='signed')[0]
        
def morph_apply(source_epochs, subject, src_to, subjects_dir):
    for epoch in source_epochs:
        morph = mne.compute_source_morph(epoch, subject_from=subject,
                                 subject_to='fsaverage',
                                 src_to=src_to,
                                 subjects_dir=subjects_dir)
        yield morph.apply(epoch)
        
def parcel_labels(hemisphere, labels, all_labels, brain=None, plot=False):
    """
    """
    if hemisphere == 'left':
        labnames = [f'L_{parcel}_ROI-lh' for parcel in labels]
    elif hemisphere == 'right':
        labnames = [f'R_{parcel}_ROI-rh' for parcel in labels]
    all_labnames = [lab.name for lab in all_labels]

    parcels = []
        
    for parcel in labnames:
        idx = all_labnames.index(parcel)
        if plot:
            brain.add_label(all_labels[idx])
  
        parcels.append(all_labels[idx])

    return parcels
