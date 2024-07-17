#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:52:22 2022

@author: sopsla
"""
import re
import os
import numpy as np
import pandas as pd
from scipy.signal import gaussian
from scipy.io.wavfile import read as wavread
from pyeeg.utils import signal_envelope
import textgrid as tg

stim_map = pd.read_csv('/home/lacnsg/sopsla/Documents/audiobook-transcripts/syntax/stim_map.csv',names=['id', 'file', 'lang'])

def get_audio_envelope(story, sfreq=120, N=None):
    """
    Directly compute the envelope, resample and truncate/pad to the right length.
    
    Parameters
    ----------
    sent_id: int
        Sentence number id.
    sfreq : int or float
        Target sampling frequency
    N : int
        Final length wanted
    audio_path : str
        Location of all .wav files (each file should start with)
        
    Returns
    -------
    env : ndarray
        Broad band envelope for this story.
    """
    fname = f'/home/lacnsg/sopsla/Documents/audiobook-transcripts/audio/{story}_normalized.wav'
    #print("Loading envelope for " + fname)
    fs, y = wavread(fname)
    env = signal_envelope(y, fs, resample=sfreq, verbose=False)
    if N is not None and len(env)!=N:
        if len(env) < N:
            env = np.pad(env, (N-len(env), 0))
        else:
            env = env[:N]
    return env

def get_word_frequency(word, wf_file='SUBTLEX-NL.cd-above2.txt', fallback=0.301):
    """
    Author: Hugo Weissbart
    Get word frequencies as given by the SUBTLEX-NL corpus.

    Parameters
    ----------
    word : str
        Word (lowered characters)
    fallback : float
        Value to fall back to if word is not in corpus. Default to the minimum 
        value encountered in the corpus.

    Returns
    -------
    float
        -log(Word frequency) of a word

    """
    if "df_wf" not in globals():
        global df_wf
        df_wf = pd.read_csv(os.path.join('/project/3027005.01/wordfreq/', wf_file), delim_whitespace=True)
    if word.lower() not in df_wf.Word.to_list():
        return fallback
    else:
        return df_wf.loc[df_wf.Word==word.lower(), 'Lg10WF'].to_numpy()[0]

def temporal_smoother(x, win=None, fs=200, std_time=0.015, axis=0):
    """
    Author: Hugo Weissbart
    Smooth (by convolution) a time series (possibly n-dimensional).
    
    Default smoothing window is gaussian (a.k.a gaussian filtering), which has the nice property of being an ideal
    temporal filter, it is equivalent to low-pass filtering with a gaussian window in fourier domain. The window
    is symmetrical, hence this is non causal. To control for the cut-off edge, or standard deviation in the frequency
    domain, use the following formula to go from stndard deviation in time domain to fourier domain:
    
    .. math::
           \sigma_f = \frac{1}{2\pi\sigma_t}
           
    Hence for a standard deviation of 15ms (0.015) in time domain we will get 42Hz standard deviation in the frequency
    domain. Hence roughly cutting off frequencies above 20Hz (as the gaussian spread both negative and positive 
    frequencies).
    
    Parameters
    ----------
    x: ndarray (n_time, any)
        Time series, axis of time is assumed to be the first one (axis=0).
        Change `axis` argument if not.
    win : ndarray (n_win,)
        Smoothing kernel, if None (default) will apply a gaussian window
        with std given by the `std_time` parameter.
    fs : int
        Sampling frequency
    std_tim : float
        temporal standard deviation of gaussian window (default=0.015 seconds)
    axis : int (default 0)
    
    Returns
    -------
    x : ndarray
        Same dimensionality as input x.
    """
    assert np.asarray(x).ndim <= 2, "Toom may dimensions (maximum 2)"
    if win is None:
        win = gaussian(fs, std=std_time * fs)
    x = np.apply_along_axis(np.convolve, axis, x, win, 'same')
    return x

def story_to_triggers(story):
    """
    Returns trigger onset and offset values
    Author: Hugo Weissbart
    """
    sid = str(stim_map.loc[stim_map.file == story  +'.wav', 'id'].to_list()[0])
    if int(sid[0]) < 6:
        return int('1' + sid), int('2' + sid)
    else:
        return int('1' + sid), int('2' + sid[::-1])
    
def get_boundaries(story, onset=True, phonemes=False, tg_path = "/home/lacnsg/sopsla/Documents/audiobook-transcripts/syntax/alignment/"):
    """
    Author: Hugo Weissbart
    """
    tgdata = tg.TextGrid.fromFile(os.path.join(tg_path, story + '.TextGrid'))
    intervals = tgdata.getFirst('MAU' if phonemes else 'ORT-MAU')
    wordlist = []
    times = []
    for k in range(len(intervals)):
        w, ton, toff = intervals[k].mark, intervals[k].minTime, intervals[k].maxTime
        if w is not None and (w != '<p:>') and (w != ''):
            wordlist.append(w)
            times.append(ton if onset else toff)

    return wordlist, times

def remove_punctuation(dataframe):
    """
    Removes all rows in which the 'word' column contains punctuation
    
    input
    dataframe : pd.DataFrame | lexical column must be called 'word'
    
    output
    dataframe : pd.DataFrame
    """
    punctuation_bools = pd.Series([bool(re.search("[a-zA-Z]", wrd)) for wrd in dataframe['word']], name='bools')
    return dataframe[punctuation_bools.values]