"""
A set of functions for data augmentation of EEG data.
@author: Seonghun Park / s.park7532@gmail.com
"""

import numpy as np
import librosa as lib
from noisy import *
import scipy.signal as signal
import math as mat


## Amplitude Perturbation : Generate data by adding gauss, salt&pepper, or speckle noise to amplitude spectrum
def Amplitude_Perturbation(X, Y, Aug_Multiplier, NFFT, type, trial_dim=0, var_par=0):
    # X: real data  | X.shape = [trial, ..., time-series]    | last dimension has to be the time-series data
    # Y: label      | Y.shape = [total trial*2, ]
    # Aug_Multiplier = Times to generate additional data
    # NFFT: number of samples of each epoch for STFT
    # trial_dim = dimension of trials (default: 0)
    # type: 1)gauss 2)s&p 3)speckle
    # var_par: variance of Gaussian noise

    aug_X = X
    
    for aug_idx in range(Aug_Multiplier):
        
        step_size = int(NFFT/2)
        
        STFT = lib.stft(X, n_fft = NFFT, hop_length=step_size, win_length=NFFT)
        STFT.real = noisy(type, STFT.real, var_par) # addition of noise to the amplitude spectrum
        signal = lib.istft(STFT, hop_length=step_size, win_length=NFFT, window="hann")
        aug_X = np.concatenate([aug_X, signal], axis = trial_dim)

    label = np.tile(Y, reps=[Aug_Multiplier+1,1])
    
    return aug_X, label    
    
   

# Time-domain recombination : Generate data by shuffling time-domain epochs in time-frequency map
def TDR(X, Y, win_len, fs, Aug_Multiplier, trial_dim=0):
    # X: real data  | X.shape = [trial, ..., time-series]    | last dimension has to be the time-series data
    # Y: label      | Y.shape = [total trial*2, ]
    # win_len: length of each epoch in seconds
    # fs: sampling rate
    # Aug_Multiplier = Times to generate additional data
    # trial_dim = dimension of trials (default: 0)

    nSeg = win_len*fs
    _, _, Zxx = signal.stft(X, fs, nperseg=nSeg, noverlap=0, boundary=None, padded=False)
    
    trial_perm_idx = np.random.randint(X.shape[trial_dim], size=(Aug_Multiplier, X.shape[trial_dim], Zxx.shape[-1]))
    
    Zxx_permuted = np.zeros([Aug_Multiplier, X.shape[trial_dim], Zxx.shape[1], Zxx.shape[2], Zxx.shape[-1]])
    for aug_multiple_idx in range(Aug_Multiplier):
        for trial_idx in range(X.shape[trial_dim]):
            for t_seg_idx in range(Zxx.shape[-1]):
                Zxx_permuted[aug_multiple_idx, trial_idx, :, :, t_seg_idx] = \
                    Zxx[trial_perm_idx[aug_multiple_idx, trial_idx, t_seg_idx], :, :, t_seg_idx]
                
    _, iZxx = signal.istft(Zxx_permuted, fs, nperseg=nSeg, noverlap=0, boundary=None)
    
    aug_X = iZxx.reshape(np.hstack([-1, iZxx.shape[2:]]))
    aug_X = np.concatenate([X, aug_X], axis=trial_dim)
    aug_Y = (np.ones(aug_X.shape[trial_dim])*Y[0]).astype(np.int64)
    
    return aug_X, aug_Y
    


# Surrogate: Generate data by randomizing the phase of Fourier-transformed signal 
def Surrogate(X, Y, Aug_Multiplier, trial_dim=0):
    # X: real data  | X.shape = [trial, ..., time-series]    | last dimension has to be the time-series data
    # Y: label      | Y.shape = [total trial*2, ]
    # Aug_Multiplier = Times to generate additional data
    # trial_dim = dimension of trials (default: 0)

    NFFT = X.shape[-1]
    FFT = np.fft.rfft(X, NFFT)

    aug_X = X
    
    for aug_idx in range(Aug_Multiplier):
        phi = 2*mat.pi*np.random.rand(*FFT.shape)
        
        Total_surro_FFT = abs(FFT)*np.cos(phi) + abs(FFT)*np.sin(phi)*1j
        
        surro_FFT = np.complex128(np.zeros(FFT.shape))
        surro_FFT[..., 0] = FFT[..., 0]        
        surro_FFT[..., 1:Total_surro_FFT.shape[-1]-1] = Total_surro_FFT[..., 1:Total_surro_FFT.shape[-1]-1]
        surro_FFT[..., -1] = FFT[..., -1]
    
        surro_X = np.fft.irfft(surro_FFT, NFFT)
        
        aug_X = np.concatenate([aug_X, surro_X], axis=trial_dim)
    
    label = (np.ones(aug_X.shape[trial_dim])*Y[0]).astype(np.int64)
    
    return aug_X, label    
    


    
   