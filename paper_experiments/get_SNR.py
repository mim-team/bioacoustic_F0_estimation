import pandas as pd, numpy as np
from scipy import signal
from p_tqdm import p_umap
from metadata import species
from glob import glob
import librosa

def fun(fn):
    df = pd.read_csv(fn[:-4]+'_preds.csv')
    sig, fs = librosa.load(fn)

    if df[df.annot > 0].annot.min() >= fs / 2:
        return fn, None

    sos = signal.butter(3, df[df.annot>0].annot.min() * 2 / fs, 'highpass', output='sos')
    sig = signal.sosfiltfilt(sos, sig)

    start = df[df.annot > 0].time.min()
    end = df[df.annot > 0].time.max()
    S = np.std(sig[int(start*fs):int(end*fs)])

    if end - start < len(sig)/fs/2: # if the voiced section is smaller than half the signal duration, we estimate the noise over the same duration only
        N = np.std(np.concatenate([ sig[ int((start-(end-start)/2)*fs) : int(start*fs) ], sig[ int(end*fs) : int((end + (end - start)/2)*fs)] ]))
    else:
        N = np.std(np.concatenate([sig[:int(start*fs)], sig[int(end*fs):]]))

    if S < N:
        return fn, None
    else:
        return fn, 10 * np.log10(S/N -1)

for specie in species:
    ret = p_umap(fun, glob(species[specie]['wavpath']), desc=specie)
    fns, SNRs = zip(*ret)
    df = pd.DataFrame({'fn':fns, 'SNR':SNRs})
    print(df.SNR.describe())
    df.to_csv(f'SNRs/{specie}.csv')
