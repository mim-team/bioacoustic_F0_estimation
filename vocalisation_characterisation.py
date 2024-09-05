from glob import glob
from p_tqdm import p_umap
import pandas as pd, numpy as np, argparse
from metadata import species
import librosa, os
np.seterr(divide = 'ignore')

parser = argparse.ArgumentParser()
parser.add_argument('specie', type=str, help="Species to treat specifically", default='all')
args = parser.parse_args()

for specie in species if args.specie == 'all' else [args.specie]:
    wavpath, FS, nfft, downsample, step = species[specie].values()
    Hz2bin = lambda f: np.floor(f / FS * nfft).astype(int)
    dt = nfft *step
    hann = np.hanning(nfft)
    get_spectrum = lambda x : np.abs(np.fft.rfft(hann * x))

    def fun(fn):
#    for fn in glob(wavpath):
        sig, fs = librosa.load(fn, sr=FS)
        df = pd.read_csv(f'{fn[:-4]}_preds.csv')
        df.SNR, df.SHR, df.salience, df.SNR_, df.SHR_, df.salience_ = None, None, None, None, None, None
        shr_ceil = min(fs/2, df.annot.max() * 5)
        # compute median background noise for unvoiced frames
        unvoiced = df.annot.isna()
        if not unvoiced.any(): # if there aren't any unvoiced frames, we take the whole signal to estimate the noise
            unvoiced = [True]*len(df)
        spectrums = np.vstack([get_spectrum(sig[t - nfft//2 : t + nfft//2]) for t in (df[unvoiced].time * fs).round().astype(int)]).T
        mednoise, stdnoise = np.median(spectrums, axis=1), np.std(spectrums, axis=1)
        # computer saliency and SHR for each voiced frame
        for i, r in df[df.annot>0].iterrows():
            if FS*r.time < nfft//2 or FS*r.time > len(sig) - nfft//2 or (sig[int(FS*r.time) - nfft//2 : int(FS*r.time) + nfft//2] == 0).all():
                continue
            spec = get_spectrum(sig[int(FS*r.time) - nfft//2 : int(FS*r.time) + nfft//2])
            spec = np.clip((spec-mednoise)/stdnoise, 1e-12, 1e3)
            f0 = r.annot

            df.loc[i, 'harmonicity'] = sum(spec[Hz2bin(np.arange(f0*2, shr_ceil, f0))]) /  sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 *2 < fs / 2 else None
            df.loc[i, 'salience'] = sum(spec[Hz2bin(f0*2**(-1/12)):Hz2bin(f0*2**(1/12))+1]) / sum(spec[Hz2bin(f0*2**(-6/12)):Hz2bin(f0*2**(6/12))+1])
            df.loc[i, 'SHR'] = sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0)-f0/2)]) / sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 < fs/2 else None

        df.to_csv(f'{fn[:-4]}_preds.csv', index=False)

    p_umap(fun, glob(wavpath), desc=specie)
