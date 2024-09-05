import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import librosa, mir_eval
from p_tqdm import p_umap
from glob import glob
import os, argparse
from metadata import species
np.seterr(divide = 'ignore')

algos = ['pesto_ft']
parser = argparse.ArgumentParser()
parser.add_argument('specie', type=str, help="Species to treat specifically", default=None)
args = parser.parse_args()

for specie in species if args.specie is None else [args.specie]:
    wavpath, FS, nfft, downsample, step = species[specie].values()
    thrs = pd.read_csv(f'scores/{specie}_scores.csv', index_col=0).threshold
    dt = nfft * step / FS # winsize / 8
    # for fn in glob(wavpath):
    def fun(fn):
        # if os.path.isfile(f'annot_pngs/{fn[:-4]}.png'):
        #     return
        if not os.path.isdir(f'pred_pngs/{fn.rsplit("/",1)[0]}'):
            os.mkdir(f'pred_pngs/{fn.rsplit("/",1)[0]}')
        # load signal and compute spetrogram
        sig, fs = librosa.load(fn, sr=FS)
        df = pd.read_csv(f'{fn[:-4]}_preds.csv')

        S, freqs, times, ax = plt.specgram(sig, Fs=FS, NFFT=nfft, noverlap=int(nfft-dt*fs))
        S = 10*np.log10(S+1e-10)
        plt.scatter(df.dropna(subset='annot').time, df.dropna(subset='annot').annot, c='k', alpha=.2, label='annot')
        for algo in algos:
            if not df[algo+'_f0'].isna().all():
                plt.scatter(df[df[algo+'_conf']>thrs[algo]].time, df[df[algo+'_conf']>thrs[algo]][algo+'_f0'], label=algo, alpha=.2)
        plt.ylim(0, df.annot.max()*1.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'pred_pngs/{fn[:-4]}.png')
        plt.close()

    p_umap(fun, glob(wavpath), desc=specie, num_cpus=40)
