import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import librosa, mir_eval
from p_tqdm import p_umap
from glob import glob
import os, argparse
from metadata import species
np.seterr(divide = 'ignore')

parser = argparse.ArgumentParser()
parser.add_argument('specie', type=str, help="Species to treat specifically", default=None)
args = parser.parse_args()

for specie in species if args.specie is None else [args.specie]:
    wavpath, FS, nfft, downsample, step = species[specie].values()
    dt = nfft * step / FS # winsize / 8
    # for fn in glob(wavpath):
    def fun(fn):
        # if os.path.isfile(f'annot_pngs/{fn[:-4]}.png'):
        #     return
        if not os.path.isdir(f'annot_pngs/{fn[5:].rsplit("/",1)[0]}'):
            os.makedirs(f'annot_pngs/{fn[5:].rsplit("/",1)[0]}', exist_ok=True)
        # load signal and compute spetrogram
        sig, fs = librosa.load(fn, sr=FS)
#        df = pd.read_csv(f'{fn[:-4]}_preds.csv').dropna(subset='annot')
        df = pd.read_csv(f'{fn[:-4]}.csv').rename(columns={'Time':'time', 'Freq':'annot'})

        plt.specgram(sig, Fs=FS, NFFT=nfft, noverlap=int(nfft-dt*fs))

        # plot
        if 'salience' in df.columns:
            plt.scatter(df.time, df.annot, c=df.salience, s=1 if specie in {'dclde','mice'} else None, cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Salience {df.salience.quantile(.25):.2f}, SHR {df.SHR.quantile(.25):.2f}, Harmonicity {df.harmonicity.quantile(.25):.2f}')
        else:
            plt.scatter(df.time, df.annot, alpha=.2, s=1 if specie in {'dclde','mice'} else None)
        plt.ylim(0, df.annot.max()*1.5)
        plt.tight_layout()
        plt.savefig(f'annot_pngs/{fn[5:-4]}.png')
        plt.close()

    files = pd.Series(glob(wavpath))
    p_umap(fun, files, desc=specie)
