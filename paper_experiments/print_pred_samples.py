import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex as to_rgba
from matplotlib.lines import Line2D
import librosa
from scipy import signal
from glob import glob
from metadata import species

algos = ['praat', 'pesto_ft', 'tcrepe_ftsp']
algo_names = ['PRAAT', 'pesto-bio', 'crepe-consp.']
colors = ['tab:orange', 'tab:red', 'tab:purple']

species_list = [
    'canids', 'spotted_hyenas', # 2 good salience & harmonicity
    'bottlenose_dolphins', 'rodents', 'little_owls', # 3 good salience only
    'hummingbirds', 'disk-winged_bats', 'Reunion_grey_white_eyes', 'monk_parakeets', 'lions', 'orangutans', 'long-billed_hermits', # 7 good harmonicity only
    'dolphins', 'La_Palma_chaffinches'] # 2 neither

names = [
    'canids', 'spotted\nhyenas', # 2 good salience & harmonicity
    'bottlenose\ndolphins', 'rodents', 'little owls', # 3 good salience only
    'hummingbirds', 'disk-winged\nbats', 'Reuniongrey\nwhite eyes', 'monk\nparakeets', 'lions', 'orangutans', 'long-billed\nhermits', # 7 good harmonicity only
    'dolphins', 'La Palma\nchaffinches'] # 2 neither


fig, ax = plt.subplots(nrows=len(species_list), ncols=5, figsize=(12, 15), sharey='row')

for i, (specie, name) in enumerate(zip(species_list, names)):
    wavpath, FS, nfft, downsample, step = species[specie].values()
    thrs = pd.read_csv(f'scores/{specie}_scores.csv', index_col=0).threshold
    files = pd.Series(glob(wavpath)).sample(5)
    dt = nfft * step / FS # winsize / 8
    fmax = 0
    for j, fn in enumerate(files):
        # load signal and compute spetrogram
        sig, fs = librosa.load(fn, sr=FS)
        df = pd.read_csv(f'{fn[:-4]}_preds.csv')
        df = df[df.annot>0]

        freqs, times, S = signal.spectrogram(sig, fs=FS, nperseg=nfft, noverlap=int(nfft-dt*fs))
        S = 10*np.log10(S+1e-10)
        S -= np.median(S, axis=1, keepdims=True)
        plt.autoscale(False)
        ax[i, j].imshow(S, vmin=np.quantile(S, .2), vmax=np.quantile(S, .98), origin='lower', aspect='auto', extent=[0, len(sig)/fs, 0, fs/2/1000], cmap='Greys')
        plt.autoscale(True)
        ax[i, j].scatter(df.dropna(subset='annot').time, df.dropna(subset='annot').annot/1000, label='annot', color='tab:blue')
        # ax[i, j].plot(df.dropna(subset='annot').time, df.dropna(subset='annot').annot/1000, linestyle='none', marker='o', color='blue', markersize=10, fillstyle='none')
        for algo, algo_name, color in zip(algos, algo_names, colors):
            ax[i, j].scatter(df[df[algo+'_conf']>thrs[algo]].time, df[df[algo+'_conf']>thrs[algo]][algo+'_f0']/1000, label=algo_name, s=3, color=color)
        ax[i,j].set_xticks(np.arange(0, len(sig)/fs, 0.1), "")
        if j == 0:
            ax[i,j].set_ylabel(name)
        fmax = max(fmax, df.annot.max()*2/1000)
    ax[i, 0].set_ylim(0, fmax)


plt.tight_layout(rect=(0, 0, 0.95, .975), pad=0.1, w_pad=0.11, h_pad=-0.05)
leg = ax[0,2].legend(loc='lower center', ncols=4, bbox_to_anchor=(0.5, 1.05))
leg.legend_handles[1]._sizes = leg.legend_handles[0]._sizes
leg.legend_handles[2]._sizes = leg.legend_handles[0]._sizes
leg.legend_handles[3]._sizes = leg.legend_handles[0]._sizes
ax[0,2].add_artist(leg)

ax2 = plt.axes([0,0,1,1], facecolor=(1,1,1,0))
plt.axis('off')

ax2.text(.965, .86, 'Salient and Harmonic', rotation=270)
line = Line2D([.96, .96], [.965, .845], lw=3., color='k', alpha=0.4)
ax2.add_line(line)

ax2.text(.965, .67, 'Salient and non-harmonic', rotation=270)
line = Line2D([.96, .96], [.825, .637], lw=3., color='k', alpha=0.4)
ax2.add_line(line)

ax2.text(.965, .3, 'Non-salient and Harmonic', rotation=270)
line = Line2D([.96, .96], [.62, .145], lw=3., color='k', alpha=0.4)
ax2.add_line(line)

ax2.text(.965, .001, 'Non-salient and Non-harmonic', rotation=270)
line = Line2D([.96, .96], [.13, .01], lw=3., color='k', alpha=0.4)
ax2.add_line(line)


plt.savefig(f'figures/sample_spectrograms.pdf')
