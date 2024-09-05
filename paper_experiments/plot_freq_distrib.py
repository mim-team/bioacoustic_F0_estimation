from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os, mir_eval
from metadata import species
np.seterr(divide = 'ignore')

species_list = [
    'canids', 'spotted_hyenas', # 2 good salience & harmonicity
    'bottlenose_dolphins', 'rodents', 'little_owls', # 3 good salience only
    'monk_parakeets', 'lions', 'orangutans', 'long-billed_hermits', # 4 good harmonicity only
    'hummingbirds', 'disk-winged_bats', 'Reunion_grey_white_eyes', 'dolphins', 'La_Palma_chaffinches'] # 5 neither

taxas = ['M', 'M', 'M', 'M', 'A', 'A', 'M', 'M', 'A', 'A', 'M', 'A', 'M', 'A']

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 3.5))
ax[0].set_xlabel('Frequency (Hz)')
ax[1].set_xlabel('Duration (sec)')
ax[2].set_xlabel('Modulation rate (octave/sec)')
for i in range(3):
    ax[i].set_xscale('log' if i < 2 else 'symlog')
    ax[i].grid()

mod_rate = lambda x: np.log2(x[1:] / x[:-1])

for i, (specie, tax) in enumerate(zip(species_list, taxas)):
    wavpath, FS, nfft, downsample, step = species[specie].values()
    dt = nfft * step / FS
    fdistrib, tdistrib, moddistrib = [], [], []
    files = pd.Series(glob(wavpath))
    for fn in tqdm(files, desc=specie):
        annot = pd.read_csv(f'{fn[:-4]}.csv').drop_duplicates(subset='Time').fillna(0)
        f0s, mask2 = mir_eval.melody.resample_melody_series(annot.Time, annot.Freq, annot.Freq > 0,\
            np.arange(annot.Time.min()+1e-5, annot.Time.max(), dt), kind='linear', verbose=False)
        fdistrib.extend(f0s[mask2.astype(bool)])
        tdistrib.append(mask2.sum() * dt)
        moddistrib.extend(mod_rate(f0s[mask2.astype(bool)])/dt)
        #moddistrib.extend(abs(np.diff(f0s[mask2.astype(bool)]))/dt)

    for j, data in enumerate([fdistrib, tdistrib, moddistrib]):
        p = ax[j].violinplot(data, points=500, positions=[-i], vert=False) #, quantiles=[0.25, 0.5, 0.75])
        p['bodies'][-1].set_facecolor('C0' if tax == 'M' else 'C1')
        p['bodies'][-1].set_alpha(1)
        p['cbars'].set_color('C0' if tax == 'M' else 'C1')
        p['cmaxes'].set_color('C0' if tax == 'M' else 'C1')
        p['cmins'].set_color('C0' if tax == 'M' else 'C1')
        ax[j].vlines(np.quantile(data, [.25, .5, .75]), -i-.3, -i+.3, color='dimgrey')

ax[0].set_xticks(10**np.arange(1, 6))
ax[2].set_xticks([-1e4, -1e2, -1, 1, 1e2, 1e4])

plt.yticks(-np.arange(len(species_list)), [s.replace('_',' ') for s in species_list])
plt.tight_layout(rect=(0.02, 0, 1, 1))

ax2 = plt.axes([0,0,1,1], facecolor=(1,1,1,0))
plt.axis('off')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
for i, (specie, xoffset) in enumerate(zip(species_list, [.075, 0.015, -.01, .079, .061, 0.02, .09, .05, -.01, 0.01, 0.005, -.04, .07, -.02])):
    icon = parse_path(svg2paths(f'svg/{specie}.svg')[1][0]['d'])
    plt.scatter(0.05+xoffset, 0.89-i*.74/len(species_list), marker=icon, color='none', edgecolors='black', s=1000)

plt.savefig('figures/freq_distrib.pdf')
