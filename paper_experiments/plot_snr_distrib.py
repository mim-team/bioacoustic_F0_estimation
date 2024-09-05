from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_umap
from glob import glob
from metadata import species
np.seterr(divide = 'ignore')
plt.style.use('tableau-colorblind10')

species_list = [
    'canids', 'spotted_hyenas', # 2 good salience & harmonicity
    'bottlenose_dolphins', 'rodents', 'little_owls', # 3 good salience only
    'monk_parakeets', 'lions', 'orangutans', 'long-billed_hermits', # 4 good harmonicity only
    'hummingbirds', 'disk-winged_bats', 'Reunion_grey_white_eyes', 'dolphins', 'La_Palma_chaffinches'] # 5 neither

taxas = ['M', 'M', 'M', 'M', 'A', 'A', 'M', 'M', 'A', 'A', 'M', 'A', 'M', 'A']

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 3.5))
plt.yticks(-np.arange(len(species_list)), [s.replace('_',' ') for s in species_list])

for i, t in enumerate(['Salience', 'SHR', 'Harmonicity']):
    ax[i].set_xlabel(t)
    ax[i].set_xlim(0, 1)
    ax[i].set_axisbelow(True)
    ax[i].grid()

ax[0].hlines([-1.5, -4.5, -8.5], 0, 1, linestyle='dashed', color='grey')
ax[1].hlines([-1.5, -4.5, -8.5], 0, 100, linestyle='dashed', color='grey')
ax[2].hlines([-1.5, -4.5, -8.5], 0, 1, linestyle='dashed', color='grey')

for i, (specie, taxa) in enumerate(zip(species_list, taxas)):
    fun = lambda fn: pd.read_csv(fn)[['salience', 'SHR', 'harmonicity']].dropna().to_numpy().T
    ret = p_umap(fun, glob(species[specie]['wavpath'][:-4]+'_preds.csv'), desc=specie)
    salience, SHR, harmonicity = (np.clip(np.concatenate(r), 0, 1) for r in zip(*ret))
    SHR = SHR[salience > .2]
    harmonicity = harmonicity[salience > .2]
    for j, data in enumerate([salience, SHR, harmonicity]):
        p = ax[j].violinplot(data, points=500, positions=[-i], vert=False, showextrema=False)
        p['bodies'][-1].set_facecolor('C0' if taxa == 'M' else 'C1')
        p['bodies'][-1].set_alpha(1)
        ax[j].vlines(np.quantile(data, [.25, .5, .75]), -i-.3, -i+.3, color='dimgrey')

plt.tight_layout(rect=(0.02, 0, 1, 1))
ax2 = plt.axes([0,0,1,1], facecolor=(1,1,1,0))
plt.axis('off')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
for i, (specie, xoffset) in enumerate(zip(species_list, [.075, 0.015, -.01, .079, .061, 0.02, .09, .05, -.01, 0., 0.005, -.04, .07, -.02])):
    icon = parse_path(svg2paths(f'svg/{specie}.svg')[1][0]['d'])
    plt.scatter(0.05+xoffset, 0.89-i*.74/len(species_list), marker=icon, color='none', edgecolors='black', s=1000)
plt.savefig('figures/SNR_distrib.pdf')
