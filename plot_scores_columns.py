import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import pandas as pd, numpy as np
# from svg_pltmarker import get_marker_from_svg

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
# colors = list(colors.TABLEAU_COLORS)
# plt.rcParams['legend.title_fontsize'] = 'x-small'

markers = [
    ('o', 'right', colors[0]), ('d', 'none', colors[0]),
    ('s', 'none', colors[1]), ('v', 'none', colors[1]), ('H', 'none', colors[1]),
    ('<', 'none', colors[2]), ('>', 'none', colors[2]), ('^', 'none', colors[2]), ('p', 'none', colors[2]),
    ('P', 'none', colors[3]), ('*', 'none', colors[3]), ('X', 'none', colors[3]), ('D', 'none', colors[3]) , ('h', 'left', colors[3])]


species_list = [
    'canids', 'spotted_hyenas', # 2 good salience & harmonicity
    'bottlenose_dolphins', 'rodents', 'little_owls', # 3 good salience only
    'monk_parakeets', 'lions', 'orangutans', 'long-billed_hermits', # 4 good harmonicity only
    'hummingbirds', 'disk-winged_bats', 'Reunion_grey_white_eyes', 'dolphins', 'La_Palma_chaffinches'] # 5 neither

algos = ['basic', 'pyin', 'pesto', 'praat', 'pesto_ft', 'tcrepe_ftoth', 'tcrepe_ftsp'] #, 'tcrepe_ftspV']
algo_names = ['basic', 'pyin', 'pesto-music', 'praat', 'pesto-bio', 'crepe-heterosp.', 'crepe-consp.'] #, 'crepe-target\nviterbi']
#metrics = ['Pitch acc', 'Chroma acc']
metrics = ['Specificity', 'Recall', 'Vocalisation recall']

fig, ax = plt.subplots(nrows=len(metrics), ncols=1, figsize=(9, 6), sharex=True, sharey=True)
ax[0].set_ylim(0, 1.05)
ax[0].set_yticks(np.arange(0, 1.1, .2))
algo_legend = []

for i, metric in enumerate(metrics):
    m_ax = ax[i] #ax[int(i//2), i%2]
    m_ax.grid('both', axis='y')
    m_ax.set_ylabel(metric.replace('acc', 'accuracy'), fontsize='medium')

    ok = pd.DataFrame()
    for j, (specie, marker) in enumerate(zip(species_list, markers)):
        df = pd.read_csv(f'scores/{specie}_scores.csv', index_col=0)
        df['Specificity'] = 1 - df['False alarm']
        df.rename(columns={'Voc. recall':'Vocalisation recall'}, inplace=True)
        ok.loc[specie, df.index] = df[metric]

        # icon = get_marker_from_svg(filepath=f'svg/{specie}.svg')
        # icon.vertices *= -1
        tt = m_ax.plot(np.arange(len(algos))-.3+j*0.6/len(species_list), ok.loc[specie, algos],\
            marker=marker[0], color=marker[2], markersize=7, fillstyle=marker[1], label=specie.replace('_',' '), linestyle='none')
        algo_legend.append(tt[0])

    m_ax.set_xticks(range(len(algo_names)))
    m_ax.set_xticklabels(algo_names)
#ax[1].set_xticklabels(algos, rotation=22)
plt.tight_layout(rect=(0, 0, 1, .86), w_pad=0.01)

species_list = [s.replace('_', ' ') for s in species_list]
m_ax = ax[0]
leg = m_ax.legend(algo_legend[:2], species_list[:2], title='salient & harmonic', loc='lower left', bbox_to_anchor=(-0.08, 1.04), fontsize='x-small', handleheight=3.15)
m_ax.add_artist(leg)
leg = m_ax.legend(algo_legend[2:5], species_list[2:5], title='salient & non-harmonic', loc='lower left', bbox_to_anchor=(0.1, 1.04), fontsize='x-small', handleheight=1.7)
m_ax.add_artist(leg)
leg = m_ax.legend(algo_legend[5:9], species_list[5:9], ncols=2, title='non-salient & harmonic', loc='lower left', bbox_to_anchor=(.32, 1.04), fontsize='x-small', handleheight=3.15)
m_ax.add_artist(leg)
leg = m_ax.legend(algo_legend[9:], species_list[9:], ncols=2, title='non-salient & non-harmonic', loc='lower left', bbox_to_anchor=(.635, 1.04), fontsize='x-small', handleheight=1.7)
m_ax.add_artist(leg)
# ax.legend(loc='lower left', ncols=7, bbox_to_anchor=(0, 1.1), fontsize='x-small')

plt.savefig(f'figures/scatter_scores_detec.pdf')
