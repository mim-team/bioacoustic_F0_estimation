import pandas as pd, numpy as np
import mir_eval.melody
from metadata import species
import matplotlib.pyplot as plt
from glob import glob
from p_tqdm import p_umap

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

cent_thr = 50
harm_thr = 0.5

sample = lambda x, N: pd.Series(x).sample(min(len(x), N))

file_list = np.concatenate([sample(glob(species[specie]['wavpath'][:-4]+'_preds.csv'), 1000) for specie in species])
algos = ['praat', 'pyin', 'basic', 'pesto', 'pesto_ft', 'tcrepe_ftoth', 'tcrepe_ftsp']
algo_names = ['praat', 'pyin', 'basic', 'pesto-music', 'pesto-bio', 'crepe-other', 'crepe-target']

def fun(fn):
    df = pd.read_csv(fn)
    assert 'salience' in df.columns, f"missing salience value for {fn}, run python vocalisation_characterisation.py"
    df.annot = mir_eval.melody.hz2cents(df.annot)
    if df.salience.mean() < 0.15:
        return pd.DataFrame()
    out = pd.DataFrame(columns=['Pitch acc', 'Chroma acc', 'salience', 'harmonicity'])
    for algo in algos:
        if not algo+'_f0' in df.columns or df[algo+'_f0'].isna().all():
            continue
        # out.loc[algo, ['Recall', 'False alarm']] = mir_eval.melody.voicing_measures(df.annot > 0, df[algo+'_conf'] > thrs[algo])
        df[algo+'_f0'] = mir_eval.melody.hz2cents(df[algo+'_f0'])
        df[algo+'_conf'].clip(0, 1, inplace=True)
        pitch_acc = mir_eval.melody.raw_pitch_accuracy(df.annot>0, df.annot, df[algo+'_conf'], df[algo+'_f0'], cent_tolerance=50)
        if pitch_acc < 0.05:
            continue
        out.loc[algo, 'Pitch acc'] = pitch_acc
        out.loc[algo, 'Chroma acc'] = mir_eval.melody.raw_chroma_accuracy(df.annot>0, df.annot, df[algo+'_conf'], df[algo+'_f0'], cent_tolerance=50)
    out['salience'] = df.salience.mean()
    out['harmonicity'] = df.harmonicity.mean() if 'harmonicity' in df.columns else 0
    return out

df = pd.concat(p_umap(fun, file_list))
df.salience = df.salience.round(1)
# df.harmonicity = df.harmonicity > 0.5
df.reset_index(names='algo', inplace=True)
harmonic = df[df.harmonicity > harm_thr].groupby(['algo','salience'])[['Pitch acc', 'Chroma acc']].agg(['mean', 'std'])
nonharmonic = df[df.harmonicity < harm_thr].groupby(['algo','salience'])[['Pitch acc', 'Chroma acc']].agg(['mean', 'std'])

fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(10, 3.5))
for algo, name in zip(algos, algo_names):
    ax[0].plot(harmonic.loc[algo].index, harmonic.loc[algo, 'Pitch acc']['mean'])
    # ax[0].fill_between(harmonic.loc[algo].index, harmonic.loc[algo, 'Pitch acc']['mean']+harmonic.loc[algo, 'Pitch acc']['std'], harmonic.loc[algo, 'Pitch acc']['mean']-harmonic.loc[algo, 'Pitch acc']['std'], alpha=.5)
    ax[1].plot(nonharmonic.loc[algo].index, nonharmonic.loc[algo, 'Pitch acc']['mean'], label=name)
    # ax[1].fill_between(nonharmonic.loc[algo].index, nonharmonic.loc[algo, 'Pitch acc']['mean']+nonharmonic.loc[algo, 'Pitch acc']['std'], nonharmonic.loc[algo, 'Pitch acc']['mean']-nonharmonic.loc[algo, 'Pitch acc']['std'], alpha=.5)

for i in range(2):
    ax[i].set_xlabel('salience')
    ax[i].grid()
    ax[i].set_title(('H' if i==0 else 'Non-h')+'armonic vocalisations')

ax[0].set_ylabel('mean pitch acc')
ax[0].set_ylim(0, 1)
plt.tight_layout(rect=(0, 0, .87, 1))
plt.legend(bbox_to_anchor=(1,1))
plt.savefig('figures/acc_vs_salience.pdf')
