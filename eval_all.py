import pandas as pd, numpy as np
from sklearn import metrics as skmetrics
import mir_eval.melody
import os, argparse
from metadata import species
import matplotlib.pyplot as plt
from glob import glob
from p_tqdm import p_umap
from tqdm import tqdm

cent_thr = 50
metrics = ['recall', 'FA', 'pitch_acc', 'chroma_acc', 'diff_distrib']
parser = argparse.ArgumentParser()
parser.add_argument('specie', type=str, help="Species to treat specifically", default='all')
args = parser.parse_args()

for specie in species if args.specie=='all' else args.specie.split(' '):
    algos = {'pyin', 'praat', 'crepe', 'tcrepe', 'tcrepe_ftsp', 'tcrepe_ftspV', 'tcrepe_ftoth', 'basic', 'pesto', 'pesto_ft', 'pesto_ftoth'}
    # Get optimal thresholds
    confs = {k:[] for k in algos}
    confs['label'] = []
    for fn in tqdm(glob(species[specie]['wavpath'][:-4]+'_preds.csv'), desc=f'{specie} get thrs', leave=False):
        df = pd.read_csv(fn)
        for algo in algos:
            if algo+'_conf' in df.columns:
                confs[algo].extend(df[algo+'_conf'])
            else:
                confs[algo].extend([np.nan]*len(df))
        confs['label'].extend(df.annot>0)
    thrs = {}
    for algo in list(algos):
        if np.isnan(confs[algo]).all():
            algos -= {algo}
            continue
        fpr, tpr, thr = skmetrics.roc_curve(np.array(confs['label'])[~np.isnan(confs[algo])], np.array(confs[algo])[~np.isnan(confs[algo])])
        thrs[algo] = thr[np.argmin(abs(tpr + fpr - 1))]

    # Compute recall, false alarm, pitch acc and chroma acc
    def fun(fn):
        df = pd.read_csv(fn)
        df.annot = mir_eval.melody.hz2cents(df.annot)
        out = pd.DataFrame(columns=metrics)
        for algo in algos:
            if not algo+'_f0' in df.columns or df[algo+'_f0'].isna().all():
                continue
            out.loc[algo, ['Recall', 'False alarm']] = mir_eval.melody.voicing_measures(df.annot > 0, df[algo+'_conf'] > thrs[algo])
            df[algo+'_f0'] = mir_eval.melody.hz2cents(df[algo+'_f0'])
            df[algo+'_conf'].clip(0, 1, inplace=True)
            out.loc[algo, 'Pitch acc'] = mir_eval.melody.raw_pitch_accuracy(df.annot>0, df.annot, df[algo+'_conf'], df[algo+'_f0'], cent_tolerance=cent_thr)
            out.loc[algo, 'Chroma acc'] = mir_eval.melody.raw_chroma_accuracy(df.annot>0, df.annot, df[algo+'_conf'], df[algo+'_f0'], cent_tolerance=cent_thr)
            out.at[algo, 'diff_distrib'] = list(abs(df[algo+'_f0'] - df.annot))
            out.loc[algo, 'Voc. recall'] = ((df.annot > 0 ) & ( df[algo+'_conf'] > thrs[algo])).sum() > 0.33 * (df.annot > 0).sum()
        return out

    df = pd.concat(p_umap(fun, glob(species[specie]['wavpath'][:-4]+'_preds.csv'), desc=f'{specie} get perf'))

    fig, ax = plt.subplots(ncols=3, figsize=(12, 4), sharex=True)
    for i, algo in enumerate(algos):
        if algo in df.index:
            ax[0].violinplot(np.concatenate(df.loc[algo, 'diff_distrib']), positions=[i])
            ax[1].violinplot(df.loc[algo, 'Pitch acc'], positions=[i])
            ax[2].violinplot(df.loc[algo, 'Chroma acc'], positions=[i])

#    ax[0].set_yscale('log')
    ax[0].set_title('Distrib of errors in cents')
    ax[0].hlines(1200, 0, len(algos), linestyle='dashed', color='k')
    ax[1].set_title('Distrib of pitch acc per vocs in % ')
    ax[2].set_title('Distrib of chroma acc per vocs in % ')
    plt.xticks(np.arange(len(algos)), algos, rotation=45)
    plt.tight_layout()
    plt.savefig(f'scores/{specie}_report.pdf')
    plt.close()

    df['Voc. recall'] = df['Voc. recall'].astype(int)
    df = df.reset_index(names='algo').groupby('algo').agg({'algo':'count', 'Recall':'mean', 'False alarm':'mean', 'Pitch acc':'mean', 'Chroma acc':'mean', 'Voc. recall':'mean'})
    df.loc[thrs.keys(), 'threshold'] = list(thrs.values())
    df.rename(columns={'algo':'count'}, inplace=True)
    print(df)
    df.to_csv(f'scores/{specie}_scores.csv')
