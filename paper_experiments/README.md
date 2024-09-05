## Reproduce paper experiments

Python package requirements necessary to run the paper experiments are detailled in the `requirements.txt` file.

### metadata.py
Stores a dictionary of datasets and characteristics (SR, NFFT, path to access soundfiles, and downsampling factor for ultra/infra-sonic signals)
Convenient to iterate over the whole dataset
```
from metadata import species
for specie in species:
    wavpath, FS, nfft, downsample, step = species[specie].values()
    # iterate over files (one per vocalisation)
    for fn in tqdm(glob(wavpath), desc=specie):
        sig, fs = sf.read(fn) # read soundfile
        annot = pd.read_csv(f'{fn[:-4]}.csv') # read annotations (one column Time in seconds, one column Freq in Herz)
        preds = pd.read_csv(f'{fn[:-4]}_preds.csv') # read the file gathering per algorithm f0 predictions
```

### print_annot.py
For each vocalisation, prints a spectrogram and overlaid annotations as .png file stored in the `annot_pngs` folder.

### run_all.py
Runs all baseline algorithms over the dataset.
- [x] praat (praat-parselmouth implem)
- [x] pyin (librosa implem)
- [x] crepe (torchcrepe implem)
- [x] crepe finetuned (torchcrepe implem)
- [x] crepe finetuned over all species except the target
- [x] crepe finetuned only on the target species
- [x] crepe (original tensorflow implem https://arxiv.org/abs/1802.06182)
- [x] basic pitch (https://arxiv.org/abs/2203.09893)
- [x] pesto (https://arxiv.org/abs/2309.02265)
- [x] pesto finetuned over the target species

This scripts stores predictions along with resampled annotations in `{basename}_preds.csv` files

### print_annot.py
For each vocalisation, prints a spectrogram and overlaid annotations as .png file stored in the `annot_pngs` folder. Similarly, `print_preds.py` prints spectrograms for a given species but also includes predictions from a chosen algorithm.

### eval_all.py
Evaluates each algorithms over the dataset using `{basename}_preds.csv` files, with a threshold of 50 cents for accuracies.
For each algorithms and species, this outputs ROC optimal thresholds, Recall, False alarm, Pitch accuracy, and Chroma accuracy.
/!\ These metrics are mesured per vocalisation before being averaged.
Scores are stored in `scores/{specie}_scores.csv` files

### vocalisation_characterisation.py
Evaluates metrics for each annotated temporal bin:
- the presence of a sub-harmonic following [this paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cb8f47c23c74932152456a6f7a464fd3a2321259)
- the salience of the annotation as the ratio of the energy of the f0 (one tone around the annotation) and its surrounding (one octave around the annotation)
- the harmonicity of the annotation as the ratio between the energy of all harmonics and that of all harmonics except the fundamental 
These values are stored in the SHR, harmonicity and salience columns of the {basename}_preds.csv files

### get_SNR.py
Computes the SNR as the ratio of energy between voiced and unvoiced sections, and reports it with a table for each species containing filename / SNR value pairs.

### train_crepe.py
Fine tunes the crepe model using the whole dataset.
- [x] Loads 1024 sample windows and their corresponding f0 to be stored in a large `train_set.pkl` file (skip if data hasn't changed).
- [x] Applies gradient descent using the BCE following the crepe paper (this task is treated as a binary classification for each spectral bin).
- [x] The fine tuned model is stored in `crepe_ft/model_all.pth`
- [x] Train on one target species given as argument, with 5-fold validation (weights are stored in `crepe_ft/model_only-{k}_{specie}.pth`)
- [x] Train on all species except the target given as argument (weights are stored in `crepe_ft/model_omit_{specie}.pth`)

### training pesto models
[This repository](https://gitlab.lis-lab.fr/paul.best/pesto-full) was forked from the [original pesto training implementation](https://github.com/SonyCSLParis/pesto-full) to include species-specific configurations and to include the small modifications necessary to correctly load signals

### Plotting
Scripts allow to generate plots to visualise results (they are saved as `.pdf` files in the `figures` folder)
- `plot_freq_distrib.py` generates a three panel subplot with violins showing distributions of f0 annotations in Hz, number of voiced bins per vocalisation, and modulation rate in (Hz/sec)
- `plot_snr_distrib.py` generates a three panel subplot with violins showing distributions of salience, SHR and harmonicity (see `compute_salience_SHR.py`)
- `plot_scores_scatter.py` generates a subplot showing F0 estimation accuracy or detection performance, for each species/algorithm combination.
- `print_pred_samples.py` generates a plot with sampled spectrograms for each species to demonstrate predictions of different algorithms
- `plot_acc_vs_salience.py` generates a plot showing pitch accuracy for each algorithm as a function of salience





