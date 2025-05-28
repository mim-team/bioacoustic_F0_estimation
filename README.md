# Bioacoustic F0 estimation

*This repository provides a Python interface based on deep learning to analyse bioacoustic signals by detecting tonal sounds and estimating their fundamental frequency.*

For a detailled descriptions of the study corresponding to this repository, see the journal publication: https://doi.org/10.1080/09524622.2025.2500380

The data that was used to train F0 estimation models and to run experiments is available on a dryad repository: https://doi.org/10.5061/dryad.prr4xgxw8

If you use this repository, please cite the associated journal publication.

## Use a model pretrained on vocalisations of 14 different taxa to analyse your own recordings
- Clone the repo locally
```
git clone https://github.com/lamipaul/bioacoustic_F0_estimation
```
- Navigate inside the local repository and install dependencies
```
cd bioacoustic_F0_estimation
pip install -r requirements.txt`
```
- Use the `predict.py` script to run a pretrained crepe model to estimate fundamental frequency values for your own sounds.
```
python predict.py my_sound_file.wav
```
A `.csv` file will be saved with timestamped F0 values and their associated model confidence.

Several options can also be specified when using this script:
```
usage: predict.py [-h] [--model_path MODEL_PATH] [--compress COMPRESS] [--step STEP] [--decoder {argmax,weighted_argmax,viterbi}] [--no_print] [--no_characterisation]
                  [--threshold THRESHOLD] [--NFFT NFFT]
                  input

positional arguments:
  input                 Directory with sound files to process, or a single file to process

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path of model weights
  --compress COMPRESS   Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. Frequencies are divided by the given factor by artificially changing the
                        sampling rate (slowing down / speeding up the signal).
  --step STEP           Step used between each prediction (in seconds)
  --decoder {argmax,weighted_argmax,viterbi}
                        Decoder used to postprocess predictions
  --no_print            Skip printing spectrograms with overlaid F0 predictions to assess their quality
  --no_characterisation
                        Skip the computation of vocalisation characteristics (harmonicity, salience, and SHR)
  --threshold THRESHOLD
                        Confidence threshold used when printing F0 predictions on spectrograms
  --NFFT NFFT           Window size used for the spectrum computation (for printing F0 predictions and computing vocalisation characteristics)
```

## Reproducing paper experiments
Go to the `paper_experiments` folder
