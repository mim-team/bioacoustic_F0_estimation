# Cross-species F0 estimation, dataset and study of baseline algorithms

For a detailled descriptions of the study corresponding to this repository, see the journal publication: (upcoming).

If you use this repository, please cite: (upcoming).

## Use a crepe model pretrained on animal signals to analyse your own vocalisations
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
usage: predict.py [-h] [--model_path MODEL_PATH] [--compress COMPRESS] [--step STEP] [--decoder {argmax,weighted_argmax,viterbi}] [--print PRINT] [--threshold THRESHOLD] [--NFFT NFFT] input

positional arguments:
  input                 Directory with sound files to process, or a single file to process

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path of model weights
  --compress COMPRESS   Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. Frequencies are divided by the given factor by artificially changing the sampling rate (slowing down / speeding up the signal).
  --step STEP           Step used between each prediction (in seconds)
  --decoder {argmax,weighted_argmax,viterbi}
                        Decoder used to postprocess predictions
  --print PRINT         Print spectrograms with overlaid F0 predictions to assess their quality
  --threshold THRESHOLD
                        Confidence threshold used when printing F0 predictions on spectrograms
  --NFFT NFFT           Window size used for the spectrogram computation (only used for printing F0 predictions)
```

## Reproducing paper experiments
Go to the `paper_experiments` folder


## Reproduce paper experiments
Go to the `paper_experiments` folder
