# Cross-species F0 estimation, dataset and study of baseline algorithms

For a detailled descriptions of the study corresponding to this repository, see the journal publication: (upcoming).

If you use this repository, please cite: (upcoming).

## Use a crepe model pretrained on animal signals to analyse your own signals
Install the packages necessary to run a crepe model using `pip install -r predict_requirements.txt`

Use the `predict.py` script to run a pretrained crepe model to estimate fundamental frequency values for your own sounds.
```
usage: predict.py [-h] [--model_path MODEL_PATH] [--compress COMPRESS] [--step STEP] [--decoder {argmax,weighted_argmax,viterbi}] [--print PRINT] indir

positional arguments:
  indir                 Directory with sound files to process

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path of model weights
  --compress COMPRESS   Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. Frequencies are divided by the given factor by artificially changing the sampling rate (slowing down / speeding up the signal).
  --step STEP           Step used between each prediction (in seconds)
  --decoder {argmax,weighted_argmax,viterbi}
                        Decoder used to postprocess predictions
  --print PRINT         Print spectrograms with overlaid F0 predictions to assess their quality
```
