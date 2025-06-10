# Bioacoustic F0 estimation

*This repository provides a Python interface based on deep learning to analyse bioacoustic signals by detecting tonal sounds and estimating their fundamental frequency.*
![test_wolf](https://github.com/user-attachments/assets/ac1d7c30-4a8e-44fc-8a3b-b57d8f46d7ce)

For a detailled descriptions of the study corresponding to this repository, see the journal publication: https://doi.org/10.1080/09524622.2025.2500380
(Author accepted manuscript full text is available [here](https://www.ricardmarxer.com/assets/f0-examples/Accepted_Manuscript_Best_Marxer_et_al_2025.pdf))

The data that was used to train F0 estimation models and to run experiments is available on a dryad repository: https://doi.org/10.5061/dryad.prr4xgxw8

If you use this repository, please cite the associated journal publication.

## How to use a pretrained model to analyse your own recordings
- Clone the repo locally
```
git clone https://github.com/lamipaul/bioacoustic_F0_estimation
```
- Navigate inside the local repository and install dependencies
```
cd bioacoustic_F0_estimation
pip install -r requirements.txt`
```
Note that these package requirements rely on CUDA being installed (for GPU computing and faster analysis). If you do not have CUDA installed, you can use freely available GPUs on google collab, or run F0 estimation locally on CPU (it will just be slower, please then use the `cpu_requirements.txt` file).

- Run the `predict.py` script to use a pretrained crepe model to estimate F0 values for your own sounds.
```
python predict.py my_sound_file.wav
```

A `.csv` file will be saved with timestamped F0 values, their associated model confidence, along with F0-based features (salience, harmonicity, and sub-harmonic ratio).



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
