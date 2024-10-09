import matplotlib.pyplot as plt
import argparse, os, tqdm
import torchcrepe, torch, librosa, soundfile
import pandas as pd, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help="Directory with sound files to process, or a single file to process")
parser.add_argument('--model_path', type=str, help="Path of model weights", default='model_all.pth')
parser.add_argument('--compress', type=float, help="Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. \
    Frequencies are divided by the given factor by artificially changing the sampling rate (slowing down / speeding up the signal).", default=1)
parser.add_argument('--step', type=float, help="Step used between each prediction (in seconds)", default=256 / torchcrepe.SAMPLE_RATE)
parser.add_argument('--decoder', choices=['argmax', 'weighted_argmax', 'viterbi'], help="Decoder used to postprocess predictions", default='weighted_argmax')
parser.add_argument('--print', type=bool, help="Print spectrograms with overlaid F0 predictions to assess their quality", default=True)
parser.add_argument('--threshold', type=float, help="Confidence threshold used when printing F0 predictions on spectrograms ", default=0.1)
parser.add_argument('--NFFT', type=int, help="Window size used for the spectrogram computation (only used for printing F0 predictions)", default=1024)
args = parser.parse_args()

# Initialisations
device, batch_size = ('cuda', 64) if torch.cuda.is_available() else ('cpu', 1)
model = torchcrepe.Crepe('full').eval().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
decoder = torchcrepe.decode.__dict__[args.decoder]

if os.path.isdir(args.input):
    files = [os.path.join(args.input, fn) for fn in os.listdir(args.input) if fn.split('.')[-1].upper() in soundfile._formats]
elif os.path.isfile(args.input) and args.input.split('.')[-1].upper() in soundfile._formats:
    files = [args.input]

if len(files) == 0:
    raise Exception(f"The given input argument {args.input} is either not a valid directory/file path, or it doesn't contain any sound file of the supported formats (please refer to pysoundfile documentation)")

for ifile, filepath in enumerate(files):
    try:
        sig, fs = librosa.load(filepath, sr=int(torchcrepe.SAMPLE_RATE * args.compress))
    except:
        print(f'Failed to load {filepath}')
        continue

    generator = torchcrepe.core.preprocess(torch.tensor(sig).unsqueeze(0), torchcrepe.SAMPLE_RATE, \
        hop_length=int(args.step / args.compress * torchcrepe.SAMPLE_RATE), batch_size=batch_size, device=device)
    size = int(1 + (len(sig) - 1024) // (args.step * torchcrepe.SAMPLE_RATE))
    with torch.inference_mode():
        preds = torch.vstack([model(frames).cpu() for frames in tqdm.tqdm(generator, desc=f'{ifile}/{len(files)}: {filepath.split("/")[-1]}', total=size, leave=False)]).T.unsqueeze(0)
        f0 = (torchcrepe.core.postprocess(preds, decoder=decoder) * args.compress).squeeze()
    confidence = preds.max(axis=1)[0].squeeze()
    time = np.arange(0, len(sig)/fs + 1e-6, args.step)
    
    df = pd.DataFrame({'time':time, 'f0':f0, 'confidence':confidence})
    df.to_csv(filepath.rsplit('.',1)[0]+'_f0.csv', index=False)
    # Plot F0 predictions over spectrograms
    if args.print:
        plt.figure(figsize=(max(6.4, 6.4*time[-1]/2), 4.8))
        plt.specgram(sig, Fs=fs, NFFT=args.NFFT, noverlap=args.NFFT-args.NFFT//8)
        mask = confidence>args.threshold
        plt.scatter(time[mask], f0[mask], c=confidence[mask], s=5)
        plt.xlim(0, len(sig)/fs)
        plt.ylim(0, f0[mask].max() * 1.5)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filepath.rsplit('.',1)[0]+'_f0.png')
        plt.close()
