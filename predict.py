import matplotlib.pyplot as plt
import argparse, os, tqdm
import torchcrepe, torch, librosa, soundfile
import pandas as pd, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help="Directory with sound files to process, or a single file to process")
parser.add_argument('--model_path', type=str, help="Path of model weights", default=os.path.join(os.path.dirname(__file__), 'model_all.pth'))
parser.add_argument('--compress', type=float, help="Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. \
    Frequencies are divided by the given factor by artificially changing the sampling rate (slowing down / speeding up the signal).", default=1)
parser.add_argument('--step', type=float, help="Step used between each prediction (in seconds)", default=256 / torchcrepe.SAMPLE_RATE)
parser.add_argument('--decoder', choices=['argmax', 'weighted_argmax', 'viterbi'], help="Decoder used to postprocess predictions", default='weighted_argmax')
parser.add_argument('--no_print', action='store_true', help="Skip printing spectrograms with overlaid F0 predictions to assess their quality")
parser.add_argument('--no_characterisation', action='store_true', help="Skip the computation of vocalisation characteristics (harmonicity, salience, and SHR)")
parser.add_argument('--threshold', type=float, help="Confidence threshold used when printing F0 predictions on spectrograms ", default=0.1)
parser.add_argument('--NFFT', type=int, help="Window size used for the spectrum computation (for printing F0 predictions and computing vocalisation characteristics)", default=1024)
args = parser.parse_args()

# Initialisations
device, batch_size = ('cuda', 64) if torch.cuda.is_available() else ('cpu', 1)
model = torchcrepe.Crepe('full').eval().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
decoder = torchcrepe.decode.__dict__[args.decoder]

FS, nfft = int(torchcrepe.SAMPLE_RATE * args.compress), args.NFFT
Hz2bin = lambda f: np.floor(f / FS * nfft).astype(int)
hann = np.hanning(nfft)
get_spectrum = lambda x : np.abs(np.fft.rfft(hann * x))

if os.path.isdir(args.input):
    files = [os.path.join(args.input, fn) for fn in os.listdir(args.input) if fn.split('.')[-1].upper() in soundfile._formats]
elif os.path.isfile(args.input) and args.input.split('.')[-1].upper() in soundfile._formats:
    files = [args.input]
else:
    files = []

if len(files) == 0:
    raise Exception(f"The given input argument {args.input} is either not a valid directory/file path, or it doesn't contain any sound file of the supported formats (please refer to pysoundfile documentation)")

print(f'With the current compression factor of {args.compress}, the model\'s F0 estimations range from {32*args.compress}Hz to {2000*args.compress}Hz')

for ifile, filepath in enumerate(files):
    try:
        sig, fs = librosa.load(filepath, sr=FS)
    except:
        print(f'Failed to load {filepath}')
        continue

    generator = torchcrepe.core.preprocess(torch.tensor(sig).unsqueeze(0), torchcrepe.SAMPLE_RATE, \
        hop_length=int(args.step * args.compress * torchcrepe.SAMPLE_RATE), batch_size=batch_size, device=device)
    size = int(1 + len(sig) // (args.step * args.compress * torchcrepe.SAMPLE_RATE)) // batch_size
    with torch.inference_mode():
        preds = torch.vstack([model(frames).cpu() for frames in tqdm.tqdm(generator, desc=f'{ifile+1}/{len(files)}: {filepath.split("/")[-1]}', total=size, leave=False)]).T.unsqueeze(0)
        f0s = (torchcrepe.core.postprocess(preds, decoder=decoder) * args.compress).squeeze()
    confidence = preds.max(axis=1)[0].squeeze()
    time = np.arange(0, len(sig), int(args.step * args.compress * torchcrepe.SAMPLE_RATE)) / fs
    
    df = pd.DataFrame({'time':time, 'f0':f0s, 'confidence':confidence})
    # Vocalisation characterisation (harmonicity, salience, SHR)
    if not args.no_characterisation:
        spectrums = np.vstack([get_spectrum(sig[t - nfft//2 : t + nfft//2]) for t in (df.time.sample(min(len(df), 100)) * fs).round().astype(int) if t > nfft/2 and t < len(sig)-nfft/2]).T
        mednoise, stdnoise = np.median(spectrums, axis=1), np.std(spectrums, axis=1)
        shr_ceil = min(fs/2, df.f0.max() * 5)
        for irow, row in tqdm.tqdm(df.iterrows(), desc=f'{ifile+1}/{len(files)}: {filepath.split("/")[-1]} - voc. characterisation', total=len(df), leave=False):
            if FS*row.time < nfft//2 or FS*row.time > len(sig) - nfft//2:
                continue
            spec = get_spectrum(sig[int(FS * row.time) - nfft//2 : int(FS * row.time) + nfft//2])
            spec = np.clip((spec-mednoise)/stdnoise, 1e-12, 1e3)
            f0 = row.f0
            
            df.loc[irow, 'harmonicity'] = sum(spec[Hz2bin(np.arange(f0*2, shr_ceil, f0))]) /  sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 *2 < fs / 2 else None
            df.loc[irow, 'salience'] = sum(spec[Hz2bin(f0*2**(-1/12)):Hz2bin(f0*2**(1/12))+1]) / sum(spec[Hz2bin(f0*2**(-6/12)):Hz2bin(f0*2**(6/12))+1])
            df.loc[irow, 'SHR'] = sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0)-f0/2)]) / sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 < fs/2 else None

    df.to_csv(filepath.rsplit('.',1)[0]+'_f0.csv', index=False)
    # Plot F0 predictions over spectrograms
    if not args.no_print and len(sig)/fs < 60:
        mask = confidence > args.threshold
        try:
            if mask.any():
                plt.figure(figsize=(6.4*time[-1]/3*args.compress, 4.8))
                plt.specgram(sig, Fs=fs, NFFT=nfft, noverlap=nfft-nfft//8, cmap='Greys')
                plt.scatter(time[mask], f0s[mask], c=confidence[mask], s=5)
                plt.xlim(0, len(sig)/fs)
                plt.ylim(0, f0s[mask].max() * 1.5)
                plt.colorbar(label="Confidence")
                plt.xlabel('Time (sec)')
                plt.ylabel('Frequency (Hz)')
                plt.tight_layout()
                plt.savefig(filepath.rsplit('.',1)[0]+'_f0.png')
                plt.close()
            else:
                print(f'With the chosen confidence threshold {args.threshold}, no section was detected as voiced')
        except:
            print(f'Failed to create figure for {filepath}, but results are still saved in the .csv table')
