import matplotlib.pyplot as plt
import argparse, os, tqdm
import torchcrepe, torch, librosa, soundfile
import pandas as pd, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('indir', type=str, help="Directory with sound files to process")
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

files = [fn for fn in os.listdir(args.indir) if fn.split('.')[-1].upper() in soundfile._formats]

for filename in tqdm.tqdm(files):
    try:
        sig, fs = librosa.load(os.path.join(args.indir, filename), sr=int(torchcrepe.SAMPLE_RATE * args.compress))
    except:
        print(f'Failed to load {filename}')
        continue
    
    generator = torchcrepe.core.preprocess(torch.tensor(sig).unsqueeze(0), torchcrepe.SAMPLE_RATE, \
        hop_length=int(args.step / args.compress * torchcrepe.SAMPLE_RATE), batch_size=batch_size, device=device)
    with torch.inference_mode():
        preds = torch.vstack([model(frames).cpu() for frames in generator]).T.unsqueeze(0)
        f0 = (torchcrepe.core.postprocess(preds, decoder=decoder) * args.compress).squeeze()
    confidence = preds.max(axis=1)[0].squeeze()
    time = np.arange(0, len(sig)/fs, args.step)
    
    df = pd.DataFrame({'time':time, 'f0':f0, 'confidence':confidence})
    df.to_csv(os.path.join(args.indir, filename.rsplit('.',1)[0]+'_f0.csv'), index=False)
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
        plt.savefig(os.path.join(args.indir, filename.rsplit('.',1)[0])+'_f0.png')
        plt.close()