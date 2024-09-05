from metadata import species
import pandas as pd, numpy as np, os, argparse, librosa, parselmouth, mir_eval
from glob import glob
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crepe, pesto, torchcrepe, torch, tensorflow as tf, basic_pitch.inference, basic_pitch.constants

device, batch_size = ('cuda', 64) if torch.cuda.is_available() else ('cpu', 1)

# LOAD MODELS
basic_pitch_model = tf.saved_model.load(str(basic_pitch.ICASSP_2022_MODEL_PATH))

tcrepe_model = torchcrepe.Crepe('full').eval().to(device)
tcrepe_model.load_state_dict(torch.load('/home/paul.best/.local/lib/python3.9/site-packages/torchcrepe/assets/full.pth', map_location='cuda'))

cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

def run_tcrepe(model, sig, fs, dt, viterbi=False):
    generator = torchcrepe.core.preprocess(torch.tensor(sig).unsqueeze(0), fs, hop_length=dt*fs if fs != torchcrepe.SAMPLE_RATE else int(dt*fs),\
                                           batch_size=batch_size, device=device, pad=False)
    with torch.no_grad():
        preds = np.vstack([model(frames).cpu().numpy() for frames in generator])
    if viterbi:
        f0 = 10 * 2 ** (crepe.core.to_viterbi_cents(preds) / 1200)
    else:
        f0 = 10 * 2 ** (crepe.core.to_local_average_cents(preds) / 1200)
    confidence = np.max(preds, axis=1)
    time = np.arange(torchcrepe.WINDOW_SIZE/2, len(sig)/fs*torchcrepe.SAMPLE_RATE - torchcrepe.WINDOW_SIZE/2 + 1e-9, dt*torchcrepe.SAMPLE_RATE) / torchcrepe.SAMPLE_RATE
    return time, f0, confidence

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('specie', type=str, help="Species to treat specifically", default=None)
parser.add_argument('--overwrite', type=bool, help="Overwrite previous pedictions", default=False)
parser.add_argument('--quick', type=bool, help="Skip pyin and crepe to make things quick", default=False)
parser.add_argument('--split', type=int, help="Section to test on between 0 and 4", default=None)
args = parser.parse_args()

algos = ['praat_f0','pyin_f0','crepe_f0','tcrepe_f0','tcrepe_ft_f0','tcrepe_ftsp_f0','tcrepe_ftoth_f0', 'basic_f0','pesto_f0', 'pesto_ft_f0', 'pesto_ftoth_f0']
quick_algos = ['praat_f0','pyin_f0','tcrepe_f0','tcrepe_ft_f0','tcrepe_ftsp_f0','tcrepe_ftspV_f0','tcrepe_ftoth_f0', 'basic_f0','pesto_f0', 'pesto_ft_f0']

if args.overwrite:
    print('Overwriting previous results')

# Iterate over species, then files, then run each algorithm and save the predictions
for specie in species if args.specie =='all' else args.specie.split(' '):
    wavpath, FS, nfft, downsample, step = species[specie].values()
    dt = round(nfft * step / FS * downsample, 3) # winsize / 8
    # Load species specific pre-trained models
    tcrepe_ftoth_model, tcrepe_ftsp_model = None, None
    if os.path.isfile(f'crepe_weights/model_only-{args.split}_{specie}.pth'):
        tcrepe_ftsp_model = torchcrepe.Crepe('full').eval().to(device)
        tcrepe_ftsp_model.load_state_dict(torch.load(f'crepe_weights/model_only-{args.split}_{specie}.pth', map_location=device))
    if os.path.isfile(f'crepe_weights/model_omit_{specie}.pth'):
        tcrepe_ftoth_model = torchcrepe.Crepe('full').eval().to(device)
        tcrepe_ftoth_model.load_state_dict(torch.load(f'crepe_weights/model_omit_{specie}.pth', map_location=device))
    # initialise the file list to iterate on
    fns = glob(wavpath)
    if type(args.split) == int:
        fns = fns[int(len(fns)/5*args.split) : int(len(fns)/5*(args.split+1))]
    # iterate over files
    for fn in tqdm(fns, desc=specie):
        if args.overwrite or not os.path.isfile(f'{fn[:-4]}_preds.csv') or os.path.getsize(f'{fn[:-4]}_preds.csv') < 300:
            # load original annotation file
            annot = pd.read_csv(f'{fn[:-4]}.csv').drop_duplicates(subset='Time')
            # add a 0 at starts and ends for large gaps to avoid interpolating between vocalisations
            med_diff = annot.Time.diff().median()
            rgaps, lgaps = annot.Time[annot.Time.diff() > med_diff*4], annot.Time[annot.Time.diff(-1) < - med_diff * 4]
            annot = pd.concat([annot, pd.DataFrame({'Time':np.concatenate([lgaps+med_diff, rgaps-med_diff]), 'Freq':[0]*(len(lgaps)+len(rgaps))})]).sort_values('Time')
            # load the waveform and create the dataframe for storing predictions
            sig, fs = librosa.load(fn, sr=FS)
            out = pd.DataFrame({'time':np.arange(nfft/fs/2, (len(sig) - nfft/2)/fs, dt / downsample)})
            mask = ((out.time > annot.Time.min())&(out.time < annot.Time.max()))
            out.loc[mask, 'annot'] = mir_eval.melody.resample_melody_series(annot.Time, annot.Freq, annot.Freq>0, out[mask].time, verbose=False)[0]
        else:
            out = pd.read_csv(f'{fn[:-4]}_preds.csv')
            for algo in algos: # drop a column if all values are None
                if algo in out.columns and out[algo].isna().all():
                    out.drop(algo, axis=1, inplace=True)

            # check if everything has already been computed, and if yes skip the file
            if pd.Series(algos).isin(out.columns).all() or (args.quick and pd.Series(quick_algos).isin(out.columns).all()):
               continue
            sig, fs = librosa.load(fn, sr=FS)

        out.time *= downsample
        fs /= downsample

        if not 'praat_f0' in out.columns: # PRAAT
            sndpitches0 = parselmouth.Sound(sig, fs).to_pitch_ac(pitch_floor=27.5, pitch_ceiling=fs//2, voicing_threshold=0.20, time_step=dt)
            time, f0, confidence = sndpitches0.xs(), sndpitches0.selected_array['frequency'], sndpitches0.selected_array['strength']
            mask = ((out.time>=time[0])&(out.time<=time[-1]))
            out.loc[mask, 'praat_f0'], out.loc[mask, 'praat_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out.loc[mask, 'time'])
            out.praat_f0 *= downsample

        if not 'pyin_f0' in out.columns: # PYIN
            f0, voiced, prob = librosa.pyin(sig, sr=fs, fmin=27.5, fmax=fs//2, frame_length=nfft, hop_length=int(fs*dt), center=False)
            out['pyin_f0'], out['pyin_conf'] = f0[:len(out)], prob[:len(out)]
            out.pyin_f0 *= downsample

        if not 'tcrepe_f0' in out.columns: # torch crepe out-of-the-box
            time, f0, confidence = run_tcrepe(tcrepe_model, sig, fs, dt)
            mask = ((out.time > time[0])&(out.time < time[-1]))
            out.loc[mask, 'tcrepe_f0'], out.loc[mask, 'tcrepe_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out[mask].time)
            out.tcrepe_f0 *= downsample

        if not 'tcrepe_ftsp_f0' in out.columns and tcrepe_ftsp_model: # torch crepe finetuned on the target species
            time, f0, confidence = run_tcrepe(tcrepe_ftsp_model, sig, fs, dt)
            mask = ((out.time > time[0])&(out.time < time[-1]))
            out.loc[mask, 'tcrepe_ftsp_f0'], out.loc[mask, 'tcrepe_ftsp_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out[mask].time)
            out.tcrepe_ftsp_f0 *= downsample

        if not 'tcrepe_ftspV_f0' in out.columns and tcrepe_ftsp_model: # torch crepe finetuned on the target species
            time, f0, confidence = run_tcrepe(tcrepe_ftsp_model, sig, fs, dt, viterbi=True)
            mask = ((out.time > time[0])&(out.time < time[-1]))
            out.loc[mask, 'tcrepe_ftspV_f0'], out.loc[mask, 'tcrepe_ftspV_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out[mask].time)
            out.tcrepe_ftspV_f0 *= downsample

        if not 'tcrepe_ftoth_f0' in out.columns and tcrepe_ftoth_model: # torch crepe finetuned on other species than the target
            time, f0, confidence = run_tcrepe(tcrepe_ftoth_model, sig, fs, dt)
            mask = ((out.time > time[0])&(out.time < time[-1]))
            out.loc[mask, 'tcrepe_ftoth_f0'], out.loc[mask, 'tcrepe_ftoth_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out[mask].time)
            out.tcrepe_ftoth_f0 *= downsample

        if not args.quick and not 'crepe_f0' in out.columns: # CREPE out-of-the-box tensorflow
            time, f0, confidence, activation = crepe.predict(sig, fs, step_size=int(dt*1e3), center=False, verbose=0) # step_size in ms
            out['crepe_f0'], out['crepe_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out.time)
            out.crepe_f0 *= downsample

        if not 'basic_f0' in out.columns: # basic_pitch
            S = basic_pitch.inference.run_inference(fn, basic_pitch_model)['contour']
            time = np.arange(len(S)) * basic_pitch.constants.FFT_HOP / basic_pitch.constants.AUDIO_SAMPLE_RATE
            f0 = basic_pitch.constants.FREQ_BINS_CONTOURS[np.argmax(S, axis=1)]
            confidence = np.max(S, axis=1)
            out['basic_f0'], out['basic_conf'] = mir_eval.melody.resample_melody_series(time, f0, confidence, out.time)
            out.basic_f0 *= downsample

        if not 'pesto_f0' in out.columns: # pesto out-of-the-box
            try:
                time, f0, confidence, activation = pesto.predict(torch.tensor(sig).unsqueeze(0), fs, step_size=int(dt*1e3), convert_to_freq=True) # step_size in ms
                out['pesto_f0'], out['pesto_conf'] = mir_eval.melody.resample_melody_series(time/1000, f0[0], confidence.numpy(), out.time, verbose=False)
                out.pesto_f0 *= downsample
            except:
                out['pesto_f0'], out['pesto_conf'] = None, None

        if not 'pesto_ft_f0' in out.columns and os.path.isfile(f'pesto_ft/{specie}.pth'): # pesto finetuned
            try:
                time, f0, confidence, activation = pesto.predict(torch.tensor(sig).unsqueeze(0), fs, model_name=f'pesto_ft/{specie}.pth', step_size=int(dt*1e3), convert_to_freq=True) # step_size in ms
                out['pesto_ft_f0'], out['pesto_ft_conf'] = mir_eval.melody.resample_melody_series(time/1000 + 1e-6, f0[0], confidence.numpy(), out.time, verbose=False)
                out.pesto_ft_f0 *= downsample
            except Exception as inst:
                out['pesto_ft_f0'], out['pesto_ft_conf'] = None, None

        out.time /= downsample
        out.to_csv(f'{fn[:-4]}_preds.csv', index=False)
