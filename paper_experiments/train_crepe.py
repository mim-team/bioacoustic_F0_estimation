import glob, tqdm, os, os, argparse, mir_eval
from metadata import species
import pandas as pd, numpy as np, librosa, resampy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch, torchcrepe
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--omit', type=str, help="Species to rule out of the training set", default=None)
parser.add_argument('--only', type=str, help="Train only on the given species", default=None)
parser.add_argument('--split', type=int, help="Portion out of between 0 and 4 to use as test set", default=0)
args = parser.parse_args()

suffix = "omit_"+args.omit if args.omit else f"only-{args.split}_"+args.only if args.only else "all"

writer = SummaryWriter('runs/'+suffix)
model = torchcrepe.Crepe('full')
model.load_state_dict(torch.load('/home/paul.best/.local/lib/python3.9/site-packages/torchcrepe/assets/full.pth', map_location='cuda'))
model = model.train().to('cuda')

FS, n_in, f0 = 16000, 1024, 10
norm = lambda s : (s-np.mean(s))/max(1e-8, np.std(s))

if not os.path.isfile(f'crepe_ft/train_set_{suffix}.pkl'):
    df = []
    for specie in ([args.only] if args.only else set(species)-{args.omit} if args.omit else species):
        wavpath, fs, nfft, downsample, step = species[specie].values()
        dt = int(n_in * step) # winsize / 8
        files = glob.glob(wavpath)
        if args.only:
            files = files[:int(len(files)/5*args.split)] + files[int(len(files)/5*(args.split+1)):]
        for fn in tqdm.tqdm(pd.Series(files).sample(min(len(files), 1000)), desc='Peparing dataset for '+specie):
            annot = pd.read_csv(fn[:-4]+'.csv').drop_duplicates(subset='Time')
            sig, fs = librosa.load(fn, sr=None)
            sig = resampy.resample(sig, fs//downsample, FS)
            annot.Time, annot.Freq = annot.Time * downsample, annot.Freq / downsample
            out = pd.DataFrame({'time':np.arange(n_in//2, len(sig)-n_in//2, dt)/FS,
                                'sig':[norm(sig[t-n_in//2:t+n_in//2]) for t in np.arange(n_in//2, len(sig)-n_in//2, dt)]})
            out['fn'] = fn
            mask = ((out.time > annot.Time.min())&(out.time < annot.Time.max()))
            out.loc[mask, 'label_Hz'] = mir_eval.melody.resample_melody_series(annot.Time, annot.Freq, annot.Freq>0, out.loc[mask, 'time'], verbose=False)[0]
            df.append(out)
    df = pd.concat(df, ignore_index=True)
    df.drop(df[(( df.label_Hz != 0 )&( df.label_Hz < 32.7 )&( df.label_Hz > 1975.5 ))].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['specie'] = df.fn.str.split('/').str[0]
    df.to_pickle(f'crepe_ft/train_set_{suffix}.pkl')
else:
    df = pd.read_pickle(f'crepe_ft/train_set_{suffix}.pkl')

weights = torch.ones(len(df))
for s, grp in df.groupby('specie'):
    weights[grp.index] = len(df) / len(grp)
sampler = torch.utils.data.WeightedRandomSampler(weights, len(df), replacement=True, generator=None)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super(Dataset).__init__()
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        f = self.df.iloc[i].label_Hz
        if f == 0 or np.isnan(f):
            label = torch.zeros(360, dtype=torch.float32)
        else:
            fbin = (1200*np.log2(f/f0) - 1997.3794084376191) / torchcrepe.CENTS_PER_BIN
            label = torch.exp(-(torch.arange(360, dtype=torch.float32)-fbin)**2 / (2*1.25**2)) # Gaussian blur with std == 1.25 (in out_bins, equivalent to 25 cents)
        return i, self.df.iloc[i].sig, label

loader = torch.utils.data.DataLoader(Dataset(df), batch_size=32, sampler=sampler, num_workers=12) # add shuffle=True if sampler is removed
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
n_iter, minloss, minloss_iter = 0, 1, 0

for epoch in range(100):
    for idxs, x, label in tqdm.tqdm(loader, desc=str(epoch), leave=False):
        optimizer.zero_grad()
        pred = model(x.to('cuda')).cpu()
        score = loss(pred, label)
        score.backward()
        optimizer.step()
        writer.add_scalar('loss', score.item(), n_iter)
        writer.add_scalar('acc', torch.mean((torch.argmax(pred, axis=1)==torch.argmax(label, axis=1)).float()), n_iter)
        if score.item() < minloss:
            minloss = score.item()
            minloss_iter = n_iter
        if n_iter > minloss_iter + 32 * 500:
            print('early stop')
            torch.save(model.state_dict(), f'crepe_ft/model_{suffix}.pth')
            exit()
        n_iter += 1
    torch.save(model.state_dict(), f'crepe_ft/model_{suffix}.pth')
