species = {
    'canids':{
        'wavpath': 'data/wolves/*/*.wav',
        'FS': 16000,
        'nfft': 1024,
        'downsample':1,
        'step': 1/8
    },
    'dolphins':{
        'wavpath': 'data/dclmmpa2011/cut_no_overlap/*.wav',
        'FS': 192000,
        'nfft':1536, # according to silbido paper, 8ms windows and 125Hz resolution
        'downsample':20,
        'step': 1/8
    },
    'spotted_hyenas':{
        'wavpath': 'data/hyenas/lehmann hyena whoop traces/*.wav',
        'FS': 8000,
        'nfft': 2048,
        'downsample':1,
        'step': 1/8
    },
    'orangutans':{
        'wavpath': 'data/orangs/*.wav',
        'FS': 44100,
        'nfft':2048,
        'downsample':1,
        'step': 1/8
    },
    'rodents':{
        'wavpath': 'data/LiuLabData/cut_no_overlap/*.wav',
        'FS': 250000,
        'nfft':512,
        'downsample':50,
        'step': 1/8
    },
    'lions':{
        'wavpath': 'data/Lion Roar Data/Biologger Roars Expanded (500ms)/*.WAV',
        'FS':16000,
        'nfft':2048, # was zero-padded to 4 times 2048 in the original
        'downsample':0.5,
        'step': 1/8
    },
    'monk_parakeets':{
        'wavpath': 'data/monk parakeet/pre-processed_calls/*.WAV',
        'FS':44100,
        'nfft':512,
        'downsample':3,
        'step': 1/16
    },
    'La_Palma_chaffinches':{
        'wavpath': 'data/FCPalmae/cut/*.wav',
        'FS':44100,
        'nfft':1024,
        'downsample':5,
        'step': 1/16
    },
    'little_owls':{
        'wavpath': 'data/little_owl/cut/*.wav',
        'FS':4000,
        'nfft':512,
        'downsample':1,
        'step': 1/8
    },
    'Reunion_grey_white_eyes':{
        'wavpath': 'data/white_eye/cut/*.wav',
        'FS':44100,
        'nfft':1024,
        'downsample':5,
        'step': 1/16
    },
    'long-billed_hermits':{
        'wavpath':'data/marcelo/long_billed_hermit_songs/*.wav',
        'FS':44100,
        'nfft':512,
        'downsample':5,
        'step': 1/16
    },
    'hummingbirds':{
        'wavpath':'data/marcelo/hummingbird_songs/*.wav',
        'FS':44100,
        'nfft':512,
        'downsample':5,
        'step':1/16
    },
    'disk-winged_bats':{
        'wavpath':'data/marcelo/spixs_disc_winged_bat_*/*.wav',
        'FS':400000,
        'nfft':512,
        'downsample':20,
        'step':1/16
    },
    'bottlenose_dolphins':{
        'wavpath':'data/sayigh/*/*.wav',
        'FS':96000,
        'nfft':1024,
        'downsample':20,
        'step':1/8
    },
#    'spider_monkeys':{
#        'wavpath':'data/spider_monkeys/audio/*wav',
#        'FS':44100,
#        'nfft':4096,
 #        'downsample':1,
 #       'step':1/8
#    }
}
