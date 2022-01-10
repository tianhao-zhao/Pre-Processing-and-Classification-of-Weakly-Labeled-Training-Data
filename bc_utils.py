from os.path import join
import numpy as np
import random
import os

import random

random.seed(2021)
np.random.seed(2021)


# Standardize audio
def change_audio_rate(audio_fname, src_dir, new_audio_rate, dest_dir=None):
    '''If the desired file doesn't exist, calls ffmpeg to change the sample rate
    of an audio file.
    eg : change_audio_rate('audio.wav', '/tmp/', 16000)

    Parameters
    ----------
    audio_fname : str
        name of the audio file
    src_dir : str
        Directory where the audio is stored
    new_audio_rate : int
        Desired sample rate
    dest_dir : str
        Destination directory (defaults to : src_dir + new_audio_rate)
    '''
    import subprocess
    if dest_dir is None:
        dest_dir = os.path.join(src_dir, str(new_audio_rate))
    wav_path_orig = os.path.join(src_dir, audio_fname)
    wav_path_dest = os.path.join(dest_dir, audio_fname)

    if not os.path.isfile(wav_path_dest):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        cmd = 'ffmpeg -i {} -ar {} -b:a 16k -ac 1 {}'.format(
            wav_path_orig,
            new_audio_rate,
            wav_path_dest)
        subprocess.call(cmd, shell=True)


# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


def noiseAugment(opt):
    data_path = opt.data
    npz_path = join(data_path, 'noise', 'wav{}.npz'.format(opt.fs // 1000))
    dataset = dict(np.load(npz_path).items())
    train, valid = dataset['train'][0], dataset['valid'][0]
    valid = (valid / np.percentile(train, 95)).clip(-1, 1)
    train = (train / np.percentile(train, 95)).clip(-1, 1)

    def f(is_train, audio_len):
        ds = train if is_train else valid
        ds = ds.astype(np.float32)
        rand_idx = np.random.randint(0, len(ds) - audio_len - 1)
        return ds[rand_idx: rand_idx + audio_len]

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound


def kl_divergence(y, t):
    '''Wont work with keras'''
    entropy = - F.sum(t[t.data.nonzero()] * F.log(t[t.data.nonzero()]))
    crossEntropy = - F.sum(t * F.log_softmax(y))

    return (crossEntropy - entropy) / y.shape[0]


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line


