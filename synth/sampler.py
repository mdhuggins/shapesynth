from common.audio import *
from common.writer import *
from common.wavesrc import WaveFile

import numpy as np

import time
from .pool import *

# TODO Duration

# If True, use a synchronous pool instead of a multiprocessed one
DEBUG_SAMPLER = False

# Helpers

def stft(x, fft_len, hop_size, zp_factor=1):
    padded_x = np.concatenate((np.zeros(int(hop_size / 2)), x))

    result = []

    i = 0
    while i < len(padded_x):
        window = padded_x[i:i + fft_len]
        window = np.hanning(len(window)) * window  # Apply hanning
        window = np.concatenate(
            (window, np.zeros(fft_len - len(window))))  # Pad if too short
        window = np.concatenate((window, np.zeros(fft_len*(zp_factor-1))))  # Pad  # TODO

        fft = np.fft.rfft(window)#[:int(np.ceil(1 + fft_len / 2))]
        result.append(fft)

        i += hop_size

    return np.column_stack(result)


def apply_in_window(x, x_h, position, centered=True):
    """Adds x_h to the given position in x, centering if applicable."""
    if centered:
        if position - len(x_h) // 2 < 0:
            x[:position + len(x_h) // 2] += x_h[len(x_h) // 2 - position:]
        else:
            x[position - len(x_h) // 2: position + len(x_h) // 2] += x_h[:min(
                len(x_h), len(x) - position + len(x_h) // 2)]
    else:
        x[position: position + len(x_h)] += x_h[
                                            :min(len(x_h), len(x) - position)]


def istft(X, hop_size, zp_factor=1, centered=True):
    N = (X.shape[0] - 1) * 2 // zp_factor
    x = np.zeros((hop_size * (X.shape[1] - 1) + N,))
    for col in range(X.shape[1]):
        x_h = np.fft.irfft(X[:, col])
        x_h = x_h[:len(x_h) // zp_factor]
        apply_in_window(x, x_h, col * hop_size, centered)
    W = np.zeros_like(x)
    for h in range(0, W.shape[0], hop_size):
        sub_w = np.hanning(N)
        apply_in_window(W, sub_w, h, centered)

    W = np.where(np.abs(W) < 0.001, 0.001, W)
    return x/ W

def pitch_to_freq(pitch):
    """ Convert MIDI pitch to frequency.

    :param pitch: MIDI pitch (float)
    :return: frequency, in Hz (float)
    """
    return 440 * 2 ** ((pitch - 69) / 12)

def freq_to_bin(freq, fft_len, fs):
    return fft_len * freq / fs



wav_glock = WaveFile("./samples/11084__angstrom__e2.wav").get_frames(0, 441000)
wav_piano = WaveFile("./samples/piano-e6.wav").get_frames(0, 441000)
wav_bass = WaveFile("./samples/bass-g3.wav").get_frames(0, 441000)  # Cut to 10 seconds
wav_cello = WaveFile("./samples/cello-48.wav").get_frames(0, 441000)

# Pad waves
pad = max([len(wav_glock),len(wav_bass),len(wav_piano), len(wav_cello)])

wav_glock = np.concatenate((wav_glock, np.zeros(pad-len(wav_glock))))
wav_piano = np.concatenate((wav_piano, np.zeros(pad-len(wav_piano))))
wav_bass = np.concatenate((wav_bass, np.zeros(pad-len(wav_bass))))
wav_cello = np.concatenate((wav_cello, np.zeros(pad-len(wav_cello))))

fft_len = 1024 * 6
hop_size = 512 * 6
bin_radius = 4
zp_factor = 1

# Get spectra
spectra_piano = stft(wav_piano, fft_len, hop_size, zp_factor=zp_factor)
spectra_glock = stft(wav_glock, fft_len, hop_size, zp_factor=zp_factor)
spectra_bass = stft(wav_bass, fft_len, hop_size, zp_factor=zp_factor)
spectra_cello = stft(wav_cello, fft_len, hop_size, zp_factor=zp_factor)

# Bins to extract
f0_bin = freq_to_bin(pitch_to_freq(88), fft_len, Audio.sample_rate)
bins = [round(f0_bin*i) for i in range(1,9)]

f0_bin_bass = freq_to_bin(pitch_to_freq(55), fft_len, Audio.sample_rate)
bins_bass = [round(f0_bin_bass * i) for i in range(1, 9)]

f0_bin_cello = freq_to_bin(pitch_to_freq(48), fft_len, Audio.sample_rate)
bins_cello = [round(f0_bin_cello * i) for i in range(1, 9)]

bin_spectra_piano = [spectra_piano[b - bin_radius:b + bin_radius + 1,:] for b in bins]
bin_spectra_glock = [spectra_glock[b - bin_radius:b + bin_radius + 1,:] for b in bins]
bin_spectra_bass = [spectra_bass[b - bin_radius:b + bin_radius + 1,:] for b in bins_bass]
bin_spectra_cello = [spectra_cello[b - bin_radius:b + bin_radius + 1,:] for b in bins_cello]

samplers = {}

def sampler_worker(requester, params):
    sampler_params, pitch, gain = params

    if sampler_params not in samplers:
        samplers[sampler_params] = Sampler(*sampler_params)

    sampler = samplers[sampler_params]
    gen = sampler.play_note(pitch, gain)
    return gen.frames

class SamplerManager(object):
    """
    Manages a pool of workers that render waveforms using Samplers.
    """

    pool = None

    @classmethod
    def initialize(cls):
        if DEBUG_SAMPLER:
            cls.pool = SynchronousPool(20 * Audio.sample_rate, sampler_worker)
        else:
            cls.pool = SharedArrayPool(20 * Audio.sample_rate, sampler_worker)

    @classmethod
    def stop_workers(cls):
        cls.pool.stop()

    @classmethod
    def on_update(cls):
        cls.pool.on_update()

class Sampler(object):
    """
    Class that works in a background thread to generate waveforms based on
    combinations of sampled spectra.
    """

    def __init__(self, gain, coords):
        super(Sampler, self).__init__()

        self.gain = gain

        self._coords = coords
        self.mix = [0,0,0,0]  # Bass, Cello, Piano, Glock
        self.swell_factor = .001  # What percentage of the envelope is attack?
        self.duration = 0
        self.update_mix()

        self.fft_len = fft_len
        self.hop_size = hop_size

        self.spectra_piano = spectra_piano
        self.spectra_glock = spectra_glock
        self.spectra_bass = spectra_bass
        self.spectra_cello = spectra_cello

        self.bin_spectra_piano = bin_spectra_piano
        self.bin_spectra_glock = bin_spectra_glock
        self.bin_spectra_bass = bin_spectra_bass
        self.bin_spectra_cello = bin_spectra_cello

        self.cache = {}

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = coords
        self.update_mix()
        self.cache = {}

    def update_mix(self):
        x, y = self.coords
        bass = 1/(x**2 + y**2)
        cello = 1/(x**2 + (1-y)**2)
        piano = 1/((1-x)**2 + (1-y)**2)
        glock = 1/((1-x)**2 + y**2)

        norm = bass + cello + piano + glock

        # Swell
        self.swell_factor = (y + (1-x))/4 if x < 0.5 < y else .001

        # Duration
        self.duration = 2*(bass) + 10*(cello) + 3*(piano) + 2*(glock)
        self.duration /= (bass) + (cello) + (piano) + (glock)
        print("duration", self.duration)

        self.mix = [bass/norm, cello/norm, piano/norm, glock/norm]
        print("mix", self.mix)

    def play_note(self, pitch, gain):

        # if pitch in self.cache:
            # return SpectraGenerator(self.gain * gain, gen_spectra,
            #                         self.hop_size, env)

        # Calculate new bins
        new_f0_bin = freq_to_bin(pitch_to_freq(pitch), self.fft_len, Audio.sample_rate)
        new_bins = [int(round(new_f0_bin * i)) for i in range(1, 9)]

        # Assemble new spectra
        gen_spectra = np.zeros_like(self.spectra_piano)

        for i in range(len(new_bins)):
            for mix_coef, spectrum in zip(self.mix, [self.bin_spectra_bass[i], self.bin_spectra_cello[i], self.bin_spectra_piano[i], self.bin_spectra_glock[i]]):
                if new_bins[i] < bin_radius: # Too low
                    gen_spectra[:new_bins[i] + bin_radius + 1, :] += spectrum[:len(spectrum) - (bin_radius - new_bins[i])] * mix_coef
                elif new_bins[i] + bin_radius + 1 >= len(gen_spectra): # Too high
                    gen_spectra[new_bins[i] - bin_radius:, :] += spectrum[len(spectrum) - (new_bins[i] + bin_radius + 1 - len(gen_spectra)):] * mix_coef
                else: # Just right
                    gen_spectra[new_bins[i] - bin_radius:new_bins[i] + bin_radius + 1, :] += spectrum * mix_coef

        swell_frames = int(self.duration*self.swell_factor*Audio.sample_rate)
        release_frames = Audio.sample_rate
        sustain_frames = int(self.duration*Audio.sample_rate) - release_frames - swell_frames

        env = np.concatenate((np.linspace(0,1,swell_frames), np.ones(sustain_frames), np.linspace(1,0, release_frames)))

        # Crop spectra
        gen_spectra = gen_spectra[:,:1+np.ceil(len(env)/self.hop_size).astype('int')]

        # Re-synthesize
        # frames = istft(gen_spectra, self.hop_size)


        # Apply envelope
        # env = np.concatenate((env, np.zeros(len(frames)-len(env))))
        # frames *= env

        # self.cache[pitch] = gen_spectra  #frames

        #hash = (self.duration, self.coords, pitch)
        return SpectraGenerator(self.gain * gain, gen_spectra, self.hop_size, env)

class SpectraGenerator(object):
    def __init__(self, gain, spectra, hop_size, env):
        self.gain = gain
        self.spectra = spectra
        self.hop_size = hop_size
        self.env = env

        self.frames = istft(self.spectra, self.hop_size, zp_factor=zp_factor)

        # Apply envelope
        env = np.concatenate((self.env, np.zeros(len(self.frames)-len(self.env))))
        self.frames *= env

        # State information
        self.frame = 0  # Keep angle continuous between generate calls
        self.playing = True

    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.playing = False

    def generate(self, num_frames, num_channels) :
        # Get frames
        output = self.frames[self.frame: self.frame + num_frames]

        # Check for end of buffer
        actual_num_frames = len(output) // num_channels

        self.frame += actual_num_frames

        # Pad if output is too short
        padding = num_frames * num_channels - len(output)
        if padding > 0:
            output = np.append(output, np.zeros(padding))

        # return
        return output * self.gain, actual_num_frames == num_frames
