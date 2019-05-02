from common.audio import *
from common.wavesrc import WaveFile

import numpy as np

import time
from threading import Thread

# TODO Duration


# Helpers

def stft(x, fft_len, hop_size):
    padded_x = np.concatenate((np.zeros(int(hop_size / 2)), x))

    result = []

    i = 0
    while i < len(padded_x):
        window = padded_x[i:i + fft_len]
        window = np.hanning(len(window)) * window  # Apply hanning
        window = np.concatenate(
            (window, np.zeros(fft_len - len(window))))  # Pad if too short
        # window = np.concatenate((window, np.zeros(fft_len*3)))  # Pad  # TODO

        fft = np.fft.fft(window)[:int(np.ceil(1 + fft_len / 2))]
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


def istft(X, hop_size, centered=True):
    N = (X.shape[0] - 1) * 2
    x = np.zeros((hop_size * (X.shape[1] - 1) + N,))
    for col in range(X.shape[1]):
        x_h = np.fft.irfft(X[:, col])
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

# Get spectra
spectra_piano = stft(wav_piano, fft_len, hop_size)
spectra_glock = stft(wav_glock, fft_len, hop_size)
spectra_bass = stft(wav_bass, fft_len, hop_size)
spectra_cello = stft(wav_cello, fft_len, hop_size)

# Bins to extract
f0_bin = freq_to_bin(pitch_to_freq(88), fft_len, Audio.sample_rate)
bins = [round(f0_bin*i) for i in range(1,9)]

f0_bin_bass = freq_to_bin(pitch_to_freq(55), fft_len, Audio.sample_rate)
bins_bass = [round(f0_bin_bass * i) for i in range(1, 9)]

f0_bin_cello = freq_to_bin(pitch_to_freq(48), fft_len, Audio.sample_rate)
bins_cello = [round(f0_bin_cello * i) for i in range(1, 9)]

bin_spectra_piano = [spectra_piano[b-2:b+3,:] for b in bins]
bin_spectra_glock = [spectra_glock[b-2:b+3,:] for b in bins]
bin_spectra_bass = [spectra_bass[b-2:b+3,:] for b in bins_bass]
bin_spectra_cello = [spectra_cello[b-2:b+3,:] for b in bins_cello]

class Sampler(object):
    def __init__(self, gain, coords):
        super(Sampler, self).__init__()

        self.gain = gain

        self._coords = coords
        self.mix = [0,0,0,0]  # Bass, Cello, Piano, Glock
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
            # TODO What if bins are too low?
            gen_spectra[new_bins[i] - 2:new_bins[i] + 3, :] += self.bin_spectra_bass[i] * self.mix[0]
            gen_spectra[new_bins[i] - 2:new_bins[i] + 3, :] += self.bin_spectra_cello[i] * self.mix[1]
            gen_spectra[new_bins[i] - 2:new_bins[i] + 3, :] += self.bin_spectra_piano[i] * self.mix[2]
            gen_spectra[new_bins[i] - 2:new_bins[i] + 3, :] += self.bin_spectra_glock[i] * self.mix[3]


        env = np.concatenate((np.ones(int(round(self.duration-1)*Audio.sample_rate)), np.linspace(1,0,Audio.sample_rate)))

        # Crop spectra
        gen_spectra = gen_spectra[:,:1+np.ceil(len(env)/self.hop_size).astype('int')]

        # Re-synthesize
        # frames = istft(gen_spectra, self.hop_size)


        # Apply envelope
        # env = np.concatenate((env, np.zeros(len(frames)-len(env))))
        # frames *= env

        # self.cache[pitch] = gen_spectra  #frames

        return BufferedSpectraGenerator(self.gain * gain, gen_spectra, self.hop_size, env)


class BufferedSpectraGenerator(object):
    def __init__(self, gain, spectra, hop_size, env):
        self.gain = gain
        self.spectra = spectra
        self.hop_size = hop_size
        self.env = env

        # hops = 1
        # self.frames = istft(self.spectra[:, :hops], self.hop_size)
        # self.frames = np.fft.irfft(self.spectra[:,0])#:self.hops+hops])
        # self.frames /= np.hanning(len(self.frames))


        # hann = np.hanning(len(self.frames))
        #
        # self.frames *= hann
        # self.frames = self.frames[:len(self.frames) // 2]

        # TODO Apply envelope
        # self.env = np.concatenate((self.env, np.zeros(len(self.frames)-len(self.env))))

        # State information
        self.frame = 0  # Keep angle continuous between generate calls
        self.hops = 0
        self.playing = True

        self.N = (self.spectra.shape[0] - 1) * 2
        self.x = np.zeros((hop_size * (self.spectra.shape[1] - 1) + self.N,))

        self.rendered_frame = self.hop_size//2

        W = np.zeros_like(self.x)
        for h in range(0, W.shape[0], hop_size):
            sub_w = np.hanning(self.N)
            apply_in_window(W, sub_w, h, True)

        W = np.where(np.abs(W) < 0.001, 0.001, W)
        self.W = W

        self.istft_step_fn(5)

        self.t = None

    def istft_step_fn(self, hops=10):
        for col in range(self.hops, min(self.spectra.shape[1], self.hops+hops)):
            x_h = np.fft.irfft(self.spectra[:, col])
            apply_in_window(self.x, x_h, col * self.hop_size, True)

        self.frames = self.x / self.W
        self.hops += hops
        self.rendered_frame += self.hop_size*hops

        # import matplotlib.pyplot as plt
        # plt.plot(self.frames)
        # plt.show()

    def istft_step(self):
        self.istft_step_fn()
        # def f(s):
        #     return s.istft_step_fn
        # t = Thread(target=f(self))
        # t.start()
        # self.t = t


    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.playing = False

    def generate(self, num_frames, num_channels) :
        # Add to buffer
        if self.rendered_frame - self.frame < 2048 and (not self.t or not self.t.isAlive()):  # TODO
            hops = 1 #np.ceil(num_frames/self.hop_size).astype('int')

            self.istft_step()


        # Get frames
        # output = np.random.random(num_frames)
        output = self.frames[self.frame:self.frame+num_frames]

        # TODO apply envelope

        # Check for end of buffer
        actual_num_frames = len(output) // num_channels

        self.frame += actual_num_frames

        # Pad if output is too short
        padding = num_frames * num_channels - len(output)
        if padding > 0:
            output = np.append(output, np.zeros(padding))

        # return
        return output * self.gain, self.frame <= len(self.env)

class SpectraGenerator(object):
    def __init__(self, gain, spectra, hop_size, env):
        self.gain = gain
        self.spectra = spectra
        self.hop_size = hop_size
        self.env = env

        self.frames = istft(self.spectra, self.hop_size)

        # Apply envelope
        env = np.concatenate((self.env, np.zeros(len(self.frames)-len(self.env))))
        self.frames *= env
        import matplotlib.pyplot as plt
        plt.plot(self.frames)
        plt.show()
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


