from common.audio import *
from common.wavesrc import WaveFile

import numpy as np

# TODO Duration

class Sampler(object):
    def __init__(self, gain, coords):
        super(Sampler, self).__init__()

        self.gain = gain

        self._coords = coords
        self.mix = [0,0,0,0]  # Bass, Cello, Piano, Glock
        self.duration = 0
        self.update_mix()

        self.wav_glock = WaveFile("./samples/11084__angstrom__e2.wav").get_frames(0, 441000)
        self.wav_piano = WaveFile("./samples/piano-e6.wav").get_frames(0, 441000)
        self.wav_bass = WaveFile("./samples/bass-g3.wav").get_frames(0, 441000)  # Cut to 10 seconds
        self.wav_cello = WaveFile("./samples/cello-48.wav").get_frames(0, 441000)

        # Pad waves
        pad = max([len(self.wav_glock),len(self.wav_bass),len(self.wav_piano), len(self.wav_cello)])

        self.wav_glock = np.concatenate((self.wav_glock, np.zeros(pad-len(self.wav_glock))))
        self.wav_piano = np.concatenate((self.wav_piano, np.zeros(pad-len(self.wav_piano))))
        self.wav_bass = np.concatenate((self.wav_bass, np.zeros(pad-len(self.wav_bass))))
        self.wav_cello = np.concatenate((self.wav_cello, np.zeros(pad-len(self.wav_cello))))

        self.fft_len = 1024 * 8
        self.hop_size = 512 * 8

        # Get spectra
        self.spectra_piano = stft(self.wav_piano, self.fft_len, self.hop_size)
        self.spectra_glock = stft(self.wav_glock, self.fft_len, self.hop_size)
        self.spectra_bass = stft(self.wav_bass, self.fft_len, self.hop_size)
        self.spectra_cello = stft(self.wav_cello, self.fft_len, self.hop_size)

        # Bins to extract
        f0_bin = freq_to_bin(pitch_to_freq(88), self.fft_len, Audio.sample_rate)
        bins = [round(f0_bin*i) for i in range(1,9)]

        f0_bin_bass = freq_to_bin(pitch_to_freq(55), self.fft_len, Audio.sample_rate)
        bins_bass = [round(f0_bin_bass * i) for i in range(1, 9)]

        f0_bin_cello = freq_to_bin(pitch_to_freq(48), self.fft_len, Audio.sample_rate)
        bins_cello = [round(f0_bin_cello * i) for i in range(1, 9)]

        self.bin_spectra_piano = [self.spectra_piano[b-2:b+3,:] for b in bins]
        self.bin_spectra_glock = [self.spectra_glock[b-2:b+3,:] for b in bins]
        self.bin_spectra_bass = [self.spectra_bass[b-2:b+3,:] for b in bins_bass]
        self.bin_spectra_cello = [self.spectra_cello[b-2:b+3,:] for b in bins_cello]

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
        self.duration = 3*(1/bass) + 10*(1/cello) + 6*(1/piano) + 5*(1/glock)
        self.duration /= (1/bass) + (1/cello) + (1/piano) + (1/glock)
        print("duration", self.duration)

        self.mix = [bass/norm, cello/norm, piano/norm, glock/norm]
        print("mix", self.mix)

    def play_note(self, pitch, gain):

        if pitch in self.cache:
            return FramesGenerator(self.gain * gain, self.cache[pitch])

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

        # Re-synthesize
        frames = istft(gen_spectra, self.hop_size)

        # Apply envelope
        env = np.concatenate((np.ones(int(round(self.duration-1)*Audio.sample_rate)), np.linspace(1,0,Audio.sample_rate)))
        env = np.concatenate((env, np.zeros(len(frames)-len(env))))

        frames *= env

        self.cache[pitch] = frames

        return FramesGenerator(self.gain*gain, frames)


class FramesGenerator(object):
    def __init__(self, gain, frames):
        self.gain = gain
        self.frames = frames

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
    return x / W


def pitch_to_freq(pitch):
    """ Convert MIDI pitch to frequency.

    :param pitch: MIDI pitch (float)
    :return: frequency, in Hz (float)
    """
    return 440 * 2 ** ((pitch - 69) / 12)

def freq_to_bin(freq, fft_len, fs):
    return fft_len * freq / fs