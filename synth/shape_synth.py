
# common import
import sys
sys.path.append('..')

from common.mixer import *

from .noise import *
from .filter import *
from .fm import *
from .envelope import *
from .util import pitch_to_freq


class ShapeSynth(object):
    def __init__(self, x, y, gain):
        """ Create a new ShapeSynth.

        :param x: the relative x coordinate of the shape (float in range [0,1])
        :param y: the relative y coordinate of the shape (float in range [0,1])
        :param gain: in range [0,1]
        """
        super(ShapeSynth, self).__init__()

        self.x = x
        self.y = y
        self.gain = gain

    def make_note(self, pitch, velocity, duration):
        """ Creates a generator to play a note.

        :param pitch: MIDI
        :param velocity: in range [0,1]
        :param duration: in seconds
        :return: a generator
        """
        x = self.x
        y = self.y

        gain = self.gain * velocity

        # Carrier Params
        carrier_p = max(0, 1-y-(1-x)/4)
        carrier_env_params = Envelope.magic_envelope(carrier_p, duration=duration)
        carrier_gain = gain * ((y) * (1 - x)) ** (1 / 2.5)

        # Modulator Params
        modulator_env_params = carrier_env_params

        # FM Factory
        fm_fact = FMFactory(carrier_gain, 0, 1, 1, carrier_env_params, modulator_env_params)

        # Noise Params
        noise_p = max(0, 1 - x**2/3.5 - (1-y)*x/15)
        noise_env_params = Envelope.magic_envelope(noise_p)
        noise_gain = gain * 0.2 * (1-y)**2

        # Noise Generator
        noise = NoiseGenerator(noise_gain)
        noise = Envelope(noise, *noise_env_params)

        # Noise Filter
        f0 = pitch_to_freq(pitch)
        noise_cutoffs = [0.1 * f0, min(1.9 * f0 ** 1.3, Audio.sample_rate / 2 - 1)]
        noise = Filter(noise, 'bandpass', noise_cutoffs)

        # Combine with FM
        mixer = Mixer()
        mixer.add(fm_fact.create_fm(pitch))
        mixer.add(noise)

        return mixer
