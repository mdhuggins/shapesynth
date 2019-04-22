
# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.synth import *
from common.clock import *
from common.mixer import *

import numpy as np

from .noise import *
from .filter import *
import time
from .fm import *
from .envelope import *

from .util import pitch_to_freq


class ShapeSynth(object):
    def __init__(self, x, y, gain):
        """

        :param x: in range [0,1]
        :param y: in range [0,1]
        :param gain: in range [0,1]
        """
        super(ShapeSynth, self).__init__()

        self.x = x
        self.y = y
        self.gain = gain


    def make_note(self, pitch, velocity, duration):
        """

        :param pitch: MIDI
        :param velocity: in range [0,1]
        :param duration: seconds
        :return: a generator
        """
        x = self.x
        y = self.y

        ce = Envelope.magic_envelope(max(0, 1 - y - (1 - x) / 4))
        me = ce

        gain = self.gain * velocity

        cgain = gain * ((y) * (1 - x)) ** (1 / 2.5)

        fm_fact = FMFactory(cgain, 0, 1, 1, ce, me)

        ne = Envelope.magic_envelope(
            max(0, 1 - x ** 2 / 3.5 - (1 - y) * x / 15))

        noise = NoiseGenerator(gain * 0.2 * (1 - y) ** 2)
        noise = Envelope(noise, *ne)
        f0 = pitch_to_freq(pitch)


        noise = Filter(noise, 'bandpass', [0.1 * f0, min(1.9 * f0 ** 1.3,
                                                         Audio.sample_rate / 2 - 1)])

        mixer = Mixer()
        mixer.add(fm_fact.create_fm(pitch))
        mixer.add(noise)

        return mixer
