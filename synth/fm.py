from common.audio import *
from generator import *
from envelope import *
import numpy as np


class FM(object):
    def __init__(self,
                 pitch,
                 carrier_gain, modulator_gain,
                 carrier_freq_ratio, modulator_freq_ratio,
                 carrier_env_params, modulator_env_params):
        super(FM, self).__init__()

        assert 0 < carrier_gain <= 1
        # assert 0 < modulator_gain <= 1
        assert 1 <= carrier_freq_ratio
        assert 1 <= modulator_freq_ratio

        self.mod = NoteGenerator.sine_wave_generator(pitch+12*(modulator_freq_ratio-1), modulator_gain)
        self.mod_env = Envelope(self.mod, *modulator_env_params)
        self.carrier = ModulatedGenerator.sine_wave_generator(pitch+12*(carrier_freq_ratio-1), carrier_gain, self.mod_env)  # TODO
        self.carrier_env = Envelope(self.carrier, *carrier_env_params)

        # State information
        self.playing = True

    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.carrier_env.note_off()

    def generate(self, num_frames, num_channels):
        frames, playing = self.carrier_env.generate(num_frames, num_channels)
        self.playing = playing
        return frames, self.playing


class FMFactory(object):
    def __init__(self,
                 carrier_gain, modulator_gain,
                 carrier_freq_ratio, modulator_freq_ratio,
                 carrier_env_params, modulator_env_params):
        super(FMFactory, self).__init__()
        self.carrier_gain = carrier_gain
        self.modulator_gain = modulator_gain
        self.carrier_freq_ratio = carrier_freq_ratio
        self.modulator_freq_ratio = modulator_freq_ratio
        self.carrier_env_params = carrier_env_params
        self.modulator_env_params = modulator_env_params

    def create_fm(self, pitch):
        return FM(pitch,
                  self.carrier_gain, self.modulator_gain,
                  self.carrier_freq_ratio, self.modulator_freq_ratio,
                  self.carrier_env_params, self.modulator_env_params)
