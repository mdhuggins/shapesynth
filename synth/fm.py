from common.audio import *
from generator import *
from envelope import *
import numpy as np


class FM(object):
    def __init__(self):
        super(FM, self).__init__()

        self.mod = NoteGenerator.square_wave_generator(84, 0.05)  # TODO
        self.mod_env = Envelope.magic_envelope(self.mod, 0.1)  # TODO
        self.carrier = ModulatedGenerator.sine_wave_generator(60, 0.3, self.mod_env)  # TODO
        self.carrier_env = Envelope.magic_envelope(self.carrier, 0.1)  # TODO

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
