import numpy as np
from common.audio import *


class Reverb(object):
    def __init__(self, generator):
        super(Reverb, self).__init__()

        self.generator = generator

        # self.fx = (AudioEffectsChain().reverb(wet_gain=10))

        self.delays = [(4410*(i**2), (i+1)**2) for i in range(10)]
        self.buffer_len = 4410*10**2
        self.buffer = np.zeros(self.buffer_len)

        # State information
        self.frame = 0
        self.playing = True

    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.playing = False

    def generate(self, num_frames, num_channels):
        """ Generate frames.

        :param num_frames: The number of frames to generate (int >= 0)
        :param num_channels: The number of channels to use (must be 1) (int)
        :return: frames, whether the note is playing (np.array([float]), bool)
        """
        # Validate args
        assert num_channels == 1
        assert type(num_frames) is int and num_frames >= 0

        # If not playing, return zeros
        if not self.playing:
            return np.zeros(num_frames), False

        # Generate frames
        output, playing = self.generator.generate(num_frames, num_channels)
        self.playing = playing

        # Add delays
        for df, denom in self.delays:
            self.buffer[df:df+num_frames] += output / denom

        # Add buffer to current output
        output += self.buffer[:num_frames]

        # Advance buffer
        self.buffer = np.concatenate((self.buffer[num_frames:], np.zeros(num_frames)))

        # Save position so next call starts from where this one left off
        self.frame += num_frames
        return output, self.playing
