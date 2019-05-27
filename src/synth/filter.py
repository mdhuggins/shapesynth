import numpy as np
from common.audio import *
from scipy.signal import lfilter, butter


class Filter(object):
    def __init__(self, generator, filter_type, freq):
        super(Filter, self).__init__()

        assert type(freq) in (int, float, list)
        assert filter_type in ['lowpass', 'highpass', 'bandpass']
        if filter_type == "bandpass":
            assert len(freq) == 2

        self.generator = generator
        self.filter_type = filter_type

        nyq = 0.5 * Audio.sample_rate
        cutoffs = freq / nyq if type(freq) is not list else [f/nyq for f in freq]
        self.b, self.a = butter(1, cutoffs, btype=filter_type)

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

        # Apply filter
        filtered = lfilter(self.b, self.a, output)

        # Save position so next call starts from where this one left off
        self.frame += num_frames
        return filtered, self.playing
