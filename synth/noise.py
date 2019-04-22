import numpy as np


class NoiseGenerator(object):
    def __init__(self, gain):
        """ Make a new note generator.

        :param gain: the amplitude of the output (float >=0)
        """
        super(NoiseGenerator, self).__init__()

        # Validate args
        assert type(gain) in (float, int) and gain >= 0

        # Setup generator
        self.gain = gain

        # State information
        self.frame = 0  # Keep angle continuous between generate calls
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
        output = np.random.rand(num_frames)
        output *= self.gain

        # Save position so next call starts from where this one left off
        self.frame += num_frames
        return output, self.playing