from random import random


import sys
sys.path.append('..')
from common.audio import *

import numpy as np


class Envelope(object):
    def __init__(self, generator, attack_time, n1, decay_time, sustain_time, sustain_gain, release_time, n2):
        """ Make a new enveloped generator.

        :param generator: the generator to apply the envelope to (Generator)
        :param attack_time: the duration of the attack, in seconds (float >= 0)
        :param n1: the steepness of the attack (float > 0)
        :param sustain_time: the duration of the decay, in seconds (float >= 0)
        :param sustain_time: the duration of the sustain, in seconds (float >= 0)
        :param sustain_time: the gain during the sustain, in range [0,1]
        :param release_time: the duration of the release, in seconds (float >= 0)
        :param n2: the steepness of the release (float > 0)
        """
        super(Envelope, self).__init__()

        # Validate args
        assert attack_time >= 0
        assert n1 > 0
        assert sustain_time >= 0
        assert release_time >= 0
        assert n2 > 0

        self.generator = generator
        self.envelope = self.make_envelope(attack_time, n1, decay_time, sustain_time, sustain_gain, release_time, n2)

        # TODO Improve (used for modulation)
        if hasattr(self.generator, 'freq'):
            self.freq = self.generator.freq
        else:
            self.freq = None

        # State information
        self.frame = 0
        self.playing = True

    @staticmethod
    def make_envelope(attack_time, n1, decay_time, sustain_time, sustain_gain, release_time, n2):
        """ Generate an envelope curve. Returns a numpy array, where the first
            value is zero, the maximum value is 1, and the final value is the
            end of the envelope at 0.

        :param attack_time: the duration of the attack, in seconds (float >= 0)
        :param n1: the steepness of the attack (float > 0)
        :param sustain_time: the duration of the decay, in seconds (float >= 0)
        :param sustain_time: the duration of the sustain, in seconds (float >= 0)
        :param sustain_time: the gain during the sustain, in range [0,1]
        :param release_time: the duration of the release, in seconds (float >= 0)
        :param n2: the steepness of the release (float > 0)
        :return: the envelope (np.array([float]))
        """
        # Validate args
        assert attack_time >= 0
        assert n1 > 0
        assert attack_time >= 0
        assert n2 > 0

        # Convert durations from seconds to frames
        fs = Audio.sample_rate
        attack_len = np.floor(attack_time * fs).astype('int')
        release_len = np.ceil(release_time * fs).astype('int')

        # Generate each piece
        attack_t = np.arange(attack_len)
        attack = (attack_t / attack_len) ** (1 / n1)

        decay = np.linspace(1, sustain_gain, int(decay_time*fs))

        sustain = np.ones(int(sustain_time * fs)) * sustain_gain

        release_t = np.arange(release_len + 1)  # Make sure last value is 0
        release = (1.0 - (release_t / release_len) ** (1 / n2)) * sustain_gain

        # Join attack and delay together to form complete envelope
        envelope = np.concatenate((attack, decay, sustain, release))

        return envelope

    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.generator.note_off()
        self.playing = False

    def generate(self, num_frames, num_channels):
        """ Generate frames.

        :param num_frames: The number of frames to generate (int >= 0)
        :param num_channels: The number of channels to use (must be 1) (int)
        :return: frames, whether the note is playing (np.array([float]), bool)
        """
        # Validate args
        assert num_channels == 1
        assert num_frames >= 0

        # Get generator output
        raw_frames, playing = self.generator.generate(num_frames, num_channels)

        # Stop if generator has stopped
        if not playing:
            self.playing = False
            return np.zeros(num_frames), self.playing

        # Pad envelope to length
        raw_env = self.envelope[self.frame:self.frame+num_frames]
        env_padding = np.zeros(num_frames - len(raw_env))
        env = np.concatenate((raw_env, env_padding))

        # Apply envelope to frames
        frames = raw_frames * env

        # If envelope is completely done, stop playing
        if len(env_padding) == num_frames:
            self.playing = False

        # Save position so next call starts from where this one left off
        self.frame += num_frames

        return frames, self.playing

    @staticmethod
    def magic_envelope(p, duration=0):
        """ Create envelope parameters from a single "percussive-ness"
            parameter. A higher p will result in an envelope with faster attack
            and release, and a lower p will result in a slightly longer attack,
            and much longer release.

        :param p: the "percussive-ness" parameter (float in range [0,1])
        :return: attack, attack slope, release, release slope
        """
        assert 0 <= p <= 1

        # Attack time
        min_attack = 0.01
        variable_attack = 0.2
        attack_rand = 0.01
        attack = min_attack + variable_attack * (1-p) + attack_rand * (2*random()-1)
        attack = max(min_attack, attack)

        # Attack slope
        min_attack_slope = 0.5
        variable_attack_slope = 0.5
        attack_slope_rand = 0.05
        attack_slope = min_attack_slope + variable_attack_slope * (1-p) + attack_slope_rand * (2*random()-1)
        attack_slope = max(min_attack_slope, attack_slope)

        # Decay time
        min_decay = 0.01
        variable_decay = 0.1
        decay_rand = 0.01
        decay = min_decay + variable_decay * (1-p) + decay_rand * (2*random()-1)
        decay = max(min_decay, decay)

        # Release time
        min_release = 0.1
        variable_release = 0.8
        release_rand = 0.1
        release = min_release + variable_release * (1-p) + release_rand * (2*random()-1)
        release = max(min_release, release)

        # Release slope
        min_release_slope = 1
        variable_release_slope = 1
        release_slope_rand = 0.1
        release_slope = min_release_slope + variable_release_slope * p + release_slope_rand * (2*random()-1)
        release_slope = max(min_release_slope, release_slope)

        # Sustain
        sustain = max(0, duration - attack - decay - release) * (1-p)

        min_sustain_gain = 0.1
        max_sustain_gain = 0.8
        variable_sustain_gain = 0.7
        sustain_gain_rand = 0.05
        sustain_gain = min_sustain_gain + variable_sustain_gain * (1-p) + sustain_gain_rand * (2*random()-1)
        sustain_gain = min(max(min_sustain_gain, sustain_gain), max_sustain_gain)

        return attack, attack_slope, decay, sustain, sustain_gain, release, release_slope


# import matplotlib.pyplot as plt
#
# for pp in range(10):
#     p = pp/10
#     print("p",p)
#     params = Envelope.magic_envelope(p, 2)
#     plt.plot(Envelope(None,*params).envelope)
#     plt.show()